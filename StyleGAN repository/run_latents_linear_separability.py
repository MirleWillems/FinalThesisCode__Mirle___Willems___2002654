# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Linear Separability (LS)."""
import pickle
from collections import defaultdict
import numpy as np
import sklearn.svm
from sklearn import metrics
import tensorflow as tf
import dnnlib.tflib as tflib
from tensorflow import keras
from joblib import dump, load

from metrics import metric_base
from training import misc

import PIL


tf.keras.backend.set_image_data_format('channels_first')

#----------------------------------------------------------------------------

classifier_urls = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Jeans jacket/saved_model/ResNet_Jeans_jacket_5.h5'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Capuchon/saved_model/Capuchon_4.h5'

    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Length/Length - Above hip length/saved_model/ResNet_Length_Above_hip_length_4.h5'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Length/Length - Hip length/saved_model/ResNet_Length_Hip_length_5.h5'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Length/Length - Below hip length/saved_model/ResNet_Length_Below_hip_length_3.h5'

    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Button/saved_model/ResNet_Closure_Button_4.h5'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Zipper/saved_model/ResNet_Closure_Zipper_2.h5'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Hidden zipper/saved_model/ResNet_Closure_Hidden_zipper_2.h5'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Open front/saved_model/ResNet_Closure_Open_front_2.h5'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Waistband/saved_model/ResNet_Closure_Waistband_3.h5'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - None closed/saved_model/ResNet_Closure_None_closed_4.h5'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Other/saved_model/ResNet_Closure_Other_2.h5'

    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Jeans jacket/saved_model/ResNet_Jeans_jacket_5.h5'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Blazer/saved_model/ResNet_Blazer_4.h5'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Leather jacket/saved_model/ResNet_Leather_jacket_1.h5'

save_images_path = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Jeans jacket/saved_model/Jeans_jacket_z_w_classified.pkl'

    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Capuchon/saved_model/Capuchon_z_w_classified.pkl'

    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Length/Length - Above hip length/saved_model/Above_Hip_Length_z_w_classified.pkl'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Length/Length - Hip length/saved_model/Hip_Length_z_w_classified.pkl'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Length/Length - Below hip length/saved_model/Below_Hip_Length_z_w_classified.pkl'

    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Button/saved_model/Closure_Button_z_w_classified.pkl'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Zipper/saved_model/Closure_Zipper_z_w_classified.pkl'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Hidden zipper/saved_model/Closure_Hidden_Zipper_z_w_classified.pkl'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Open front/saved_model/Closure_Open_Front_z_w_classified.pkl'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Waistband/saved_model/Closure_Waistband_z_w_classified.pkl'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - None closed/saved_model/None_Closed_z_w_classified.pkl'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Other/saved_model/Other_z_w_classified.pkl'

    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Jeans jacket/saved_model/Jeans_jacket_z_w_classified.pkl'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Blazer/saved_model/Blazer_z_w_classified.pkl'
    #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Leather jacket/saved_model/Leather_jacket_z_w_classified.pkl'

#----------------------------------------------------------------------------

class LS(metric_base.MetricBase):
    def __init__(self, num_samples, num_keep, attrib_indices, minibatch_per_gpu, **kwargs):
        assert num_keep <= num_samples
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.num_keep = num_keep
        self.attrib_indices = attrib_indices
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        results = []
        #gpu_idx = 0
        attrib_idx = 0
        i=0
        results_dic = {}

        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):

                #Load in generator and classifier
                Gs_clone = Gs.clone()
                classifier = tf.keras.models.load_model(classifier_urls)


                print('IT IS USING THE RIGHT CODE')

                for _ in range(0, self.num_samples, minibatch_size):

                    # Generate images
                    #Generate random latent as numpy arrays
                    latents = np.random.normal(size=[self.minibatch_per_gpu] + Gs_clone.input_shape[1:])  # tf.random.normal

                    #Feed them through the mapping network and store results in a array (eval)
                    dlatents = Gs.components.mapping.get_output_for(latents, None, is_validation=True) #latents, None, is_validation=True
                    dlatents = dlatents.eval()

                    #Generate images using the synthesis.run
                    synthesis_kwargs = dict(
                        output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
                        minibatch_size=self.minibatch_per_gpu)
                    images = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)

                    #Transpose the images to change shape (batch, 512, 512, 3) to (batch, 3, 512, 512)
                    images = np.transpose(images, (0, 3, 1, 2))

                    #Make the numpy.ndarray into a tensor to make the values from uint8 into float32
                    images = tf.convert_to_tensor(images, dtype=tf.float32)

                    #preprocessing step: tensorflow.keras.applications.resnet50.preprocess_input
                    images = tf.keras.applications.resnet50.preprocess_input(images)


                    #Pseudo-label images
                    result_dict = dict(latents=latents, dlatents=dlatents[:,-1])
                    for attrib_idx in self.attrib_indices:

                        predictions = classifier.predict(images, steps=1)

                        #Output softmax layer e.g. [9.8694420e-01 1.3055802e-02]

                        result_dict[attrib_idx] = predictions
                    results.append(result_dict)

                    print(i)
                    i+=1

                print('Images made = DONE')

                #van lijst met dics naar dic van lijsten
                results = {key: np.concatenate([value[key] for value in results], axis=0) for key in results[0].keys()}

                #TO PICKLE - latents, dlatents and sigmoid outcome
                with open(save_images_path, 'wb') as handle:
                    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # Calculate conditional entropy for each attribute.
                conditional_entropies = defaultdict(list)
                for attrib_idx in self.attrib_indices:
                    # Prune the least confident samples.
                    pruned_indices = list(range(self.num_samples))
                    pruned_indices = sorted(pruned_indices, key=lambda i: -np.max(results[attrib_idx][i]))
                    pruned_indices = pruned_indices[:self.num_keep]

                    #print('PRUNED', pruned_indices)
                    print('Pruned indices = DONE')

                    #Convert sigmoid output layer to 0 or 1
                    svm_targets = np.argmax(results[attrib_idx][pruned_indices], axis=1)

                    # Fit SVM to the remaining samples.
                    for space in ['latents', 'dlatents']:
                        svm_inputs = results[space][pruned_indices]

                        if space == 'dlatents':
                            results_dic['dlatents'] = svm_inputs
                            results_dic['label'] = svm_targets
                            #print(results_dic)
                            # TO CSV or PICKLE
                            with open(f'{save_images_path}_pruned_dlatents_results.pkl', 'wb') as handle:
                                pickle.dump(results_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        if space == 'latents':
                            results_dic['latents'] = svm_inputs
                            results_dic['label'] = svm_targets
                            #print(results_dic)
                            # TO CSV or PICKLE
                            with open(f'{save_images_path}_pruned_latents_results.pkl', 'wb') as handle:
                                pickle.dump(results_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

#----------------------------------------------------------------------------
