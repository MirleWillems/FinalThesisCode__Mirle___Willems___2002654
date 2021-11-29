# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import numpy

import tensorflow as tf

save_images_path = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Jeans jacket/saved_model/Jeans_jacket_z_w_classified.pkl_pruned_dlatents_results.pkl'

        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Capuchon/saved_model/Capuchon_z_w_classified.pkl_pruned_dlatents_results.pkl'

        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Length/Length - Above hip length/saved_model/Above_Hip_Length_z_w_classified.pkl_pruned_dlatents_results.pkl'
        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Length/Length - Hip length/saved_model/Hip_Length_z_w_classified.pkl_pruned_dlatents_results.pkl'
        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Length/Length - Below hip length/saved_model/Below_Hip_Length_z_w_classified.pkl_pruned_dlatents_results.pkl'

        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Button/saved_model/Closure_Button_z_w_classified.pkl_pruned_dlatents_results.pkl'
        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Zipper/saved_model/Closure_Zipper_z_w_classified.pkl_pruned_dlatents_results.pkl'
        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Hidden zipper/saved_model/Closure_Hidden_Zipper_z_w_classified.pkl_pruned_dlatents_results.pkl'
        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Open front/saved_model/Closure_Open_front_z_w_classified.pkl_pruned_dlatents_results.pkl'
        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Waistband/saved_model/Closure_Waistband_z_w_classified.pkl_pruned_dlatents_results.pkl'
        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - None closed/saved_model/None_Closed_z_w_classified.pkl_pruned_dlatents_results.pkl'
        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Other/saved_model/Other_z_w_classified.pkl_pruned_dlatents_results.pkl'

        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Jeans jacket/saved_model/Jeans_jacket_z_w_classified.pkl_pruned_dlatents_results.pkl'
        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Blazer/saved_model/Blazer_z_w_classified.pkl_pruned_dlatents_results.pkl'
        #'//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Leather jacket/saved_model/Leather_jacket_z_w_classified.pkl_pruned_dlatents_results.pkl'


def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    url = os.path.abspath('results/00009-sgan-custom_dataset_3-1gpu/network-snapshot-006566.pkl ')
    with open(url, 'rb') as f:
        _G, _D, Gs = pickle.load(f)

    # Print network details.
    Gs.print_layers()


    #Inladen van dlatents (gesorteerd pruned).
    with open(save_images_path, 'rb') as handle:
        results = pickle.load(handle)

    #Load in data
    results_latents = results['latents']
    latents_labels = results['label']


    for i, latent in enumerate(results_latents):

        label = latents_labels[i]

        # #FROM DLATENT
        # dlatent = Gs.components.mapping.get_output_for(latent, None, is_validation=True)  # latents, None, is_validation=True
        # dlatent = dlatent.eval()
        #
        # synthesis_kwargs = dict(
        #     output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        #     minibatch_size=1)
        # images = Gs.components.synthesis.run(dlatent, randomize_noise=False, **synthesis_kwargs)

        #FROM LATENT
        latent = np.array([latent])

        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latent, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, f'images/Button try/Jeans_jacket_{label}_index_{i}.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()
