# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle

# import pickle5 as pickle

import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import numpy

import tensorflow as tf

# ### HOOD ###
# hood = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Capuchon/saved_model/3th ROUND CAPUCHON - 5000 images - 2.500 keep/svm_coef_dlatent.pkl'
#
# ### BUTTON ###
# buttons = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Button/saved_model/3th ROUND BUTTON- 5.000 images - 2.500 keep/svm_coef_dlatent.pkl'
#
# ### BLAZER ###
# blazer = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Blazer/saved_model/3th ROUND BLAZER - 5.000 images - 2.500 keep/svm_coef_dlatent.pkl'
#
# ### LEATHER JACKET ###
# leather = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Leather jacket/saved_model/3th ROUND LEATHER JACKET - 5.000 images - 2.500 keep/svm_coef_dlatent.pkl'
#
# ### JEANS JACKET ###
# jeans = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Jeans jacket/saved_model/3th ROUND JEANS JACKET - 5.000 images - 2.500 keep/svm_coef_dlatent.pkl'

###BLAZER####
# save_image_path_latent = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Blazer/saved_model/3th ROUND BLAZER - 5.000 images - 2.500 keep/latent_image.pkl'
#
# save_image_path_dlatent = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Blazer/saved_model/3th ROUND BLAZER - 5.000 images - 2.500 keep/dlatent_image.pkl'

###HOOD###
# save_image_path_latent = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Capuchon/saved_model/3th ROUND CAPUCHON - 5000 images - 2.500 keep/latent_image.pkl'
#
# save_image_path_dlatent = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Capuchon/saved_model/3th ROUND CAPUCHON - 5000 images - 2.500 keep/dlatent_image.pkl'

###BUTTONS###
# save_image_path_latent = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Button/saved_model/3th ROUND BUTTON- 5.000 images - 2.500 keep/latent_image.pkl'
#
# save_image_path_dlatent = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Closures/Closure - Button/saved_model/3th ROUND BUTTON- 5.000 images - 2.500 keep/dlatent_image.pkl'

###LEATHER###
# save_image_path_latent = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Leather jacket/saved_model/3th ROUND LEATHER JACKET - 5.000 images - 2.500 keep/latent_image.pkl'
#
# save_image_path_dlatent = '//drive/Algemeen/Interns/Mirle Willems/CLASSIFIER DATA/Style/Style - Leather jacket/saved_model/3th ROUND LEATHER JACKET - 5.000 images - 2.500 keep/dlatent_image.pkl'

#changeall is False pakt die de 'layers' values om te veranderen met coef
changeall = True
layers = [2, 3, 4]


def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    url = os.path.abspath('results/00009-sgan-custom_dataset_3-1gpu/network-snapshot-006566.pkl ')
    with open(url, 'rb') as f:
        _G, _D, Gs = pickle.load(f)

    # Print network details.
    Gs.print_layers()

    # Open attribute vectors
    with open(hood, 'rb') as handle:
        hood_attribute_vector = pickle.load(handle)

    with open(buttons, 'rb') as handle:
        buttons_attribute_vector = pickle.load(handle)

    with open(blazer, 'rb') as handle:
        blazer_attribute_vector = pickle.load(handle)

    with open(leather, 'rb') as handle:
        leather_attribute_vector = pickle.load(handle)

    with open(jeans, 'rb') as handle:
        jeans_attribute_vector = pickle.load(handle)

    with open(save_image_path_latent, 'rb') as handle:
        latent_initial_image = pickle.load(handle)

    with open(save_image_path_dlatent, 'rb') as handle:
        dlatent_initial_image = pickle.load(handle)


    #Check if dlatent is same as latent converted into dlatent
    check_dlatent = latent_initial_image[np.newaxis, :]
    check_dlatent = Gs.components.mapping.get_output_for(check_dlatent, None,
                                                     is_validation=True)  # latents, None, is_validation=True
    check_dlatent = check_dlatent.eval()
    check_dlatent = check_dlatent[0, 0, :]

    assert np.all(np.round(dlatent_initial_image, 3) == np.round(check_dlatent, 3))


    #1.  ### WITH ALL LAYERS  AND WITH SOME SAME LAYERS DLATENTS###
    results_dlatents = []
    dlatent_initial_image = dlatent_initial_image[np.newaxis, :]
    for i in range(0, 5, 1):
        i = i * 0.13
        dlatent_first = dlatent_initial_image + (i * buttons_attribute_vector)
        print('here', dlatent_first[0][0])
        results_dlatents.append(dlatent_first)

    dlatent_first = results_dlatents[-1] #when -,1
    print(dlatent_first[0][0])

    for i in range(0, 5, 1):
        print(i)
        i = i * 0.23
        dlatent_second = dlatent_first + (i * hood_attribute_vector)
        print(dlatent_second[0][0])
        results_dlatents.append(dlatent_second)


    # Generate images dlatents
    for i, scaled_dlatent in enumerate(results_dlatents):

        if changeall:
            scaled_dlatent = np.tile(scaled_dlatent, (16, 1))
            scaled_dlatent = scaled_dlatent[np.newaxis, :]
        else:
            dlatent = np.tile(dlatent_initial_image, (16, 1)) #dlatent_initial_image

            for j in layers: #wanneer je 0,1, 2, 3, 4, 5, ... 16 invoert verander je alles
                dlatent[j,:] = scaled_dlatent
            dlatent = dlatent[np.newaxis, :]
            scaled_dlatent = dlatent

        synthesis_kwargs = dict(
            output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
            minibatch_size=1)
        images = Gs.components.synthesis.run(scaled_dlatent, randomize_noise=False, **synthesis_kwargs)

        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, f'images/Dlatent try/Capuchon_{i}.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


    #2. ### WITH SOME DIFFERNT LAYERS DLATENTS###
    changeall = False
    first_layers = [2, 3, 4] #2, 3, 4
    second_layers = [13, 14, 15]

    results_dlatents = []
    dlatent_initial_image = dlatent_initial_image[np.newaxis, :]
    for i in range(0, 5, 1):
        i = i * 0.28
        dlatent_first = dlatent_initial_image + (i * blazer_attribute_vector)
        results_dlatents.append(dlatent_first)

    first_w_altered = results_dlatents[-1]

    results_dlatents = []
    for i in range(0, 5, 1):
        i = i * 0.15
        dlatent_first = dlatent_initial_image + (i * jeans_attribute_vector)
        results_dlatents.append(dlatent_first)

    second_w_altered = results_dlatents[-1]

    w_altered = []
    w_altered.append(first_w_altered)
    w_altered.append(second_w_altered)


    # Generate images dlatents
    for i, scaled_dlatent in enumerate(w_altered):

        if changeall:
            scaled_dlatent = np.tile(scaled_dlatent, (16, 1))
            scaled_dlatent = scaled_dlatent[np.newaxis, :]
        else:
            dlatent = np.tile(dlatent_initial_image, (16, 1)) #dlatent_initial_image

            for j in first_layers: #wanneer je 0,1, 2, 3, 4, 5, ... 16 invoert verander je alles
                dlatent[j,:] = first_w_altered

            for j in second_layers: #wanneer je 0,1, 2, 3, 4, 5, ... 16 invoert verander je alles
                dlatent[j,:] = second_w_altered

            dlatent = dlatent[np.newaxis, :]
            scaled_dlatent = dlatent

        synthesis_kwargs = dict(
            output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
            minibatch_size=1)
        images = Gs.components.synthesis.run(scaled_dlatent, randomize_noise=False, **synthesis_kwargs)

        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, f'images/Dlatent try/Capuchon_{i}.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)




    #3. ### WITH ALL LAYERS LATENTS ###
    # Create a list with the initial latent vector - and - some attribute vectors of choice
    results_latents = []
    latent_initial_image = latent_initial_image[np.newaxis, :]
    for i in range(-4, 0, 1):  # -19, 20
        i = i * 0.54
        latent = latent_initial_image + (i * blazer_attribute_vector)
        results_latents.append(latent)

    # Generate images latents
    for i, scaled_latent in enumerate(results_latents):

        scaled_dlatent = Gs.components.mapping.get_output_for(scaled_latent, None,
                                                              is_validation=True)  # latents, None, is_validation=True
        scaled_dlatent = scaled_dlatent.eval()

        synthesis_kwargs = dict(
            output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
            minibatch_size=1)
        images = Gs.components.synthesis.run(scaled_dlatent, randomize_noise=False, **synthesis_kwargs)

        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, f'images/Latent try/Capuchon_{i}.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


if __name__ == "__main__":
    main()
