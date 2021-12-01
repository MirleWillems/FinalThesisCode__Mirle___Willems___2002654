# Final Thesis Code - Mirle Willems - 2002654

This repository exists of the following elements: (If applicable, it is indicated where the code originated from or which code or video/article was taken as base/inspiration)

## Attribute classifiers:
This folder contains code for training all Attribute classifiers. These classifiers are divided into three main attributes: Hood, Closure, Length and Style. Moreover, it contains, per main attribute, the code to split the dataset into train, test and valiation set. 

The code to train the classifiers is based on https://www.youtube.com/watch?v=1Gbcp66yYX4&t=139s 

## Data
This folder contains subsets of the original PTTRNS.ai dataset (zalando_dataset_filtered_2 (sample)) and the scraped images dataset (zalando_dataset_resized_total (sample))


## Linear SVMs
This folder contains code to fit all Linear SVMs, calculate the conditional entropy and plot the t-SNE plot, divided per attribute. 

The code to calculate the conditional entropy score originated from the StyleGAN repository.


## Splitting data - One vs Rest approach
This folder contains the code to split the dataset of n-ary attributes into binary ones and includes the csv files with the images and corresponding labels.


## StyleGAN respository
This folder contains all scripts originated from the StyleGAN repository that has been altered or added to perform this research.

### attribute_vector_images.py
to perform (controlled) single attribute manipulation

### multiple_attribute_manipulation.py 
to perform (controlled) multiple attribute manipulation

### run_latents_linear_separability.py 
generate images and perform pseudo-labelling

### pretrained_example.py 
generate pseudo-labelled images to investigate

#### Scripts that were minimally changed to train our StyleGAN
train.py, run_metrics.py, metric_base.py

#### Scripts originating from the original StyleGAN repository that were alted for our research
pretrained_example.py, run_latents_linear_separability.py

#### Script that are written for our research with pieces of codes originating from the original StyleGAN repository
attribute_vector_images.py, multiple_attribute_manipulation.py

#### Article used to train StyleGAN with own dataset
https://evigio.com/post/how-to-use-custom-datasets-with-stylegan-tensorFlow-implementation


## Attribute Vectors Correlation Matrix
Contains the code to create the correlation matrix of the attribute vectors.


## Exploration PIM Dataset
Contains the code that explored the PIM dataset.


## Resize and Reshape scraped images Zalando
Contains the code to preprocess the scraped images of Zalando.
