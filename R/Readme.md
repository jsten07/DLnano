## Analysis of Nano Satellite Imagery of Areas Affected from War Destruction with Deep Learning Methods 
### File Description

#### This folder contains the different code files we used to calculate our results. Here we want to give you a short overview what purpose the files fulfill:


- model.R
  - In this file you can inspect the construction of a model. We saved the model structure in this file to use it for the other files.

- pixelwise_classification.R
  - This file performs a pixelwise classification with JPG images. It prepares the data, i.e. adds the mask to the images. Then it augments the data and makes it ready to use for the training. 
The script also performs the training and then plots the results.


- Run_classification_as_tif.R
   - This file allows to run a pixelwise classification with TIF files. It also performs the preparation, augmentation, training and plotting. The functions used to work with TIF files 
are also used for the blockwise detection.
 
- preparation_blockwise.R
   - Here we prepare the TIF for the blockwise training. It splits the complete TIF in different tiles with the wanted number of pixels. 
It also proves whether a war destruction happened in this tile and sorts it into the appropriate folders (true / false).


- augmentation.R
   - This file takes a stack of normalized tiles and augments them. So it turns the images and changes spectral information like the brightness. It is used by the following training files. 


- imagewise_model_training.R
  - The main goal is to train and plot the results. It uses the split images, delivered by the preparation_blockwise.R file. It then augments the data with the augmentation.R script,  performs the training of the model and plots the final results. At the beginning of the script, we can create models for different tile sizes and number of bands. So with this file 
also, the multitemporal images were used

- pretrainedModel.R
   - This file works very similar to the imagewise_model_training.R file. As a difference, it uses the pre-trained model and just fine-tunes it. It also makes the input data useable
 for the pretrained model, which just works on three bands.

- pretrainedModelMulti.R
  - This file adjusts the multitemporal input data so that they fit to the pretrained model. It then performs the same steps as the imagewise_model_training.R file, so augmentation, training and plotting

- test.R
    - This file was mainly used for our own test purposes. Here we tried different things and plotted some results.
