library(keras)
library(tensorflow)
library(tfdatasets)
library(purrr)
library(ggplot2)
library(rsample)
library(stars)
library(raster)
library(reticulate)
library(mapview)

# links:
# - https://cran.r-project.org/web/packages/tfdatasets/tfdatasets.pdf
# - https://www.tensorflow.org/api_docs/python/tf/image/flip_left_right

#setwd("/Users/tomniers/Desktop/Msc.Geoinformatik/2.Semester/DeepLearningForTheAnalysisOfRemoteSensingImageryFromNanoSatellites/augmentation/")

#imageSize = 32

#read_tif <- function(f,mask=FALSE) {
#  out = array(NA)
#  out = unclass(read_stars(f))[[1]]
#  if(mask==T){
#    dim(out) <- c(dim(out),1)
#  }
#  return(out)
#}

spectral_augmentation <- function(img) {
  img <- tf$image$random_brightness(img, max_delta = 0.3)
  img <- tf$image$random_contrast(img, lower = 0.8, upper = 1.2)
  # img <- tf$image$random_saturation(img, lower = 0.8, upper = 1.2)
  # make sure we still are between 0 and 1
  img <- tf$clip_by_value(img, 0, 1)
}

#filesList <- list.files("./data/true", full.names = T, pattern = "*.tif")
#dataset <- data.frame(img=filesList, lbl=0)
#dataset$img <- lapply(dataset$img, read_tif)
#dataset$mask <- lapply(dataset$mask, read_tif, TRUE)


dl_prepare_data = function(dataset,
                           augmentation,
                           train,
                           subset_size,
                           batch_size = 10L) {
  dataset <- tensor_slices_dataset(dataset)
  
  #convert to float32:
  #for each record in dataset, both its list items are modyfied by the result of applying convert_image_dtype to them
  dataset <-
    dataset_map(dataset, function(.x)
      list_modify(
        .x,
        img = tf$image$convert_image_dtype(.x$img, dtype = tf$float64)
      ))
  
  dataset <-
    dataset_map(dataset, function(.x)
      list_modify(.x, img = tf$image$resize(
        .x$img, size = shape(subset_size[1], subset_size[2])
      )))
  
  
  if (augmentation) {
    #augmentation 1: flip left right, including random change of saturation, brightness and contrast
    augmentation <-
      dataset_map(dataset, function(.x)
        list_modify(.x, img = spectral_augmentation(.x$img)))
    augmentation <-
      dataset_map(augmentation, function(.x)
        list_modify(.x, img = tf$image$flip_left_right(.x$img)))
    dataset_augmented <- dataset_concatenate(dataset, augmentation)
    
    #augmentation 2: flip up down, including random change of saturation, brightness and contrast
    augmentation <-
      dataset_map(dataset, function(.x)
        list_modify(.x, img = spectral_augmentation(.x$img)))
    augmentation <-
      dataset_map(augmentation, function(.x)
        list_modify(.x, img = tf$image$flip_up_down(.x$img)))
    dataset_augmented <-
      dataset_concatenate(dataset_augmented, augmentation)
    
    #augmentation 3: flip left right AND up down, including random change of saturation, brightness and contrast
    augmentation <-
      dataset_map(dataset, function(.x)
        list_modify(.x, img = spectral_augmentation(.x$img)))
    augmentation <-
      dataset_map(dataset, function(.x)
        list_modify(.x, img = tf$image$flip_left_right(.x$img)))
    augmentation <-
      dataset_map(augmentation, function(.x)
        list_modify(.x, img = tf$image$flip_up_down(.x$img)))
    dataset_augmented <-
      dataset_concatenate(dataset_augmented, augmentation)
    
    
    dataset = dataset_augmented
  }
  
  
  
  # shuffling on training set only
  if (train) {
    dataset <-
      dataset_shuffle(dataset_augmented, buffer_size = batch_size * 128)
  }
  
  
  dataset <- dataset_batch(dataset, 10L)
  dataset <- dataset_map(dataset, unname)
  return(dataset)
  
}
