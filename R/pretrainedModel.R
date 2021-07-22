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

source("~/augmentation.R")

path_true_data <- "~/data/Mossul/prepared/64/true"
path_false_data <- "~/data/Mossul/prepared/64/false"


img <- stack("~/data/Mossul/planet/Mosul_cropped.tif")


# load vgg16 as basis for feature extraction
vgg16_feat_extr <- application_vgg16(include_top = F,input_shape = c(64,64,3),weights = "imagenet")
#freeze weights
freeze_weights(vgg16_feat_extr)
#use only layers 1 to 15
pretrained_model <- keras_model_sequential(vgg16_feat_extr$layers[1:15])



# add flatten and dense layers for classification 
# -> these dense layers are going to be trained on our data only
pretrained_model <- layer_flatten(pretrained_model)
pretrained_model <- layer_dense(pretrained_model,units = 256,activation = "relu")
pretrained_model <- layer_dense(pretrained_model,units = 1,activation = "sigmoid")

pretrained_model


read_tif <- function(f,mask=FALSE) {
  out = array(NA)
  out = unclass(read_stars(f))[[1]]
  if(mask==T){
    dim(out) <- c(dim(out),1)
  }
  return(out[,,1:3])
}


#reading subsets
subset_list <- list.files(path_true_data, full.names = T, pattern= "*tif")
data_true <- data.frame(img=subset_list,lbl=rep(1L,length(subset_list)))

subset_list <- list.files(path_false_data, full.names = T,pattern= "*tif")
#subset_list <- sample(subset_list, length(data_true[,1])) # drop false values to have same amount of true and false
data_false <- data.frame(img=subset_list,lbl=rep(0L,length(subset_list)))



data_original <- rbind(data_true,data_false)

#split into training & validation
#set.seed(2020)
data <- initial_split(data_original,prop = 0.75, strata = "lbl")


training_dataset_list <- training(data)
training_dataset = training(data)
training_dataset$img <- lapply(training_dataset$img, read_tif)
imgs_mean <- mean(unlist(lapply(training_dataset$img,mean)),na.rm=T)
imgs_sd <- sd(unlist(lapply(training_dataset$img,mean)),na.rm=T)
training_dataset$img <- lapply(training_dataset$img, function(x){(x-imgs_mean)/imgs_sd})
training_dataset$img <- lapply(training_dataset$img, function(x) { x[is.na(x)] <- 0; return(x) })

#get input shape expected by image_model
subset_size <- pretrained_model$input_shape[2:3]

# apply function on each dataset element: function is list_modify()
#->modify list item "img" three times:
set.seed(2020)
#dataset <- tensor_slices_dataset(training_dataset)

training_dataset = dl_prepare_data(training_dataset, T, T, subset_size)



#validation
validation_dataset_list <- testing(data)
validation_dataset <- testing(data)
validation_dataset$img <- lapply(validation_dataset$img, read_tif)
imgs_mean <- mean(unlist(lapply(validation_dataset$img,mean)),na.rm=T)
imgs_sd <- sd(unlist(lapply(validation_dataset$img,mean)),na.rm=T)
validation_dataset$img <- lapply(validation_dataset$img, function(x){(x-imgs_mean)/imgs_sd})
validation_dataset$img <- lapply(validation_dataset$img, function(x) { x[is.na(x)] <- 0; return(x) })

validation_dataset =  dl_prepare_data(validation_dataset, F, F, subset_size)



compile(
  pretrained_model,
  optimizer = optimizer_rmsprop(lr = 1e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

diagnostics <- fit(pretrained_model,
                   training_dataset,
                   epochs = 20,
                   class_weight = list("0" = 1, "1" = 5),
                   validation_data = validation_dataset)
plot(diagnostics)


rebuild_img <- function(subsets,out_path,target_rst){
  require(raster)
  require(gdalUtils)
  require(stars)
  
  
  subset_pixels_x <- ncol(subsets[1,,,])
  subset_pixels_y <- nrow(subsets[1,,,])
  tiles_rows <- nrow(target_rst)/subset_pixels_y
  tiles_cols <- ncol(target_rst)/subset_pixels_x
  
  # load target image to determine dimensions
  target_stars <- st_as_stars(target_rst,proxy=F)
  #prepare subfolder for output
  result_folder <- paste0(out_path,"out")
  if(dir.exists(result_folder)){
    unlink(result_folder,recursive = T)
  }
  dir.create(path = result_folder)
  
  #for each tile, create a stars from corresponding predictions, 
  #assign dimensions using original/target image, and save as tif: 
  for (crow in 1:tiles_rows){
    for (ccol in 1:tiles_cols){
      i <- (crow-1)* floor(tiles_cols) + (ccol-1) +1
      
      dimx <- c(((ccol-1)*subset_pixels_x+1),(ccol*subset_pixels_x))
      dimy <- c(((crow-1)*subset_pixels_y+1),(crow*subset_pixels_y))
      cstars <- st_as_stars(t(subsets[i,,,1]))
      attr(cstars,"dimensions")[[2]]$delta=-1
      #set dimensions using original raster
      st_dimensions(cstars) <- st_dimensions(target_stars[,dimx[1]:dimx[2],dimy[1]:dimy[2]])[1:2]
      
      write_stars(cstars,dsn = paste0(result_folder,"/_out_",i,".tif")) 
    }
  }
  
  #mosaic the created tifs
  
  starstiles <- as.vector(list.files(result_folder,full.names = T),mode = "character")
  starstiles <- order(as.numeric(tools::file_path_sans_ext(basename(starstiles))))
  starstiles = paste0(result_folder, "/_out_", starstiles, ".tif")
  print(starstiles)
  gdalbuildvrt(starstiles, paste0(result_folder, "./mosaic" ,".vrt"))
  gdalwarp(paste0(result_folder, "/mosaic" ,".vrt"), paste0(result_folder, "/mosaic",".tif"))

  

}






data_original$img <- lapply(data_original$img, read_tif)
imgs_mean <- mean(unlist(lapply(data_original$img,mean)),na.rm=T)
imgs_sd <- sd(unlist(lapply(data_original$img,mean)),na.rm=T)
data_original$img <- lapply(data_original$img, function(x){(x-imgs_mean)/imgs_sd})

completeData = dl_prepare_data(data_original, F, F, c(64,64,4))

predictions <- predict(pretrained_model, completeData)

input_img =  stack("~/data/Mossul/planet/Mosul_cropped.tif")
predictions <- array(data= rep(predictions,64*64),dim = c(length(predictions),64,64,1))
rebuild_img(predictions,out_path = "./data/Mossul/predictions_pretrained/",target_rst = input_img)

#gdalwarp("./mosaic.vrt","./mosaic.tif")
starstiles <- as.vector(list.files("~/data/Mossul/predictions_blockwise/out",full.names = T),mode = "character")
starstiles <- order(as.numeric(tools::file_path_sans_ext(basename(starstiles))))
starstiles = paste0("./data/Mossul/predictions_blockwise/out", "/_out_", starstiles, ".tif")
print(starstiles)
gdalbuildvrt(starstiles[0:1000], paste0( "./mosaicPredict" ,".vrt"))
gdalwarp(paste0( "./mosaicPredict" ,".vrt"), paste0( "./mosaicPredict",".tif"))

result_map0 <- raster("./data/Mossul/predictions_pretrained/mosaic0.tif")%>%readAll()
result_map1 <- raster("./data/Mossul/predictions_pretrained/mosaic1.tif")%>%readAll()
result_map2 <- raster("./data/Mossul/predictions_pretrained/mosaic2.tif")%>%readAll()
result_map3 <- raster("./mosaic3.tif")%>%readAll()
result_map4 <- raster("./mosaic4.tif")%>%readAll()
result_map5 <- raster("./mosaic5.tif")%>%readAll()

result_map = merge(result_map0,result_map1, result_map2)



result_map <- raster("./mosaicPredict.tif")%>%readAll()

#result_map[result_map[[1]]<0.5] <- NA

agg <- suppressMessages(aggregate(result_map[[1]],c(64,64),fun="max"))
result_scratch <- suppressMessages(rasterToPolygons(agg))

#result_map2 <- raster("./mosaic2.tif")%>%readAll()
#result_map[result_map[[1]]<0.5] <- NA
#agg2 <- suppressMessages(aggregate(result_map2[[1]],c(32,32),fun="max"))
#result_scratch2 <- suppressMessages(rasterToPolygons(agg2))

viewRGB(input_img,layer.name = "Input image", quantiles = c(0,1),r=1,g=2,b=3)+ mapview(sf_destroyed)+ mapview(result_scratch,layer.name="Damage Prediction",alpha.regions=0.4,na.alpha=0,col.regions =c("blue","red","yellow"))


