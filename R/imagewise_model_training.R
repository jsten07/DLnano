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
library(raster)
library(gdalUtils)
library(stars)

source("~/augmentation.R")


path_true_data <- "~/data/AlRaqqah/prepared/64/true"
path_false_data <- "~/data/AlRaqqah/prepared/64/false"


path_true_data <- "~/data/Deir_Ez_Zor/prepared/128/true"
path_false_data <- "~/data/Deir_Ez_Zor/prepared/128/false"

path_true_data <- "~/data/Mossul/prepared/multitemporal/64/true"
path_false_data <- "~/data/Mossul/prepared/multitemporal/64/false"

#modell for 128
input_size <- c(64,64,8)
image_model <- keras_model_sequential()


layer_conv_2d(image_model,filters = 32,kernel_size = 3, activation = "relu",input_shape = input_size)
layer_max_pooling_2d(image_model, pool_size = c(2, 2)) 
layer_conv_2d(image_model, filters = 64, kernel_size = c(3, 3), activation = "relu") 
layer_max_pooling_2d(image_model, pool_size = c(2, 2)) 
layer_conv_2d(image_model, filters = 128, kernel_size = c(3, 3), activation = "relu") 
layer_max_pooling_2d(image_model, pool_size = c(2, 2)) 
layer_conv_2d(image_model, filters = 128, kernel_size = c(3, 3), activation = "relu")
layer_max_pooling_2d(image_model, pool_size = c(2, 2)) 
layer_flatten(image_model) 
layer_dense(image_model, units = 256, activation = "relu")
layer_dense(image_model, units = 1, activation = "sigmoid")

summary(image_model)


#modell for 64
input_size <- c(64,64,4)
image_model <- keras_model_sequential()


layer_conv_2d(image_model,filters = 32,kernel_size = 3, activation = "relu",input_shape = input_size) # 30
layer_max_pooling_2d(image_model, pool_size = c(2, 2)) # 15
layer_conv_2d(image_model, filters = 64, kernel_size = c(3, 3), activation = "relu") # 13
layer_max_pooling_2d(image_model, pool_size = c(2, 2)) # 6 / 6.5?
layer_conv_2d(image_model, filters = 128, kernel_size = c(3, 3), activation = "relu") # 4
layer_max_pooling_2d(image_model, pool_size = c(2, 2)) # 2
layer_flatten(image_model) 
layer_dense(image_model, units = 256, activation = "relu")
layer_dense(image_model, units = 1, activation = "sigmoid")

summary(image_model)

read_tif <- function(f,mask=FALSE) {
  out = array(NA)
  out = unclass(read_stars(f))[[1]]
  if(mask==T){
    dim(out) <- c(dim(out),1)
  }
  return(out)
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
data <- initial_split(data_original,prop = 0.8, strata = "lbl")


training_dataset_list <- training(data)
training_dataset = training(data)
training_dataset$img <- lapply(training_dataset$img, read_tif)
imgs_mean <- mean(unlist(lapply(training_dataset$img,mean)),na.rm=T)
imgs_sd <- sd(unlist(lapply(training_dataset$img,mean)),na.rm=T)
training_dataset$img <- lapply(training_dataset$img, function(x){(x-imgs_mean)/imgs_sd})
training_dataset$img <- lapply(training_dataset$img, function(x) { x[is.na(x)] <- 0; return(x) })

#get input shape expected by image_model
subset_size <- image_model$input_shape[2:3]

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



#Training
compile(
  image_model,
  optimizer = optimizer_rmsprop(lr = 5e-5),
  loss = "binary_crossentropy",
  metrics = "accuracy"
)

weight= length(data_false)/length(data_true)

diagnostics <- fit(image_model,
                   training_dataset,
                   epochs = 10,
                   class_weight = list("0" = 1, "1" = 3),
                   validation_data = validation_dataset)

plot(diagnostics)




#Prediction
predictions <- predict(image_model,validation_dataset)
head(predictions)

par(mfrow=c(1,3),mai=c(0.1,0.1,0.3,0.1),cex=0.8)
for(i in 1:3){
  sample <- floor(runif(n = 1,min = 1,max = 350))
  img_path <- as.character(testing(data)[[sample,1]])
  img <- stack(img_path)
  plotRGB(img,  r=3, g=2, b=1, stretch = "lin",margins=T,main = paste("prediction:",round(predictions[sample],digits=3)," | ","label:",as.character(testing(data)[[sample,2]])))
  # plot(sf_destroyed[1], add=TRUE)
}



rebuild_img <- function(subsets,out_path,target_rst){

  
  
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
  gdalbuildvrt(starstiles, paste0(result_folder, "./mosaic3" ,".vrt"))
  gdalwarp(paste0(result_folder, "/mosaic3" ,".vrt"), paste0(result_folder, "/mosaic3",".tif"))

  
}



data_original$img <- lapply(data_original$img, read_tif)
imgs_mean <- mean(unlist(lapply(data_original$img,mean)),na.rm=T)
imgs_sd <- sd(unlist(lapply(data_original$img,mean)),na.rm=T)
data_original$img <- lapply(data_original$img, function(x){(x-imgs_mean)/imgs_sd})

completeData = dl_prepare_data(data_original, F, F, c(64,64,8))
#completeData = dl_prepare_data(data_original, F, F, c(128,128,4))

predictions <- predict(image_model,completeData)

input_img = stack("./data/Mossul/planet/Mosul_multitemporal.tif")
#input_img = stack("./data/Deir_Ez_Zor/planet/Deir_cropped.tif")
predictions <- array(data= rep(predictions,64*64),dim = c(length(predictions),64,64,1))
rebuild_img(predictions,out_path = "./data/Mossul/predictions_blockwise/",target_rst = input_img)


starstiles <- as.vector(list.files("./data/Mossul/predictions_blockwise/out",full.names = T),mode = "character")
starstiles <- order(as.numeric(tools::file_path_sans_ext(basename(starstiles))))
starstiles = paste0("./data/Mossul/predictions_blockwise/out", "/_out_", starstiles, ".tif")
print(starstiles)
gdalbuildvrt(starstiles, paste0("./data/Mossul/predictions_blockwise/out", "/mosaic3" ,".vrt"))
gdalwarp(paste0("./data/Mossul/predictions_blockwise/out", "/mosaic3" ,".vrt"), paste0("./data/Mossul/predictions_blockwise/out", "/mosaic3",".tif"))


#gdalwarp("./mosaic.vrt","./mosaic.tif")

result_map <- raster("./data/Mossul/predictions_blockwise/out/mosaic3.tif")%>%readAll()
#result_map1 <- raster("./mosaic1.tif")%>%readAll()
#result_map2 <- raster("./mosaic2.tif")%>%readAll()
#result_map3 <- raster("./mosaic3.tif")%>%readAll()
#result_map4 <- raster("./mosaic4.tif")%>%readAll()
#result_map5 <- raster("./mosaic5.tif")%>%readAll()

result_map = merge(result_map0,result_map1, result_map2, result_map3, result_map4,result_map5)

#result_map[result_map[[1]]<0.5] <- NA

agg <- suppressMessages(aggregate(result_map[[1]],c(64,64),fun="max"))
result_scratch <- suppressMessages(rasterToPolygons(agg))

#result_map2 <- raster("./mosaic2.tif")%>%readAll()
#result_map[result_map[[1]]<0.5] <- NA
#agg2 <- suppressMessages(aggregate(result_map2[[1]],c(32,32),fun="max"))
#result_scratch2 <- suppressMessages(rasterToPolygons(agg2))
sf_destroyed <- st_read("~/data/Mossul/shp/Mosul_damaged.shp")
viewRGB(input_img,layer.name = "Input image", quantiles = c(0,1),r=1,g=2,b=3)+ mapview(sf_destroyed)+ mapview(result_scratch,layer.name="Damage Prediction",alpha.regions=0.4,na.alpha=0,col.regions =c("blue","red","yellow"))

