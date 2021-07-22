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
library(jpeg)

imageSize=32

dl_subsets <- function(inputrst, targetsize, targetdir, targetname="", img_info_only = FALSE, is_mask = FALSE){
  require(jpeg)
  require(raster)
  
  #determine next number of quadrats in x and y direction, by simple rounding
  targetsizeX <- targetsize[1]
  targetsizeY <- targetsize[2]
  inputX <- ncol(inputrst)
  inputY <- nrow(inputrst)
  
  #determine dimensions of raster so that 
  #it can be split by whole number of subsets (by shrinking it)
  while(inputX%%targetsizeX!=0){
    inputX = inputX-1  
  }
  while(inputY%%targetsizeY!=0){
    inputY = inputY-1    
  }
  
  #determine difference
  diffX <- ncol(inputrst)-inputX
  diffY <- nrow(inputrst)-inputY
  
  #determine new dimensions of raster and crop, 
  #cutting evenly on all sides if possible
  newXmin <- floor(diffX/2)
  newXmax <- ncol(inputrst)-ceiling(diffX/2)-1
  newYmin <- floor(diffY/2)
  newYmax <- nrow(inputrst)-ceiling(diffY/2)-1
  rst_cropped <- suppressMessages(crop(inputrst, extent(inputrst,newYmin,newYmax,newXmin,newXmax)))
  #writeRaster(rst_cropped,filename = target_dir_crop)
  
  #return (list(ssizeX = ssizeX, ssizeY = ssizeY, nsx = nsx, nsy =nsy))
  agg <- suppressMessages(aggregate(rst_cropped[[1]],c(targetsizeX,targetsizeY)))
  agg[]    <- suppressMessages(1:ncell(agg))
  agg_poly <- suppressMessages(rasterToPolygons(agg))
  names(agg_poly) <- "polis"
  
  pb <- txtProgressBar(min = 0, max = ncell(agg), style = 3)
  for(i in 1:ncell(agg)) {
    
    # rasterOptions(tmpdir=tmpdir)
    setTxtProgressBar(pb, i)
    e1  <- extent(agg_poly[agg_poly$polis==i,])
    
    subs <- suppressMessages(crop(rst_cropped,e1))
    #rescale to 0-1, for jpeg export
    if(is_mask==FALSE){
      
      subs <- suppressMessages((subs-cellStats(subs,"min"))/(cellStats(subs,"max")-cellStats(subs,"min")))
    } 
    #write jpg
    
    
    #writeJPEG(as.array(subs),target = paste0(targetdir,targetname,i,".jpg"),quality = 1)
    
    writeRaster(subs,filename=paste0(targetdir,targetname,i,".tif"),overwrite=TRUE) 
    #return(c(extent(rst_cropped),crs(rst_cropped)))
  }
  close(pb)
  #img_info <- list("tiles_rows"=nrow(rst_cropped)/targetsizeY, "tiles_cols"=ncol(rst_cropped)/targetsizeX,"crs"= crs(rst_cropped),"extent"=extent(rst_cropped))
  #writeRaster(rst_cropped,filename = paste0(targetdir,"input_rst_cropped.tif"))
  rm(subs,agg,agg_poly)
  gc()
  return(rst_cropped)
  
}

rebuild_img <- function(pred_subsets,out_path,target_rst){
  require(raster)
  require(gdalUtils)
  require(stars)
  
  
  subset_pixels_x <- ncol(pred_subsets[1,,,])
  subset_pixels_y <- nrow(pred_subsets[1,,,])
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
      i <- (crow-1)*tiles_cols + (ccol-1) +1 
      
      dimx <- c(((ccol-1)*subset_pixels_x+1),(ccol*subset_pixels_x))
      dimy <- c(((crow-1)*subset_pixels_y+1),(crow*subset_pixels_y))
      cstars <- st_as_stars(t(pred_subsets[i,,,1]))
      attr(cstars,"dimensions")[[2]]$delta=-1
      #set dimensions using original raster
      st_dimensions(cstars) <- st_dimensions(target_stars[,dimx[1]:dimx[2],dimy[1]:dimy[2]])[1:2]
      
      write_stars(cstars,dsn = paste0(result_folder,"/_out_",i,".tif")) 
    }
  }
  
  starstiles <- as.vector(list.files(result_folder,full.names = T),mode = "character")
  gdalbuildvrt(starstiles,paste0(result_folder,"/mosaic.vrt"))
  gdalwarp(paste0(result_folder,"/mosaic.vrt"), paste0(result_folder,"/mosaic.tif"))
}

spectral_augmentation <- function(img) {
  img <- tf$image$random_brightness(img, max_delta = 0.3) 
  img <- tf$image$random_contrast(img, lower = 0.8, upper = 1.2)
  img <- tf$image$random_saturation(img, lower = 0.8, upper = 1.2) 
  # make sure we still are between 0 and 1
  img <- tf$clip_by_value(img,0, 1) 
}

dl_prepare_data_tif <- function(files, train, predict=FALSE, subsets_path=NULL, model_input_shape = c(448,448), batch_size = 10L) {
  
  if (!predict){
    
    #function for random change of saturation,brightness and hue, will be used as part of the augmentation
    spectral_augmentation <- function(img) {
      img %>% 
        tf$image$random_brightness(max_delta = 0.3) %>% 
        tf$image$random_contrast(lower = 0.5, upper = 0.7) %>% 
        #tf$image$random_saturation(lower = 0.5, upper = 0.7) %>%  --> not supported for >3 bands - you can uncomment in case you use only 3band images
        # make sure we still are between 0 and 1
        tf$clip_by_value(0, 1) 
    }
    
    
    #create a tf_dataset from the first two coloumns of data.frame (ignoring area number used for splitting during data preparation),
    #right now still containing only paths to images 
    dataset <- tensor_slices_dataset(files[,1:2])
    
    
    #the following (replacing tf$image$decode_jpeg by the custom read_tif function) doesn't work, since read_tif cannot be used with dataset_map -> dl_prepare_data_tif therefore expects a data.frame with arrays (i.e. images already loaded)
    #dataset <- dataset_map(dataset, function(.x) list_modify(.x,
    #                                                         img = read_tif(.x$img)/10000,
    #                                                         mask = read_tif(.x$mask)#[1,,,][,,1,drop=FALSE]
    #)) 
    
    
    #convert to float32:
    #for each record in dataset, both its list items are modyfied by the result of applying convert_image_dtype to them
    dataset <- dataset_map(dataset, function(.x) list_modify(.x,
                                                             img = tf$image$convert_image_dtype(.x$img, dtype = tf$float64),
                                                             mask = tf$image$convert_image_dtype(.x$mask, dtype = tf$float64)
    )) 
    
    #resize:
    #for each record in dataset, both its list items are modified by the results of applying resize to them 
    dataset <- 
      dataset_map(dataset, function(.x) 
        list_modify(.x, img = tf$image$resize(.x$img, size = shape(model_input_shape[1], model_input_shape[2])),
                    mask = tf$image$resize(.x$mask, size = shape(model_input_shape[1], model_input_shape[2]))))
    
    
    # data augmentation performed on training set only
    if (train) {
      
      #augmentation 1: flip left right, including random change of saturation, brightness and contrast
      
      #for each record in dataset, only the img item is modified by the result of applying spectral_augmentation to it
      augmentation <- dataset_map(dataset, function(.x) list_modify(.x,
                                                                    img = spectral_augmentation(.x$img)
      ))
      #...as opposed to this, flipping is applied to img and mask of each record
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_left_right(.x$img),
                                                                         mask = tf$image$flip_left_right(.x$mask)
      ))
      dataset_augmented <- dataset_concatenate(dataset,augmentation)
      
      #augmentation 2: flip up down, including random change of saturation, brightness and contrast
      augmentation <- dataset_map(dataset, function(.x) list_modify(.x,
                                                                    img = spectral_augmentation(.x$img)
      ))
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_up_down(.x$img),
                                                                         mask = tf$image$flip_up_down(.x$mask)
      ))
      dataset_augmented <- dataset_concatenate(dataset_augmented,augmentation)
      
      #augmentation 3: flip left right AND up down, including random change of saturation, brightness and contrast
      augmentation <- dataset_map(dataset, function(.x) list_modify(.x,
                                                                    img = spectral_augmentation(.x$img)
      ))
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_left_right(.x$img),
                                                                         mask = tf$image$flip_left_right(.x$mask)
      ))
      augmentation <- dataset_map(augmentation, function(.x) list_modify(.x,
                                                                         img = tf$image$flip_up_down(.x$img),
                                                                         mask = tf$image$flip_up_down(.x$mask)
      ))
      dataset_augmented <- dataset_concatenate(dataset_augmented,augmentation)
      
    }
    
    # shuffling on training set only
    if (train) {
      dataset <- dataset_shuffle(dataset_augmented, buffer_size = batch_size*128)
    }
    
    # train in batches; batch size might need to be adapted depending on
    # available memory
    dataset <- dataset_batch(dataset, batch_size)
    
    # output needs to be unnamed
    dataset <-  dataset_map(dataset, unname) 
    
  }else{
    #make sure subsets are read in in correct order so that they can later be reassambled correctly
    #needs files to be named accordingly (only number)
    #o <- order(as.numeric(tools::file_path_sans_ext(basename(list.files(subsets_path)))))
    #subset_list <- list.files(subsets_path, full.names = T)[o]
    print(files)
    dataset <- tensor_slices_dataset(files)
    print(dataset)
    #dataset <- dataset_map(dataset, function(.x) tf$image$decode_jpeg(tf$io$read_file(.x))) 
    dataset <- dataset_map(dataset, function(.x) tf$image$convert_image_dtype(.x, dtype = tf$float32)) 
    dataset <- dataset_map(dataset, function(.x) tf$image$resize(.x, size = shape(model_input_shape[1], model_input_shape[2]))) 
    dataset <- dataset_batch(dataset, batch_size)
    dataset <-  dataset_map(dataset, unname)
    
  }
  
}

img <- stack("data_cropped.tif")
img
spplot(img)
#spectral_augmentation((img))
#spplot(img)

# Subset satellite image
subsets <- dl_subsets(inputrst = img, targetsize = c(imageSize, imageSize), targetdir = "./subsets/")
plt <- stack("./subsets/28.tif")
plt
# plot(plt)
plotRGB(plt, r=3, g=2, b=1, stretch = "lin")

# TODO: remove empty tiles; compare with mask
# TODO: data augmentation

# subset masks
buffer <- raster("./destroyed_cropped.tif")
buffer[is.na(buffer)] <- 0 # assign zero to NAs
plot(buffer)

subsets <- dl_subsets(inputrst = buffer, targetsize = c(imageSize, imageSize), targetdir = "./masks/", is_mask = TRUE)
plt_b <- raster("./masks/27.tif")
plot(plt_b)

plotRGB(plt, r=3, g=2, b=1, stretch = "lin")
plot(plt_b, add=TRUE)


###Unet
# TODO: Improve / adjust model
input_tensor <- layer_input(shape = c(imageSize,imageSize,4))

#conv block 1
unet_tensor <- layer_conv_2d(input_tensor,filters = 64,kernel_size = c(3,3), padding = "same",activation = "relu")
conc_tensor2 <- layer_conv_2d(unet_tensor,filters = 64,kernel_size = c(3,3), padding = "same",activation = "relu")
unet_tensor <- layer_max_pooling_2d(conc_tensor2)

#conv block 2
unet_tensor <- layer_conv_2d(unet_tensor,filters = 128,kernel_size = c(3,3), padding = "same",activation = "relu")
conc_tensor1 <- layer_conv_2d(unet_tensor,filters = 128,kernel_size = c(3,3), padding = "same",activation = "relu")
unet_tensor <- layer_max_pooling_2d(conc_tensor1)

#"bottom curve" of unet
unet_tensor <- layer_conv_2d(unet_tensor,filters = 256,kernel_size = c(3,3), padding = "same",activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor,filters = 256,kernel_size = c(3,3), padding = "same",activation = "relu")

##  this is where the expanding path begins ##

# upsampling block 1
unet_tensor <- layer_conv_2d_transpose(unet_tensor,filters = 128,kernel_size = c(2,2),strides = 2,padding = "same") 
unet_tensor <- layer_concatenate(list(conc_tensor1,unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3),padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 128, kernel_size = c(3,3),padding = "same", activation = "relu")

# upsampling block 2
unet_tensor <- layer_conv_2d_transpose(unet_tensor,filters = 64,kernel_size = c(2,2),strides = 2,padding = "same")
unet_tensor <- layer_concatenate(list(conc_tensor2,unet_tensor))
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3),padding = "same", activation = "relu")
unet_tensor <- layer_conv_2d(unet_tensor, filters = 64, kernel_size = c(3,3),padding = "same", activation = "relu")

# output
unet_tensor <- layer_conv_2d(unet_tensor,filters = 1,kernel_size = 1, activation = "sigmoid")

# combine final unet_tensor (carrying all the transformations applied through the layers) 
# with input_tensor to create model

unet_model <- keras_model(inputs = input_tensor, outputs = unet_tensor)




files <- data.frame(
  img = list.files("./subsets/", full.names = TRUE, pattern = "*.tif"),
  mask = list.files("./masks/", full.names = TRUE, pattern = "*.tif")
)

read_tif <- function(f,mask=FALSE) {
  out = array(NA)
  out = unclass(read_stars(f))[[1]]
  if(mask==T){
    dim(out) <- c(dim(out),1)
  }
  return(out)
}

files <- initial_split(files, prop = 0.8)

files_train=training(files)
files_val=testing(files)

#training
files_train$img <- lapply(files_train$img, read_tif)
imgs_mean <- mean(unlist(lapply(files_train$img,mean)),na.rm=T)
imgs_sd <- sd(unlist(lapply(files_train$img,mean)),na.rm=T)
files_train$img <- lapply(files_train$img, function(x){(x-imgs_mean)/imgs_sd})
files_train$mask <- lapply(files_train$mask, read_tif, TRUE)

#same for validation
files_val$img <- lapply(files_val$img, read_tif)
imgs_mean <- mean(unlist(lapply(files_val$img,mean)),na.rm=T)
imgs_sd <- sd(unlist(lapply(files_val$img,mean)),na.rm=T)
files_val$img <- lapply(files_val$img, function(x){(x-imgs_mean)/imgs_sd})
files_val$mask <- lapply(files_val$mask, read_tif, TRUE)

plot(files_train$img)


training_dataset <- dl_prepare_data_tif(files_train,train = TRUE,model_input_shape = c(imageSize,imageSize),batch_size = 10L)
validation_dataset <- dl_prepare_data_tif(files_val,train = FALSE,model_input_shape = c(imageSize,imageSize),batch_size = 10L)

training_tensors <- training_dataset%>%as_iterator()%>%iterate()


compile(
  unet_model,
  optimizer = optimizer_rmsprop(lr = 1e-5),
  loss = "binary_crossentropy",
  metrics = c(metric_binary_accuracy)
)


diagnostics <- fit(unet_model,
                   training_dataset,
                   epochs = 18,
                   validation_data = validation_dataset)

plot(diagnostics)

save_model_hdf5(unet_model,filepath = "./unet2")
unet_model<- load_model_hdf5("./unet2")

sample <- floor(runif(n = 1,min = 1,max = 20))
img_path <- as.character(testing(files)[[sample,1]])
mask_path <- as.character(testing(files)[[sample,2]])
img <- magick::image_read(img_path)
mask <- magick::image_read(mask_path)
pred <- magick::image_read(as.raster(predict(object = unet_model,validation_dataset)[sample,,,]))

out <- magick::image_append(c(
  magick::image_append(mask, stack = TRUE),
  magick::image_append(img, stack = TRUE), 
  magick::image_append(pred, stack = TRUE)
)
)

plot(out)


### Prediction
files <- data.frame(
  img = list.files("./subsets/", full.names = TRUE, pattern = "*.tif"),
  mask = list.files("./masks/", full.names = TRUE, pattern = "*.tif")
)

files_all= files
files_all$img <- lapply(files$img, read_tif)
imgs_mean <- mean(unlist(lapply(files_all$img,mean)),na.rm=T)
imgs_sd <- sd(unlist(lapply(files_all$img,mean)),na.rm=T)
files_all$img <- lapply(files_all$img, function(x){(x-imgs_mean)/imgs_sd})
files_all$mask <- lapply(files_all$mask, read_tif, TRUE)
test_dataset <- dl_prepare_data_tif(files_all, train = F,predict = F,model_input_shape = c(imageSize,imageSize),batch_size = 10L)
test_dataset
system.time(predictions <- predict(unet_model,test_dataset))



rebuild_img <- function(pred_subsets,out_path,target_rst){
  require(raster)
  require(gdalUtils)
  require(stars)
  
  
  subset_pixels_x <- ncol(pred_subsets[1,,,])
  subset_pixels_y <- nrow(pred_subsets[1,,,])
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
      # floor added
      # TODO: check results
      i <- (crow-1)* floor(tiles_cols) + (ccol-1) +1 
      print(i)
      dimx <- c(((ccol-1)*subset_pixels_x+1),(ccol*subset_pixels_x))
      dimy <- c(((crow-1)*subset_pixels_y+1),(crow*subset_pixels_y))
      cstars <- st_as_stars(t(pred_subsets[i,,,1]))
      attr(cstars,"dimensions")[[2]]$delta=-1
      #set dimensions using original raster
      st_dimensions(cstars) <- st_dimensions(target_stars[,dimx[1]:dimx[2],dimy[1]:dimy[2]])[1:2]
      
      write_stars(cstars,dsn = paste0(result_folder,"/_out_",i,".tif")) 
    }
  }

  starstiles <- as.vector(list.files(result_folder,full.names = T),mode = "character")
  gdalbuildvrt(starstiles,paste0(result_folder,"/mosaic.vrt"))
  gdalwarp(paste0(result_folder,"/mosaic.vrt"), paste0(result_folder,"/mosaic.tif"))
}

predictions
img <- stack("data_cropped.tif")
rebuild_img(predictions,out_path ="./predictions/", target_rst = img)

result_map <- raster("./predictions/out/mosaic.tif")%>%readAll()
#result_map[result_map[[1]]<0.5] <- NA


viewRGB(img,layer.name = "input image", quantiles = c(0,1),r=1,g=2,b=3)+mapview(result_map,layer.name="damaged", alpha.regions=0.4,na.alpha=0)


rebuild_img(test_dataset)

arrayT= c()
for(i in 1:length(files$img))
{
  imgArray= as.array(raster(files$img[i]))
  arrayT = c(arrayT, list(imgArray) )
}

arrayT2= array(as.numeric(unlist(arrayT)), dim=c(35,imageSize,imageSize,1))
rebuild_img(arrayT2,out_path ="./data/Deir_Ez_Zor/originalImage/", target_rst = img)
original_map <- raster("./data/Deir_Ez_Zor/originalImage/out/mosaic.tif")%>%readAll()
viewRGB(original_map,layer.name = "input image", quantiles = c(0,1),r=1,g=2,b=3)+mapview(result_map,layer.name="damaged", alpha.regions=0.4,na.alpha=0)
plot(original_map)


plot_layer_activations <- function(img_path, model, activations_layers,channels){
  
  
  model_input_size <- c(model$input_shape[[2]], model$input_shape[[3]]) 
  
  #preprocess image for the model
  img <- image_load(img_path, target_size =  model_input_size) %>%
    image_to_array() %>%
    array_reshape(dim = c(1, model_input_size[1], model_input_size[2], 3)) %>%
    imagenet_preprocess_input()
  
  layer_outputs <- lapply(model$layers[activations_layers], function(layer) layer$output)
  activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
  activations <- predict(activation_model,img)
  if(!is.list(activations)){
    activations <- list(activations)
  }
  
  #function for plotting one channel of a layer, adopted from: Chollet (2018): "Deep learning with R"
  plot_channel <- function(channel,layer_name,channel_name) {
    rotate <- function(x) t(apply(x, 2, rev))
    image(rotate(channel), axes = FALSE, asp = 1,
          col = terrain.colors(12),main=paste("layer:",layer_name,"channel:",channel_name))
  }
  
  for (i in 1:length(activations)) {
    layer_activation <- activations[[i]]
    layer_name <- model$layers[[activations_layers[i]]]$name
    n_features <- dim(layer_activation)[[4]]
    for (c in channels){
      
      channel_image <- layer_activation[1,,,c]
      plot_channel(channel_image,layer_name,c)
      
    }
  } 
  
}

par(mfrow=c(3,4),mar=c(1,1,1,1),cex=0.5)
plot_layer_activations(img_path = "./data/Deir_Ez_Zor/prepared/imageSize_jpg/25.jpg", model=unet_model ,activations_layers = c(2,3,5,6,8,9,10,12,13,14), channels = 1:4)
