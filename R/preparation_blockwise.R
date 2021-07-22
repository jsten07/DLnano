# library(stars)
library(raster)
library(sf)
library(sp)

# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#' Subset images and split in true / false data
#' 
#' @param inputrst Raster to be subsetted
#' @param targetsize Vector: size of subsets (e.g. c(64, 64))
#' @param targetdir String: Directory to save the images
#' @param sf_points St_points: Points to split tiles in true / false; true if point within tile, false if not
#' @return Cropped raster
dl_subsets <- function(inputrst, targetsize, targetdir, targetname="", img_info_only = FALSE, is_mask = FALSE, sf_points){
  require(jpeg)
  require(raster)
  
  point_coord <- st_coordinates(sf_points)
  
  dir.create(file.path(targetdir, "true"), showWarnings = FALSE)
  dir.create(file.path(targetdir, "false"), showWarnings = FALSE)
  
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
    
    pol <- matrix(c(xmin(subs), ymin(subs), xmax(subs), ymin(subs), xmax(subs), ymax(subs), xmin(subs), ymax(subs), xmin(subs), ymin(subs)), ncol=2, byrow = TRUE)
    within <- point.in.polygon(point_coord[,1], point_coord[,2], pol.x = pol[,1], pol.y = pol[,2])
    
    if("1" %in% within) {
      writeRaster(subs,filename=paste0(targetdir, "/true/",targetname,i,".tif"),overwrite=TRUE) 
    } else {
      writeRaster(subs,filename=paste0(targetdir, "/false/",targetname,i,".tif"),overwrite=TRUE) 
    }
    
    
    #return(c(extent(rst_cropped),crs(rst_cropped)))
  }
  close(pb)
  #img_info <- list("tiles_rows"=nrow(rst_cropped)/targetsizeY, "tiles_cols"=ncol(rst_cropped)/targetsizeX,"crs"= crs(rst_cropped),"extent"=extent(rst_cropped))
  #writeRaster(rst_cropped,filename = paste0(targetdir,"input_rst_cropped.tif"))
  rm(subs,agg,agg_poly)
  gc()
  return(rst_cropped)
  
}


# Crop Mosul
im_size = 64
targetdir = "~/data/Mossul/prepared/"
 
img <- stack("~/data/Mossul/planet/Mosul_cropped.tif")
sf_destroyed <- st_read("~/data/Mossul/shp/Mosul_damaged.shp")
# sf_destroyed <- st_transform(sf_destroyed, crs(img))

# crop Deir
# img <- stack("~/data_cropped.tif")
# sf_destroyed <- st_read("~/data/Deir_Ez_Zor/shp/Deir_destroyed_buildings.shp")

dir.create(file.path(targetdir, im_size), showWarnings = TRUE)

subsets <- dl_subsets(inputrst = img, targetsize = c(im_size, im_size), targetdir = paste(targetdir, im_size, "/", sep = ""), sf_points = sf_destroyed)


# plot(sf_destroyed)
# 
# test <- stack("prepared/32/4138.tif")
# plot(test)
# 
# point_coord <- st_coordinates(sf_destroyed)
# 
# pol <- matrix(c(xmin(test), ymin(test), xmax(test), ymin(test), xmax(test), ymax(test), xmin(test), ymax(test), xmin(test), ymin(test)), ncol=2, byrow = TRUE)
# within <- point.in.polygon(point_coord[,1], point_coord[,2], pol.x = pol[,1], pol.y = pol[,2])
# within 
# "1" %in% within
# 
# 
# plot(test$X4138.1) 
# plot(sf_destroyed[1], add=TRUE)
# 
