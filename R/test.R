library(raster)
library(mapview)
library(sf)

idlib <- stack("~/data/test/Idlib_Abdin_20190526.tif")

plot(idlib)
spplot(idlib)

viewRGB(idlib,layer.name = "input image",quantiles = c(0,1),r=4,g=2,b=1)

damage_deir <- st_read("~/data/test/Damage_Sites_Deir_ez_Zor_CDA.shp")
plot(damage_dir)

plot(st_geometry(damage_deir))
