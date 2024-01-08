"""
Statically defined values to help with well defined processing flows
"""
from rasterio import Affine
from rasterio.crs import CRS


##############################################################
# General Landsat information
##############################################################
LANDSAT_SENSORS = ('oli-tirs', 'etm', 'tm')
LANDSAT_PLATFORMS = ('LC09', 'LC08', 'LE07', 'LT05', 'LT04')

##############################################################
# Landsat ARD defined values
##############################################################
LS_ARD_C1_CU_TileAff = Affine.from_gdal(-2565585, 150000, 0, 3314805, 0, -150000)
LS_ARD_C1_AK_TileAff = Affine.from_gdal(-851715, 150000, 0, 2474325, 0, -150000)
LS_ARD_C1_HI_TileAff = Affine.from_gdal(-444345, 150000, 0, 2168895, 0, -150000)
LS_ARD_C1_CU_CRS = CRS.from_wkt('PROJCS["Albers",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378140,298.2569999999957,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]')
LS_ARD_C1_AK_CRS = CRS.from_wkt('PROJCS["Albers",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378140,298.2569999999986,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",55],PARAMETER["standard_parallel_2",65],PARAMETER["latitude_of_center",50],PARAMETER["longitude_of_center",-154],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]')
LS_ARD_C1_HI_CRS = CRS.from_wkt('PROJCS["Albers",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378140,298.2569999999986,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",8],PARAMETER["standard_parallel_2",18],PARAMETER["latitude_of_center",3],PARAMETER["longitude_of_center",-157],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]')

LS_ARD_C2_CU_TileAff = Affine.from_gdal(-2565585, 150000, 0, 3314805, 0, -150000)
LS_ARD_C2_AK_TileAff = Affine.from_gdal(-2201715, 150000, 0, 2474325, 0, -150000)
LS_ARD_C2_HI_TileAff = Affine.from_gdal(-444345, 150000, 0, 2168895, 0, -150000)
LS_ARD_C2_CU_CRS = CRS.from_wkt('PROJCS["Albers",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",29.5],PARAMETER["standard_parallel_2",45.5],PARAMETER["latitude_of_center",23],PARAMETER["longitude_of_center",-96],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]')
LS_ARD_C2_AK_CRS = CRS.from_wkt('PROJCS["Albers",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",55],PARAMETER["standard_parallel_2",65],PARAMETER["latitude_of_center",50],PARAMETER["longitude_of_center",-154],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]')
LS_ARD_C2_HI_CRS = CRS.from_wkt('PROJCS["Albers",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",8],PARAMETER["standard_parallel_2",18],PARAMETER["latitude_of_center",3],PARAMETER["longitude_of_center",-157],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]')

LS_ARD_C1_SR_MULT = 0.0001
LS_ARD_C1_SR_ADD = 0
LS_ARD_C1_SR_NODATA = -9999

LS_ARD_C1_BT_MULT = 0.1
LS_ARD_C1_BT_ADD = 0
LS_ARD_C1_BT_NODATA = -9999

LS_ARD_C1_TOA_MULT = 0.0001
LS_ARD_C1_TOA_ADD = 0
LS_ARD_C1_TOA_NODATA = -9999

LS_ARD_C2_SR_MULT = 0.0000275
LS_ARD_C2_SR_ADD = -0.2
LS_ARD_C2_SR_NODATA = 0

LS_ARD_C2_BT_MULT = 0.00341802
LS_ARD_C2_BT_ADD = 149.0
LS_ARD_C2_BT_NODATA = 0

LS_ARD_C2_TOA_MULT = 0.0000275
LS_ARD_C2_TOA_ADD = -0.2
LS_ARD_C2_TOA_NODATA = 0
