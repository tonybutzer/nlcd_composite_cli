#!/usr/bin/env python
# coding: utf-8

# # CODE

import datetime as dt
from datetime import timedelta
from functools import partial
import logging
import random
import multiprocessing as mp
import sys
import geopandas as gp

import numpy as np
import rasterio as rio
from rasterio.windows import Window
import fsspec
import boto3
import xarray as xr
import matplotlib.pyplot as plt

from lcnext import lsc2ard as c2ard
from lcnext.lsc2ard import LandsatARDObservation
from lcnext.static import LANDSAT_SENSORS

from tag.tagme import tagme

from typing import List, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

# logging.basicConfig(
#     level=logging.ERROR,
#     format="%(asctime)s %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     stream=sys.stderr,
# )

log = logging.getLogger()


def sr_bandnumbers(sensor: str) -> List[int]:
    """
    SR band numbers for the given sensor
    oli-tirs, tm, etm
    """
    if sensor == 'oli-tirs':
        return [2, 3, 4, 5, 6, 7]
    elif (sensor == 'tm') | (sensor == 'etm'):
        return [1, 2, 3, 4, 5, 7]
    else:
        raise ValueError

def qa_std_layers(deets: LandsatARDObservation) -> List[str]:
    """
    Standard list of QA bands associated with the given observation
    """
    return [f'{deets.root_id}_QA_PIXEL.TIF']


def sr_std_layers(deets: LandsatARDObservation) -> List[str]:
    """
    Standard list of needed bands associated with the given observation
    """
    return [f'{deets.root_id}_SR_B{b}.TIF'
            for b in sr_bandnumbers(deets.sensor)]
    
def dstack_idx(idxs: np.ndarray):
    """
    Takes 2d index returns from numpy.argmin or numpy.argmax on a 3d array where axis=0 and turns it into
    a tuple of tuples for indexing back into the 3d array
    """
    rows, cols = idxs.shape
    
    d_stack_idx = (idxs,
            np.repeat(np.arange(rows).reshape(-1, 1), repeats=rows, axis=1),
            np.repeat(np.arange(cols).reshape(1, -1), repeats=cols, axis=0))
    
    # print(f'dstack input idx: {idxs}')
    
    return d_stack_idx


def difference_absolute(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Calculate the absolute distance between the values in the two different arrays
    """
    diff_abs = np.abs(difference(arr1, arr2))
    
    return diff_abs


def difference(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Difference the two given arrays
    """ 
    return arr1 - arr2


def difference_median(arr):
    """
    Calculate the absolute difference each value is from the median
    """
    
    median = np.ma.median(arr, axis=0)
    # print(f'median: {median}')
    
    diff_median = difference_absolute(arr, median)
    # print(f'diff_median: {diff_median}')
    
    return diff_median


def sum_squares(arrs):
    """
    Square and then sum all the values (element wise) in the given arrays
    """
    sum_sqs = np.ma.sum([np.ma.power(a, 2) for a in arrs], axis=0)
    
    # print(f'sum_sqs: {sum_sqs}')
    
    return sum_sqs


def distance_overall(spectral):
    """
    Return the euclidean distance for observations that come closest to the overall median value. Where spectral 
    is the input 6 band array. The shape of the input spectral band is (#of bands, #of ARD observations/scenes, Xsize of Window, Ysize of Window).
    The 
    """

    euc_dist = np.ma.sqrt(sum_squares([difference_median(spectral[b])
                                       for b in range(spectral.shape[0])]))
    
    
    # print(f'euc_dist: {euc_dist}')
    
    idxs = dstack_idx(np.ma.argmin(euc_dist, axis=0))
    
    # print(f'idxs: {idxs}')
     
    dist_overall = np.array([spectral[b][idxs]
                     for b in range(spectral.shape[0])])
    
    
    return dist_overall
    

# def std_mask(qa_arr: np.ndarray) -> np.ndarray:
#     """
#     Standard mask for compositing stuff
#     """
    
#     std_masks = (c2ard.qa_fill(qa_arr) |
#             c2ard.qa_cirrus(qa_arr) |
#             c2ard.qa_cloud(qa_arr) |
#             c2ard.qa_cl_shadow(qa_arr) |
#             c2ard.qa_snow(qa_arr))
    
#     return std_masks


def create_qa_arrays(qa_arr: np.array):
    '''
    Create a dictionary of QA arrays.
    '''
    qa_arrays = {
        'qa_fill': c2ard.qa_fill(qa_arr),
        'qa_cirrus': c2ard.qa_cirrus(qa_arr),
        'qa_cloud': c2ard.qa_cloud(qa_arr),
        'qa_cl_shadow': c2ard.qa_cl_shadow(qa_arr),
        'qa_snow': c2ard.qa_snow(qa_arr)
    }
    return qa_arrays



def get_qa_count(qa_arr: np.ndarray)-> np.ndarray:
    """
    This function calculates the QA count i.e. the number of TRUE values per pixel location across all input ARDS.
    The input qa_arr is converted to a boolean_arrays which is a python list the size of the number of QA filters 
    used in the processing.In this function we first convert the boolean_array TRUE/FALSE list of arrays to a binary
    list of arrays (1's' and 0's') then sums across each pixel to geta the count of 1's/TRUE's per QA filter.
    
    Parameters
    ----------
    boolean_arrays : np.ndarray
        DESCRIPTION.
        Example boolean_array = [array([[[False,  True],
                 [ True,  True]],
         
                [[False,  True],
                 [ True, False]],
         
                [[False, False],
                 [ True, False]]]),
                 
         array([[[False, False],
                 [False, False]],
         
                [[False, False],
                 [False, False]],
         
                [[ True,  True],
                 [ True,  True]]])]
        
        Which is a list of 2 arrays (represents 2 QA filters) each with 3 sub arrays (boolean data for a specific 
        window in 3 ARD tiles) at 2 by 2 size. After converting all Trues to 1's and False to 0's the result 
        qa_count array for this boolean is
                [array([[0, 2],
                        [3, 1]]),
                 array([[1, 1],
                        [1, 1]])]
        In the first QA filter (first array of result) the first pixel value of 0 shows that for that pixel location
        there were 0 True values across all 3 ARD's used, likewise the value of 2 shows that for that pixel location 
        2 of 3 ARD's for this pixel location is True

    Returns
    -------
    None.

    """
    # print(f'qa_arr: {qa_arr}')
    qa_arrays = create_qa_arrays(qa_arr)
    
    # Extract the boolean arrays from the dictionary
    boolean_arrays = list(qa_arrays.values())
    # print(f'boolean_arrays: {boolean_arrays}')
    
    # Convert True to 1 and False to 0 for each array in the list
    binary_arrays = [arr1.astype(int) for arr1 in boolean_arrays]
    
    # print(f'binary_arrays: {binary_arrays}')
    
    # Sum along the 0-axis for each array in the list
    qa_count_arrays = [np.sum(arr2, axis=0) for arr2 in binary_arrays]
    
    # print(f'qa_count: {qa_count_arrays}')
    
    return qa_count_arrays


def get_band_ids(bandcombotype: str, sensor: str) -> str:
    """
    Returns the band Ids for the specified combination of bands given the sensor
    https://www.usgs.gov/media/images/common-landsat-band-combinations
    """
    
    match bandcombotype:
        case 'rgb':
            band_ids = {
            'oli-tirs': ['B4', 'B3', 'B2'],
            'tm': ['B3', 'B2', 'B1'],
            'etm': ['B3', 'B2', 'B1']
             }
        case 'vegetation':
            band_ids = {
            'oli-tirs': ['B6', 'B5', 'B4'],
            'tm': ['B5', 'B4', 'B3'],
            'etm': ['B5', 'B4', 'B3']
            }
        case _:
            raise ValueError(f'Platform not recognized: {platform}')

    return band_ids.get(sensor.lower(), None)


def std_mask(qa_arr: np.ndarray) -> np.ndarray:
    """
    Standard mask for each observation used in compositing, returns pixel locations with TRUE or FALSE for masking
    """
    # print(f'qa_arr: {qa_arr}')
    qa_arrays = create_qa_arrays(qa_arr)
    
    # Extract the boolean arrays from the dictionary
    boolean_arrays = list(qa_arrays.values())
    
    # Combine the boolean arrays using bitwise OR
    std_masks = np.bitwise_or.reduce(boolean_arrays)
    
    # print(f'std_masks: {std_masks}')
    
    return std_masks    
    


def get_cog_metadata(window: Window, ids: List[LandsatARDObservation]):
    """
    Pulls in a single ARD tile and returns the crs, transform, height, width, dtype,
    """
    env = rio_env()
    obs_layer = sr_std_layers(ids[0])[0] #select a single layer to retrieve metadata from
    path = '/'.join([ids[0].std_path, obs_layer])
    
    try:
        with env:
            with rio.open('s3://' + path) as ds:
                metadata_list = []

                window_info = {
                    'dtype': ds.dtypes[0],  # Assuming index 1
                    'transform': ds.window_transform(window),
                    'crs': ds.crs,
                    'width': window.width,
                    'height': window.height,
                    'count': ds.count
                }

                metadata_list.append(window_info)

                log.info(f"Metadata for window {window}: {window_info}\n\n")

                return metadata_list

    except Exception as e:
        logging.error(f"Error processing raster: {path}, {e}")
        raise



# def read_raster(path: str, fs: fsspec.AbstractFileSystem, env: rio.env, index=1, window=None, boundless=False, fill_value=None):
#     try:
#         with fs.open(path) as f:
#             with env:
#                 with rio.open(f) as ds:
#                     return ds.read(index, window=window, boundless=boundless, fill_value=fill_value)
#     except:
#         print(path)
#         raise


def read_raster_pure_rio(path: str, env: rio.env, index=1, window=None, boundless=False, fill_value=None):
    try:
        with env:
            with rio.open('s3://' + path) as ds:

                return ds.read(index, window=window, boundless=boundless, fill_value=fill_value)
    except:
        print(path)
        raise


        
# def comp_worker(window: Window, ids: List[LandsatARDObservation], storage_options: dict) -> np.ndarray:
#     log.info(f'Processing {window}')
#     fs = fsspec.filesystem("s3", **storage_options)
#     env = rio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR')

#     big_arr = np.zeros(shape=(6, len(ids), window.height, window.width), dtype=int)
#     qas = np.zeros(shape=(len(ids), window.height, window.width), dtype=int)

#     for idx1, d in enumerate(ids):
#         log.info(f'Pulling data for {d}')
#         layers = sr_std_layers(d)
#         qlayer = qa_std_layers(d)[0]
        
#         for idx2, b in enumerate(layers):
#             big_arr[idx2, idx1] = read_raster('/'.join([d.std_path, b]), fs, env, window=window, boundless=False)

#         qas[idx1] = read_raster('/'.join([d.std_path, qlayer]), fs, env, window=window, boundless=False)
        
#     return window, distance_overall(np.ma.masked_array(big_arr,
#                                                dtype=big_arr.dtype,
#                                                mask=np.repeat(np.expand_dims(std_mask(qas), axis=0),
#                                                               repeats=6,
#                                                               axis=0)))

def sort_percentiles(rgb_arr, percent_cut_off):
    """
    Use in cases where valid pixels are being removed due to snow or terrain shadows 
    when pixel_qa is applied. This function works by summing the red, green and blue
    arrays for all observations. It then sorts which pixels to keep using a upper and
    lower percentile threshold. I.e. each pixel location is given a TRUE if it falls 
    outside of the upper and lower threshold otherwise FALSE if its being kept.
    """
    
    # Sum the red green and blue bands
    sum_rgb = np.sum(rgb_arr, axis=1)

    # Calculate the lower and upper percentiles for each pixel location
    lower_percentiles = np.percentile(sum_rgb, percent_cut_off, axis=0)
    upper_percentiles = np.percentile(sum_rgb, 100-percent_cut_off, axis=0)

    # Find pixels outside the lower or higher 10% cut for each pixel
    sorted_pixels = np.logical_or(sum_rgb <= lower_percentiles, sum_rgb  >=  upper_percentiles)

    # # Print the result
    # print("3D Array:")
    # print(sum_rgb)
    # print("\nPixels oustide lower and higher 10% cut:")
    # print(sorted_pixels)
    
    return sorted_pixels        


def comp_worker_pure_rio(window: Window, ids: List[LandsatARDObservation], *args, **kwargs) -> np.ndarray:
    log.info(f'PROCESSING {window}')
    env = rio_env()
    
    big_arr = np.zeros(shape=(6, len(ids), window.height, window.width), dtype=int)
    rgb_arr = np.zeros(shape=(len(ids),3, window.height, window.width), dtype=int)
    qas = np.zeros(shape=(len(ids), window.height, window.width), dtype=int)
    
    for idx1, d in enumerate(ids):
        log.info(f'Pulling data for {d}')
        layers = sr_std_layers(d)
        qlayer = qa_std_layers(d)[0]
        
        qas[idx1] = read_raster_pure_rio('/'.join([d.std_path, qlayer]), env, window=window, boundless=True)
        # print(f'qas[idx1]: {qas}')
        
        band_ids = get_band_ids('rgb', d.sensor)
        # print(f'>>> band_ids: {band_ids}')
        
        for idx2, b in enumerate(layers):
            for idx3, band_id in enumerate(band_ids):
                if band_id in b:
                    # print(f'>>> b: {b}')
                    rgb_arr[idx1, idx3] = read_raster_pure_rio('/'.join([d.std_path, b]), env, window=window, boundless=True)
            # print(f'>>> rgb_arr: {rgb_arr}')
                
            big_arr[idx2, idx1] = read_raster_pure_rio('/'.join([d.std_path, b]), env, window=window, boundless=True)
                   
        # print(f'big_arr: {big_arr}')             
  
    masked_spectral_array = np.ma.masked_array(big_arr,
                                                dtype=big_arr.dtype,
                                                mask=np.repeat(np.expand_dims(std_mask(qas), axis=0),
                                                               repeats=6,
                                                               axis=0))
 
    #masked_spectral_array = np.ma.masked_array(big_arr,
                                               #dtype=big_arr.dtype,
                                               #mask=np.repeat(np.expand_dims(sort_percentiles(rgb_arr,10), axis=0),
                                                              #repeats=6,
                                                              #axis=0))
    # print(f'masked_spectral_array: {masked_spectral_array}')
    dist_overall = distance_overall(masked_spectral_array)  
    # log.info(f' DISTANCE_OVERALL: {np.count_nonzero(dist_overall)} filled pixels of {dist_overall.size}')
    # print(f'dist_overall: {dist_overall}')
    qa_count = get_qa_count(qas) 
    
    return window, dist_overall, qa_count


def year_deets(fs: fsspec.AbstractFileSystem, year: int, sensor: str, region: str, horiz: int, vert: int) -> List[LandsatARDObservation]:
    """
    Build out the details for each observation for a given year/sensor
    """
    return [c2ard.obs_deets(p)
            for p in
            fs.ls(f'usgs-landsat-ard/collection02/{sensor}/{year}/{region}/{horiz:03}/{vert:03}') if not p.endswith('.json')] 
                                                                                ##since each dir has a JSON file causing errors

def find_observations(fs: fsspec.AbstractFileSystem, start_date: dt.datetime.date, end_date: dt.datetime.date, sensor: str, region: str, horiz: int, vert: int):
    """
    Find all observations to fit within the specified start_date/end_date for a given sensor
    """
    ret = []
    for year in range(start_date.year, end_date.year + 1):
        ret.extend(filter(lambda x: start_date <= x.acquired <= end_date, year_deets(fs, year, sensor, region, horiz, vert)))
    # print(ret)
    return ret


def check_sensor_availability(sensor_name: str, year: int) -> str:
    """
    Determine sensor availability
    """
    start_year, end_year = {
        "tm": (1982, 2012),
        "oli-tirs": (2013, 2023),
        "etm": (1999, 2022)}[sensor_name]
    if start_year <= year <= end_year:
        return "available"
    return "unavailable"


def rio_env() -> rio.Env:
    return rio.Env(rio.session.AWSSession(boto3.Session(), requester_pays=True),
                 GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                 GDAL_HTTP_MAX_RETRY=30,
                 GDAL_HTTP_RETRY_DELAY=random.randint(3, 15))


# def main():
#     log.info('Initializing filesystem')
#     # storage_options = {'profile': "ceph",
#     #                    'client_kwargs': {"endpoint_url": ENDPOINT}}
    
#     storage_options = {'requester_pays': True}
    
#     fs = fsspec.filesystem("s3", **storage_options)
        
#     start = dt.datetime.strptime('20130501', '%Y%m%d').date()
#     end = dt.datetime.strptime('20130901', '%Y%m%d').date()

#     log.info('Identifying inputs')
#     deets = []
#     for sensor in LANDSAT_SENSORS:
#         if check_sensor_availability(sensor, start.year) == 'available':
#             deets.extend(find_observations(fs, start, end, sensor, 'CU', 3, 10))
        
#     # windows = [Window(col_off=x, row_off=y, width=100, height=100)
#     #            for x in range(0, 5000, 100)
#     #            for y in range(0, 5000, 100)]
    
#     windows = [Window(col_off=0, row_off=0, width=100, height=100)]
    
#     func = partial(comp_worker_pure_rio,
#                    ids=deets,
#                    storage_options=storage_options)

#     # with mp.Pool(processes=4) as pool:
#     #     log.info(f'Begin processing')
#     #     for window, arrs in pool.imap_unordered(func, windows):
#     #         log.info('Processed...')

#     log.info(f'Begin processing')
#     for window, arrs in map(func, windows):
#         log.info('Processed...')
    
# if __name__ == '__main__':
#     mp.set_start_method('forkserver')
#     main()


#%%time

# Get the current date and time
start_time = dt.datetime.now()
formatted_datetime = start_time.strftime('%Y%m%d_%H%M')

#%=====================USER INPUT========================

# start = dt.datetime.strptime('19840501', '%Y%m%d').date()
# end = dt.datetime.strptime('19840930', '%Y%m%d').date()

# horiz = 4 # horizontal location
# vert =  2 # vertical location

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Run the composite code')
    parser.add_argument('-s', '--start', help='specify start year and season example: -s 19840501 ', type=str, required=True)
    parser.add_argument('-e', '--end', help='end year and season example: --end 19840930', type=str, required=True)
    parser.add_argument('-z', '--horiz', help='ARD H horizontal grid example -h 23', type=str, required=True)
    parser.add_argument('-v', '--vert', help='ARD V veritical -v 13' , type=str, required=True)
    return parser


parser = get_parser()
args = vars(parser.parse_args())
print(args)

starta=args['start']
enda=args['end']
start = dt.datetime.strptime(starta, '%Y%m%d').date()
end = dt.datetime.strptime(enda, '%Y%m%d').date()


horiz = int(args['horiz'])
vert = int(args['vert'])

year = starta[0:4]

my_new_tag = f'Compute-{year}_{horiz}_{vert}-composite-butzer'
tagme(my_new_tag)



max_x = 5000
max_y = 5000
interval = 500

start_x = 0
start_y = 0

windows = [Window(col_off=x, row_off=y, width=interval, height=interval)
           for x in range(start_x, max_x, interval)
           for y in range(start_y, max_y, interval)]

n_qa_filters = 5
n_bands = 6


#%%%===================Setup Logging Output==============================
log.info(f'Initializing filesystem {start_time}')
# storage_options = {'profile': "ceph",
#                    'client_kwargs': {"endpoint_url": ENDPOINT}}

storage_options = {'requester_pays': True}
fs = fsspec.filesystem("s3", **storage_options)

log.info('Identifying inputs')
deets = []
for sensor in LANDSAT_SENSORS:
    # print(f'sensor: {sensor}')
    if check_sensor_availability(sensor, start.year) == 'available':
        print(f'sensor: {sensor} = available')
        deets.extend(find_observations(fs, start, end, sensor, 'CU', horiz, vert))
log.info(f'Observations found: {len(deets)}')

# if len(deets) < 10:
#     log.info(f'Adding extra year to observation list...')
#     # Increment both start and end dates by one year
#     start += timedelta(days=365)
#     end += timedelta(days=365)
#     log.info(f'New start and end dates: {start}:{end}')
#     for sensor in LANDSAT_SENSORS:
#         if check_sensor_availability(sensor, start.year) == 'available':
#             deets.extend(find_observations(fs, start, end, sensor, 'CU', horiz, vert))
#     log.info(f'New total Observations found: {len(deets)}')
        

func = partial(comp_worker_pure_rio,
               ids=deets,
               storage_options=storage_options)


# # Create a larger array filled with zeros
# out_comp_arr = np.zeros((6, max_x, max_y), dtype=int)
# for window, window_arr in map(func, windows):
#     for i in range(window_arr.shape[0]):
#         out_comp_arr[i][window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width] = window_arr[i]
#     log.info('Processed...')
#     log.info('\n\n')


#%%===================SET OUT FILE NAMES==============================
# create out tif name with current date time
OUTDIR = '/efs/danny'
out_comp_tif = f'{OUTDIR}/comp_{horiz:03d}{vert:03d}_{start.year}_{max_x}x{max_y}y{interval}i_{formatted_datetime}.tif'
out_qa_tif = f'{OUTDIR}/QA_{horiz:03d}{vert:03d}_{start.year}_{len(deets)}_{max_x}x{max_y}y{interval}i_{formatted_datetime}.tif'


#%%==================Retrieve metadata from a single input obs=======================
# The whole window since the final tif will be the whole window
cog_metadata = get_cog_metadata(Window(col_off=0, row_off=0, width=max_x, height=max_y), deets)
transform = cog_metadata[0]['transform']
dtype = cog_metadata[0]['dtype']
crs = cog_metadata[0]['crs']

log.info(f'BEGIN PROCESSING: {out_comp_tif}')



#%%====================FILL COMPOSITE and QA RASTER==================================

# # Open the output datasets outside the loop
# with rio.open(out_comp_tif, 'w', driver='GTiff', nodata=None, width=max_x, height=max_y,
#               count=6, dtype=dtype, crs=crs, transform=transform) as out_comp_dataset, \
#      rio.open(out_qa_tif, 'w', driver='GTiff', nodata=None, width=max_x, height=max_y,
#               count=n_qa_filters, dtype=dtype, crs=crs, transform=transform) as out_qa_dataset:

#     for window in windows:
#         comp_window_arr = func(window)  # func returns a tuple, thus comp_window_arr[1] is used below

#         # Write the comp array
#         out_comp_dataset.write(comp_window_arr[1], window=((window.row_off, window.row_off + window.height),
#                                                            (window.col_off, window.col_off + window.width)))

#         # Write the QA array
#         out_qa_dataset.write(np.array(comp_window_arr[2]), window=((window.row_off, window.row_off + window.height),
#                                                                    (window.col_off, window.col_off + window.width)))

#         log.info(f'PROCESSED WINDOW: {window} \n\n')


# #%%===============================FILL COMPOSITE RASTER w/ Multiprocessing Pool========================
def process_window(window):
    comp_window_arr = func(window)
    return comp_window_arr, window

# Number of CPU cores
num_cores = mp.cpu_count()

with rio.open(out_comp_tif, 'w', driver='GTiff', nodata=None, width=max_x, height=max_y,
              count=6, dtype=dtype, crs=crs, transform=transform) as out_comp_dataset, \
     rio.open(out_qa_tif, 'w', driver='GTiff', nodata=None, width=max_x, height=max_y,
              count=n_qa_filters, dtype=dtype, crs=crs, transform=transform) as out_qa_dataset:

    with mp.Pool(num_cores) as pool:
        results = pool.map(process_window, windows)

    for comp_window_arr, window in results:
        # Write the comp array
        out_comp_dataset.write(comp_window_arr[1], window=((window.row_off, window.row_off + window.height),
                                                           (window.col_off, window.col_off + window.width)))

        # Write the QA array
        out_qa_dataset.write(np.array(comp_window_arr[2]), window=((window.row_off, window.row_off + window.height),
                                                                   (window.col_off, window.col_off + window.width)))


        log.info(f'PROCESSED WINDOW: {window} \n\n')




# Record the end time and calculate the elapsed time
elapsed_time = dt.datetime.now() - start_time

# Convert elapsed time to seconds
elapsed_seconds = elapsed_time.total_seconds()

# Print the result in minutes or hours
if elapsed_time.total_seconds() < 60:
    log.info(f"Processing time: {elapsed_time.total_seconds():.2f} seconds")
elif elapsed_time.total_seconds() < 3600:
    log.info(f"Processing time: {elapsed_time.total_seconds()/60:.2f} minutes")
else:
    log.info(f"Processing time: {elapsed_time.total_seconds()/3600:.2f} hours")


band_ids = ['B3', 'B2', 'B1']
b = 'LT05_CU_003010_19840513_20210421_02_SR_B1.TIF'
# # Check if any of the band_ids is present in the string b
# if any(band_id in b for band_id in band_ids):
#     print('yes')
        
for idx3, band_id in enumerate(band_ids):
    if band_id in b:
        print(f'>>> b: {b}')





# # RUN

# !python3 nlcd_compositing_TR.py >> output_014016_1984_1985_5000xy100i_20231219_2134.txt


# # TEST

# ## Plotting Composites

storage_options = {'requester_pays': True}
fs = fsspec.filesystem("s3", **storage_options)


# fs.ls('usgs-landsat-ard/collection02/tm/1984/CU/014/016/')


def plot_ARD_composite(ARDs, bandcombotype: str):
    """
    Retrieve red, green and blue tiles for select ARDs then plot the rgb composites for each.
    
    PARAMETERS:
        ARDs (list) - a list of ARDs with parameters set by the find_observations function
    """
    log.info(f'Number of Observations Found: {len(ARDs)}')
    
    env = rio_env()
    

    for d in ARDs:
        log.info(f'Pulling data for {d}')
        bands_arr = np.zeros(shape=(3, 5000, 5000), dtype=int)
        layers = sr_std_layers(d)
        idx = 0    
        
        band_ids = get_band_ids('rgb', d.sensor)
        
        for band_id in band_ids:
            # print(f'band_id: {band_id}')
            for b in layers: 
                if band_id in b:
                    bands_arr[idx] = read_raster_pure_rio('/'.join([d.std_path, b]), env, boundless=True)
                    idx = idx+1

        # Initialize an empty list to store the RGB bands
        scaled_bands = []
        composite = []
        
        # Loop through each band data and transform
        for band_array in bands_arr:
            # Append the band data after scaling to 0-1 range
            scaled_bands.append((band_array / band_array.max()).astype(np.float32))

        # Create a natural color composite using the RGB bands
        composite = np.stack(scaled_bands, axis=-1)

        # Display the natural color composite
        plt.imshow(composite)
        plt.title(f'{bandcombotype} composite:  {d.root_id} ')
        plt.show()  



#ARDs = find_observations(fs, dt.date(1984, 5, 1), dt.date(1984, 9, 30), 'tm', 'CU', '003', '003')
#plot_ARD_composite(ARDs, 'rgb')


# # Not in Use

# qa_count_arrays = [np.array([[4, 4, 5, 5, 5],
#        [4, 4, 4, 4, 5],
#        [5, 4, 4, 4, 4],
#        [5, 5, 5, 4, 4],
#        [5, 5, 5, 5, 5]]), np.array([[0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0]]), np.array([[1, 1, 1, 1, 1],
#        [1, 1, 1, 1, 1],
#        [0, 1, 1, 1, 1],
#        [0, 0, 0, 1, 1],
#        [0, 0, 0, 0, 0]]), np.array([[0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0]]), np.array([[0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0]])]


# with rio.open('out_qa1.tif', 'w', driver='GTiff', nodata=None, width=max_x, height=max_y,
#           count=len(qa_count_arrays), dtype=dtype, crs=crs, transform=transform) as out_dataset:

#     print(f'window: {window}')
#     # Write the entire 3D array at once
#     out_dataset.write(np.array(qa_count_arrays), window=((window.row_off, window.row_off + window.height),
#                                               (window.col_off, window.col_off + window.width)))


# print(f'TYPE: {type(comp_window_arr[1])}')

# type(np.array(qa_count_arrays))


# import sys

# # Save the current system output (stdout)
# original_stdout = sys.stdout

# # Specify the file where you want to redirect the output
# output_file_path = 'output.txt'

# # Open the file in write mode
# with open(output_file_path, 'w') as file:
#     # Redirect the system output to the file
#     sys.stdout = file
    
#     # Now, any print statements will be written to the file
#     print('Hello, World!')
#     print('This is a system output.')

# # Restore the original system output
# sys.stdout = original_stdout

# # File has been closed automatically due to the 'with' statement


# boo_arr = [np.array([[[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]],

#        [[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]],

#        [[ True,  True,  True,  True,  True],
#         [ True,  True,  True,  True,  True],
#         [ True,  True,  True,  True,  True],
#         [ True,  True,  True,  True,  True],
#         [ True,  True,  True,  True,  True]]]), np.array([[[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]],

#        [[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]],

#        [[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]]]), np.array([[[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]],

#        [[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]],

#        [[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]]]), np.array([[[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]],

#        [[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]],

#        [[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]]]), np.array([[[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]],

#        [[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]],

#        [[False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False],
#         [False, False, False, False, False]]])]


# np.bitwise_or.reduce(boo_arr, axis=1) #this may tell us that the qa_fill had more of an influence on the std_mask results


# binary_qa_arrays = [arr.astype(int) for arr in boo_arr]

# # Assuming comp_window_arr.shape[0] gives the correct number of bands
# with rio.open(out_qa_tif, 'w', driver='GTiff', nodata=None, width=max_x, height=max_y,
#               count=len(binary_qa_arrays), dtype=dtype, crs=crs, transform=transform) as out_qa_tif:
  
#     for window in windows:
        
#         # Write the entire 3D array at once
#         out_qa_tif.write(comp_window_arr[1], window=((window.row_off, window.row_off + window.height),
#                                                   (window.col_off, window.col_off + window.width)))
        
#         log.info(f'PROCESSED WINDOW: {window} \n\n')


# ### **Fill Composite Xarray**

# # Define the band numbers
# band_numbers = sr_bandnumbers(sensor)

# # Create an empty xarray dataset
# comp_xarr = xr.Dataset()

# # Set coordinate values for 'x' and 'y' based on the shape of 'out_comp_arr'
# comp_xarr['x'] = np.arange(out_comp_arr.shape[2])
# comp_xarr['y'] = np.arange(out_comp_arr.shape[1])

# # Iterate through the bands and add them to the dataset
# for band_number, band_data in zip(band_numbers, out_comp_arr):
#     data_var_name = f'band_{band_number}'
#     comp_xarr[data_var_name] = (['y', 'x'], band_data)  # Note the order of dimensions

# # Optionally, set metadata or attributes for the dataset or data variables
# comp_xarr.attrs['description'] = (f'{sensor} Landsat Sensor Bands')

# # Display the resulting xarray dataset
# print(comp_xarr)


# fig, ax = plt.subplots(figsize=(5, 5))

# comp_xarr[["band_3", "band_2", "band_1"]].to_array().plot.imshow(robust=True, ax=ax)
# ax.set_title("Composite");


# !find ~/ -name "lsc2ard.py"


# import shutil

# # Source directory: lcnext-core/lcnext
# source_directory = 'lcnext-core/lcnext'

# # Destination directory: lcnext-trr
# destination_directory = 'lcnext_trr'

# # Copy everything from the source to the destination
# shutil.copytree(source_directory, destination_directory)


# !python3 /home/jovyan/delete_all.py -d '/home/jovyan/LCNEXT/lcnext-trr'


# ids=deets

# env = rio_env()

# big_arr = np.zeros(shape=(6, len(ids), 5000, 5000), dtype=int) 

# for idx1, d in enumerate(ids):
#     log.info(f'Pulling data for {d}')
#     layers = sr_std_layers(d)

#     for idx2, b in enumerate(layers):
#         log.info(f'Retrieving layer {b}')
#         big_arr[idx2, idx1] = read_raster_pure_rio('/'.join([d.std_path, b]), env, boundless=True)


# from shapely.geometry import box
# def read_bounds(path: str, env: rio.env, index=1, window=None, boundless=False, fill_value=None):
#     try:
#         with env:
#             with rio.open('s3://' + path) as src:
#                 bounds = src.bounds
#                 outline_polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
#                 return gp.GeoDataFrame(geometry=[outline_polygon], crs=src.crs)
#     except:
#         print(path)
#         raise


# import rasterio

# def get_tif_extents(tif_path):
#     with rasterio.open(tif_path) as src:
#         bounds = src.bounds
#     return bounds

# # Replace 'your_file.tif' with the actual path to your GeoTIFF file
# tif_extents = get_tif_extents('your_file.tif')
# print(f'The extents of the GeoTIFF file are: {tif_extents}')


# bounds = []

# env = rio_env()

# for d in deets:
#     log.info(f'Pulling data for {d}')
#     layers = sr_std_layers(d)
    
#     b = layers[0]
    
#     log.info(f'Retrieving bounds from first layer {b}')
#     bounds.append(read_bounds('/'.join([d.std_path, b]), env, boundless=False))


# import geopandas as gp

# # Specify the path to the shapefile
# shapefile_path = 'CONUS_ARD_shapefile/conus_c2_ard_grid.shp'

# # Open the shapefile using GeoPandas
# gdf = gp.read_file(shapefile_path)

# # Display basic information about the GeoDataFrame
# print("Shape of the GeoDataFrame:", gdf.shape)
# print("\nColumns in the GeoDataFrame:", gdf.columns)

# # Display the GeoDataFrame
# print("\nFirst few rows of the GeoDataFrame:")
# print(gdf.head())


# gdf[(gdf['h'] == horiz) & (gdf['v'] == vert)]['geometry']


# import folium
# from folium.plugins import MarkerCluster
# from shapely.geometry import mapping

# # Define an HTML color code (e.g., blue color)
# aoi_color = "#0489B1"
# scene_color = "#A4A4A4"

# gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84 (latitude, longitude)

# # Filter the GeoDataFrame based on h and v values
# filtered_gdf = gdf[(gdf['h'] == horiz) & (gdf['v'] == vert)]

# # Create a Folium map centered at the mean of the filtered geometries
# center_lat, center_lon = filtered_gdf['geometry'].centroid.y.mean(), filtered_gdf['geometry'].centroid.x.mean()
# m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

# # Create a MarkerCluster layer for better visualization (optional)
# marker_cluster = MarkerCluster().add_to(m)

# # Iterate over filtered GeoDataFrame rows and add each geometry to the map
# for idx, row in filtered_gdf.iterrows():
#     geojson_data = mapping(row['geometry'])
#     folium.GeoJson(geojson_data, 
#                    tooltip=f"h: {row['h']},v: {row['v']}", 
#                    style_function=lambda x: {
#                        "fillColor": aoi_color,  # Change the fill color to blue
#                        "color" : aoi_color
#                    }
#                   ).add_to(marker_cluster)


# # Add extent of scenes
# for b in bounds:
#     b = b.to_crs(epsg=4326)  # Convert to WGS84 (latitude, longitude)
#     folium.GeoJson(
#         b['geometry'],
#         name="Additional Polygon",
#         style_function=lambda feature: {
#             "fillColor": None,  # NO fill
#             "color": scene_color,  # Change the border color to black
#             "weight" : 1, # border line thickness
#         },
#     ).add_to(m)
    

# # Display the map
# m


# Record the start time
#start_time = datetime.now()

# Your code to be measured for processing time
# ...

# # Record the end time and calculate the elapsed time
# elapsed_time = datetime.now() - start_time

# # Convert elapsed time to seconds
# elapsed_seconds = elapsed_time.total_seconds()

# # Print the result in minutes or hours
# if elapsed_time.total_seconds() < 60:
#     print(f"Processing time: {elapsed_time.total_seconds():.2f} seconds")
# elif elapsed_time.total_seconds() < 3600:
#     print(f"Processing time: {elapsed_time.total_seconds()/60:.2f} minutes")
# else:
#     print(f"Processing time: {elapsed_time.total_seconds()/3600:.2f} hours")


# def calculate_qa_percent_true(qa_arrays: np.ndarray, ids) -> np.ndarray:
  
#     """ 
#     The calculate_qa_percent_true script is designed to calculate the percentage of true pixel values 
#     within each observation (ARD tiles stored as matrices) and grouped in by assessment (QA) layers. The QA layers, 
#     including parameters such as fill, cirrus, cloud, cloud shadow, and snow, are derived from the lsc2ard library. 
#     The script computes the percentage of true values for each array for each observation within these QA layers and organizes the results 
#     into a dictionary, providing key-wise lists of indices and corresponding percentages.
    
#     Input is a dictionary of qa_arrays using the repective qa_filter to fill in values from lsc2ard imported as c2ard in this case
#             e.g: qa_arrays = {
#             'qa_fill': c2ard.qa_fill(qa_arr),
#             'qa_cirrus': c2ard.qa_cirrus(qa_arr),
#             'qa_cloud': c2ard.qa_cloud(qa_arr),
#             'qa_cl_shadow': c2ard.qa_cl_shadow(qa_arr),
#             'qa_snow': c2ard.qa_snow(qa_arr)
#             }
            
#     Example result:
#             {'qa_fill': [{'index': 0, 'percent_true': 0.0},
#           {'index': 1, 'percent_true': 100.0},
#           {'index': 2, 'percent_true': 0.0}],
#          'qa_cirrus': [{'index': 0, 'percent_true': 0.0},
#           {'index': 1, 'percent_true': 0.0},
#           {'index': 2, 'percent_true': 0.0}],
#          'qa_cloud': [{'index': 0, 'percent_true': 0.0},
#           {'index': 1, 'percent_true': 0.0},
#           {'index': 2, 'percent_true': 0.0}],
#          'qa_cl_shadow': [{'index': 0, 'percent_true': 0.0},
#           {'index': 1, 'percent_true': 0.0},
#           {'index': 2, 'percent_true': 0.0}],
#          'qa_snow': [{'index': 0, 'percent_true': 0.0},
#           {'index': 1, 'percent_true': 0.0},
#           {'index': 2, 'percent_true': 0.0}]}
      
#       Where the qa_fill is 100% TRUE for pixels in the observation at index '1' (2nd ARD tile)
#     """
#     # New dictionary to store results for each key
#     results_dict = {}
    

#     # Loop through the dictionary
#     for key, qa_array in qa_arrays.items():
#         print(f"{key.upper()}:")
            
#         # Initialize an empty list to store the results
#         results = []

#         # Loop through the collection of arrays
#         for idx, qa_arr in enumerate(qa_array):
            
#             # Calculate the percentage of true values
#             percent_true = (np.sum(qa_arr) / qa_arr.size) * 100
            
#             if percent_true > 0:
#                 print(f"    {ids[idx].root_id}: {percent_true:.2f}%")
            
#             # Append the results to the list
#             results.append({
#                 'index': idx,
#                 'root_id' : ids[idx].root_id,
#                 'percent_true': percent_true
#             })
            
#         results_dict[key] = results;

#     return results_dict





