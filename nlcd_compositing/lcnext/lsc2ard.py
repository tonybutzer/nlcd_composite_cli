"""
Functionality specifically targetted towards Landsat Collection 2 ARD data, stored with
specific pathing.
"""

import datetime as dt
from dataclasses import dataclass
import re

from rasterio.transform import rowcol

from lcnext.static import LS_ARD_C2_CU_TileAff
from lcnext.static import LS_ARD_C2_HI_TileAff
from lcnext.static import LS_ARD_C2_AK_TileAff
from lcnext.static import LS_ARD_C1_SR_MULT
from lcnext.static import LS_ARD_C1_SR_ADD
from lcnext.static import LS_ARD_C1_SR_NODATA
from lcnext.static import LS_ARD_C1_BT_MULT
from lcnext.static import LS_ARD_C1_BT_ADD
from lcnext.static import LS_ARD_C1_BT_NODATA
from lcnext.static import LS_ARD_C1_TOA_MULT
from lcnext.static import LS_ARD_C1_TOA_ADD
from lcnext.static import LS_ARD_C1_TOA_NODATA
from lcnext.static import LS_ARD_C2_SR_MULT
from lcnext.static import LS_ARD_C2_SR_ADD
from lcnext.static import LS_ARD_C2_SR_NODATA
from lcnext.static import LS_ARD_C2_BT_MULT
from lcnext.static import LS_ARD_C2_BT_ADD
from lcnext.static import LS_ARD_C2_BT_NODATA
from lcnext.static import LS_ARD_C2_TOA_MULT
from lcnext.static import LS_ARD_C2_TOA_ADD
from lcnext.static import LS_ARD_C2_TOA_NODATA

# Typing imports
from typing import List
from typing import Tuple

from fsspec import AbstractFileSystem
import numpy as np
from rasterio import Affine


C2ARD_PATTERN = re.compile(r"(?P<platform>L[CEOT]0[45789])_"
                           r"(?P<region>[A-Z]{2})_"
                           r"(?P<horiz>[0-9]{3})(?P<vert>[0-9]{3})_"
                           r"(?P<acquired>[0-9]{8})_"
                           r"(?P<prod_date>[0-9]{8})_02")


@dataclass(frozen=True, slots=True)
class LandsatARDObservation:
    """
    Keep track of the details for a particular Landsat ARD observation stored somewhere.
    """
    root_id: str
    std_path: str
    acquired: dt.datetime.date
    region: str
    platform: str
    sensor: str
    horiz: int
    vert: int
    tileid: str
    prod_date: dt.datetime.date


def id_search(path_or_id: str) -> dict:
    """
    Apply the C2 ARD base id regex to search for and identify characteristics from the name
    """
    match = C2ARD_PATTERN.search(path_or_id)

    if not match:
        raise ValueError(f'No matching C2 ARD string in {path_or_id}')

    info = match.groupdict()
    info['root_id'] = match.group()
    info['sensor'] = platform_to_sensor(info['platform'])

    return info

def platform_to_sensor(platform: str) -> str:
    """
    Match the platform ID to the onboard sensor ID
    LC09, LC08, LT05, LE07, LT04
    """
    match platform:
        case 'LC09' | 'LC08':
            return 'oli-tirs'
        case 'LE07':
            return 'etm'
        case 'LT05' | 'LT04':
            return 'tm'
        case _:
            raise ValueError(f'Platform not recognized: {platform}')


def std_obs_path(sensor: str, year: str | int, region: str, horiz: str, vert: str, root_id: str):
    """
    Build the standard C2 ARD root pathing based on the given observation information
    """
    return f'usgs-landsat-ard/collection02/{sensor}/{year}/{region}/{horiz}/{vert}/{root_id}'


def obs_deets(path_or_id: str) -> LandsatARDObservation:
    """
    Build the LandsatARDObservation object based around how Landsat ARD is stored on S3.
    """
    details = id_search(path_or_id)

    return LandsatARDObservation(root_id=details['root_id'],
                                 std_path=std_obs_path(details['sensor'],
                                                       details['acquired'][:4],
                                                       details['region'],
                                                       details['horiz'],
                                                       details['vert'],
                                                       details['root_id']),
                                 acquired=dt.datetime.strptime(details['acquired'], '%Y%m%d').date(),
                                 region=details['region'],
                                 platform=details['platform'],
                                 sensor=details['sensor'],
                                 horiz=int(details['horiz']),
                                 vert=int(details['vert']),
                                 tileid=''.join([details['horiz'], details['vert']]),
                                 prod_date=dt.datetime.strptime(details['prod_date'], '%Y%m%d').date())

def year_deets(fs: AbstractFileSystem, year: int, sensor: str, region: str, horiz: int, vert: int) -> List[LandsatARDObservation]:
    """
    Build out the details for each observation for a given year/sensor
    """
    return [obs_deets(p)
            for p in
            fs.ls(f'usgs-landsat-ard/collection02/{sensor}/{year}/{region}/{horiz:03}/{vert:03}')]

def find_observations(fs: AbstractFileSystem, start_date: dt.datetime.date, end_date: dt.datetime.date, sensor: str, region: str, horiz: int, vert: int):
    """
    Find all observations to fit within the specified start_date/end_date for a given sensor
    """
    ret = []
    for year in range(start_date.year, end_date.year + 1):
        ret.extend(filter(lambda x: start_date <= x.acquired <= end_date, year_deets(fs, year, sensor, region, horiz, vert)))

    return ret

def sr_bandnumbers(sensor: str) -> List[int]:
    """
    SR band numbers for the given sensor
    oli-tirs, tm, etm
    """
    if sensor == 'oli-tirs':
        return [2, 3, 4, 5, 6, 7]
    elif (sensor == 'tm') | (sensor == 'etm'):
        return [1, 2, 3, 4, 5, 7]

    raise ValueError

def bt_bandnumbers(sensor: str) -> List[int]:
    """
    BT band numbers for the given sensor
    oli-tirs, tm, etm
    """
    if sensor == 'oli-tirs':
        return [10, 11]
    elif (sensor == 'tm') | (sensor == 'etm'):
        return [6]

    raise ValueError


def regional_affine(region: str) -> Affine:
    """
    Pair up the region string with the defined spatial Affine
    """
    if region == 'CU':
        return LS_ARD_C2_CU_TileAff
    elif region == 'AK':
        return LS_ARD_C2_AK_TileAff
    elif region == 'HI':
        return LS_ARD_C2_HI_TileAff

    raise ValueError

def contains_hv(region: str, xs: List[float] | float, ys: List[float] | float) -> Tuple[List[int]] | Tuple[int]:
    """
    Determine the H/V that the point(s) fall in
    """
    rows, cols = rowcol(regional_affine(region),
                        xs,
                        ys)

    return cols, rows

########################################################
# Helper functions for scaling and descaling L1/2 products such as SR/BT/TOA
########################################################
def find_nodata(arr: np.ndarray, nodata: float = np.nan) -> np.ndarray:
    """
    Helper function for finding the nodata values within the given array due to the
    special nature of nan's

    If precision becomes an issue, then logic can be added to take advantage of numpy isclose
    https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
    """
    if np.isnan(nodata):
        return np.isnan(arr)
    else:
        return arr == nodata

def descale(data: np.ndarray,
            mult: float,
            add: float,
            in_nodata: float = np.nan,
            out_nodata: float = np.nan) -> np.ndarray:
    """
    Descale some array of values (usually SR/TOA/BT)
    f(x) = x * mult + add
    """
    out = (data.astype(float) * mult + add)
    out[find_nodata(data, in_nodata)] = out_nodata

    return out

def scale(data: np.ndarray,
          mult: float,
          add: float,
          in_nodata: float = np.nan,
          out_nodata: float = np.nan) -> np.ndarray:
    """
    Scale some array of values (usually SR/TOA/BT)
    f(x) = (x + add) / mult
    """
    out = (data.astype(float) + add) / mult
    out[find_nodata(data, in_nodata)] = out_nodata

    return out

def sr_to_c1(sr_data: np.ndarray) -> np.ndarray:
    """
    Rescale the Collection 2 surface reflectance values to match Collection 1 scaling
    """
    sr = descale(sr_data,
                 mult=LS_ARD_C2_SR_MULT,
                 add=LS_ARD_C2_SR_ADD,
                 in_nodata=LS_ARD_C2_SR_NODATA,
                 out_nodata=LS_ARD_C1_SR_NODATA)

    return scale(sr,
                 mult=LS_ARD_C1_SR_MULT,
                 add=LS_ARD_C1_SR_ADD,
                 in_nodata=LS_ARD_C1_SR_NODATA,
                 out_nodata=LS_ARD_C1_SR_NODATA).astype(np.int16)

def bt_to_c1(bt_data: np.ndarray) -> np.ndarray:
    """
    Rescale the Collection 2 brightness temperature values to match Collection 1 scaling
    """
    bt = descale(bt_data,
                 mult=LS_ARD_C2_BT_MULT,
                 add=LS_ARD_C2_BT_ADD,
                 in_nodata=LS_ARD_C2_BT_NODATA,
                 out_nodata=LS_ARD_C1_BT_NODATA)

    return scale(bt,
                 mult=LS_ARD_C1_BT_MULT,
                 add=LS_ARD_C1_BT_ADD,
                 in_nodata=LS_ARD_C1_BT_NODATA,
                 out_nodata=LS_ARD_C1_BT_NODATA).astype(np.int16)

################################################
# QA related functions
################################################
def qa_bitmask(qa_arr: np.ndarray, bit: int) -> np.ndarray:
    """
    Create a boolean mask for where the bit is set in the given array
    """
    return (qa_arr & 1 << bit) > 0

def qa_fill(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for where the fill QA bit is set
    """
    return qa_bitmask(qa_arr, 0)

def qa_cl_dilated(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for where the dilated cloud QA bit is set
    """
    return qa_bitmask(qa_arr, 1)

def qa_cirrus(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for where the cirrus QA bit is set
    """
    return qa_bitmask(qa_arr, 2)

def qa_cloud(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for where the cloud QA bit is set
    """
    return qa_bitmask(qa_arr, 3)

def qa_cl_shadow(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for where the cloud shadow QA bit is set
    """
    return qa_bitmask(qa_arr, 4)

def qa_snow(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for where the snow QA bit is set
    """
    return qa_bitmask(qa_arr, 5)

def qa_clear(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for where the clear QA bit is set
    """
    return qa_bitmask(qa_arr, 6)

def qa_water(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for where the water QA bit is set
    """
    return qa_bitmask(qa_arr, 7)

def qa_cl_lconf(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for low confidence clouds
    """
    return qa_bitmask(qa_arr, 8) & ~qa_bitmask(qa_arr, 9)

def qa_cl_mconf(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for medium confidence clouds
    """
    return ~qa_bitmask(qa_arr, 8) & qa_bitmask(qa_arr, 9)

def qa_cl_hconf(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for high confidence clouds
    """
    return qa_bitmask(qa_arr, 8) & qa_bitmask(qa_arr, 9)

def qa_clsh_lconf(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for low confidence cloud shadow
    """
    return qa_bitmask(qa_arr, 10) & ~qa_bitmask(qa_arr, 11)

def qa_clsh_mconf(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for medium confidence cloud shadow
    """
    return ~qa_bitmask(qa_arr, 10) & qa_bitmask(qa_arr, 11)

def qa_clsh_hconf(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for high confidence cloud shadow
    """
    return qa_bitmask(qa_arr, 10) & qa_bitmask(qa_arr, 11)

def qa_snice_lconf(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for low confidence snow/ice
    """
    return qa_bitmask(qa_arr, 12) & ~qa_bitmask(qa_arr, 13)

def qa_snice_mconf(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for medium confidence snow/ice
    """
    return ~qa_bitmask(qa_arr, 12) & qa_bitmask(qa_arr, 13)

def qa_snice_hconf(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for high confidence snow/ice
    """
    return qa_bitmask(qa_arr, 12) & qa_bitmask(qa_arr, 13)

def qa_cirrus_lconf(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for low confidence cirrus
    """
    return qa_bitmask(qa_arr, 14) & ~qa_bitmask(qa_arr, 15)

def qa_cirrus_mconf(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for medium confidence cirrus
    """
    return ~qa_bitmask(qa_arr, 14) & qa_bitmask(qa_arr, 15)

def qa_cirrus_hconf(qa_arr: np.ndarray) -> np.ndarray:
    """
    Check pixels for high confidence cirrus
    """
    return qa_bitmask(qa_arr, 14) & qa_bitmask(qa_arr, 15)
