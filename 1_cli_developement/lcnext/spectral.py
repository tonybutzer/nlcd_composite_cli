"""
Calculate spectral indices or other common spectral operations
"""
import numpy as np


########################################################
# Helper functions
########################################################
def divide(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """
    Help function for dividing two arrays
    provide some handling of divide by 0
    """
    out = np.zeros_like(top, dtype=float)
    return np.divide(top,
                     bottom,
                     where=bottom != 0,
                     out=out)


def normalized_difference(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Generalized function for calculating most standard spectral indices
    (x1 - x2) / (x1 + x2)
    """
    return divide(x1 - x2,
                  x1 + x2)

########################################################
# spectral indices
########################################################
def nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Normalized Burn Ratio

    NBR is used to identify burned areas and provide a measure of burn severity. It is
    calculated as a ratio between the NIR and SWIR values in traditional fashion.

    Landsat 4-7: NBR = (Band 4 – Band 7) / (Band 4 + Band 7)
    Landsat 8-9: NBR = (Band 5 – Band 7) / (Band 5 + Band 7)
    """
    return normalized_difference(nir, swir)


def nbr2(swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    Normalized Burn Ratio 2

    NBR2 modifies the Normalized Burn Ratio (NBR) to highlight water sensitivity in
    vegetation and may be useful in post-fire recovery studies. NBR2 is calculated as
    a ratio between the SWIR values, substituting the SWIR1 band for the NIR band used in NBR.

    Landsat 4-7: NBR2 = (Band 5 – Band 7) / (Band 5 + Band 7)
    Landsat 8-9: NBR2 = (Band 6 – Band 7) / (Band 6 + Band 7)
    """
    return normalized_difference(swir1, swir2)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index

    NDVI is used to quantify vegetation greenness and is useful in understanding vegetation
    density and assessing changes in plant health. NDVI is calculated as a ratio between the
    red and near infrared values in traditional fashion.

    Landsat 4-7: NDVI = (Band 4 – Band 3) / (Band 4 + Band 3)
    Landsat 8-9: NDVI = (Band 5 – Band 4) / (Band 5 + Band 4)
    """
    return normalized_difference(nir, red)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray, g: float, c1: float, c2: float, l: float) -> np.ndarray:
    """
    Enhanced Vegetation Index

    EVI is similar to Normalized Difference Vegetation Index (NDVI) and can be used to
    quantify vegetation greenness. However, EVI corrects for some atmospheric conditions
    and canopy background noise and is more sensitive in areas with dense vegetation. It
    incorporates an “L” value to adjust for canopy background, “C” values as coefficients
    for atmospheric resistance, and values from the blue band (B).  These enhancements
    allow for index calculation as a ratio between the R and NIR values, while reducing
    the background noise, atmospheric noise, and saturation in most cases.

    Landsat 4-7: EVI = 2.5 * ((Band 4 – Band 3) / (Band 4 + 6 * Band 3 – 7.5 * Band 1 + 1))
    Landsat 8-9: EVI = 2.5 * ((Band 5 – Band 4) / (Band 5 + 6 * Band 4 – 7.5 * Band 2 + 1))
    """
    return g * divide(nir - red,
                      nir + c1 * red - c2 * blue + l)


def savi(nir: np.ndarray, red: np.ndarray, l: float) -> np.ndarray:
    """
    Soil Adjusted Vegetation Index

    SAVI is used to correct Normalized Difference Vegetation Index (NDVI) for the influence of soil brightness
    in areas where vegetative cover is low.

    Landsat 4-7: SAVI = ((Band 4 – Band 3) / (Band 4 + Band 3 + 0.5)) * (1.5)
    Landsat 8-9: SAVI = ((Band 5 – Band 4) / (Band 5 + Band 4 + 0.5)) * (1.5)
    """
    return divide(nir - red,
                  (nir + red + l)) * (1 + l)


def msavi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Modified Soil Adjusted Vegetation Index

    MSAVI minimizes the effect of bare soil on the Soil Adjusted Vegetation Index (SAVI). MSAVI is
    calculated as a ratio between the R and NIR values with an inductive L function applied to
    maximize reduction of soil effects on the vegetation signal.

    Landsat 4-7: MSAVI = (2 * Band 4 + 1 – sqrt ((2 * Band 4 + 1)2 – 8 * (Band 4 – Band 3))) / 2
    Landsat 8-9: MSAVI = (2 * Band 5 + 1 – sqrt ((2 * Band 5 + 1)2 – 8 * (Band 5 – Band 4))) / 2
    """
    return ((2 * nir + 1 -
             np.sqrt((2 * nir + 1) ** 2 -
                     8 * (nir - red)))
            / 2)


def ndmi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Moisture Index

    NDMI is used to determine vegetation water content. It is calculated as a ratio between
    the NIR and SWIR values in traditional fashion.

    Landsat 4-7: NDMI = (Band 4 – Band 5) / (Band 4 + Band 5)
    Landsat 8-9: NDMI = (Band 5 – Band 6) / (Band 5 + Band 6)
    """
    return normalized_difference(nir, swir)


def ndsi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Snow Index

    The normalized difference between spectral bands green (G) and the shortwave infrared (SWIR).
    The NDSI is particularly useful for separating snow from vegetation, soils, and lithology endmembers.

    Landsat 4-7, NDSI = (Band 2 – Band 5) / (Band 2 + Band 5)
    Landsat 8-9, NDSI = (Band 3 – Band 6) / (Band 3 + Band 6)
    """
    return normalized_difference(green, swir)


########################################################
# spectral indices for specific sensors
########################################################
def landsat_sr_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Enhanced Vegetation Index for Landsat surface reflectance

    provides default values for "g", "c1", "c2", "l"

    https://www.usgs.gov/landsat-missions/landsat-enhanced-vegetation-index
    """
    return evi(nir, red, blue, g=2.5, c1=6, c2=7.5, l=1)


def landsat_sr_savi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Soil Adjusted Vegetation Index for Landsat surface reflectance

    provides default value for "l"

    https://www.usgs.gov/landsat-missions/landsat-soil-adjusted-vegetation-index
    """
    return savi(nir, red, l=0.5)
