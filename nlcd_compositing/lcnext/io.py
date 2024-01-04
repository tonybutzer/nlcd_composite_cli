"""
Handling opening data, in whatever format (zarr, tif, whatever) where special logic
is required due to environment or other challenges
"""

import rasterio as rio

# Typing imports
import fsspec
import numpy as np
from rasterio.windows import Window


def fs_read_raster(path: str, fs: fsspec.AbstractFileSystem, index: int = 1, window: Window = None) -> np.ndarray:
    """
    General function to let fsspec do the filesystem lifting instead of rasterio/gdal

    Will also allow for better function chaining using fsspec
    """
    with fs.open(path) as f:
        return rio.open(f).read(index, window=window)
