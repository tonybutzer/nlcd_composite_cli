from rasterio import Affine


def std_affine(ul_x: int, ul_y: int, resolution: int = 30.0) -> Affine:
    """
    Build a standard north up affine for transformations based on upper-left coordinates
    """
    return Affine.from_gdal(ul_x, resolution, 0.0, ul_y, 0.0, -resolution)
