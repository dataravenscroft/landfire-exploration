"""
Download a digital elevation model (DEM) and derive terrain variables.

Elevation data source: USGS 3D Elevation Program (3DEP)
  - 1/3 arc-second (~10 m) or 1 arc-second (~30 m) products
  - py3dep provides a clean Python interface to the 3DEP elevation service
  - Reference: https://www.usgs.gov/3d-elevation-program

Terrain variables
-----------------
Elevation  : metres above sea level (directly from DEM)
Slope      : steepness in degrees — key fire-spread driver (steeper = faster)
Aspect     : compass direction of slope face (0/360 = north, 90 = east, etc.)
             North-facing slopes are typically cooler/moister; south-facing are
             drier and more fire-prone at mid-latitudes in the Northern Hemisphere.

Slope and aspect are computed from the DEM using finite-difference gradients
(numpy.gradient), which is equivalent to the standard ArcGIS / GDAL algorithm
for moderate resolutions.

Integration with the conceptual framework
-----------------------------------------
Topography is a quasi-static conditioning factor:
  fuel + weather + topography → fire-behavior environment

Unlike fuel models (which change after disturbance) or weather (daily variation),
topography is fixed. It modulates fire spread primarily through:
  1. Slope-driven spread acceleration (Rothermel model)
  2. Aspect-driven differences in fuel moisture and curing rate
  3. Elevation-driven differences in fuel type and phenology
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import Transformer

from . import BBOX_WGS84, LANDFIRE_CRS, PROCESSED_DIR


# ---------------------------------------------------------------------------
# DEM download via py3dep
# ---------------------------------------------------------------------------

def download_dem(
    bbox_wgs84: Tuple[float, float, float, float] = BBOX_WGS84,
    resolution: int = 30,
    output_path: Optional[Path] = None,
) -> xr.DataArray:
    """Download a DEM for the study area from the USGS 3DEP service.

    Uses py3dep, which queries the 3DEP elevation API and returns an
    xarray DataArray. The DEM is in the WGS84 CRS by default; we reproject
    to match the LANDFIRE grid CRS (UTM Zone 13N) before saving.

    Parameters
    ----------
    bbox_wgs84  : (lon_min, lat_min, lon_max, lat_max)
    resolution  : target resolution in metres (30 = native Sentinel-1 LANDFIRE res)
    output_path : if provided, save the reprojected DEM as a GeoTIFF

    Returns
    -------
    xr.DataArray with elevation in metres, CRS=LANDFIRE_CRS
    """
    try:
        import py3dep
    except ImportError:
        raise ImportError(
            "py3dep is required for DEM download. "
            "Install with: pip install py3dep"
        )

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    print(f"Downloading 3DEP DEM at {resolution} m resolution...")
    print(f"  AOI: {bbox_wgs84}")

    # py3dep expects (lon_min, lat_min, lon_max, lat_max) with CRS="EPSG:4326"
    dem_wgs84 = py3dep.get_map(
        "DEM",
        (lon_min, lat_min, lon_max, lat_max),
        resolution=resolution,
        crs="EPSG:4326",
    )

    print(f"  Raw DEM shape: {dem_wgs84.shape}  CRS: {dem_wgs84.rio.crs}")

    # Reproject to LANDFIRE CRS
    dem = dem_wgs84.rio.reproject(LANDFIRE_CRS, resolution=resolution)
    dem.name = "elevation"
    dem.attrs["units"] = "metres"
    dem.attrs["long_name"] = "Elevation above sea level"
    dem.attrs["source"] = "USGS 3DEP"

    print(f"  Reprojected DEM shape: {dem.shape}  CRS: {dem.rio.crs}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dem.rio.to_raster(output_path, compress="lzw")
        print(f"  Saved to: {output_path}")

    return dem


def load_or_download_dem(
    bbox_wgs84: Tuple[float, float, float, float] = BBOX_WGS84,
    resolution: int = 30,
    processed_dir: Path = PROCESSED_DIR,
) -> xr.DataArray:
    """Load a cached DEM if available, otherwise download."""
    dem_path = processed_dir / "elevation.tif"
    if dem_path.exists():
        print(f"Loading cached DEM from {dem_path}")
        dem = xr.open_dataarray(dem_path, engine="rasterio")
        dem = dem.squeeze("band", drop=True)
        dem.name = "elevation"
        return dem
    return download_dem(bbox_wgs84, resolution, output_path=dem_path)


# ---------------------------------------------------------------------------
# Terrain derivatives
# ---------------------------------------------------------------------------

def compute_slope(dem: xr.DataArray) -> xr.DataArray:
    """Compute slope in degrees from a projected DEM.

    Uses numpy.gradient to estimate the rise/run in x and y directions,
    then converts to degrees. The DEM must be in a metric projection so
    that cell spacing is in metres.

    Parameters
    ----------
    dem : elevation DataArray in metres, projected CRS (e.g. UTM)

    Returns
    -------
    slope DataArray in degrees (0 = flat, 90 = vertical cliff)
    """
    # Cell size in metres from the spatial coordinates
    dy = float(abs(dem.y.values[1] - dem.y.values[0]))  # row spacing (m)
    dx = float(abs(dem.x.values[1] - dem.x.values[0]))  # column spacing (m)

    elev = dem.values.astype(float)
    # gradient returns [row_gradient, col_gradient] i.e. [N-S, W-E]
    grad_y, grad_x = np.gradient(elev, dy, dx)

    # Slope magnitude: tan(slope) = sqrt(dz/dx^2 + dz/dy^2)
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_deg = np.degrees(slope_rad)

    slope = xr.DataArray(
        slope_deg,
        coords=dem.coords,
        dims=dem.dims,
        name="slope",
        attrs={
            "units":     "degrees",
            "long_name": "Terrain slope",
            "valid_min": 0.0,
            "valid_max": 90.0,
        },
    )
    slope.rio.write_crs(dem.rio.crs, inplace=True)
    return slope


def compute_aspect(dem: xr.DataArray) -> xr.DataArray:
    """Compute aspect (slope direction) in degrees clockwise from north.

    Convention: 0/360 = north, 90 = east, 180 = south, 270 = west.
    Flat areas (slope ≈ 0) are assigned -1 by convention.

    Parameters
    ----------
    dem : elevation DataArray in metres, projected CRS

    Returns
    -------
    aspect DataArray in degrees (clockwise from north). Flat cells = -1.
    """
    dy = float(abs(dem.y.values[1] - dem.y.values[0]))
    dx = float(abs(dem.x.values[1] - dem.x.values[0]))

    elev   = dem.values.astype(float)
    grad_y, grad_x = np.gradient(elev, dy, dx)

    # atan2 convention: angle from +x axis, counterclockwise
    # Convert to clockwise-from-north for geospatial convention
    aspect_rad = np.arctan2(-grad_x, -grad_y)  # Note: y-axis points south in rasters
    aspect_deg = np.degrees(aspect_rad) % 360

    # Mark flat cells
    slope_mag = np.sqrt(grad_x**2 + grad_y**2)
    aspect_deg[slope_mag < 1e-6] = -1.0

    aspect = xr.DataArray(
        aspect_deg,
        coords=dem.coords,
        dims=dem.dims,
        name="aspect",
        attrs={
            "units":     "degrees clockwise from north",
            "long_name": "Terrain aspect",
            "valid_min": 0.0,
            "valid_max": 360.0,
            "flag_value": -1.0,
            "flag_meaning": "flat terrain",
        },
    )
    aspect.rio.write_crs(dem.rio.crs, inplace=True)
    return aspect


def compute_terrain_layers(
    dem: xr.DataArray,
    output_dir: Optional[Path] = None,
) -> Dict[str, xr.DataArray]:
    """Compute elevation, slope, and aspect from a DEM.

    Returns a dict with keys 'elevation', 'slope', 'aspect'.
    Optionally saves each layer as a GeoTIFF.
    """
    slope  = compute_slope(dem)
    aspect = compute_aspect(dem)

    layers = {"elevation": dem, "slope": slope, "aspect": aspect}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, da in layers.items():
            out = output_dir / f"{name}.tif"
            da.rio.to_raster(out, compress="lzw")
            print(f"  Saved: {out.name}")

    return layers


def add_terrain_to_dataset(
    ds: xr.Dataset,
    terrain_layers: Dict[str, xr.DataArray],
    reference_key: str = "evt",
) -> xr.Dataset:
    """Add terrain variables to an existing LANDFIRE Dataset.

    Terrain layers are reprojected to match the LANDFIRE grid if needed.
    Only cells overlapping the LANDFIRE extent are retained.
    """
    from .align_rasters import align

    reference = {k: ds[k] for k in [reference_key] if reference_key in ds}
    if not reference:
        reference = {k: ds[list(ds.data_vars)[0]] for k in [reference_key]}
        reference = {list(ds.data_vars)[0]: ds[list(ds.data_vars)[0]]}

    ref_da = ds[reference_key] if reference_key in ds else ds[list(ds.data_vars)[0]]

    for key, da in terrain_layers.items():
        # float64 + no nodata causes rasterio warp to silently return all NaN;
        # cast to float32 and set nodata explicitly before reprojecting.
        da = da.astype(np.float32)
        da.rio.write_crs(da.rio.crs, inplace=True)
        da.rio.write_nodata(np.nan, inplace=True)
        da = da.rio.reproject_match(ref_da)
        # Force exact coordinate labels to prevent xarray NaN-fill on assignment
        da = xr.DataArray(
            da.values,
            coords=ref_da.coords,
            dims=ref_da.dims,
            name=key,
        )
        da.rio.write_crs(ref_da.rio.crs, inplace=True)
        ds[key] = da

    return ds
