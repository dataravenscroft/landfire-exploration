"""
Align all LANDFIRE raster layers to a common CRS, resolution, and grid extent.

LANDFIRE layers delivered by LFPS should already share a common grid, but this
module verifies alignment explicitly and reprojects/resamples where needed.
It also handles clipping to a bounding box and building the final xr.Dataset
suitable for analysis.

Why alignment matters
---------------------
Even within the same LANDFIRE delivery, subtle differences can arise:
  - Floating-point coordinate offsets between bands
  - Different nodata encodings
  - Elevation (DEM) and ERA5 weather data arrive on entirely different grids
    and must be reprojected / interpolated before any combined analysis.

This module makes the alignment step explicit and testable rather than leaving
it implicit in analysis code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def _crs_string(da: xr.DataArray) -> str:
    crs = da.rio.crs
    return crs.to_epsg() and f"EPSG:{crs.to_epsg()}" or str(crs)


def verify_alignment(layers: Dict[str, xr.DataArray]) -> bool:
    """Check that all layers share CRS, resolution, and grid dimensions.

    Prints a summary table and returns True if fully aligned, False otherwise.
    Tolerates small floating-point differences in coordinates (< 1 mm).
    """
    if not layers:
        print("No layers to verify.")
        return True

    ref_key = next(iter(layers))
    ref = layers[ref_key]
    ref_crs  = ref.rio.crs
    ref_res  = ref.rio.resolution()
    ref_shape = ref.shape

    print(f"\nAlignment check (reference: '{ref_key}')")
    print(f"  CRS       : {_crs_string(ref)}")
    print(f"  Resolution: {ref_res[0]:.2f} × {ref_res[1]:.2f} m")
    print(f"  Shape     : {ref_shape}")
    print()

    header = f"  {'Layer':<16}  {'CRS':>12}  {'Shape':>14}  {'OK':>4}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_ok = True
    for key, da in layers.items():
        crs_ok    = CRS(da.rio.crs) == CRS(ref_crs)
        shape_ok  = da.shape == ref_shape
        coords_ok = (
            np.allclose(da.x.values, ref.x.values, atol=1e-3) and
            np.allclose(da.y.values, ref.y.values, atol=1e-3)
        )
        ok = crs_ok and shape_ok and coords_ok
        flag = "OK" if ok else "MISMATCH"
        print(f"  {key:<16}  {_crs_string(da):>12}  "
              f"{'×'.join(str(s) for s in da.shape):>14}  {flag:>4}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All layers are aligned.")
    else:
        print("\n  Mismatches detected — call align() to fix.")
    return all_ok


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align(
    layers: Dict[str, xr.DataArray],
    reference_key: Optional[str] = None,
    target_crs: Optional[str] = None,
    target_resolution: Optional[float] = None,
    resampling_categorical: str = "nearest",
    resampling_continuous: str = "bilinear",
    categorical_keys: Optional[List[str]] = None,
) -> Dict[str, xr.DataArray]:
    """Reproject and resample all layers to a common grid.

    By default, the first layer is used as the reference grid. Pass
    reference_key to choose a different anchor (e.g. the EVT layer).

    Parameters
    ----------
    layers               : dict of {key: DataArray} from load_data.load_all_layers()
    reference_key        : which layer defines the target grid
    target_crs           : override target CRS (EPSG string or pyproj-compatible)
    target_resolution    : override target resolution in metres (uses reference if None)
    resampling_categorical : rasterio resampling method for categorical layers
    resampling_continuous  : rasterio resampling method for continuous layers
    categorical_keys     : layer keys that should use nearest-neighbour resampling.
                           Defaults to ['evt', 'fuel_model', 'disturbance'].

    Returns
    -------
    dict of aligned {key: DataArray} — same keys as input
    """
    from rasterio.enums import Resampling

    if not layers:
        return {}

    if categorical_keys is None:
        categorical_keys = ["evt", "fuel_model", "disturbance"]

    resample_map = {
        "nearest":  Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic":    Resampling.cubic,
    }

    if reference_key is None:
        reference_key = next(iter(layers))
    ref = layers[reference_key]

    if target_crs is None:
        target_crs = str(ref.rio.crs)

    if target_resolution is None:
        res = abs(ref.rio.resolution()[0])
    else:
        res = target_resolution

    print(
        f"Aligning {len(layers)} layers → CRS={target_crs}  "
        f"resolution={res:.0f} m  (reference: '{reference_key}')"
    )

    aligned: Dict[str, xr.DataArray] = {}
    for key, da in layers.items():
        method_str = (
            resampling_categorical if key in categorical_keys
            else resampling_continuous
        )
        method = resample_map.get(method_str, Resampling.nearest)

        current_crs = str(da.rio.crs)
        current_res = abs(da.rio.resolution()[0])
        needs_warp  = (current_crs != target_crs) or (abs(current_res - res) > 0.1)
        needs_match = da.shape != ref.shape or not (
            np.allclose(da.x.values[:2], ref.x.values[:2], atol=1e-3)
        )

        if needs_warp:
            da = da.rio.reproject(
                target_crs, resolution=res, resampling=method
            )
        elif needs_match:
            da = da.rio.reproject_match(ref, resampling=method)

        aligned[key] = da
        print(f"  {key:<16}  {method_str}")

    return aligned


def clip_to_bbox(
    layers: Dict[str, xr.DataArray],
    bbox_wgs84: Tuple[float, float, float, float],
) -> Dict[str, xr.DataArray]:
    """Clip all layers to a bounding box given in WGS84.

    This reprojects the bbox into each layer's native CRS before clipping,
    so layers do not all need to share a CRS (though they should after align()).
    """
    import pyproj
    from shapely.geometry import box
    from shapely.ops import transform

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    wgs84_box = box(lon_min, lat_min, lon_max, lat_max)

    clipped: Dict[str, xr.DataArray] = {}
    for key, da in layers.items():
        layer_crs = pyproj.CRS(da.rio.crs)
        wgs84_crs = pyproj.CRS("EPSG:4326")

        if not layer_crs.equals(wgs84_crs):
            project = pyproj.Transformer.from_crs(
                wgs84_crs, layer_crs, always_xy=True
            ).transform
            clipping_box = transform(project, wgs84_box)
        else:
            clipping_box = wgs84_box

        clipped[key] = da.rio.clip_box(
            minx=clipping_box.bounds[0],
            miny=clipping_box.bounds[1],
            maxx=clipping_box.bounds[2],
            maxy=clipping_box.bounds[3],
            crs=da.rio.crs,
        )
    return clipped


def build_aligned_dataset(
    layers: Dict[str, xr.DataArray],
) -> xr.Dataset:
    """Merge an aligned dict of DataArrays into an xr.Dataset.

    Assumes all layers have already been aligned (same grid). The CRS is
    preserved via the rio accessor on the Dataset variables.
    """
    ds = xr.Dataset({key: da for key, da in layers.items()})
    return ds


def save_processed(
    layers: Dict[str, xr.DataArray],
    output_dir: Path,
    compress: bool = True,
) -> None:
    """Save aligned layers as GeoTIFFs in output_dir for fast reloading."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoding = {"compress": "lzw"} if compress else {}
    for key, da in layers.items():
        out_path = output_dir / f"{key}.tif"
        da.rio.to_raster(out_path, **encoding)
        print(f"  Saved: {out_path.name}")
