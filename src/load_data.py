"""
Load LANDFIRE GeoTIFFs into xarray DataArrays and build a combined Dataset.

Each layer is loaded with rioxarray, which preserves CRS metadata and spatial
coordinates. Nodata values are masked to NaN. The dataset can be inspected
immediately for alignment consistency before any reprojection is applied.

LANDFIRE layer encodings (LF2022)
----------------------------------
EVT  : Integer class codes (hundreds of classes). Special codes:
         0      = background / nodata
         7295   = Southern Rocky Mountain Dry-Mesic Montane Mixed Conifer
         7296   = Rocky Mountain Subalpine Dry-Mesic Spruce-Fir Forest
         7292   = Rocky Mountain Subalpine-Montane Mesic Meadow
         (Full lookup CSV available: https://www.landfire.gov/CSV/)

EVC  : Percent canopy cover for the dominant life form, or a special code.
         11–30 = non-vegetated (water, snow, developed, agriculture, barren)
         31    = sparse/barren
         40–70 = Cover class (values approximate % cover)
       Note: actual cover % for vegetated cells is the pixel value directly
       (e.g. 65 = 65% cover) in many LF versions. Verify from layer metadata.

EVH  : Vegetation height in metres (vegetated cells) or special codes.
       Non-vegetated cells share the same code scheme as EVC.
       Vegetated cells: value in metres (e.g. 20 = 20 m canopy height).

FBFM40: Scott & Burgan 40 fire behavior fuel model codes.
         91–99  = Non-burnable (NB)
         101–109 = Grass (GR)
         121–124 = Grass-Shrub (GS)
         141–149 = Shrub (SH)
         161–165 = Timber Understory (TU)
         181–189 = Timber Litter (TL)
         201–204 = Slash-Blowdown (SB)

DIST : Disturbance codes encoding the disturbance type and year for the most
       recent event (LF2022 vintage covers events through 2022).
         0         = undisturbed
         1000–4999 = fire (code × 10 ≈ year, varies by product)
         See LANDFIRE disturbance legend CSV for full encoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr

from . import LANDFIRE_NODATA, RAW_DIR


# ---------------------------------------------------------------------------
# FBFM40 lookup table
# ---------------------------------------------------------------------------
# Structure: code → (short_name, group, description)
FBFM40_LOOKUP: Dict[int, Tuple[str, str, str]] = {
    91:  ("NB1", "Non-burnable", "Urban/suburban — no fire spread expected"),
    92:  ("NB2", "Non-burnable", "Snow/ice"),
    93:  ("NB3", "Non-burnable", "Agricultural"),
    98:  ("NB8", "Non-burnable", "Open water"),
    99:  ("NB9", "Non-burnable", "Bare ground"),
    101: ("GR1", "Grass",        "Short sparse dry-climate grass"),
    102: ("GR2", "Grass",        "Low load dry-climate grass"),
    103: ("GR3", "Grass",        "Low load moist-climate grass"),
    104: ("GR4", "Grass",        "Moderate load dry-climate grass"),
    105: ("GR5", "Grass",        "Low load humid-climate grass"),
    106: ("GR6", "Grass",        "Moderate load humid-climate grass"),
    107: ("GR7", "Grass",        "High load dry-climate grass"),
    108: ("GR8", "Grass",        "High load very coarse dry-climate grass"),
    109: ("GR9", "Grass",        "Very high load humid-climate grass"),
    121: ("GS1", "Grass-Shrub",  "Low load dry-climate grass-shrub"),
    122: ("GS2", "Grass-Shrub",  "Moderate load dry-climate grass-shrub"),
    123: ("GS3", "Grass-Shrub",  "Moderate load humid-climate grass-shrub"),
    124: ("GS4", "Grass-Shrub",  "High load humid-climate grass-shrub"),
    141: ("SH1", "Shrub",        "Low load dry-climate shrub"),
    142: ("SH2", "Shrub",        "Moderate load dry-climate shrub"),
    143: ("SH3", "Shrub",        "Moderate load chaparral"),
    144: ("SH4", "Shrub",        "Low load moist-climate shrub"),
    145: ("SH5", "Shrub",        "High load dry-climate shrub"),
    146: ("SH6", "Shrub",        "Low load humid-climate shrub"),
    147: ("SH7", "Shrub",        "Very high load dry-climate shrub"),
    148: ("SH8", "Shrub",        "High load humid-climate shrub"),
    149: ("SH9", "Shrub",        "Very high load humid-climate shrub"),
    161: ("TU1", "Timber-Understory", "Low load dry-climate timber-grass-shrub"),
    162: ("TU2", "Timber-Understory", "Moderate load humid-climate timber-shrub"),
    163: ("TU3", "Timber-Understory", "Moderate load humid-climate timber-grass-shrub"),
    164: ("TU4", "Timber-Understory", "Dwarf conifer with understory"),
    165: ("TU5", "Timber-Understory", "Very high load dry-climate timber-shrub"),
    181: ("TL1", "Timber Litter", "Low load compact conifer litter"),
    182: ("TL2", "Timber Litter", "Low load broadleaf litter"),
    183: ("TL3", "Timber Litter", "Moderate load conifer litter"),
    184: ("TL4", "Timber Litter", "Small downed logs"),
    185: ("TL5", "Timber Litter", "High load conifer litter"),
    186: ("TL6", "Timber Litter", "Moderate load broadleaf litter"),
    187: ("TL7", "Timber Litter", "Large downed logs"),
    188: ("TL8", "Timber Litter", "Long-needle litter"),
    189: ("TL9", "Timber Litter", "Very high load broadleaf litter"),
    201: ("SB1", "Slash-Blowdown", "Low load activity fuel"),
    202: ("SB2", "Slash-Blowdown", "Moderate load activity fuel, small debris"),
    203: ("SB3", "Slash-Blowdown", "Moderate load activity fuel, large debris"),
    204: ("SB4", "Slash-Blowdown", "High load activity fuel"),
}

# Colors for each fuel group (for categorical mapping)
FBFM40_GROUP_COLORS: Dict[str, str] = {
    "Non-burnable":      "#9E9E9E",
    "Grass":             "#FDD835",
    "Grass-Shrub":       "#9CCC65",
    "Shrub":             "#388E3C",
    "Timber-Understory": "#A1887F",
    "Timber Litter":     "#E64A19",
    "Slash-Blowdown":    "#7B1FA2",
}

# Codes that represent non-vegetated / non-burnable pixels across EVC/EVH
NON_VEGETATED_CODES = frozenset([0, 11, 12, 13, 14, 15, 16, 17, 18,
                                  20, 21, 22, 23, 24, 25, 26, 27,
                                  31, 32, 36, 37])


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_layer_file(raw_dir: Path, layer_key: str) -> Optional[Path]:
    """Search raw_dir for a GeoTIFF matching the LANDFIRE layer key.

    Searches recursively and matches on common LANDFIRE filename fragments.
    Returns the first match, or None if no file is found.
    """
    patterns: Dict[str, List[str]] = {
        # LF2024 Fuel Vegetation layers (FVT/FVC/FVH) and LF2022 fallbacks (EVT/EVC/EVH)
        "evt":         ["*FVT*", "*fvt*", "*EVT*", "*evt*"],
        "evc":         ["*FVC*", "*fvc*", "*EVC*", "*evc*"],
        "evh":         ["*FVH*", "*fvh*", "*EVH*", "*evh*"],
        "fuel_model":  ["*FBFM40*", "*F40*", "*fbfm40*", "*f40*"],
        "disturbance": ["*FDST*", "*fdst*", "*DIST*", "*dist*", "*Dist*"],
    }
    raw_dir = Path(raw_dir)
    for pattern in patterns.get(layer_key, [f"*{layer_key}*"]):
        for ext in [".tif", ".TIF"]:
            # pattern is like "*FVT*"; glob as **/*FVT*<ext>
            glob_pattern = f"**/{pattern}*{ext}" if not pattern.endswith("*") else f"**/{pattern}{ext}"
            matches = [p for p in raw_dir.glob(glob_pattern) if p.suffix.lower() == ".tif"]
            if matches:
                return sorted(matches)[0]
    return None


# ---------------------------------------------------------------------------
# Single-layer loading
# ---------------------------------------------------------------------------

def load_raster(
    path: Path,
    name: str,
    nodata: int = LANDFIRE_NODATA,
) -> xr.DataArray:
    """Load a single LANDFIRE GeoTIFF as an xarray DataArray.

    The file is opened with rioxarray so CRS and transform are preserved.
    The band dimension is squeezed (all LANDFIRE layers are single-band).
    Nodata values are set to NaN.

    Parameters
    ----------
    path   : path to GeoTIFF
    name   : variable name assigned to the DataArray
    nodata : nodata value to mask (overrides file metadata if differs)
    """
    da = rioxarray.open_rasterio(path, masked=True)
    # Squeeze the band dimension; LANDFIRE layers are always single-band.
    da = da.squeeze("band", drop=True)
    da.name = name

    # Honour the file's own nodata in addition to the override value.
    file_nodata = da.rio.nodata
    for nd in {nodata, file_nodata}:
        if nd is not None:
            da = da.where(da != nd)

    return da


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def load_all_layers(raw_dir: Path = RAW_DIR) -> Dict[str, xr.DataArray]:
    """Load all available LANDFIRE layers from raw_dir.

    Returns a dict of {layer_key: DataArray}. Missing files are skipped
    with a warning rather than raising an error, so partial datasets work.
    """
    from . import LANDFIRE_LAYERS

    layers: Dict[str, xr.DataArray] = {}
    for key in LANDFIRE_LAYERS:
        path = find_layer_file(raw_dir, key)
        if path is None:
            print(f"  [MISSING] {key:14s}  — no file found in {raw_dir}")
            continue
        da = load_raster(path, name=key)
        layers[key] = da
        print(f"  [loaded ] {key:14s}  {path.name}  shape={da.shape}  crs={da.rio.crs}")
    return layers


def build_dataset(raw_dir: Path = RAW_DIR) -> xr.Dataset:
    """Load all layers and merge into a single xr.Dataset.

    Layers are NOT reprojected or resampled here — use align_rasters.align()
    to put all layers on a common grid before building the Dataset for analysis.

    Returns an xr.Dataset with one variable per LANDFIRE layer.
    CRS and spatial metadata are preserved via the .rio accessor.
    """
    layers = load_all_layers(raw_dir)
    if not layers:
        raise FileNotFoundError(
            f"No LANDFIRE GeoTIFFs found in {raw_dir}. "
            "Run download_landfire.download_landfire() or place files manually."
        )
    ds = xr.Dataset(layers)
    return ds


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def fbfm40_name(code: int) -> str:
    """Return the short fuel model name for an FBFM40 code (e.g. 'TL3')."""
    return FBFM40_LOOKUP.get(int(code), (str(code), "Unknown", ""))[0]


def fbfm40_group(code: int) -> str:
    """Return the fuel group for an FBFM40 code (e.g. 'Timber Litter')."""
    return FBFM40_LOOKUP.get(int(code), ("?", "Unknown", ""))[1]


def is_vegetated(code: int) -> bool:
    """Return True if the EVC/EVH code represents a vegetated cell."""
    return int(code) not in NON_VEGETATED_CODES and int(code) > 0


def load_evt_csv(csv_path: Path) -> Dict[int, str]:
    """Load an EVT lookup CSV downloaded from LANDFIRE into a code→name dict.

    The full EVT CSV (hundreds of rows) is available at:
        https://www.landfire.gov/CSV/LF2022/LF22_EVT_220.csv

    Expected columns: VALUE, EVT_NAME (or similar; first two columns used).
    """
    import pandas as pd
    df = pd.read_csv(csv_path, encoding="latin-1")
    code_col  = df.columns[0]
    label_col = df.columns[1]
    return dict(zip(df[code_col].astype(int), df[label_col].astype(str)))
