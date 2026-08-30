"""
landfire-exploration: Geospatial analysis of LANDFIRE vegetation, fuel, and
fire-weather data for a Northern Colorado mountain landscape.

Study area overview
-------------------
The study area spans ~36 × 39 km across Larimer and Grand counties, Colorado.
It covers subalpine spruce-fir forest, montane mixed conifer, ponderosa pine
woodland, shrubland, and alpine tundra. The extent overlaps the eastern
perimeter of the 2020 Cameron Peak Fire — Colorado's largest recorded wildfire —
making disturbance effects on fuel model assignment clearly visible.

Conceptual framework
--------------------
LANDFIRE products encode two conceptually distinct representations:

  Vegetation state  →  EVT + EVC + EVH
  Fuel representation → FBFM40

The fuel model (FBFM40) is not simply a vegetation label. It is a fire-behavior
representation derived from vegetation type, structure, ecological context, and
disturbance history, calibrated for input to fire-behavior simulation models
such as FARSITE, FlamMap, and Phoenix. Current weather and fuel moisture are
separate dynamic inputs, not encoded in LANDFIRE.

References
----------
  Rollins, M.G. (2009). LANDFIRE: A nationally consistent vegetation, wildland
  fire, and fuel assessment. International Journal of Wildland Fire, 18, 235-249.

  Scott, J.H. & Burgan, R.E. (2005). Standard fire behavior fuel models: a
  comprehensive set for use with Rothermel's surface fire spread model.
  USDA Forest Service RMRS-GTR-153.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR   = PROJECT_ROOT / "figures"
RESULTS_DIR   = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Study area
# ---------------------------------------------------------------------------
# Bounding box in WGS84: (lon_min, lat_min, lon_max, lat_max)
BBOX_WGS84 = (-105.632, 38.4287, -105.3533, 38.6395)

STUDY_AREA_NAME = "Central Colorado Mountain Landscape"
STUDY_AREA_DESCRIPTION = (
    "~26 × 23 km area in central Colorado spanning the Royal Gorge / Cañon City "
    "foothills through the Wet Mountains and Sangre de Cristo front. "
    "Covers lower-montane ponderosa pine, mixed conifer, and shrubland vegetation zones."
)


# ---------------------------------------------------------------------------
# LANDFIRE metadata
# ---------------------------------------------------------------------------
LANDFIRE_VERSION   = "240"   # LF2024
LANDFIRE_YEAR      = 2024
LANDFIRE_CRS       = "EPSG:32613"   # UTM Zone 13N (CONUS standard for Colorado)
LANDFIRE_NODATA    = -9999
LANDFIRE_RESOLUTION = 30            # native resolution, metres

# LFPS API product codes for LF2024.
# LF2024 uses Fuel Vegetation layers (FVT/FVC/FVH) rather than the traditional
# Existing Vegetation layers (EVT/EVC/EVH) as inputs to fuel model assignment.
LANDFIRE_LAYERS = {
    "evt":         "US_240FBFVT",  # Fuel Vegetation Type
    "evc":         "US_240FBFVC",  # Fuel Vegetation Cover (% canopy cover)
    "evh":         "US_240FBFVH",  # Fuel Vegetation Height (m)
    "fuel_model":  "US_240FBFM40", # Scott & Burgan 40 fire behavior fuel model
    "disturbance": "US_240FDST",   # Fuel Disturbance (2024 vintage)
}

# Layer descriptions for documentation and figure labels
LAYER_DESCRIPTIONS = {
    "evt":         "Existing Vegetation Type (EVT)",
    "evc":         "Existing Vegetation Cover — % canopy cover",
    "evh":         "Existing Vegetation Height — m",
    "fuel_model":  "Scott & Burgan 40 Fuel Model (FBFM40)",
    "disturbance": "Disturbance History (most recent event)",
    "elevation":   "Elevation — m",
    "slope":       "Slope — degrees",
    "aspect":      "Aspect — degrees from north",
}


# ---------------------------------------------------------------------------
# ERA5 fire-weather configuration
# ---------------------------------------------------------------------------
# A representative fire-weather day for the study area.
# August 13, 2020 coincided with active fire-weather conditions across the
# Rocky Mountain region, during the period preceding major fire growth events.
ERA5_DATE = "2020-08-13"
ERA5_HOUR = 18          # 18:00 UTC ≈ 12:00 local (MDT)

ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]
