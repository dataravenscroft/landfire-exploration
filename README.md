# LANDFIRE Vegetation and Fuel Exploration

 This project focuses on fire ecology: using LANDFIRE's multi-layer vegetation and fuel data to ask how much of LANDFIRE's expert fuel model assignments can be recovered from vegetation inputs alone, and what the errors reveal about where that approach breaks down.

I used Claude Code to generate most of this codebase. I'm not going to pretend I wrote every line. I'm working through it now — reading the design decisions, making changes where things don't make sense to me — and that's the actual work in progress.

Built on LANDFIRE LF2024 raster data, USGS 3DEP elevation, and ERA5 atmospheric reanalysis.

---

## Questions driving this

> *How do vegetation type, structure, and disturbance history map onto Scott & Burgan fuel model assignments across a heterogeneous mountain landscape — and how much of that mapping can a simplified classifier recover from the vegetation layers alone?*

> *What do the classifier's errors reveal about which fuel classes require contextual or structural information beyond what aggregate remote-sensing variables can capture?*

---

## Background

### LANDFIRE

LANDFIRE (Landscape Fire and Resource Management Planning Tools) is a shared program of the USDA Forest Service and U.S. Department of the Interior. It provides nationally consistent geospatial data products describing vegetation, fuel, and disturbance conditions across the contiguous United States, Alaska, and Hawaii, at 30 m resolution, updated annually.

LANDFIRE products support wildland fire management planning, fire-behavior modeling (FARSITE, FlamMap, Phoenix), landscape ecology research, and post-fire rehabilitation planning.

**Reference:** Rollins, M.G. (2009). LANDFIRE: A nationally consistent vegetation, wildland fire, and fuel assessment. *International Journal of Wildland Fire*, 18, 235–249.

### Vegetation vs. fuel products

**Vegetation products (EVT, EVC, EVH)** describe the current biological state of the landscape — what plant communities are present, how dense the canopy is, how tall the vegetation is.

**Fuel products (FBFM40)** describe how surface fuels are expected to *behave* under fire — what fuel loading and arrangement is present, and which standardized fire-behavior model best represents potential surface fire behavior.

A fuel model is not simply a vegetation label. The same vegetation type can map to different fuel models depending on canopy cover, height, disturbance history, and ecological context. Adjacent fuel classes may look nearly identical in a satellite image but behave very differently under fire. That gap is what this project examines.

---

## LANDFIRE products used

| Layer | Product code | Description |
|---|---|---|
| Fuel Vegetation Type | US_240FBFVT | Ecological system classification calibrated for fuel assignment |
| Fuel Vegetation Cover | US_240FBFVC | % canopy cover for dominant life form |
| Fuel Vegetation Height | US_240FBFVH | Vegetation height in metres |
| FBFM40 | US_240FBFM40 | Scott & Burgan 40 fire behavior fuel model |
| Fuel Disturbance | US_240FDST | Most recent disturbance event type and year |

**LANDFIRE version:** LF2024 (version code 240) | **CRS:** UTM Zone 13N (EPSG:32613) | **Resolution:** 30 m | **Nodata:** −9999

### EVT, EVC, EVH

EVT classifies each 30 m cell into an ecological system — a recurring combination of plant species, processes, and environmental setting defined by NatureServe. For this study area (Northern Colorado mountains), dominant EVT classes include Rocky Mountain subalpine spruce-fir forest, montane mixed conifer, ponderosa pine woodland, Rocky Mountain shrubland, and subalpine-montane meadows.

EVC is the estimated percent canopy cover of the dominant life form; EVH is height in metres. Both drive fuel model assignment — dense tall forest typically maps to Timber Litter or Timber Understory models; open grassland maps to Grass models; recovering post-fire areas may retain their pre-fire EVT but receive disturbance-adjusted fuel models.

### Scott & Burgan 40 Fuel Models (FBFM40)

The Scott & Burgan 40 models (2005) are the standard set used with Rothermel's surface fire spread model, organized into seven groups:

| Group | Codes | Typical setting |
|---|---|---|
| Non-burnable (NB) | NB1–NB9 | Urban, water, snow, bare ground |
| Grass (GR) | GR1–GR9 | Open grasslands; fast-spreading surface fire |
| Grass-Shrub (GS) | GS1–GS4 | Transitional grass-shrub mix |
| Shrub (SH) | SH1–SH9 | Shrub-dominated landscapes |
| Timber Understory (TU) | TU1–TU5 | Forest with significant understory component |
| Timber Litter (TL) | TL1–TL9 | Closed forest; litter-dominated surface fuels |
| Slash-Blowdown (SB) | SB1–SB4 | Logging slash or wind-thrown material |

**Reference:** Scott, J.H. & Burgan, R.E. (2005). Standard fire behavior fuel models: a comprehensive set for use with Rothermel's surface fire spread model. USDA Forest Service RMRS-GTR-153.

### Disturbance history

The disturbance layer records the most recent disturbance event (fire, harvest, insect/disease, weather) and year per cell. Post-fire cells may transition from Timber Litter to Grass-Shrub or Slash-Blowdown depending on severity and time since fire; beetle-killed forest may transition toward higher-load litter models; mechanically treated areas may be assigned non-burnable classes.

---

## Study area

**Northern Colorado Mountain Landscape** | (−105.80°W, 40.30°N) to (−105.35°W, 40.65°N) | ~36 × 39 km | ~1,600–3,800 m elevation

This area spans lower-montane ponderosa pine and shrubland through montane mixed conifer and subalpine spruce-fir into alpine tundra. It overlaps the eastern perimeter of the **2020 Cameron Peak Fire** (311,000 acres — Colorado's largest recorded wildfire as of 2020), making disturbance effects on fuel model assignment clearly visible in the data.

---

## Conceptual framework

```
Vegetation state (relatively static)
  EVT — what ecological system is present
  EVC — how dense is the canopy
  EVH — how tall is the vegetation
  + ecological context + disturbance history
  ↓
Fuel representation (annually updated)
  FBFM40 — surface fuel behavior model
  + topography (slope, aspect, elevation)
  ↓
Physical fire-behavior environment
  + dynamic weather / fuel moisture (ERA5: Temp, RH, Wind, VPD)
  ↓
Inputs to fire-behavior simulation models (FARSITE, FlamMap, Phoenix)
```

A high fuel load or dense forest fuel model is not inherently "high fire risk." Risk depends on the combination of fuel load, fuel moisture state, wind speed, slope, and ignition probability. Atmospheric conditions drive whether fuels burn, not the fuel model alone.

---

## Repository structure

```
landfire-exploration/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          # LANDFIRE GeoTIFFs (downloaded or manually placed)
│   └── processed/    # Aligned, clipped layers (generated by notebook 01)
├── src/
│   ├── __init__.py          # Project constants: study area, layer metadata, ERA5 config
│   ├── download_landfire.py # LFPS API download + manual fallback instructions
│   ├── load_data.py         # GeoTIFF loading, FBFM40 lookup table, xr.Dataset assembly
│   ├── align_rasters.py     # CRS verification, reprojection, clipping, alignment
│   ├── terrain.py           # 3DEP DEM download, slope and aspect derivation
│   ├── analysis.py          # DataFrame conversion, summary statistics, all plots
│   └── modeling.py          # RF classifier, evaluation, feature importance
├── notebooks/
│   ├── 01_landfire_exploration.ipynb  # Data acquisition → maps → EDA → classifier → terrain
│   └── 02_weather_integration.ipynb  # ERA5 fire weather → integrated visualization
├── figures/          # Generated figures (committed selectively)
└── results/          # JSON metrics and summary tables
```

---

## Installation and quickstart

```bash
git clone https://github.com/dataravenscroft/landfire-exploration
cd landfire-exploration
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/01_landfire_exploration.ipynb`. The first cell will attempt to download LANDFIRE data automatically via the LFPS API. If the API is unavailable, follow the printed manual download instructions and place GeoTIFFs in `data/raw/`.

### ERA5 setup (notebook 02)

ERA5 download requires a free account at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu). After registering:

```bash
echo "url: https://cds.climate.copernicus.eu/api/v2
key: <your-uid>:<your-api-key>
verify: 0" > ~/.cdsapirc
```

Or place a pre-downloaded ERA5 netCDF at `data/raw/era5_fire_weather.nc` and the notebook will use it directly.

---

## Analyses

### 1. Data acquisition and alignment

LANDFIRE rasters downloaded via the LFPS REST API (or loaded from manually placed GeoTIFFs), verified to share the same CRS, resolution, and extent, then clipped to the study area bounding box.

### 2. Landscape visualization

Five-panel maps showing EVT, EVC, EVH, FBFM40, and disturbance history. FBFM40 is displayed by fuel group rather than individual model code to keep the legend readable.

### 3. Vegetation–fuel exploratory analysis

Raster cells flattened into a DataFrame for statistical analysis: EVC and EVH distributions per fuel group (boxplots), EVT composition within each fuel group (heatmap), fuel group distribution in disturbed vs. undisturbed cells.

### 4. Classifier

A Random Forest trained to predict FBFM40 from EVT (one-hot encoded), EVC, EVH, and disturbance presence. Evaluation includes overall accuracy, macro F1, per-class F1, confusion matrix, and feature importance.

LANDFIRE uses a complex expert-knowledge ruleset incorporating ecological context, disturbance type, and field-validated relationships. The classifier can't reproduce that — and the errors show exactly where the vegetation layers alone are insufficient.

### 5. Terrain analysis

Slope and aspect derived from USGS 3DEP elevation, visualized alongside fuel models and joined to the cell DataFrame for fuel-by-terrain breakdowns.

### 6. ERA5 fire-weather integration

ERA5 single-level fields (2m temperature, dewpoint, 10m wind, precipitation) for a representative fire-weather afternoon. Derived variables include relative humidity, wind speed, and vapour pressure deficit. The spatial scale mismatch between ERA5 (0.25° ≈ 27.8 km) and LANDFIRE (30 m) is explicitly shown.

---

## Limitations

- **Random train/test split inflates accuracy.** Adjacent raster cells share environmental context, so nearby cells appear in both training and test sets. Spatially blocked cross-validation would give a more conservative estimate.
- **FBFM40 is not fully derivable from EVT + EVC + EVH.** Fuel loading and ecological context require information these layers don't capture.
- **ERA5 at 0.25° is very coarse** for a 36 × 39 km area. A production analysis would use downscaled weather or station-based interpolation.
- **Fuel moisture is not encoded in FBFM40.** The fuel model specifies potential behavior under standardized conditions; actual moisture is dynamic.

---

## Prior work

I built a forest cover type classifier for a large landscape in northeastern Minnesota using Landsat imagery and forest composition maps — traditional spectral classification, manually engineered features. That work also used DEMs to modify species establishment and productivity parameters per forest cell, so terrain as an ecological driver rather than just a backdrop is a thread that runs through both projects.

This project applies similar conceptual goals — extract ecologically meaningful structure from remote sensing data — to a different domain (fire ecology, fuel mapping) and a richer data infrastructure. The companion CNN project ([sentinel-vegetation-cnn](https://github.com/dataravenscroft/sentinel-vegetation-cnn)) approaches vegetation mapping through deep learning on Sentinel-2.

Comparing all three is the point.

---

## Data sources

**LANDFIRE** — Rollins, M.G. (2009). *International Journal of Wildland Fire*, 18, 235–249. https://www.landfire.gov

**Scott & Burgan 40 Fuel Models** — Scott, J.H. & Burgan, R.E. (2005). USDA Forest Service RMRS-GTR-153.

**ERA5** — Hersbach, H. et al. (2020). *Quarterly Journal of the Royal Meteorological Society*, 146, 1999–2049. https://doi.org/10.1002/qj.3803

**USGS 3DEP** — https://www.usgs.gov/3d-elevation-program
