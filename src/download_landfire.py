"""
Programmatic download of LANDFIRE raster products via the LFPS REST API.

The LANDFIRE Product Service (LFPS) is an ArcGIS GP REST service that
clips and reprojects LANDFIRE rasters to a user-specified bounding box.
A job is submitted asynchronously, polled until complete, and the result
ZIP is downloaded and extracted.

LFPS documentation:
  https://www.landfire.gov/data/lf-product-service

If automated download is unavailable (network, authentication, API changes),
the workflow falls back gracefully: place manually downloaded GeoTIFFs in
data/raw/ and the rest of the pipeline will discover them by name pattern.

Manual download portal:
  https://www.landfire.gov/viewer/
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import requests

from . import (
    BBOX_WGS84,
    LANDFIRE_CRS,
    LANDFIRE_LAYERS,
    LANDFIRE_RESOLUTION,
    LANDFIRE_VERSION,
    RAW_DIR,
)


# ---------------------------------------------------------------------------
# LFPS API constants
# ---------------------------------------------------------------------------
LFPS_BASE = (
    "https://lfps.usgs.gov/arcgis/rest/services/"
    "LandfireProductService/GPServer/LandfireProductService"
)
LFPS_SUBMIT   = f"{LFPS_BASE}/submitJob"
LFPS_JOBS     = f"{LFPS_BASE}/jobs"

# EPSG → ESRI WKID mapping for common output projections.
# LFPS accepts ESRI WKIDs rather than EPSG codes for Output_Projection.
_EPSG_TO_WKID: Dict[str, int] = {
    "EPSG:32613": 32613,   # UTM Zone 13N
    "EPSG:32612": 32612,   # UTM Zone 12N
    "EPSG:4326":  4326,    # WGS84 geographic
    "EPSG:5070":  5070,    # Albers CONUS (USGS standard)
}


# ---------------------------------------------------------------------------
# LFPS job submission and polling
# ---------------------------------------------------------------------------

def _submit_lfps_job(
    layer_list: List[str],
    bbox_wgs84: tuple,
    output_crs: str,
    resolution: int,
) -> str:
    """Submit a clip-and-download job to the LANDFIRE Product Service.

    Parameters
    ----------
    layer_list   : LFPS product codes, e.g. ["US_220EVT", "US_220EVC"]
    bbox_wgs84   : (lon_min, lat_min, lon_max, lat_max) in WGS84
    output_crs   : EPSG string, e.g. "EPSG:32613"
    resolution   : output resolution in metres

    Returns
    -------
    jobId string
    """
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    # LFPS expects a JSON AOI string
    aoi = (
        f'{{"xmin":{lon_min},"ymin":{lat_min},'
        f'"xmax":{lon_max},"ymax":{lat_max},'
        f'"spatialReference":{{"wkid":4326}}}}'
    )
    wkid = _EPSG_TO_WKID.get(output_crs, 32613)

    payload = {
        "Layer_list":          " ".join(layer_list),
        "Area_of_Interest":    aoi,
        "Output_Projection":   str(wkid),
        "Resample_Resolution": str(resolution),
        "f":                   "json",
    }

    resp = requests.post(LFPS_SUBMIT, data=payload, timeout=60)
    resp.raise_for_status()
    result = resp.json()

    if "jobId" not in result:
        raise RuntimeError(f"LFPS did not return a jobId. Response: {result}")

    return result["jobId"]


def _poll_lfps_job(
    job_id: str,
    poll_interval: int = 30,
    timeout: int = 1800,
) -> str:
    """Poll the LFPS job endpoint until completion. Returns output URL."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(
            f"{LFPS_JOBS}/{job_id}", params={"f": "json"}, timeout=30
        )
        resp.raise_for_status()
        status_data = resp.json()
        status = status_data.get("jobStatus", "")

        if status == "esriJobSucceeded":
            # Fetch the output file URL
            out_resp = requests.get(
                f"{LFPS_JOBS}/{job_id}/results/Output_File",
                params={"f": "json"},
                timeout=30,
            )
            out_resp.raise_for_status()
            url = out_resp.json().get("value", {}).get("url")
            if not url:
                raise RuntimeError(
                    f"Job succeeded but no output URL found. Response: {out_resp.json()}"
                )
            return url

        elif status in ("esriJobFailed", "esriJobCancelled", "esriJobTimedOut"):
            msgs = status_data.get("messages", [])
            raise RuntimeError(
                f"LFPS job {job_id} ended with status '{status}'. "
                f"Messages: {msgs}"
            )

        elif status in ("esriJobExecuting", "esriJobSubmitted", "esriJobWaiting"):
            print(f"  LFPS job {job_id[:8]}… status={status}, waiting {poll_interval}s")
            time.sleep(poll_interval)

        else:
            print(f"  Unknown LFPS status '{status}', retrying in {poll_interval}s")
            time.sleep(poll_interval)

    raise TimeoutError(
        f"LFPS job {job_id} did not complete within {timeout // 60} minutes."
    )


def _download_and_extract(url: str, output_dir: Path) -> None:
    """Download a ZIP from the LFPS result URL and extract to output_dir."""
    print(f"  Downloading LFPS output from: {url}")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    zip_path = output_dir / "lfps_download.zip"
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = 100 * downloaded / total
                print(f"\r  {pct:.0f}% ({downloaded // 1024 // 1024} MB)", end="")
    print()

    print(f"  Extracting to {output_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    zip_path.unlink()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_landfire(
    bbox_wgs84: tuple = BBOX_WGS84,
    layers: Optional[Dict[str, str]] = None,
    output_dir: Path = RAW_DIR,
    output_crs: str = LANDFIRE_CRS,
    resolution: int = LANDFIRE_RESOLUTION,
    poll_interval: int = 30,
    timeout: int = 1800,
    force: bool = False,
) -> bool:
    """Download LANDFIRE LF2022 rasters for the study area via LFPS.

    Parameters
    ----------
    bbox_wgs84   : (lon_min, lat_min, lon_max, lat_max) bounding box in WGS84
    layers       : dict of {key: product_code} — defaults to all LANDFIRE_LAYERS
    output_dir   : directory to write extracted GeoTIFFs
    output_crs   : output coordinate reference system (EPSG string)
    resolution   : output cell size in metres (default 30)
    poll_interval: seconds between LFPS status checks
    timeout      : maximum wait time in seconds before giving up
    force        : re-download even if TIFFs already present

    Returns
    -------
    True if download succeeded, False if it failed (with instructions printed).

    Notes
    -----
    The LFPS API can be slow and occasionally unavailable. If this function
    returns False, download data manually from:
        https://www.landfire.gov/viewer/
    and place the GeoTIFFs in data/raw/. The rest of the pipeline will find
    them automatically.
    """
    if layers is None:
        layers = LANDFIRE_LAYERS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        existing = list(output_dir.glob("**/*.tif")) + list(output_dir.glob("**/*.TIF"))
        if len(existing) >= len(layers):
            print(
                f"Found {len(existing)} GeoTIFF(s) in {output_dir}. "
                "Skipping download (pass force=True to re-download)."
            )
            return True

    product_codes = list(layers.values())
    print(f"Submitting LFPS job for {len(product_codes)} layers:")
    for key, code in layers.items():
        print(f"  {code:20s}  ({key})")
    print(f"  AOI: {bbox_wgs84}")
    print(f"  CRS: {output_crs}  |  Resolution: {resolution} m")

    try:
        job_id = _submit_lfps_job(product_codes, bbox_wgs84, output_crs, resolution)
        print(f"  Job submitted: {job_id}")
        download_url = _poll_lfps_job(job_id, poll_interval, timeout)
        _download_and_extract(download_url, output_dir)
        tifs = list(output_dir.glob("**/*.tif")) + list(output_dir.glob("**/*.TIF"))
        print(f"\nDownload complete. {len(tifs)} GeoTIFF(s) in {output_dir}:")
        for t in tifs:
            print(f"  {t.name}")
        return True

    except Exception as exc:
        print(f"\n[LFPS download failed]: {exc}")
        _print_manual_instructions(layers, bbox_wgs84)
        return False


def _print_manual_instructions(layers: Dict[str, str], bbox_wgs84: tuple) -> None:
    """Print clear manual-download instructions if the API fails."""
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    sep = "=" * 65
    lines = [
        "",
        sep,
        "MANUAL DOWNLOAD INSTRUCTIONS",
        sep,
        "The LFPS API is unavailable. To obtain LANDFIRE data manually:",
        "",
        "  1. Visit: https://www.landfire.gov/viewer/",
        "  2. Zoom to your study area:",
        f"       West: {lon_min}  East: {lon_max}",
        f"       South: {lat_min}  North: {lat_max}",
        "  3. Use the Draw Area of Interest tool to select the region.",
        "  4. Request these LF2022 (version 220) products:",
        "",
    ]
    for key, code in layers.items():
        lines.append(f"       {code:25s}  ({key})")
    lines += [
        "",
        "  5. Choose output format: GeoTIFF, UTM Zone 13N (EPSG:32613).",
        "  6. Download the ZIP and extract GeoTIFFs to:",
        "       data/raw/",
        "",
        "The pipeline will find the files automatically by name pattern.",
        sep,
    ]
    print("\n".join(lines))


def check_raw_data(raw_dir: Path = RAW_DIR) -> Dict[str, Optional[Path]]:
    """Scan raw_dir for LANDFIRE GeoTIFFs and report which layers are present."""
    from .load_data import find_layer_file

    status = {}
    for key in LANDFIRE_LAYERS:
        path = find_layer_file(raw_dir, key)
        status[key] = path
        flag = "OK" if path else "MISSING"
        name = path.name if path else "—"
        print(f"  [{flag:7s}] {key:12s}  {name}")
    return status
