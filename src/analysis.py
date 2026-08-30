"""
Exploratory analysis and visualization for LANDFIRE vegetation and fuel data.

Functions
---------
raster_to_dataframe     Convert an xr.Dataset to a flat pandas DataFrame for
                        cell-level analysis. Supports subsampling for large datasets.

summarize_fuel_classes  For each FBFM40 code: cell count, dominant EVT, mean
                        EVC/EVH, disturbance fraction.

plot_landscape_maps     Multi-panel map showing EVT, EVC, EVH, FBFM40, disturbance.

plot_fuel_distributions Violin/boxplots of EVC and EVH per fuel group.

plot_fuel_evt_heatmap   Heatmap of cell count by fuel group × EVT group.

plot_spectral_means     Per-class mean channel values (for spectral baseline).

Design note
-----------
All plotting functions accept an optional `figures_dir` argument and save PNG
files when provided. They also return the Figure object for notebook display.
Figures are generated with matplotlib's Agg backend when no display is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .load_data import (
    FBFM40_GROUP_COLORS,
    FBFM40_LOOKUP,
    NON_VEGETATED_CODES,
    fbfm40_group,
    fbfm40_name,
)


# ---------------------------------------------------------------------------
# Color schemes
# ---------------------------------------------------------------------------

def _fbfm40_group_colormap(
    codes: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """Map FBFM40 integer codes to RGBA colors by fuel group.

    Returns an RGBA image array and a legend {label: color} dict.
    """
    rgba = np.zeros((*codes.shape, 4), dtype=float)
    rgba[..., 3] = 1.0  # fully opaque by default

    for code, (short, group, _) in FBFM40_LOOKUP.items():
        color = FBFM40_GROUP_COLORS.get(group, "#CCCCCC")
        r, g, b = _hex_to_rgb(color)
        mask = codes == code
        rgba[mask, 0] = r
        rgba[mask, 1] = g
        rgba[mask, 2] = b

    # Transparent for NaN / nodata
    nan_mask = np.isnan(codes.astype(float))
    rgba[nan_mask, 3] = 0.0

    legend = {
        f"{group}": color
        for group, color in FBFM40_GROUP_COLORS.items()
    }
    return rgba, legend


def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------

def raster_to_dataframe(
    ds: xr.Dataset,
    max_pixels: int = 500_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Flatten an xr.Dataset into a pandas DataFrame for cell-level analysis.

    Each row is one raster cell; each variable becomes a column.
    Cells where ALL LANDFIRE variables are NaN are dropped.

    Parameters
    ----------
    ds         : aligned xr.Dataset (output of align_rasters.build_aligned_dataset)
    max_pixels : maximum number of cells to include (random subsample if larger)
    seed       : random seed for reproducible subsampling

    Returns
    -------
    pd.DataFrame with spatial coordinates and one column per dataset variable.
    Columns include 'x', 'y', and all data variable names.
    """
    records = {}
    # Flatten each variable to 1D
    for var in ds.data_vars:
        arr = ds[var].values.ravel()
        records[var] = arr

    df = pd.DataFrame(records)

    # Add spatial coordinates if 2D layers are available
    first_var = list(ds.data_vars)[0]
    if "y" in ds[first_var].dims and "x" in ds[first_var].dims:
        yy, xx = np.meshgrid(
            ds[first_var].y.values,
            ds[first_var].x.values,
            indexing="ij",
        )
        df["y"] = yy.ravel()
        df["x"] = xx.ravel()

    # Drop rows where all LANDFIRE vars (not terrain) are NaN
    landfire_vars = [v for v in df.columns if v not in ("x", "y", "elevation", "slope", "aspect")]
    df = df.dropna(subset=landfire_vars, how="all").reset_index(drop=True)

    if len(df) > max_pixels:
        print(
            f"Dataset has {len(df):,} valid cells; subsampling to "
            f"{max_pixels:,} for analysis."
        )
        df = df.sample(max_pixels, random_state=seed).reset_index(drop=True)

    print(f"DataFrame: {len(df):,} cells × {len(df.columns)} variables")
    return df


# ---------------------------------------------------------------------------
# Fuel-class summary statistics
# ---------------------------------------------------------------------------

def summarize_fuel_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for each FBFM40 class present in the data.

    Returns a DataFrame indexed by fuel model code with columns:
      short_name, group, n_cells, pct_cells, mean_evc, mean_evh,
      pct_disturbed, dominant_evt (most common code)
    """
    if "fuel_model" not in df.columns:
        raise KeyError("DataFrame must have a 'fuel_model' column.")

    fuel_df = df.dropna(subset=["fuel_model"])
    rows = []

    for code in sorted(fuel_df["fuel_model"].dropna().unique()):
        code_int = int(code)
        subset   = fuel_df[fuel_df["fuel_model"] == code]
        info     = FBFM40_LOOKUP.get(code_int, (str(code_int), "Unknown", ""))

        row = {
            "code":       code_int,
            "short_name": info[0],
            "group":      info[1],
            "n_cells":    len(subset),
            "pct_cells":  100 * len(subset) / max(len(fuel_df), 1),
        }
        if "evc" in subset.columns:
            veg_evc = subset["evc"].replace({c: np.nan for c in NON_VEGETATED_CODES})
            row["mean_evc"] = veg_evc.dropna().mean()
        if "evh" in subset.columns:
            veg_evh = subset["evh"].replace({c: np.nan for c in NON_VEGETATED_CODES})
            row["mean_evh"] = veg_evh.dropna().mean()
        if "disturbance" in subset.columns:
            row["pct_disturbed"] = 100 * (subset["disturbance"] > 0).mean()
        if "evt" in subset.columns:
            mode = subset["evt"].dropna().mode()
            row["dominant_evt"] = int(mode.iloc[0]) if len(mode) > 0 else np.nan

        rows.append(row)

    summary = pd.DataFrame(rows).set_index("code")
    summary = summary.sort_values("n_cells", ascending=False)
    return summary


# ---------------------------------------------------------------------------
# Landscape maps
# ---------------------------------------------------------------------------

def plot_landscape_maps(
    ds: xr.Dataset,
    figures_dir: Optional[Path] = None,
    figsize: Tuple[float, float] = (18, 12),
) -> plt.Figure:
    """Create a 5-panel landscape map: EVT, EVC, EVH, FBFM40, disturbance.

    EVT is shown with a random qualitative colormap (many classes).
    EVC and EVH use continuous colormaps.
    FBFM40 uses categorical fuel-group colors.
    Disturbance uses a binary mask (disturbed / undisturbed).
    """
    vars_present = list(ds.data_vars)
    n_panels = len(vars_present)
    ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).ravel()

    for idx, var in enumerate(vars_present):
        ax  = axes_flat[idx]
        arr = ds[var].values.astype(float)
        arr[arr <= -9000] = np.nan   # mask any remaining nodata

        if var == "fuel_model":
            rgba, legend = _fbfm40_group_colormap(arr)
            ax.imshow(rgba, origin="upper", interpolation="nearest")
            patches = [
                mpatches.Patch(facecolor=c, label=lbl)
                for lbl, c in legend.items()
                if any(
                    fbfm40_group(code) == lbl
                    for code in np.unique(arr[~np.isnan(arr)]).astype(int)
                    if code in FBFM40_LOOKUP
                )
            ]
            ax.legend(handles=patches, loc="lower right",
                      fontsize=6, framealpha=0.8)
            ax.set_title("FBFM40 Fuel Model Groups", fontsize=10, fontweight="bold")

        elif var == "disturbance":
            dist = np.where(np.isnan(arr), np.nan, (arr > 0).astype(float))
            cmap = matplotlib.colors.ListedColormap(["#E8F5E9", "#C62828"])
            ax.imshow(dist, cmap=cmap, origin="upper", interpolation="nearest")
            ax.set_title("Recent Disturbance", fontsize=10, fontweight="bold")
            patches = [
                mpatches.Patch(color="#E8F5E9", label="Undisturbed"),
                mpatches.Patch(color="#C62828", label="Disturbed"),
            ]
            ax.legend(handles=patches, loc="lower right", fontsize=8)

        elif var == "evt":
            # Show EVT as categorical; use tab20c colormap across unique classes
            uniq = np.unique(arr[~np.isnan(arr)])
            cmap_base = matplotlib.colormaps.get_cmap("tab20c").resampled(len(uniq))
            code_to_idx = {int(c): i for i, c in enumerate(uniq)}
            mapped = np.full_like(arr, np.nan)
            for c, i in code_to_idx.items():
                mapped[arr == c] = i
            ax.imshow(mapped, cmap=cmap_base, origin="upper",
                      interpolation="nearest")
            ax.set_title(f"Existing Vegetation Type\n({len(uniq)} classes in AOI)",
                         fontsize=10, fontweight="bold")

        elif var == "evc":
            vmin = np.nanpercentile(arr, 2)
            vmax = np.nanpercentile(arr, 98)
            im = ax.imshow(arr, cmap="YlGn", origin="upper",
                           interpolation="nearest", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.7, label="% cover")
            ax.set_title("Existing Vegetation Cover", fontsize=10, fontweight="bold")

        elif var == "evh":
            vmin = np.nanpercentile(arr, 2)
            vmax = np.nanpercentile(arr, 98)
            im = ax.imshow(arr, cmap="BuGn", origin="upper",
                           interpolation="nearest", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.7, label="height (m)")
            ax.set_title("Existing Vegetation Height", fontsize=10, fontweight="bold")

        elif var == "elevation":
            vmin = np.nanpercentile(arr, 1)
            vmax = np.nanpercentile(arr, 99)
            im = ax.imshow(arr, cmap="terrain", origin="upper",
                           interpolation="bilinear", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.7, label="m")
            ax.set_title("Elevation", fontsize=10, fontweight="bold")

        elif var == "slope":
            im = ax.imshow(arr, cmap="Oranges", origin="upper",
                           interpolation="bilinear",
                           vmin=0, vmax=np.nanpercentile(arr, 98))
            plt.colorbar(im, ax=ax, shrink=0.7, label="°")
            ax.set_title("Slope", fontsize=10, fontweight="bold")

        elif var == "aspect":
            im = ax.imshow(arr, cmap="hsv", origin="upper",
                           interpolation="nearest", vmin=0, vmax=360)
            plt.colorbar(im, ax=ax, shrink=0.7, label="° from N")
            ax.set_title("Aspect", fontsize=10, fontweight="bold")

        else:
            vmin = np.nanpercentile(arr, 2)
            vmax = np.nanpercentile(arr, 98)
            im = ax.imshow(arr, cmap="viridis", origin="upper",
                           interpolation="nearest", vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.7)
            ax.set_title(var, fontsize=10, fontweight="bold")

        ax.axis("off")

    # Hide unused axes
    for idx in range(len(vars_present), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "LANDFIRE LF2022 — Northern Colorado Mountain Landscape",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    if figures_dir is not None:
        Path(figures_dir).mkdir(parents=True, exist_ok=True)
        out = Path(figures_dir) / "landscape_maps.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")

    return fig


# ---------------------------------------------------------------------------
# Vegetation–fuel relationship plots
# ---------------------------------------------------------------------------

def plot_fuel_distributions(
    df: pd.DataFrame,
    figures_dir: Optional[Path] = None,
) -> plt.Figure:
    """Boxplots of EVC and EVH distributions per fuel model group."""
    import seaborn as sns

    df_plot = df.dropna(subset=["fuel_model"]).copy()
    df_plot["fuel_group"] = df_plot["fuel_model"].apply(
        lambda c: fbfm40_group(int(c)) if not np.isnan(c) else "Unknown"
    )
    # Mask non-vegetated EVC/EVH codes
    for col in ("evc", "evh"):
        if col in df_plot.columns:
            df_plot[col] = df_plot[col].where(
                ~df_plot[col].isin(NON_VEGETATED_CODES)
            )

    group_order = [g for g in FBFM40_GROUP_COLORS if g in df_plot["fuel_group"].unique()]
    palette = {g: FBFM40_GROUP_COLORS[g] for g in group_order}

    n_panels = sum(1 for c in ("evc", "evh") if c in df_plot.columns)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    for col, label in [("evc", "Vegetation Cover (%)"), ("evh", "Vegetation Height (m)")]:
        if col not in df_plot.columns:
            continue
        ax = axes[ax_idx]
        sub = df_plot.dropna(subset=[col])
        sns.boxplot(
            data=sub, x="fuel_group", y=col,
            order=group_order, palette=palette,
            ax=ax, linewidth=0.8, fliersize=1.5,
        )
        ax.set_xticklabels(group_order, rotation=35, ha="right", fontsize=9)
        ax.set_xlabel("Fuel model group", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} by Fuel Group", fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax_idx += 1

    fig.suptitle(
        "Vegetation Structure vs. Fuel Model Group",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    if figures_dir is not None:
        out = Path(figures_dir) / "fuel_distributions.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")

    return fig


def plot_fuel_evt_heatmap(
    df: pd.DataFrame,
    top_n_evt: int = 15,
    figures_dir: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap of cell count by fuel group × top-N EVT class codes."""
    df_plot = df.dropna(subset=["fuel_model", "evt"]).copy()
    df_plot["fuel_group"] = df_plot["fuel_model"].apply(
        lambda c: fbfm40_group(int(c)) if not np.isnan(c) else "Unknown"
    )

    top_evt = (
        df_plot["evt"].value_counts().head(top_n_evt).index.tolist()
    )
    df_plot = df_plot[df_plot["evt"].isin(top_evt)]

    pivot = (
        df_plot.groupby(["fuel_group", "evt"])
        .size()
        .unstack(fill_value=0)
    )
    # Normalize rows to show composition within each fuel group
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(10, len(top_evt) * 0.7), 6))
    im = ax.imshow(pivot_norm.values, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=pivot_norm.values.max())
    plt.colorbar(im, ax=ax, shrink=0.7, label="Fraction of fuel-group cells")

    ax.set_xticks(range(len(pivot_norm.columns)))
    ax.set_xticklabels(
        [f"EVT {int(c)}" for c in pivot_norm.columns],
        rotation=40, ha="right", fontsize=8,
    )
    ax.set_yticks(range(len(pivot_norm.index)))
    ax.set_yticklabels(pivot_norm.index, fontsize=9)
    ax.set_title(
        f"EVT Composition within Fuel Model Groups (top {top_n_evt} EVT classes)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    if figures_dir is not None:
        out = Path(figures_dir) / "fuel_evt_heatmap.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")

    return fig


def plot_disturbance_effect(
    df: pd.DataFrame,
    figures_dir: Optional[Path] = None,
) -> plt.Figure:
    """Show how disturbance shifts the fuel model group distribution."""
    df_plot = df.dropna(subset=["fuel_model", "disturbance"]).copy()
    df_plot["fuel_group"]  = df_plot["fuel_model"].apply(
        lambda c: fbfm40_group(int(c)) if not np.isnan(c) else "Unknown"
    )
    df_plot["disturbed"] = df_plot["disturbance"] > 0

    groups = [g for g in FBFM40_GROUP_COLORS if g in df_plot["fuel_group"].unique()]
    palette = {g: FBFM40_GROUP_COLORS[g] for g in groups}

    undist = df_plot[~df_plot["disturbed"]]["fuel_group"].value_counts(normalize=True)
    dist   = df_plot[df_plot["disturbed"]]["fuel_group"].value_counts(normalize=True)

    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    bars_u = ax.bar(x - w/2, [undist.get(g, 0) * 100 for g in groups],
                    w, label="Undisturbed", color=[palette[g] for g in groups],
                    alpha=0.9, edgecolor="white")
    bars_d = ax.bar(x + w/2, [dist.get(g, 0) * 100 for g in groups],
                    w, label="Recently disturbed", color=[palette[g] for g in groups],
                    alpha=0.45, edgecolor=[palette[g] for g in groups],
                    linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of cells in group")
    ax.set_title(
        "Fuel Model Group Distribution: Undisturbed vs. Recently Disturbed Cells",
        fontsize=11, fontweight="bold",
    )
    ax.legend()
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if figures_dir is not None:
        out = Path(figures_dir) / "disturbance_effect.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")

    return fig
