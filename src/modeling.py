"""
Educational predictive modeling: predict FBFM40 from vegetation variables.

Scientific question
-------------------
"How much of the mapped fuel class can be recovered from LANDFIRE vegetation
and disturbance layers alone, and where does that simplified relationship fail?"

The purpose is explicitly NOT to reproduce LANDFIRE's fuel-model assignment
process. LANDFIRE derives FBFM40 from a complex ruleset that incorporates EVT,
EVC, EVH, disturbance type, ecological context, and expert knowledge developed
over decades. A Random Forest trained on those same inputs will partially
recover the relationships but cannot replicate the full decision logic.

Errors are scientifically informative: they reveal which fuel classes require
ecological context, disturbance history, or other information not captured by
the aggregate vegetation variables in this experiment.

Disclaimer
----------
This classifier is an educational exercise. It should not be used as a
replacement for LANDFIRE fuel assignments or as a basis for operational
fire-behavior analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .load_data import FBFM40_LOOKUP, FBFM40_GROUP_COLORS, fbfm40_group, fbfm40_name


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    target_col: str = "fuel_model",
    feature_cols: Optional[List[str]] = None,
    min_class_size: int = 20,
) -> Tuple[np.ndarray, np.ndarray, List[str], LabelEncoder]:
    """Prepare feature matrix and encoded label vector for fuel-model prediction.

    Features
    --------
    By default: EVT (one-hot encoded), EVC, EVH, disturbance presence flag.
    EVT is treated as categorical (one-hot) because vegetation type codes are
    class identifiers, not ordinal values — their numeric magnitude is meaningless.

    Parameters
    ----------
    df             : DataFrame from analysis.raster_to_dataframe()
    target_col     : column name for the target label (FBFM40 code)
    feature_cols   : override the default feature list
    min_class_size : drop fuel-model classes with fewer than this many samples

    Returns
    -------
    X : feature matrix (n_samples, n_features)
    y : encoded integer labels (n_samples,)
    feature_names : list of feature column names
    encoder : LabelEncoder for decoding predictions back to fuel codes
    """
    if feature_cols is None:
        feature_cols = [c for c in ("evt", "evc", "evh", "disturbance")
                        if c in df.columns]

    df_clean = df.dropna(subset=[target_col] + feature_cols).copy()

    # Drop rare classes for stable evaluation
    class_counts = df_clean[target_col].value_counts()
    valid_classes = class_counts[class_counts >= min_class_size].index
    df_clean = df_clean[df_clean[target_col].isin(valid_classes)]
    print(f"Features: {feature_cols}")
    print(f"Samples after filtering: {len(df_clean):,}  "
          f"Fuel classes retained: {len(valid_classes)}")

    # Encode disturbance as binary presence flag
    if "disturbance" in df_clean.columns:
        df_clean["disturbed"] = (df_clean["disturbance"] > 0).astype(float)
        feature_cols = [c if c != "disturbance" else "disturbed" for c in feature_cols]
        df_clean["disturbed"] = (df_clean["disturbance"] > 0).astype(float)

    # One-hot encode EVT (many categories; use top N by frequency)
    if "evt" in feature_cols:
        top_evt = df_clean["evt"].value_counts().head(50).index
        for code in top_evt:
            col = f"evt_{int(code)}"
            df_clean[col] = (df_clean["evt"] == code).astype(float)
        feat_cols_final = (
            [c for c in feature_cols if c != "evt"]
            + [f"evt_{int(c)}" for c in top_evt]
        )
    else:
        feat_cols_final = feature_cols

    X = df_clean[feat_cols_final].values.astype(float)
    y_raw = df_clean[target_col].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    return X, y, feat_cols_final, encoder


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fuel_classifier(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    seed: int = 42,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    n_jobs: int = -1,
) -> Tuple[RandomForestClassifier, Dict]:
    """Train a Random Forest classifier for FBFM40 prediction.

    Spatial autocorrelation note
    ----------------------------
    A simple random split is used here. Because adjacent raster cells share
    environmental context, nearby cells may appear in both train and test sets,
    which likely inflates accuracy estimates. A spatially blocked split
    (e.g. grid blocks or geographic hold-outs) would give a more conservative
    estimate of generalization to new landscapes.

    Returns
    -------
    (fitted RandomForestClassifier, evaluation metrics dict)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(
        f"Training Random Forest  ({n_estimators} trees)\n"
        f"  Train: {len(X_train):,} samples  |  Test: {len(X_test):,} samples"
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced",  # handle unequal class sizes
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    mac_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\nTest accuracy : {acc:.4f}")
    print(f"Macro F1      : {mac_f1:.4f}")

    metrics = {
        "accuracy":  round(acc, 4),
        "macro_f1":  round(mac_f1, 4),
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "n_classes": int(y.max()) + 1,
        "y_test":    y_test.tolist(),
        "y_pred":    y_pred.tolist(),
    }
    return clf, metrics


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_test:      np.ndarray,
    y_pred:      np.ndarray,
    encoder:     LabelEncoder,
    figures_dir: Optional[Path] = None,
    normalize:   bool = True,
) -> plt.Figure:
    """Plot a confusion matrix for FBFM40 fuel-class prediction."""
    classes   = encoder.classes_
    labels    = [fbfm40_name(int(c)) for c in classes]
    cm        = confusion_matrix(y_test, y_pred)
    cm_plot   = (cm / cm.sum(axis=1, keepdims=True)) if normalize else cm.astype(float)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.65), max(7, n * 0.6)))
    im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=1.0 if normalize else None)
    plt.colorbar(im, ax=ax, shrink=0.75,
                 label="Recall (fraction)" if normalize else "Count")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(
        "FBFM40 Prediction — Confusion Matrix (row-normalised recall)",
        fontsize=11, fontweight="bold",
    )

    thresh = cm_plot.max() / 2
    for i in range(n):
        for j in range(n):
            val = f"{cm_plot[i,j]:.2f}" if normalize else str(int(cm_plot[i,j]))
            ax.text(j, i, val, ha="center", va="center",
                    fontsize=5, color="white" if cm_plot[i, j] > thresh else "black")

    fig.tight_layout()
    if figures_dir is not None:
        out = Path(figures_dir) / "fuel_confusion_matrix.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    return fig


def plot_feature_importance(
    clf:           RandomForestClassifier,
    feature_names: List[str],
    top_n:         int = 20,
    figures_dir:   Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of the top-N most important features."""
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    importances = importances.nlargest(top_n)[::-1]

    # Color EVT indicators differently from continuous variables
    colors = [
        "#2E7D32" if name.startswith("evt_") else "#1565C0"
        for name in importances.index
    ]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    ax.barh(range(len(importances)), importances.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances.index, fontsize=8)
    ax.set_xlabel("Mean decrease in impurity (feature importance)")
    ax.set_title(
        f"Top {top_n} Features for FBFM40 Prediction\n"
        "(green = EVT indicator, blue = continuous variable)",
        fontsize=11, fontweight="bold",
    )
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if figures_dir is not None:
        out = Path(figures_dir) / "feature_importance.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    return fig


def plot_per_class_f1(
    y_test:      np.ndarray,
    y_pred:      np.ndarray,
    encoder:     LabelEncoder,
    figures_dir: Optional[Path] = None,
) -> plt.Figure:
    """Per-class F1 scores colored by fuel group."""
    classes = encoder.classes_
    report = classification_report(
        y_test, y_pred,
        labels=range(len(classes)),
        target_names=[fbfm40_name(int(c)) for c in classes],
        output_dict=True,
        zero_division=0,
    )
    names  = [fbfm40_name(int(c)) for c in classes]
    f1s    = [report[n]["f1-score"] for n in names]
    colors = [FBFM40_GROUP_COLORS.get(fbfm40_group(int(c)), "#888888") for c in classes]

    order = np.argsort(f1s)[::-1]

    fig, ax = plt.subplots(figsize=(12, max(4, len(names) * 0.38)))
    ax.barh(range(len(names)),
            [f1s[i] for i in order],
            color=[colors[i] for i in order],
            edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 1.05)
    ax.set_title(
        "Per-class F1: FBFM40 Prediction from EVT + EVC + EVH + Disturbance",
        fontsize=11, fontweight="bold",
    )
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Legend for fuel groups
    patches = [
        mpatches.Patch(color=c, label=g)
        for g, c in FBFM40_GROUP_COLORS.items()
        if any(fbfm40_group(int(cls)) == g for cls in classes)
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.8)
    import matplotlib.patches as mpatches  # already imported at top
    fig.tight_layout()

    if figures_dir is not None:
        out = Path(figures_dir) / "per_class_f1.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    return fig


def save_metrics(metrics: Dict, results_dir: Path) -> None:
    """Save evaluation metrics to a JSON file."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "fuel_model_prediction_metrics.json"
    # Convert numpy arrays to lists for JSON serialization
    json_metrics = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in metrics.items()
        if k not in ("y_test", "y_pred")
    }
    with open(out, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"  Metrics saved to {out}")
