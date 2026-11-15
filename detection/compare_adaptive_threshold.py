
"""
Compare global and adaptive thresholds for PatchCore anomaly maps.

Methods:
- Global baseline threshold
- Mean + k * Standard Deviation
- Per-image percentile thresholds

Results:
- Global metrics
- Per-defect metrics
- False positives on normal images
- CSV report
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from anomalib.engine import Engine

from detection.common import checkpoint_path
from detection.evaluate import (
    load_model,
    build_datamodule,
    collect_predictions,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASELINE_THRESHOLD = 50.5652885

K_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
PERCENTILES = [99.0, 99.5, 99.9]


# ============================================================
# DATA PREPARATION
# ============================================================

def load_config():
    config_path = Path("config.yaml")

    if not config_path.exists():
        raise FileNotFoundError("config.yaml introuvable.")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_arrays(predictions):

    maps = np.asarray(predictions["maps"], dtype=np.float32)
    masks = np.asarray(predictions["masks"])

    # Remove channel dimension if present: (N, 1, H, W)
    if maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if maps.ndim != 3 or masks.ndim != 3:
        raise ValueError(
            f"Unexpected shapes: maps={maps.shape}, masks={masks.shape}"
        )

    if maps.shape != masks.shape:
        raise ValueError(
            f"Maps and masks have different shapes: "
            f"{maps.shape} vs {masks.shape}"
        )

    masks = masks > 0

    defect_types = predictions["defect_types"]

    if defect_types is None:
        defect_types = ["unknown"] * len(maps)

    return maps, masks, np.asarray(defect_types, dtype=str)


# ============================================================
# THRESHOLD METHODS
# ============================================================

def generate_thresholds(anomaly_map):

    thresholds = {
        "global_baseline": BASELINE_THRESHOLD
    }

    mean = float(np.mean(anomaly_map))
    std = float(np.std(anomaly_map))

    for k in K_VALUES:
        thresholds[f"mean_plus_{k}_std"] = mean + k * std

    for percentile in PERCENTILES:
        thresholds[f"percentile_{percentile}"] = np.percentile(
            anomaly_map,
            percentile
        )

    return thresholds


def generate_predictions(maps):

    methods = {
        "global_baseline": np.full(
            maps.shape[0],
            BASELINE_THRESHOLD,
            dtype=np.float32
        )
    }

    for k in K_VALUES:
        methods[f"mean_plus_{k}_std"] = (
            np.mean(maps, axis=(1, 2))
            + k * np.std(maps, axis=(1, 2))
        )

    for percentile in PERCENTILES:
        methods[f"percentile_{percentile}"] = np.percentile(
            maps,
            percentile,
            axis=(1, 2)
        )

    predictions = {}

    for method, thresholds in methods.items():
        predictions[method] = (
            maps >= thresholds[:, None, None]
        )

    return predictions


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(predicted, ground_truth):

    predicted = predicted.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = int(np.logical_and(predicted, ground_truth).sum())
    fp = int(np.logical_and(predicted, ~ground_truth).sum())
    fn = int(np.logical_and(~predicted, ground_truth).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "pixel_f1": f1,
        "iou": iou,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_methods(maps, masks, defect_types):

    method_predictions = generate_predictions(maps)

    results = []

    for method, predicted_masks in method_predictions.items():

        # Global metrics
        metrics = calculate_metrics(
            predicted_masks,
            masks
        )

        results.append({
            "method": method,
            "category": "ALL",
            **metrics
        })

        # Normal images: false positive analysis
        good_indices = np.where(defect_types == "good")[0]

        if len(good_indices) > 0:

            normal_predictions = predicted_masks[good_indices]

            false_positive_pixels = int(normal_predictions.sum())

            total_pixels = normal_predictions.size

            results.append({
                "method": method,
                "category": "GOOD",
                "precision": np.nan,
                "recall": np.nan,
                "pixel_f1": np.nan,
                "iou": np.nan,
                "TP": 0,
                "FP": false_positive_pixels,
                "FN": 0,
                "false_positive_rate": (
                    false_positive_pixels / total_pixels
                    if total_pixels else 0.0
                )
            })

        # Per-defect metrics
        categories = sorted(set(defect_types) - {"good"})

        for category in categories:

            indices = np.where(defect_types == category)[0]

            if len(indices) == 0:
                continue

            category_metrics = calculate_metrics(
                predicted_masks[indices],
                masks[indices]
            )

            results.append({
                "method": method,
                "category": category,
                **category_metrics
            })

    return pd.DataFrame(results)


# ============================================================
# DISPLAY
# ============================================================

def display_results(df):

    print("\n" + "=" * 75)
    print("ADAPTIVE THRESHOLD COMPARISON")
    print("=" * 75)

    global_results = df[df["category"] == "ALL"]

    print("\nGLOBAL RESULTS:\n")

    print(
        global_results[
            ["method", "precision", "recall", "pixel_f1", "iou", "FP"]
        ].to_string(index=False)
    )

    print("\nNORMAL IMAGES - FALSE POSITIVE ANALYSIS:\n")

    normal_results = df[df["category"] == "GOOD"]

    if not normal_results.empty:
        print(
            normal_results[
                ["method", "FP", "false_positive_rate"]
            ].to_string(index=False)
        )

    print("\nPER-DEFECT RESULTS:\n")

    defect_results = df[
        ~df["category"].isin(["ALL", "GOOD"])
    ]

    print(
        defect_results[
            ["method", "category", "precision", "recall", "pixel_f1", "iou"]
        ].to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--category",
        default="leather",
        help="MVTec category"
    )

    args = parser.parse_args()

    category = args.category

    print("\nLoading configuration...")
    cfg = load_config()

    print("\nLoading PatchCore checkpoint...")

    ckpt = checkpoint_path(cfg, category)

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = load_model(
        ckpt,
        map_location="cpu"
    )

    print("\nBuilding test datamodule...")

    datamodule = build_datamodule(cfg, category)

    # Required before accessing test_dataloader()
    datamodule.setup(stage="test")

    engine = Engine()

    print("\nCollecting predictions...")
    print("This may take a few minutes on CPU.")

    predictions = collect_predictions(
        model,
        datamodule,
        engine
    )

    maps, masks, defect_types = prepare_arrays(predictions)

    print(f"\nImages evaluated: {len(maps)}")
    print(f"Anomaly maps shape: {maps.shape}")
    print(f"Baseline threshold: {BASELINE_THRESHOLD}")

    print("\nEvaluating threshold methods...")

    df = evaluate_methods(
        maps,
        masks,
        defect_types
    )

    display_results(df)

    # Save CSV
    output_dir = Path(cfg["paths"]["results_dir"]) / category
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "adaptive_threshold_comparison.csv"

    df.to_csv(output_file, index=False)

    print("\n" + "=" * 75)
    print("EXPERIMENT COMPLETED")
    print("=" * 75)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
