
"""Comparaison du Morphological Post-processing pour PatchCore."""

import argparse
import csv

import cv2
import numpy as np

from .common import (
    build_datamodule,
    build_engine,
    checkpoint_path,
    load_config,
    quiet,
    resolve,
    results_dir,
)
from .evaluate import load_model, collect_predictions


# Configurations à tester.
CONFIGURATIONS = [
    {"name": "baseline", "operation": "none", "kernel_size": 0},
    {"name": "opening_3x3", "operation": "opening", "kernel_size": 3},
    {"name": "opening_5x5", "operation": "opening", "kernel_size": 5},
    {"name": "closing_3x3", "operation": "closing", "kernel_size": 3},
    {"name": "closing_5x5", "operation": "closing", "kernel_size": 5},
    {"name": "opening_closing_3x3", "operation": "opening_closing", "kernel_size": 3},
]


def calculate_metrics(predicted, masks):
    """Calcule les métriques pixel sur un ensemble d'images."""

    predicted = np.asarray(predicted, dtype=bool)
    ground_truth = np.asarray(masks, dtype=bool)

    tp = int(np.logical_and(predicted, ground_truth).sum())
    fp = int(np.logical_and(predicted, ~ground_truth).sum())
    fn = int(np.logical_and(~predicted, ground_truth).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0

    gt_area = int(ground_truth.sum())
    pred_area = int(predicted.sum())

    return {
        "n_images": len(predicted),
        "ground_truth_pixels": gt_area,
        "predicted_pixels": pred_area,
        "area_ratio": pred_area / gt_area if gt_area else None,
        "precision": precision,
        "recall": recall,
        "pixel_f1": f1,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def apply_morphology(binary_mask, operation, kernel_size):
    """Applique une opération morphologique à un masque binaire."""

    if operation == "none":
        return binary_mask.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    mask_uint8 = binary_mask.astype(np.uint8) * 255

    if operation == "opening":
        processed = cv2.morphologyEx(
            mask_uint8,
            cv2.MORPH_OPEN,
            kernel,
        )

    elif operation == "closing":
        processed = cv2.morphologyEx(
            mask_uint8,
            cv2.MORPH_CLOSE,
            kernel,
        )

    elif operation == "opening_closing":
        processed = cv2.morphologyEx(
            mask_uint8,
            cv2.MORPH_OPEN,
            kernel,
        )

        processed = cv2.morphologyEx(
            processed,
            cv2.MORPH_CLOSE,
            kernel,
        )

    else:
        raise ValueError(f"Opération inconnue : {operation}")

    return processed > 0


def process_maps(maps, threshold, operation, kernel_size):
    """Transforme les cartes d'anomalies en masques post-traités."""

    processed_masks = []

    for anomaly_map in maps:
        anomaly_map = np.squeeze(anomaly_map)

        # Seuillage identique à la baseline.
        binary_mask = anomaly_map >= threshold

        # Post-processing morphologique.
        processed_mask = apply_morphology(
            binary_mask,
            operation,
            kernel_size,
        )

        processed_masks.append(processed_mask)

    return np.asarray(processed_masks, dtype=bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="leather")
    args = parser.parse_args()

    quiet()
    cfg = load_config()
    category = args.category

    print(f"\n=== Morphological Post-processing : {category} ===")

    # 1. Charger le checkpoint existant.
    ckpt = checkpoint_path(cfg, category)
    print(f"Checkpoint : {ckpt}")

    model = load_model(ckpt, map_location="cpu")

    # 2. Préparer les données de test.
    dm = build_datamodule(cfg, category)
    dm.setup(stage="test")

    engine = build_engine(
        cfg,
        resolve(cfg["paths"]["outputs_dir"]),
    )

    # 3. Récupérer les prédictions PatchCore.
    pred = collect_predictions(model, dm, engine)

    maps = np.asarray(pred["maps"], dtype=np.float32)
    masks = np.asarray(pred["masks"], dtype=bool)
    defect_types = np.asarray(pred["defect_types"])

    threshold = float(model.post_processor.pixel_threshold)

    print(f"\nNombre d'images : {len(maps)}")
    print(f"Seuil fixe : {threshold:.6f}")
    print(f"Configurations : {len(CONFIGURATIONS)}")

    # 4. Comparer les configurations.
    all_results = []

    for config in CONFIGURATIONS:

        name = config["name"]
        operation = config["operation"]
        kernel_size = config["kernel_size"]

        print("\n" + "=" * 60)
        print(f"CONFIGURATION : {name}")
        print("=" * 60)

        processed_masks = process_maps(
            maps,
            threshold,
            operation,
            kernel_size,
        )

        # Résultat global.
        global_metrics = calculate_metrics(
            processed_masks,
            masks,
        )

        row = {
            "configuration": name,
            "operation": operation,
            "kernel_size": kernel_size,
            "defect_type": "ALL",
            **global_metrics,
        }

        all_results.append(row)

        print(
            f"GLOBAL | F1={global_metrics['pixel_f1']:.4f} | "
            f"Precision={global_metrics['precision']:.4f} | "
            f"Recall={global_metrics['recall']:.4f} | "
            f"IoU={global_metrics['iou']:.4f} | "
            f"FP={global_metrics['fp']:,}"
        )

        # Résultats par type de défaut.
        for defect_type in sorted(set(defect_types)):

            indices = np.where(defect_types == defect_type)[0]

            category_metrics = calculate_metrics(
                processed_masks[indices],
                masks[indices],
            )

            row = {
                "configuration": name,
                "operation": operation,
                "kernel_size": kernel_size,
                "defect_type": defect_type,
                **category_metrics,
            }

            all_results.append(row)

            if category_metrics["ground_truth_pixels"] == 0:
                print(
                    f"{defect_type.upper():6s} | "
                    f"FP={category_metrics['fp']:,} | "
                    "Pixel F1=N/A"
                )
            else:
                print(
                    f"{defect_type.upper():6s} | "
                    f"F1={category_metrics['pixel_f1']:.4f} | "
                    f"Precision={category_metrics['precision']:.4f} | "
                    f"Recall={category_metrics['recall']:.4f} | "
                    f"IoU={category_metrics['iou']:.4f} | "
                    f"Area ratio={category_metrics['area_ratio']:.3f}"
                )

    # 5. Sauvegarder les résultats dans un CSV.
    output_dir = results_dir(cfg, category)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "morphology_comparison.csv"

    fieldnames = list(all_results[0].keys())

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # 6. Résumé global.
    global_results = [
        row for row in all_results
        if row["defect_type"] == "ALL"
    ]

    baseline = next(
        row for row in global_results
        if row["configuration"] == "baseline"
    )

    best = max(
        global_results,
        key=lambda row: row["pixel_f1"],
    )

    improvement = (
        best["pixel_f1"] - baseline["pixel_f1"]
    ) * 100

    print("\n" + "=" * 60)
    print("RÉSUMÉ GLOBAL")
    print("=" * 60)

    for row in global_results:
        print(
            f"{row['configuration']:25s} | "
            f"F1={row['pixel_f1']:.4f} | "
            f"Precision={row['precision']:.4f} | "
            f"Recall={row['recall']:.4f} | "
            f"IoU={row['iou']:.4f}"
        )

    print(f"\nMeilleure configuration exploratoire : {best['configuration']}")
    print(f"Pixel F1 baseline : {baseline['pixel_f1']:.4f}")
    print(f"Meilleur Pixel F1 : {best['pixel_f1']:.4f}")
    print(f"Différence : {improvement:+.2f} points")

    print(f"\nCSV sauvegardé : {output_file}")


if __name__ == "__main__":
    main()
