
"""Comparaison de Gaussian smoothing pour les cartes d'anomalies PatchCore."""

import argparse
import csv

import numpy as np
from scipy.ndimage import gaussian_filter

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


SIGMAS = [0, 0.5, 1.0, 1.5]


def calculate_metrics(maps, masks, threshold):
    """Calcule les métriques pixel sur un ensemble d'images."""

    predicted = maps >= threshold
    ground_truth = masks.astype(bool)

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
        "n_images": len(maps),
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


def smooth_maps(maps, sigma):
    """Applique un Gaussian smoothing séparément à chaque carte."""

    if sigma == 0:
        return maps

    smoothed = np.empty_like(maps, dtype=np.float32)

    for i in range(len(maps)):
        smoothed[i] = gaussian_filter(
            maps[i],
            sigma=sigma,
            mode="nearest",
        )

    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="leather")
    args = parser.parse_args()

    quiet()
    cfg = load_config()
    category = args.category

    print(f"\n=== Comparaison Post-processing : {category} ===")

    # 1. Charger le checkpoint existant
    ckpt = checkpoint_path(cfg, category)
    print(f"Checkpoint : {ckpt}")

    model = load_model(ckpt, map_location="cpu")

    # 2. Préparer les données de test
    dm = build_datamodule(cfg, category)
    dm.setup(stage="test")

    engine = build_engine(
        cfg,
        resolve(cfg["paths"]["outputs_dir"]),
    )

    # 3. Récupérer les prédictions PatchCore
    pred = collect_predictions(model, dm, engine)

    maps = np.asarray(pred["maps"], dtype=np.float32)
    masks = np.asarray(pred["masks"], dtype=bool)
    defect_types = np.asarray(pred["defect_types"])

    threshold = float(model.post_processor.pixel_threshold)

    print(f"\nNombre d'images : {len(maps)}")
    print(f"Seuil fixe       : {threshold:.6f}")
    print(f"Configurations   : {SIGMAS}")

    # 4. Comparer les configurations
    all_results = []

    for sigma in SIGMAS:
        print(f"\n{'=' * 55}")
        print(f"GAUSSIAN SIGMA = {sigma}")
        print(f"{'=' * 55}")

        processed_maps = smooth_maps(maps, sigma)

        # Résultat global, toutes les images incluses
        global_metrics = calculate_metrics(
            processed_maps,
            masks,
            threshold,
        )

        row = {
            "sigma": sigma,
            "defect_type": "ALL",
            **global_metrics,
        }
        all_results.append(row)

        print(
            f"GLOBAL | F1={global_metrics['pixel_f1']:.4f} | "
            f"Precision={global_metrics['precision']:.4f} | "
            f"Recall={global_metrics['recall']:.4f} | "
            f"IoU={global_metrics['iou']:.4f}"
        )

        # Résultats par type de défaut
        for defect_type in sorted(set(defect_types)):
            indices = np.where(defect_types == defect_type)[0]

            category_metrics = calculate_metrics(
                processed_maps[indices],
                masks[indices],
                threshold,
            )

            row = {
                "sigma": sigma,
                "defect_type": defect_type,
                **category_metrics,
            }
            all_results.append(row)

            if category_metrics["ground_truth_pixels"] == 0:
                print(
                    f"{defect_type.upper():6s} | "
                    f"FP={category_metrics['fp']:,} | "
                    "Pixel F1=N/A (aucun défaut réel)"
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

    # 5. Sauvegarder les résultats dans un CSV
    output_dir = results_dir(cfg, category)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "postprocessing_comparison.csv"

    fieldnames = list(all_results[0].keys())

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # 6. Comparaison finale des F1 globaux
    global_results = [
        row for row in all_results
        if row["defect_type"] == "ALL"
    ]

    best = max(global_results, key=lambda row: row["pixel_f1"])
    baseline = next(row for row in global_results if row["sigma"] == 0)

    improvement = (
        best["pixel_f1"] - baseline["pixel_f1"]
    ) * 100

    print(f"\n{'=' * 55}")
    print("RÉSUMÉ GLOBAL")
    print(f"{'=' * 55}")

    for row in global_results:
        print(
            f"Sigma={row['sigma']:<3} | "
            f"Pixel F1={row['pixel_f1']:.4f} | "
            f"Precision={row['precision']:.4f} | "
            f"Recall={row['recall']:.4f}"
        )

    print(f"\nMeilleur sigma exploratoire : {best['sigma']}")
    print(f"Pixel F1 baseline           : {baseline['pixel_f1']:.4f}")
    print(f"Meilleur Pixel F1            : {best['pixel_f1']:.4f}")
    print(f"Différence                  : {improvement:+.2f} points")

    print(f"\nCSV sauvegardé : {output_file}")


if __name__ == "__main__":
    main()
