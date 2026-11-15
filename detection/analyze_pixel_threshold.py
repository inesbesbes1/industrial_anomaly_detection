
"""Analyse exploratoire des seuils Pixel F1 pour PatchCore."""

import argparse
import csv
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="leather")
    args = parser.parse_args()

    quiet()
    cfg = load_config()

    category = args.category

    print(f"\n=== Analyse des seuils Pixel F1 : {category} ===")

    # 1. Charger le checkpoint existant
    ckpt = checkpoint_path(cfg, category)
    print(f"Checkpoint : {ckpt}")

    model = load_model(ckpt, map_location="cpu")

    # 2. Préparer les données et récupérer les prédictions
    dm = build_datamodule(cfg, category)

# Initialiser les datasets de test
    dm.setup(stage="test")

    engine = build_engine(cfg, resolve(cfg["paths"]["outputs_dir"]))

    pred = collect_predictions(model, dm, engine)

    maps = pred["maps"]
    masks = pred["masks"]

    # 3. Récupérer le seuil actuel d'Anomalib
    current_threshold = float(model.post_processor.pixel_threshold)

    # Aplatir les cartes et les masques
    scores = maps.ravel()
    ground_truth = masks.ravel()

    print(f"Nombre de pixels : {len(scores):,}")
    print(f"Seuil actuel : {current_threshold:.6f}")
    print(f"Score minimum : {scores.min():.6f}")
    print(f"Score maximum : {scores.max():.6f}")

    # 4. Construire une liste de seuils à tester
    lower = np.percentile(scores, 80)
    upper = np.percentile(scores, 99.99)

    thresholds = np.linspace(lower, upper, 101)
    thresholds = np.unique(
        np.append(thresholds, current_threshold)
    )

    results = []

    # 5. Calculer Precision, Recall et F1 pour chaque seuil
    for threshold in thresholds:
        predicted = scores >= threshold

        tp = np.logical_and(predicted, ground_truth).sum()
        fp = np.logical_and(predicted, ~ground_truth).sum()
        fn = np.logical_and(~predicted, ground_truth).sum()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        results.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "pixel_f1": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "is_current_threshold": (
                abs(threshold - current_threshold) < 1e-6
            ),
        })

    # 6. Trier les résultats par Pixel F1 décroissant
    results.sort(key=lambda x: x["pixel_f1"], reverse=True)

    # 7. Sauvegarder le CSV
    output_dir = results_dir(cfg, category)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "pixel_threshold_analysis.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    # 8. Afficher les meilleurs seuils
    best = results[0]

    current = next(
        r for r in results if r["is_current_threshold"]
    )

    print("\n=== Seuil actuel ===")
    print(f"Threshold : {current['threshold']:.6f}")
    print(f"Precision : {current['precision']:.4f}")
    print(f"Recall    : {current['recall']:.4f}")
    print(f"Pixel F1  : {current['pixel_f1']:.4f}")

    print("\n=== Meilleur seuil exploratoire ===")
    print(f"Threshold : {best['threshold']:.6f}")
    print(f"Precision : {best['precision']:.4f}")
    print(f"Recall    : {best['recall']:.4f}")
    print(f"Pixel F1  : {best['pixel_f1']:.4f}")

    print("\n=== Top 5 des seuils ===")
    for i, result in enumerate(results[:5], 1):
        print(
            f"{i}. Threshold={result['threshold']:.4f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['pixel_f1']:.4f}"
        )

    print(f"\nCSV sauvegardé : {output_file}")


if __name__ == "__main__":
    main()
