
"""Comparaison du filtrage Connected Components pour PatchCore."""

import argparse
import csv

import numpy as np
from scipy.ndimage import label

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


MIN_AREAS = [0, 2, 5, 10, 20]


def calculate_metrics(predicted, masks):
    """Calcule les métriques pixel à partir de masques binaires."""

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


def filter_small_components(binary_mask, min_area):
    """Supprime les composantes connexes de surface inférieure à min_area."""

    if min_area == 0:
        return binary_mask.copy()

    # Connexité 8 : les pixels diagonaux sont aussi considérés voisins.
    structure = np.ones((3, 3), dtype=np.uint8)

    component_labels, number_of_components = label(
        binary_mask,
        structure=structure,
    )

    if number_of_components == 0:
        return np.zeros_like(binary_mask, dtype=bool)

    # Surface de chaque composante.
    component_sizes = np.bincount(component_labels.ravel())

    # Conserver uniquement les composantes suffisamment grandes.
    keep = component_sizes >= min_area

    # Le label 0 correspond au fond : il ne doit pas être conservé.
    keep[0] = False

    return keep[component_labels]


def analyze_component_sizes(maps, threshold):
    """Analyse la distribution des tailles des composantes connexes."""

    structure = np.ones((3, 3), dtype=np.uint8)

    all_sizes = []

    for anomaly_map in maps:
        binary_mask = np.squeeze(anomaly_map) >= threshold

        component_labels, number = label(
            binary_mask,
            structure=structure,
        )

        if number == 0:
            continue

        sizes = np.bincount(component_labels.ravel())[1:]
        all_sizes.extend(sizes.tolist())

    all_sizes = np.asarray(all_sizes, dtype=np.int64)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC DES COMPOSANTES CONNEXES")
    print("=" * 60)

    if len(all_sizes) == 0:
        print("Aucune composante détectée.")
        return

    print(f"Nombre total de composantes : {len(all_sizes)}")
    print(f"Taille minimale : {all_sizes.min()} pixels")
    print(f"Taille médiane : {np.median(all_sizes):.1f} pixels")
    print(f"Taille maximale : {all_sizes.max()} pixels")

    for limit in [2, 5, 10, 20, 50, 100]:
        small = all_sizes[all_sizes < limit]

        print(
            f"Composantes < {limit:3d} pixels : "
            f"{len(small):6d} | "
            f"Pixels concernés : {small.sum():8d}"
        )

def apply_connected_components(maps, threshold, min_area):
    """Transforme les cartes d'anomalies en masques filtrés."""

    filtered_masks = []

    for anomaly_map in maps:
        anomaly_map = np.squeeze(anomaly_map)

        binary_mask = anomaly_map >= threshold

        filtered_mask = filter_small_components(
            binary_mask,
            min_area,
        )

        filtered_masks.append(filtered_mask)

    return np.asarray(filtered_masks, dtype=bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="leather")
    args = parser.parse_args()

    quiet()
    cfg = load_config()
    category = args.category

    print(f"\n=== Connected Components : {category} ===")

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
    print(f"Surfaces minimales : {MIN_AREAS}")
    print(f"Dimensions des cartes : {maps.shape}")
    print(f"Dimensions des masques : {masks.shape}")
    analyze_component_sizes(maps, threshold)
    # 4. Comparer les configurations.
    all_results = []

    for min_area in MIN_AREAS:

        print("\n" + "=" * 60)
        print(f"CONNECTED COMPONENTS - MIN AREA = {min_area}")
        print("=" * 60)

        filtered_masks = apply_connected_components(
            maps,
            threshold,
            min_area,
        )

        # Résultat global.
        global_metrics = calculate_metrics(
            filtered_masks,
            masks,
        )

        row = {
            "min_area": min_area,
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
                filtered_masks[indices],
                masks[indices],
            )

            row = {
                "min_area": min_area,
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

    output_file = output_dir / "connected_components_comparison.csv"

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
        if row["min_area"] == 0
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
            f"Min area={row['min_area']:2d} | "
            f"F1={row['pixel_f1']:.4f} | "
            f"Precision={row['precision']:.4f} | "
            f"Recall={row['recall']:.4f} | "
            f"IoU={row['iou']:.4f}"
        )

    print(f"\nMeilleur filtrage exploratoire : {best['min_area']} pixels")
    print(f"Pixel F1 baseline : {baseline['pixel_f1']:.4f}")
    print(f"Meilleur Pixel F1 : {best['pixel_f1']:.4f}")
    print(f"Différence : {improvement:+.2f} points")

    print(f"\nCSV sauvegardé : {output_file}")


if __name__ == "__main__":
    main()
