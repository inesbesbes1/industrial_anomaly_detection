
"""Visualisation qualitative des prédictions PatchCore."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


def select_examples(pred):
    """Sélectionne une image normale et une image par type de défaut."""
    selected = []

    # Une image normale
    normal_indices = np.where(pred["labels"] == 0)[0]
    if len(normal_indices):
        selected.append(int(normal_indices[0]))

    # Une image défectueuse par type
    defect_types = sorted(set(pred["defect_types"]))
    for defect_type in defect_types:
        if defect_type == "good":
            continue

        indices = np.where(
            (pred["defect_types"] == defect_type)
            & (pred["labels"] == 1)
        )[0]

        if len(indices):
            selected.append(int(indices[0]))

    return selected


def visualize_one(index, pred, threshold, output_dir):
    image_path = Path(pred["paths"][index])
    defect_type = pred["defect_types"][index]
    label = "Anormale" if pred["labels"][index] == 1 else "Normale"
    # Image originale
    image = Image.open(image_path).convert("RGB")

    anomaly_map = pred["maps"][index]
    ground_truth = pred["masks"][index].astype(bool)

    height, width = anomaly_map.shape

    # Adapter l'image originale à la résolution de la carte
    image = image.resize(
        (width, height),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(image)

    # Masque prédit avec le seuil actuel
    predicted_mask = anomaly_map >= threshold
    # Calcul des métriques pour cette im    age
    tp = np.logical_and(predicted_mask, ground_truth).sum()
    fp = np.logical_and(predicted_mask, ~ground_truth).sum()
    fn = np.logical_and(~predicted_mask, ground_truth).sum()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0

    gt_area = int(ground_truth.sum())
    pred_area = int(predicted_mask.sum())

    area_ratio = pred_area / gt_area if gt_area else 0.0

    print(f"\n--- Analyse : {image_path.name} ---")
    print(f"Type          : {defect_type}")
    print(f"Ground Truth  : {gt_area} pixels")
    print(f"Masque prédit : {pred_area} pixels")
    print(f"Area ratio    : {area_ratio:.3f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"Pixel F1      : {f1:.4f}")
    print(f"IoU           : {iou:.4f}")
    print(f"TP={tp} | FP={fp} | FN={fn}")
    

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image)
    axes[0].set_title("Image originale")

    axes[1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    heatmap = axes[2].imshow(anomaly_map, cmap="jet")
    axes[2].set_title("Anomaly Map")
    fig.colorbar(heatmap, ax=axes[2], fraction=0.046)

    axes[3].imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Masque prédit")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{image_path.name} | Type : {defect_type} | "
        f"{label} | Seuil : {threshold:.4f}"
    )

    fig.tight_layout()

    output_file = output_dir / f"{index}_{defect_type}_{image_path.stem}.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Image sauvegardée : {output_file}")


def analyze_all_defects(pred, threshold):
    """Calcule les métriques globales pour chaque type de défaut."""

    defect_types = np.asarray(pred["defect_types"])
    labels = np.asarray(pred["labels"])

    print("\n========== ANALYSE GLOBALE PAR TYPE ==========")

    for defect_type in sorted(set(defect_types)):

        indices = np.where(defect_types == defect_type)[0]

        maps = pred["maps"][indices]
        ground_truth = pred["masks"][indices].astype(bool)
        predicted = maps >= threshold

        tp = np.logical_and(predicted, ground_truth).sum()
        fp = np.logical_and(predicted, ~ground_truth).sum()
        fn = np.logical_and(~predicted, ground_truth).sum()

        gt_area = int(ground_truth.sum())
        pred_area = int(predicted.sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0

        print(f"\n--- {defect_type.upper()} ---")
        print(f"Nombre d'images : {len(indices)}")
        print(f"Ground Truth    : {gt_area:,} pixels")
        print(f"Masque prédit   : {pred_area:,} pixels")

        if gt_area > 0:
            print(f"Area ratio      : {pred_area / gt_area:.3f}")
            print(f"Precision       : {precision:.4f}")
            print(f"Recall          : {recall:.4f}")
            print(f"Pixel F1        : {f1:.4f}")
            print(f"IoU             : {iou:.4f}")
        else:
            print("Area ratio      : N/A")
            print("Pixel F1        : N/A (aucun pixel anormal réel)")

        print(f"TP={tp:,} | FP={fp:,} | FN={fn:,}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="leather")
    args = parser.parse_args()

    quiet()
    cfg = load_config()
    category = args.category

    print(f"\n=== Visualisation PatchCore : {category} ===")

    # Charger le modèle existant
    ckpt = checkpoint_path(cfg, category)
    model = load_model(ckpt, map_location="cpu")

    # Préparer le DataModule
    dm = build_datamodule(cfg, category)
    dm.setup(stage="test")

    engine = build_engine(
        cfg,
        resolve(cfg["paths"]["outputs_dir"])
    )

    # Récupérer les prédictions
    pred = collect_predictions(model, dm, engine)

    threshold = float(model.post_processor.pixel_threshold)
    analyze_all_defects(pred, threshold)
    # Sélectionner les exemples
    indices = select_examples(pred)

    output_dir = results_dir(cfg, category) / "figures" / "qualitative"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSeuil utilisé : {threshold:.6f}")
    print(f"Nombre d'images sélectionnées : {len(indices)}")

    for index in indices:
        visualize_one(index, pred, threshold, output_dir)

    print(f"\nVisualisations terminées : {output_dir}")


if __name__ == "__main__":
    main()
