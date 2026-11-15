"""Étape 3 - Évaluation de PatchCore sur le jeu de test MVTec AD.

Métriques calculées (sur les scores bruts du modèle) :
  - AUROC image   : capacité à séparer images normales / défectueuses (indépendante du seuil)
  - AUROC pixel   : qualité de la localisation (carte d'anomalie vs masque de vérité terrain)
  - Précision, rappel, F1, exactitude, matrice de confusion au seuil choisi par Anomalib
  - Détail par type de défaut (AUROC, taux de détection, AUROC pixel)
Les métriques "officielles" d'Anomalib (engine.test) sont aussi enregistrées pour contrôle.

Usage :
    python -m detection.evaluate
    python -m detection.evaluate --category bottle

Produit, pour chaque catégorie :
    results/<cat>/metrics.json, predictions.csv, predictions.npz
    results/<cat>/figures/{roc,score_hist,confusion,per_defect}.png
et results/results_table.md (tableau récapitulatif pour le rapport).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from .common import (build_datamodule, build_engine, checkpoint_path, load_config, load_json,
                     normalize, quiet, resolve, results_dir, save_json)


# ---------------------------------------------------------------------------
# Inférence sur le jeu de test
# ---------------------------------------------------------------------------
def load_model(ckpt: Path, map_location: str = "cpu", model_class: str = "Patchcore"):
    import anomalib.models as models

    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} introuvable. Lancez d'abord : python -m detection.train")
    model = getattr(models, model_class).load_from_checkpoint(str(ckpt), map_location=map_location)
    model.eval()
    return model


def collect_predictions(model, datamodule, engine) -> dict:
    """Prédit tout le jeu de test et renvoie scores / cartes BRUTS (non normalisés)."""
    pp = model.post_processor
    pp.enable_normalization = False  # on veut les valeurs brutes pour les métriques
    t0 = time.perf_counter()
    batches = engine.predict(model=model, dataloaders=datamodule.test_dataloader())
    elapsed = time.perf_counter() - t0
    pp.enable_normalization = True

    scores, labels, maps, masks, paths = [], [], [], [], []
    for b in batches:
        scores.append(b.pred_score.detach().cpu().float().reshape(-1).numpy())
        labels.append(b.gt_label.detach().cpu().int().reshape(-1).numpy())
        amap = b.anomaly_map.detach().cpu().float()
        maps.append(amap.reshape(amap.shape[0], *amap.shape[-2:]).numpy())
        gm = b.gt_mask.detach().cpu()
        masks.append(gm.reshape(gm.shape[0], *gm.shape[-2:]).numpy().astype(bool))
        paths.extend(str(p) for p in b.image_path)
    out = {
        "scores": np.concatenate(scores),
        "labels": np.concatenate(labels),
        "maps": np.concatenate(maps).astype(np.float32),
        "masks": np.concatenate(masks),
        "paths": paths,
        "defect_types": np.array([Path(p).parent.name for p in paths]),
        "seconds_per_image": elapsed / max(len(paths), 1),
    }
    return out


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------
def _safe_auroc(y, s) -> float | None:
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def compute_metrics(pred: dict, img_thr: float, pix_thr: float) -> dict:
    s, y, types = pred["scores"], pred["labels"], pred["defect_types"]
    maps, masks = pred["maps"], pred["masks"]
    y_hat = (s >= img_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # Pixels : F1 au seuil pixel d'Anomalib
    pm = maps >= pix_thr
    ptp = np.logical_and(pm, masks).sum()
    pfp = np.logical_and(pm, ~masks).sum()
    pfn = np.logical_and(~pm, masks).sum()
    pixel_f1 = 2 * ptp / (2 * ptp + pfp + pfn) if ptp + pfp + pfn else 0.0

    per_type = {}
    good = types == "good"
    for t in sorted(set(types) - {"good"}):
        sel = types == t
        both = sel | good
        per_type[t] = {
            "n_images": int(sel.sum()),
            "image_auroc": _safe_auroc(y[both], s[both]),
            "detection_rate": float(y_hat[sel].mean()),
            "pixel_auroc": _safe_auroc(masks[sel].ravel(), maps[sel].ravel()),
            "mean_score_raw": float(s[sel].mean()),
        }

    return {
        "n_test_images": int(len(y)),
        "n_normal": int((y == 0).sum()),
        "n_defective": int((y == 1).sum()),
        "image_auroc": _safe_auroc(y, s),
        "pixel_auroc": _safe_auroc(masks.ravel(), maps.ravel()),
        "image_threshold_raw": float(img_thr),
        "pixel_threshold_raw": float(pix_thr),
        "image_precision": float(precision),
        "image_recall": float(recall),
        "image_f1": float(f1),
        "image_accuracy": float((tp + tn) / len(y)),
        "false_alarm_rate": float(y_hat[good].mean()) if good.any() else None,
        "pixel_f1": float(pixel_f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_defect_type": per_type,
        "seconds_per_image": float(pred["seconds_per_image"]),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_all(pred: dict, m: dict, category: str, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    s, y, types = pred["scores"], pred["labels"], pred["defect_types"]
    thr = m["image_threshold_raw"]

    # Courbe ROC image
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {m['image_auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("Taux de faux positifs"), ax.set_ylabel("Taux de vrais positifs")
    ax.set_title(f"Courbe ROC (image) - {category}"), ax.legend(loc="lower right")
    fig.tight_layout(), fig.savefig(fig_dir / "roc.png", dpi=150), plt.close(fig)

    # Distribution des scores
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bins = np.linspace(s.min(), s.max(), 30)
    ax.hist(s[y == 0], bins=bins, alpha=0.65, label="normales")
    ax.hist(s[y == 1], bins=bins, alpha=0.65, label="défectueuses")
    ax.axvline(thr, color="red", ls="--", label=f"seuil = {thr:.2f}")
    ax.set_xlabel("Score d'anomalie (brut)"), ax.set_ylabel("Nombre d'images")
    ax.set_title(f"Distribution des scores - {category}"), ax.legend()
    fig.tight_layout(), fig.savefig(fig_dir / "score_hist.png", dpi=150), plt.close(fig)

    # Matrice de confusion
    cm = m["confusion_matrix"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["normal", "anomalie"]), ax.set_yticks([0, 1], ["normal", "anomalie"])
    ax.set_xlabel("Prédiction"), ax.set_ylabel("Vérité terrain")
    ax.set_title(f"Matrice de confusion - {category}")
    fig.tight_layout(), fig.savefig(fig_dir / "confusion.png", dpi=150), plt.close(fig)

    # Par type de défaut
    pt = m["per_defect_type"]
    names = list(pt)
    if names:
        fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names)), 3.8))
        x = np.arange(len(names))
        ax.bar(x - 0.2, [pt[n]["detection_rate"] for n in names], 0.4, label="taux de détection")
        ax.bar(x + 0.2, [pt[n]["pixel_auroc"] or 0 for n in names], 0.4, label="AUROC pixel")
        ax.set_xticks(x, names, rotation=25, ha="right"), ax.set_ylim(0, 1.05)
        ax.set_title(f"Résultats par type de défaut - {category}"), ax.legend(loc="lower right")
        fig.tight_layout(), fig.savefig(fig_dir / "per_defect.png", dpi=150), plt.close(fig)

        # Boîtes à moustaches des scores par type (good inclus)
        order = ["good"] + names
        fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(order)), 3.8))
        ax.boxplot([s[types == t] for t in order])
        ax.set_xticks(range(1, len(order) + 1), order, rotation=25, ha="right")
        ax.axhline(thr, color="red", ls="--", lw=1, label="seuil")
        ax.set_ylabel("Score brut"), ax.set_title(f"Scores par type - {category}"), ax.legend()
        fig.tight_layout(), fig.savefig(fig_dir / "scores_by_type.png", dpi=150), plt.close(fig)


# ---------------------------------------------------------------------------
# Sauvegardes
# ---------------------------------------------------------------------------
def save_predictions(pred: dict, info: dict, out_dir: Path) -> None:
    """CSV par image + cartes normalisées (float16) pour l'analyse des erreurs."""
    img_thr, pix_thr = info["image_threshold_raw"], info["pixel_threshold_raw"]
    s_norm = normalize(pred["scores"], *info["image_score_range_raw"], img_thr)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "defect_type", "label", "score_raw", "score_norm", "predicted"])
        for p, t, l, s, sn in zip(pred["paths"], pred["defect_types"], pred["labels"], pred["scores"], s_norm):
            w.writerow([p, t, int(l), f"{s:.5f}", f"{sn:.4f}", int(s >= img_thr)])
    maps_norm = normalize(pred["maps"], *info["pixel_score_range_raw"], pix_thr).astype(np.float16)
    np.savez_compressed(out_dir / "predictions.npz", maps_norm=maps_norm, masks=pred["masks"],
                        scores_raw=pred["scores"], labels=pred["labels"],
                        paths=np.array(pred["paths"]), defect_types=pred["defect_types"])


def results_table(cfg: dict) -> str:
    rows = ["| Catégorie | AUROC image | AUROC pixel | F1 image | Précision | Rappel | Fausses alarmes | Temps / image |",
            "|---|---|---|---|---|---|---|---|"]
    for cat in cfg["data"]["categories"]:
        p = results_dir(cfg, cat) / "metrics.json"
        if not p.exists():
            continue
        m = load_json(p)

        def pct(v):
            return "-" if v is None else f"{100 * v:.1f} %"

        rows.append(f"| {cat} | {pct(m['image_auroc'])} | {pct(m['pixel_auroc'])} | {pct(m['image_f1'])} | "
                    f"{pct(m['image_precision'])} | {pct(m['image_recall'])} | {pct(m['false_alarm_rate'])} | "
                    f"{1000 * m['seconds_per_image']:.0f} ms |")
    lines = ["## Résultats globaux\n", *rows, ""]
    for cat in cfg["data"]["categories"]:
        p = results_dir(cfg, cat) / "metrics.json"
        if not p.exists():
            continue
        m = load_json(p)
        lines += [f"\n### {cat} : détail par type de défaut\n",
                  "| Type de défaut | Images | AUROC image | Taux de détection | AUROC pixel |",
                  "|---|---|---|---|---|"]
        for t, v in m["per_defect_type"].items():
            f = (lambda x: "-" if x is None else f"{100 * x:.1f} %")
            lines.append(f"| {t} | {v['n_images']} | {f(v['image_auroc'])} | {f(v['detection_rate'])} | "
                         f"{f(v['pixel_auroc'])} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
def evaluate_category(cfg: dict, category: str, ckpt: Path | None = None, out_dir: Path | None = None,
                      model_class: str = "Patchcore", make_figures: bool = True) -> dict:
    from lightning.pytorch import seed_everything

    seed_everything(cfg["training"]["seed"], workers=True)
    ckpt = ckpt or checkpoint_path(cfg, category)
    out_dir = out_dir or results_dir(cfg, category)
    model = load_model(ckpt, model_class=model_class)
    dm = build_datamodule(cfg, category)
    engine = build_engine(cfg, resolve(cfg["paths"]["outputs_dir"]))

    official = engine.test(model=model, datamodule=dm)
    official = {k: float(v) for k, v in (official[0] if official else {}).items()}

    pred = collect_predictions(model, dm, engine)
    pp = model.post_processor
    info = {
        "image_threshold_raw": float(pp.image_threshold),
        "pixel_threshold_raw": float(pp.pixel_threshold),
        "image_score_range_raw": [float(pp.image_min), float(pp.image_max)],
        "pixel_score_range_raw": [float(pp.pixel_min), float(pp.pixel_max)],
    }
    m = compute_metrics(pred, info["image_threshold_raw"], info["pixel_threshold_raw"])
    m["anomalib_official_metrics"] = official
    m["normalization"] = info
    m["category"] = category

    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(m, out_dir / "metrics.json")
    if make_figures:
        save_predictions(pred, info, out_dir)
        plot_all(pred, m, category, out_dir / "figures")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--category", default=None)
    args = ap.parse_args()

    quiet()
    cfg = load_config(args.config)
    cats = [args.category] if args.category else cfg["data"]["categories"]
    for cat in cats:
        print(f"\n=== Évaluation : {cat} ===")
        m = evaluate_category(cfg, cat)
        print(f"[{cat}] AUROC image = {m['image_auroc']:.4f} | AUROC pixel = {m['pixel_auroc']:.4f} | "
              f"F1 = {m['image_f1']:.4f} | FP = {m['confusion_matrix']['fp']} | FN = {m['confusion_matrix']['fn']}")

    md = results_table(cfg)
    (results_dir(cfg) / "results_table.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
