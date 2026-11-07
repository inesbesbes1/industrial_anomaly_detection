"""Étape 4 - Analyse des erreurs (faux positifs / faux négatifs) et figures qualitatives.

À lancer après detection.evaluate (utilise results/<cat>/predictions.npz).

Usage :
    python -m detection.error_analysis
    python -m detection.error_analysis --category cable

Produit, pour chaque catégorie :
    results/<cat>/figures/errors_fp_fn.png        faux positifs et faux négatifs
    results/<cat>/figures/best_detections.png     meilleures détections (pour les slides)
    results/<cat>/figures/examples_<type>.png     exemples par type de défaut
    results/<cat>/error_analysis.md               résumé chiffré + liste des erreurs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import load_config, load_json, results_dir
from .visualize import draw_contours, load_rgb, overlay_heatmap, resize_like_model

GT_COLOR = (0, 255, 0)      # vert : vérité terrain
PRED_COLOR = (255, 255, 255)  # blanc : zone prédite (carte >= 0.5)


def _load(cfg: dict, category: str):
    d = results_dir(cfg, category)
    p = d / "predictions.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} introuvable. Lancez d'abord : python -m detection.evaluate")
    z = np.load(p, allow_pickle=False)
    m = load_json(d / "metrics.json")
    return z, m


def _rows_figure(z, idx: list[int], cfg: dict, title: str, out: Path, thr_raw: float) -> None:
    """Une ligne par image : image | vérité terrain | heatmap + zone prédite."""
    if not idx:
        return
    size = tuple(cfg["model"].get("image_size", (256, 256)))
    crop = cfg["model"].get("center_crop")
    n = len(idx)
    fig, axes = plt.subplots(n, 3, figsize=(8.4, 2.8 * n))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idx):
        img = resize_like_model(load_rgb(z["paths"][i]), size, crop)
        amap = z["maps_norm"][i].astype(np.float32)
        mask = z["masks"][i]
        verdict = "anomalie" if z["scores_raw"][i] >= thr_raw else "normal"
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{z['defect_types'][i]} / {Path(str(z['paths'][i])).name}", fontsize=8)
        axes[r, 1].imshow(draw_contours(img, mask, GT_COLOR) if mask.any() else img)
        axes[r, 1].set_title("vérité terrain" + ("" if mask.any() else " (aucun défaut)"), fontsize=8)
        axes[r, 2].imshow(draw_contours(overlay_heatmap(img, amap), amap >= 0.5, PRED_COLOR))
        axes[r, 2].set_title(f"heatmap - score {z['scores_raw'][i]:.2f} -> {verdict}", fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def analyse_category(cfg: dict, category: str, max_rows: int = 8) -> str:
    z, m = _load(cfg, category)
    thr = m["image_threshold_raw"]
    scores, labels, types = z["scores_raw"], z["labels"], z["defect_types"]
    pred = scores >= thr
    fig_dir = results_dir(cfg, category) / "figures"

    fp = [int(i) for i in np.where((labels == 0) & pred)[0]]
    fn = [int(i) for i in np.where((labels == 1) & ~pred)[0]]
    fp.sort(key=lambda i: -scores[i])   # faux positifs les plus "sûrs" d'abord
    fn.sort(key=lambda i: scores[i])    # faux négatifs les plus ratés d'abord

    errors = (fp[: max_rows // 2] + fn[: max_rows // 2]) or []
    _rows_figure(z, errors, cfg, f"{category} - erreurs ({len(fp)} FP, {len(fn)} FN)",
                 fig_dir / "errors_fp_fn.png", thr)

    tp = [int(i) for i in np.where((labels == 1) & pred)[0]]
    tp.sort(key=lambda i: -scores[i])
    _rows_figure(z, tp[:4], cfg, f"{category} - meilleures détections", fig_dir / "best_detections.png", thr)

    for t in sorted(set(types.tolist()) - {"good"}):
        idx = [int(i) for i in np.where(types == t)[0]]
        idx.sort(key=lambda i: scores[i])  # les plus difficiles d'abord
        _rows_figure(z, idx[:4], cfg, f"{category} - {t} (cas les plus difficiles en haut)",
                     fig_dir / f"examples_{t}.png", thr)

    # ---- Résumé texte
    pt = m["per_defect_type"]
    lines = [f"# Analyse des erreurs - {category}\n",
             f"- Images de test : {m['n_test_images']} ({m['n_normal']} normales, {m['n_defective']} défectueuses)",
             f"- Seuil (score brut) : {thr:.4f}",
             f"- Faux positifs : **{len(fp)}** (images normales signalées comme défectueuses)",
             f"- Faux négatifs : **{len(fn)}** (défauts non détectés)\n"]
    if pt:
        worst = min(pt, key=lambda k: pt[k]["detection_rate"])
        worst_loc = min(pt, key=lambda k: pt[k]["pixel_auroc"] or 1)
        lines += ["## Par type de défaut\n",
                  "| Type | Images | Taux de détection | AUROC pixel | Score moyen (brut) |", "|---|---|---|---|---|"]
        for t, v in pt.items():
            pa = "-" if v["pixel_auroc"] is None else f"{100 * v['pixel_auroc']:.1f} %"
            lines.append(f"| {t} | {v['n_images']} | {100 * v['detection_rate']:.1f} % | {pa} | "
                         f"{v['mean_score_raw']:.3f} |")
        lines += ["", f"- Type le moins bien détecté : **{worst}** "
                      f"({100 * pt[worst]['detection_rate']:.1f} % détectés)",
                  f"- Type le moins bien localisé : **{worst_loc}**\n"]
    if fp or fn:
        lines += ["## Liste des erreurs\n", "| Type d'erreur | Fichier | Catégorie de défaut | Score brut |",
                  "|---|---|---|---|"]
        for kind, ids in (("FP", fp), ("FN", fn)):
            for i in ids:
                lines.append(f"| {kind} | {Path(str(z['paths'][i])).name} | {types[i]} | {scores[i]:.4f} |")
    lines += ["", "## Pistes d'interprétation (à rédiger dans le rapport)\n",
              "- Faux négatifs : défauts petits ou peu contrastés, proches de l'apparence normale ?",
              "- Faux positifs : variations normales (reflets, position du câble, bruit) absentes du train ?",
              "- Comparer les figures `examples_<type>.png` : la heatmap pointe-t-elle au bon endroit ?",
              "- Effet du seuil : voir `score_hist.png` (chevauchement des deux distributions)."]
    md = "\n".join(lines) + "\n"
    (results_dir(cfg, category) / "error_analysis.md").write_text(md, encoding="utf-8")
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--category", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for cat in [args.category] if args.category else cfg["data"]["categories"]:
        print(f"\n=== Analyse des erreurs : {cat} ===")
        print(analyse_category(cfg, cat))


if __name__ == "__main__":
    main()
