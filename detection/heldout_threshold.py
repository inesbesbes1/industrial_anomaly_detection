"""Évaluation à seuil "hors échantillon" (held-out) - sans réentraînement.

Problème : avec val_split_mode = same_as_test, le seuil d'Anomalib est choisi sur les
mêmes images que celles qui servent à calculer F1 / FP / FN. Ces métriques sont donc
légèrement optimistes (l'AUROC, indépendant du seuil, ne l'est pas).

Ici, on coupe le jeu de test en deux moitiés (stratifiées par type de défaut), on choisit
le seuil (F1 maximal) sur une moitié, et on mesure FP / FN / F1 sur l'AUTRE moitié.
On répète sur de nombreux tirages aléatoires pour obtenir une moyenne et un écart-type.

À lancer après detection.evaluate (utilise results/<cat>/predictions.npz).

Usage :
    python -m detection.heldout_threshold
    python -m detection.heldout_threshold --category cable --repeats 500

Produit, pour chaque catégorie :
    results/<cat>/heldout_threshold.md et heldout_threshold.json
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from .common import load_config, load_json, results_dir, save_json


def f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Seuil qui maximise le F1 (image >= seuil -> anomalie).

    On place le seuil au milieu de l'intervalle entre le dernier score "positif" retenu
    et le score immédiatement inférieur, pour ne pas le coller sur une image précise.
    """
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    f1 = 2 * tp / (tp + fp + y.sum())  # 2TP / (2TP + FP + FN)
    valid = np.r_[s[1:] != s[:-1], True]  # un seuil n'a de sens qu'entre deux scores différents
    f1 = np.where(valid, f1, -1.0)
    i = int(np.argmax(f1))
    return float((s[i] + s[i + 1]) / 2) if i + 1 < len(s) else float(s[i])


def score_metrics(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred = scores >= thr
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
        "miss_rate": fn / (tp + fn) if tp + fn else 0.0,
        "fp": fp,
        "fn": fn,
    }


def run_category(cfg: dict, category: str, repeats: int, seed: int) -> str:
    d = results_dir(cfg, category)
    p = d / "predictions.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} introuvable. Lancez d'abord : python -m detection.evaluate")
    z = np.load(p, allow_pickle=False)
    m = load_json(d / "metrics.json")
    scores = z["scores_raw"].astype(np.float64)
    labels = z["labels"].astype(int)
    types = z["defect_types"]
    n_norm, n_def = int((labels == 0).sum()), int((labels == 1).sum())

    # Référence : seuil d'Anomalib, choisi sur tout le jeu de test (protocole du rapport)
    ref = score_metrics(scores, labels, m["image_threshold_raw"])

    # Tirages : seuil sur une moitié, métriques sur l'autre
    splitter = StratifiedShuffleSplit(n_splits=repeats, test_size=0.5, random_state=seed)
    rows = []
    per_type = {t: [] for t in sorted(set(types.tolist()) - {"good"})}
    for cal, test in splitter.split(scores, types):
        thr = f1_threshold(scores[cal], labels[cal])
        rows.append({**score_metrics(scores[test], labels[test], thr), "threshold": thr})
        pred = scores[test] >= thr
        for t in per_type:
            sel = types[test] == t
            if sel.any():
                per_type[t].append(float(pred[sel].mean()))

    def agg(key: str) -> tuple[float, float]:
        v = np.array([r[key] for r in rows])
        return float(v.mean()), float(v.std())

    f1_m, f1_s = agg("f1")
    far_m, far_s = agg("false_alarm_rate")
    miss_m, miss_s = agg("miss_rate")
    thr_m, thr_s = agg("threshold")

    def pct(mean: float, std: float | None = None) -> str:
        return f"{100 * mean:.1f} %" if std is None else f"{100 * mean:.1f} ± {100 * std:.1f} %"

    lines = [
        f"# Seuil hors échantillon - {category}\n",
        f"{repeats} tirages aléatoires 50/50 (stratifiés par type de défaut, graine {seed}). "
        f"Test complet : {n_norm} normales, {n_def} défectueuses.\n",
        "| Protocole | F1 image | Fausses alarmes | Défauts manqués |",
        "|---|---|---|---|",
        f"| Seuil Anomalib ({m['image_threshold_raw']:.2f}), choisi sur tout le test | {pct(ref['f1'])} | "
        f"{pct(ref['false_alarm_rate'])} ({ref['fp']}/{n_norm}) | {pct(ref['miss_rate'])} ({ref['fn']}/{n_def}) |",
        f"| Seuil choisi sur une moitié, mesuré sur l'autre (moyenne ± écart-type) | {pct(f1_m, f1_s)} | "
        f"{pct(far_m, far_s)} (≈ {far_m * n_norm:.1f}/{n_norm}) | "
        f"{pct(miss_m, miss_s)} (≈ {miss_m * n_def:.1f}/{n_def}) |",
        "",
        f"Seuil moyen retrouvé sur les moitiés : {thr_m:.2f} ± {thr_s:.2f} "
        f"(seuil Anomalib : {m['image_threshold_raw']:.2f}).\n",
    ]
    if per_type:
        lines += ["## Taux de détection par type de défaut (moitié non utilisée pour le seuil)\n",
                  "| Type | Taux de détection (moyenne ± écart-type) |", "|---|---|"]
        for t, v in per_type.items():
            lines.append(f"| {t} | {pct(float(np.mean(v)), float(np.std(v)))} |")
        lines.append("")
    lines += ["Lecture : l'AUROC ne dépend pas du seuil et reste inchangé. Un écart important entre les "
              "deux lignes du tableau mesure l'optimisme dû au choix du seuil sur les images de test.",
              "Les écarts-types reflètent la petite taille des moitiés : ils sont indicatifs."]
    md = "\n".join(lines) + "\n"

    (d / "heldout_threshold.md").write_text(md, encoding="utf-8")
    save_json({"reference": ref, "heldout_mean": {k: agg(k)[0] for k in rows[0]},
               "heldout_std": {k: agg(k)[1] for k in rows[0]},
               "per_type_detection_mean": {t: float(np.mean(v)) for t, v in per_type.items()},
               "repeats": repeats, "seed": seed}, d / "heldout_threshold.json")
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for cat in [args.category] if args.category else cfg["data"]["categories"]:
        print(f"\n=== Seuil hors échantillon : {cat} ===")
        print(run_category(cfg, cat, args.repeats, args.seed))


if __name__ == "__main__":
    main()