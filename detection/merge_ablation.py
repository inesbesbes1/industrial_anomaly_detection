"""Fusionne tous les results/<cat>/ablation/<variante>/summary.json en un seul tableau.

Utile car chaque lancement de detection.ablation réécrit ablation.md avec ses seules variantes.

Usage :
    python -m detection.merge_ablation
    python -m detection.merge_ablation --category cable

Produit : results/<cat>/ablation/ablation_all.md et ablation_all.csv
"""

from __future__ import annotations

import argparse
import csv

from .common import load_config, load_json, results_dir

ORDER = ["patchcore_ref", "patchcore_coreset_1pct", "patchcore_coreset_25pct",
         "patchcore_resnet18", "patchcore_layer3_only", "padim_r18"]


def merge(cfg: dict, category: str) -> str | None:
    abl = results_dir(cfg, category) / "ablation"
    rows = []
    for p in abl.glob("*/summary.json"):
        rows.append(load_json(p))
    if not rows:
        return None
    rows.sort(key=lambda r: ORDER.index(r["variante"]) if r["variante"] in ORDER else len(ORDER))
    cols = list(dict.fromkeys(k for r in rows for k in r))  # union des colonnes, ordre conservé
    with open(abl / "ablation_all.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |" for r in rows]
    text = "\n".join(md) + "\n"
    (abl / "ablation_all.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--category", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for cat in [args.category] if args.category else cfg["data"]["categories"]:
        print(f"\n=== Ablation complète : {cat} ===")
        print(merge(cfg, cat) or "Aucun summary.json trouvé.")


if __name__ == "__main__":
    main()