"""Bonus - Étude d'ablation : hyperparamètres de PatchCore + comparaison avec PaDiM.

Chaque variante est entraînée puis évaluée sur la même catégorie. Utile pour la
section "Discussion" du rapport (compromis précision / mémoire / vitesse).

Usage :
    python -m detection.ablation --category bottle
    python -m detection.ablation --category cable --only patchcore_ref padim_r18

Produit :
    results/<cat>/ablation/ablation.csv et ablation.md
"""

from __future__ import annotations

import argparse
import copy
import csv
import tempfile
from pathlib import Path

from .common import build_patchcore, load_config, quiet, results_dir, save_json
from .evaluate import evaluate_category
from .train import train_category


def variants(cfg: dict) -> dict:
    base = cfg["model"]

    def pc(**kw):
        m = copy.deepcopy(base)
        m.update(kw)
        return ("Patchcore", m)

    return {
        "patchcore_ref": pc(),                                     # config de référence
        "patchcore_coreset_1pct": pc(coreset_sampling_ratio=0.01),  # banque mémoire 10x plus petite
        "patchcore_coreset_25pct": pc(coreset_sampling_ratio=0.25),
        "patchcore_resnet18": pc(backbone="resnet18"),             # backbone plus léger
        "patchcore_layer3_only": pc(layers=["layer3"]),            # sans les features plus fines
        "padim_r18": ("Padim", {"backbone": "resnet18", "layers": ["layer1", "layer2", "layer3"],
                                "pre_trained": base.get("pre_trained", True)}),
    }


def build(model_class: str, mcfg: dict):
    if model_class == "Patchcore":
        return build_patchcore(mcfg)
    from anomalib.models import Padim

    pre = Padim.configure_pre_processor(image_size=tuple(cfg_image_size(mcfg)))
    return Padim(backbone=mcfg["backbone"], layers=mcfg["layers"], pre_trained=mcfg["pre_trained"],
                 pre_processor=pre)


def cfg_image_size(mcfg: dict):
    return mcfg.get("image_size", (256, 256))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--category", required=True)
    ap.add_argument("--only", nargs="+", default=None, help="noms des variantes à lancer")
    args = ap.parse_args()

    quiet()
    cfg = load_config(args.config)
    out_dir = results_dir(cfg, args.category) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (cls, mcfg) in variants(cfg).items():
        if args.only and name not in args.only:
            continue
        print(f"\n=== Ablation {args.category} : {name} ===")
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "model.ckpt"
            info = train_category(cfg, args.category, model_cfg=mcfg, ckpt_out=ckpt,
                                  workdir=Path(tmp) / "work", model=build(cls, mcfg))
            m = evaluate_category(cfg, args.category, ckpt=ckpt, out_dir=out_dir / name,
                                  model_class=cls, make_figures=False)
        bank = info["memory_bank_shape"]
        row = {
            "variante": name,
            "modele": cls,
            "backbone": mcfg["backbone"],
            "couches": "+".join(mcfg["layers"]),
            "coreset": mcfg.get("coreset_sampling_ratio", "-"),
            "auroc_image": round(m["image_auroc"], 4),
            "auroc_pixel": round(m["pixel_auroc"], 4),
            "f1_image": round(m["image_f1"], 4),
            "temps_entrainement_s": info["train_time_s"],
            "ms_par_image": round(1000 * m["seconds_per_image"], 1),
            "taille_banque": bank[0] if bank else "-",
        }
        rows.append(row)
        save_json(row, out_dir / name / "summary.json")
        print(row)

    if rows:
        with open(out_dir / "ablation.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(), w.writerows(rows)
        md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        md += ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows]
        (out_dir / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
