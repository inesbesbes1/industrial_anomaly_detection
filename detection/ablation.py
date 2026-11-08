"""Bonus - Étude d'ablation : hyperparamètres de PatchCore + comparaison avec PaDiM.

Chaque variante est entraînée puis évaluée sur la même catégorie. Utile pour la
section "Discussion" du rapport (compromis précision / mémoire / vitesse).

Comparaisons "propres" (un seul facteur change à la fois) :
    - algorithme, backbone de réf.    : padim_wrn50 vs patchcore_ref
    - algorithme, backbone resnet18   : padim_r18_layers23 vs patchcore_resnet18
    - backbone (même algo, mêmes couches) : patchcore_resnet18 vs patchcore_ref
    - coreset                          : patchcore_coreset_1pct / patchcore_ref / patchcore_coreset_25pct
    - couches                          : patchcore_layer3_only / patchcore_ref

Usage :
    python -m detection.ablation --category cable
    python -m detection.ablation --category cable --only padim_wrn50 padim_r18_layers23 patchcore_coreset_1pct --resume
    python -m detection.ablation --category cable --forward-timing

Options :
    --resume          saute les variantes qui ont déjà un summary.json (utile : ~1 h par PatchCore sur CPU)
    --forward-timing  ajoute la colonne ms_forward_cpu : temps du seul passage avant du modèle, sans
                      dataloader ni callbacks (ms_par_image, lui, couvre tout engine.predict)

Produit :
    results/<cat>/ablation/<variante>/summary.json   (écrit dès qu'une variante est terminée)
    results/<cat>/ablation/ablation.csv et ablation.md (reconstruits à partir de tous les summary.json)
"""

from __future__ import annotations

import argparse
import copy
import csv
import tempfile
import time
from pathlib import Path

from .common import build_datamodule, build_patchcore, load_config, load_json, quiet, results_dir, save_json
from .evaluate import evaluate_category, load_model
from .train import train_category

LAYERS_123 = ["layer1", "layer2", "layer3"]


def variants(cfg: dict) -> dict:
    base = cfg["model"]

    def pc(**kw):
        m = copy.deepcopy(base)
        m.update(kw)
        return ("Patchcore", m)

    def pd(**kw):
        m = {"backbone": "resnet18", "layers": list(LAYERS_123), "pre_trained": base.get("pre_trained", True)}
        m.update(kw)
        return ("Padim", m)

    return {
        "patchcore_ref": pc(),                                       # config de référence
        "patchcore_coreset_1pct": pc(coreset_sampling_ratio=0.01),    # banque mémoire 10x plus petite
        "patchcore_coreset_25pct": pc(coreset_sampling_ratio=0.25),
        "patchcore_resnet18": pc(backbone="resnet18"),               # backbone plus léger
        "patchcore_layer3_only": pc(layers=["layer3"]),              # sans les features plus fines
        "padim_r18": pd(),
        # --- Ajouts : isoler l'effet de l'algorithme (même backbone, mêmes couches des deux côtés).
        # On évite layer1 : PaDiM et PatchCore travaillent à sa résolution (64x64 = 4096 positions),
        # ce qui fait exploser la RAM (covariance PaDiM) et le temps de coreset (PatchCore).
        # n_features=300 (défaut Anomalib : 550) : la covariance PaDiM croît en n_features^2, et 550 dépasse la RAM
        "padim_wrn50": pd(backbone=base["backbone"], layers=list(base["layers"]), n_features=300),  # vs patchcore_ref
        "padim_r18_layers23": pd(layers=list(base["layers"])),                      # vs patchcore_resnet18
    }


def build(model_class: str, mcfg: dict):
    if model_class == "Patchcore":
        return build_patchcore(mcfg)
    from anomalib.models import Padim

    pre = Padim.configure_pre_processor(image_size=tuple(cfg_image_size(mcfg)))
    return Padim(backbone=mcfg["backbone"], layers=mcfg["layers"], pre_trained=mcfg["pre_trained"],
                 n_features=mcfg.get("n_features"), pre_processor=pre)


def cfg_image_size(mcfg: dict):
    return mcfg.get("image_size", (256, 256))


def forward_ms(cfg: dict, category: str, ckpt: Path, model_class: str, n_batches: int = 4, warmup: int = 1) -> float:
    """Temps moyen (ms / image, CPU) du seul passage avant du modèle, hors dataloader et callbacks.

    Seul le temps compte ici : les valeurs de pixels n'ont pas besoin d'être normalisées comme à l'évaluation.
    """
    import torch
    import torch.nn.functional as F

    model = load_model(ckpt, model_class=model_class)  # CPU, mode eval
    dm = build_datamodule(cfg, category)
    dm.setup()
    size = tuple(cfg["model"].get("image_size", (256, 256)))
    per_image = []
    with torch.no_grad():
        for i, batch in enumerate(dm.test_dataloader()):
            x = F.interpolate(batch.image.float(), size=size, mode="bilinear", antialias=True, align_corners=False)
            t0 = time.perf_counter()
            model(x)
            dt = time.perf_counter() - t0
            if i >= warmup:
                per_image.append(dt / x.shape[0])
            if len(per_image) >= n_batches:
                break
    return round(1000 * sum(per_image) / len(per_image), 1)


def write_tables(rows: list[dict], out_dir: Path) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="-")
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    md += ["| " + " | ".join(str(r.get(k, "-")) for k in fields) + " |" for r in rows]
    (out_dir / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--category", required=True)
    ap.add_argument("--only", nargs="+", default=None, help="noms des variantes à lancer")
    ap.add_argument("--resume", action="store_true", help="saute les variantes déjà terminées")
    ap.add_argument("--forward-timing", action="store_true", help="mesure aussi le passage avant seul (CPU)")
    args = ap.parse_args()

    quiet()
    cfg = load_config(args.config)
    out_dir = results_dir(cfg, args.category) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_variants = variants(cfg)
    unknown = set(args.only or []) - set(all_variants)
    if unknown:
        raise SystemExit(f"Variantes inconnues : {sorted(unknown)}. Disponibles : {list(all_variants)}")

    for name, (cls, mcfg) in all_variants.items():
        if args.only and name not in args.only:
            continue
        summary = out_dir / name / "summary.json"
        if args.resume and summary.exists():
            print(f"[{name}] déjà terminé, ignoré (--resume)")
            continue

        print(f"\n=== Ablation {args.category} : {name} ===")
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "model.ckpt"
            info = train_category(cfg, args.category, model_cfg=mcfg, ckpt_out=ckpt,
                                  workdir=Path(tmp) / "work", model=build(cls, mcfg))
            m = evaluate_category(cfg, args.category, ckpt=ckpt, out_dir=out_dir / name,
                                  model_class=cls, make_figures=False)
            fwd = None
            if args.forward_timing:
                try:
                    fwd = forward_ms(cfg, args.category, ckpt, cls)
                except Exception as e:  # la mesure est optionnelle : on ne perd pas l'ablation
                    print(f"[{name}] mesure du passage avant impossible : {e}")

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
        if "n_features" in mcfg:
            row["n_features"] = mcfg["n_features"]
        if fwd is not None:
            row["ms_forward_cpu"] = fwd
        save_json(row, summary)
        print(row)

        # Tableaux reconstruits à chaque fin de variante : rien n'est perdu si le script s'arrête ensuite
        rows = [load_json(out_dir / n / "summary.json") for n in all_variants
                if (out_dir / n / "summary.json").exists()]
        write_tables(rows, out_dir)

    rows = [load_json(out_dir / n / "summary.json") for n in all_variants
            if (out_dir / n / "summary.json").exists()]
    if rows:
        write_tables(rows, out_dir)
        print("\n" + (out_dir / "ablation.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()