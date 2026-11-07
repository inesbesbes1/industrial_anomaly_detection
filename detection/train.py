"""Étape 2 - Entraînement de PatchCore (un modèle par catégorie).

PatchCore n'apprend pas par descente de gradient : il passe une seule fois sur les
images normales, extrait les features de patchs et en garde un sous-ensemble
représentatif (coreset) dans une banque mémoire. Le seuil de décision est ensuite
fixé automatiquement sur le jeu de validation (F1 maximal).

Usage :
    python -m detection.train                      # toutes les catégories
    python -m detection.train --category cable
    python -m detection.train --coreset 0.01       # surcharger un hyperparamètre

Produit :
    models/patchcore_<categorie>.ckpt
    results/<categorie>/train_info.json
"""

from __future__ import annotations

import argparse
import copy
import shutil
import time

from .common import (build_datamodule, build_engine, build_patchcore, checkpoint_path,
                     load_config, quiet, resolve, results_dir, save_json)


def _bank_shape(model):
    bank = getattr(model.model, "memory_bank", None)
    return list(bank.shape) if hasattr(bank, "shape") else None


def train_category(cfg: dict, category: str, model_cfg: dict | None = None,
                   ckpt_out=None, workdir=None, model=None) -> dict:
    """Entraîne PatchCore sur une catégorie et copie le checkpoint final."""
    from lightning.pytorch import seed_everything

    seed_everything(cfg["training"]["seed"], workers=True)
    model_cfg = model_cfg or cfg["model"]
    dm = build_datamodule(cfg, category)
    model = model if model is not None else build_patchcore(model_cfg)
    workdir = workdir or resolve(cfg["paths"]["outputs_dir"])
    engine = build_engine(cfg, workdir)

    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=dm)
    train_time = time.perf_counter() - t0

    src = engine.trainer.checkpoint_callback.best_model_path
    if not src:
        raise RuntimeError("Aucun checkpoint produit par Anomalib.")
    dst = ckpt_out or checkpoint_path(cfg, category)
    shutil.copy(src, dst)

    pp = model.post_processor
    info = {
        "category": category,
        "checkpoint": str(dst),
        "train_time_s": round(train_time, 1),
        "memory_bank_shape": _bank_shape(model),
        "image_threshold_raw": float(pp.image_threshold),
        "pixel_threshold_raw": float(pp.pixel_threshold),
        "image_score_range_raw": [float(pp.image_min), float(pp.image_max)],
        "pixel_score_range_raw": [float(pp.pixel_min), float(pp.pixel_max)],
        "model_class": type(model).__name__,
        "model_config": copy.deepcopy(model_cfg),
        "val_split_mode": cfg["data"].get("val_split_mode"),
    }
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--category", default=None, help="une seule catégorie (défaut : toutes)")
    ap.add_argument("--coreset", type=float, default=None, help="coreset_sampling_ratio")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--layers", nargs="+", default=None)
    args = ap.parse_args()

    quiet()
    cfg = load_config(args.config)
    if args.coreset is not None:
        cfg["model"]["coreset_sampling_ratio"] = args.coreset
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.layers:
        cfg["model"]["layers"] = args.layers

    cats = [args.category] if args.category else cfg["data"]["categories"]
    for cat in cats:
        print(f"\n=== Entraînement PatchCore : {cat} ===")
        info = train_category(cfg, cat)
        save_json(info, results_dir(cfg, cat) / "train_info.json")
        print(f"[{cat}] terminé en {info['train_time_s']} s - banque mémoire {info['memory_bank_shape']}")
        print(f"[{cat}] checkpoint : {info['checkpoint']}")


if __name__ == "__main__":
    main()
