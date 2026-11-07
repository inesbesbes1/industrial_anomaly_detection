"""Fonctions partagées : configuration, chemins, construction du datamodule et du modèle."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def quiet() -> None:
    """Réduit le bruit des logs Lightning / Anomalib dans la console."""
    warnings.filterwarnings("ignore")
    for name in ("lightning", "lightning.pytorch", "anomalib", "pytorch_lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def resolve(path: str | Path) -> Path:
    """Chemin relatif au dossier du projet (ou absolu tel quel)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def results_dir(cfg: dict, category: str | None = None) -> Path:
    d = resolve(cfg["paths"]["results_dir"])
    if category:
        d = d / category
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(cfg: dict, category: str, model_name: str = "patchcore") -> Path:
    d = resolve(cfg["paths"]["models_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{model_name}_{category}.ckpt"


def build_datamodule(cfg: dict, category: str):
    """Datamodule MVTec AD d'Anomalib pour une catégorie."""
    from anomalib.data import MVTecAD

    d = cfg["data"]
    root = resolve(d["root"])
    if not (root / category / "train" / "good").is_dir():
        raise FileNotFoundError(
            f"Catégorie '{category}' introuvable dans {root}. "
            "Lancez d'abord : python -m detection.prepare_data"
        )
    return MVTecAD(
        root=root,
        category=category,
        train_batch_size=d["train_batch_size"],
        eval_batch_size=d["eval_batch_size"],
        num_workers=d["num_workers"],
        val_split_mode=d.get("val_split_mode", "same_as_test"),
        val_split_ratio=d.get("val_split_ratio", 0.5),
        seed=cfg["training"]["seed"],
    )


def build_patchcore(model_cfg: dict):
    """Instancie PatchCore avec le pré-traitement défini dans la config."""
    from anomalib.models import Patchcore

    image_size = tuple(model_cfg.get("image_size", (256, 256)))
    crop = model_cfg.get("center_crop")
    pre_processor = Patchcore.configure_pre_processor(
        image_size=image_size, center_crop_size=tuple(crop) if crop else None
    )
    return Patchcore(
        backbone=model_cfg["backbone"],
        layers=list(model_cfg["layers"]),
        pre_trained=model_cfg.get("pre_trained", True),
        coreset_sampling_ratio=model_cfg["coreset_sampling_ratio"],
        num_neighbors=model_cfg["num_neighbors"],
        pre_processor=pre_processor,
    )


def build_engine(cfg: dict, root_dir: Path):
    from anomalib.engine import Engine

    return Engine(
        default_root_dir=str(root_dir),
        accelerator=cfg["training"].get("accelerator", "auto"),
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )


def normalize(x, vmin: float, vmax: float, threshold: float):
    """Même normalisation que le PostProcessor d'Anomalib : le seuil devient 0.5, sortie dans [0, 1]."""
    import numpy as np

    rng = (vmax - vmin) or 1.0
    return np.clip((np.asarray(x, dtype=np.float32) - threshold) / rng + 0.5, 0.0, 1.0)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
