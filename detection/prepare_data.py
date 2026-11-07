"""Étape 1 - Extraction et vérification du dataset MVTec AD (Bottle, Cable).

Usage :
    python -m detection.prepare_data
    python -m detection.prepare_data --archives-dir ~/Downloads

Produit :
    data/MVTecAD/<categorie>/{train,test,ground_truth}/...
    results/dataset_stats.json, results/dataset_stats.md
    results/figures/dataset_samples_<categorie>.png
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import load_config, resolve, results_dir, save_json

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT) if folder.is_dir() else []


def extract_archive(archive: Path, root: Path, category: str) -> None:
    """Extrait <categorie>.tar.xz dans root, quel que soit le dossier racine de l'archive."""
    target = root / category
    if (target / "train" / "good").is_dir():
        print(f"[{category}] déjà extrait dans {target} : rien à faire.")
        return
    print(f"[{category}] extraction de {archive.name} (peut prendre 1 à 2 minutes)...")
    tmp = root / f"_tmp_{category}"
    tmp.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        try:
            tar.extractall(tmp, filter="data")  # Python >= 3.12 (et 3.10.12+/3.11.4+)
        except TypeError:
            tar.extractall(tmp)
    # Retrouver le dossier qui contient train/good, peu importe la profondeur
    candidates = [p.parent.parent for p in tmp.rglob("good") if p.parent.name == "train"]
    if not candidates:
        raise RuntimeError(f"Structure MVTec introuvable dans {archive}.")
    src = candidates[0]
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(src), str(target))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{category}] extrait dans {target}")


def category_stats(root: Path, category: str) -> dict:
    base = root / category
    stats = {"train_good": len(_list_images(base / "train" / "good")), "test": {}, "masks": {}}
    for d in sorted(p for p in (base / "test").iterdir() if p.is_dir()):
        stats["test"][d.name] = len(_list_images(d))
        if d.name != "good":
            stats["masks"][d.name] = len(_list_images(base / "ground_truth" / d.name))
    sample = _list_images(base / "train" / "good")[0]
    h, w = cv2.imread(str(sample)).shape[:2]
    stats["resolution"] = f"{w}x{h}"
    stats["test_good"] = stats["test"].get("good", 0)
    stats["test_defect"] = sum(v for k, v in stats["test"].items() if k != "good")
    stats["defect_types"] = [k for k in stats["test"] if k != "good"]
    # Contrôle de cohérence images / masques
    for k, n in stats["masks"].items():
        if n != stats["test"][k]:
            print(f"  ATTENTION [{category}/{k}] : {stats['test'][k]} images mais {n} masques")
    return stats


def sample_figure(root: Path, category: str, out: Path) -> None:
    """Une image normale + un exemple (avec masque) par type de défaut."""
    base = root / category
    types = sorted(p.name for p in (base / "test").iterdir() if p.is_dir() and p.name != "good")
    n = 1 + len(types)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4))
    if n == 1:
        axes = axes.reshape(2, 1)
    good = _list_images(base / "train" / "good")[0]
    axes[0, 0].imshow(cv2.cvtColor(cv2.imread(str(good)), cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("good (normal)", fontsize=9)
    axes[1, 0].axis("off")
    for i, t in enumerate(types, start=1):
        img = _list_images(base / "test" / t)[0]
        mask = base / "ground_truth" / t / f"{img.stem}_mask.png"
        axes[0, i].imshow(cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(t, fontsize=9)
        if mask.exists():
            axes[1, i].imshow(cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE), cmap="gray")
            axes[1, i].set_title("masque", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"MVTec AD - {category}", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def stats_markdown(all_stats: dict) -> str:
    lines = [
        "| Catégorie | Résolution | Train (normales) | Test normales | Test défectueuses | Types de défauts |",
        "|---|---|---|---|---|---|",
    ]
    for cat, s in all_stats.items():
        types = ", ".join(f"{k} ({s['test'][k]})" for k in s["defect_types"])
        lines.append(
            f"| {cat} | {s['resolution']} | {s['train_good']} | {s['test_good']} | {s['test_defect']} | {types} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--archives-dir", default=None, help="dossier contenant <categorie>.tar.xz")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = resolve(cfg["data"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    archives = Path(args.archives_dir).expanduser() if args.archives_dir else resolve(cfg["data"]["archives_dir"])

    all_stats = {}
    for cat in cfg["data"]["categories"]:
        if not (root / cat / "train" / "good").is_dir():
            arch = next((archives / f"{cat}{ext}" for ext in (".tar.xz", ".tar.gz", ".tar")
                         if (archives / f"{cat}{ext}").exists()), None)
            if arch is None:
                raise FileNotFoundError(f"Archive {cat}.tar.xz introuvable dans {archives}")
            extract_archive(arch, root, cat)
        else:
            print(f"[{cat}] déjà présent dans {root / cat}")
        all_stats[cat] = category_stats(root, cat)
        sample_figure(root, cat, results_dir(cfg) / "figures" / f"dataset_samples_{cat}.png")

    out = results_dir(cfg)
    save_json(all_stats, out / "dataset_stats.json")
    md = stats_markdown(all_stats)
    (out / "dataset_stats.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Statistiques : {out / 'dataset_stats.md'}")


if __name__ == "__main__":
    main()
