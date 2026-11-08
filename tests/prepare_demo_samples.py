"""Sélectionne automatiquement quelques images de test pour la démo Streamlit."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "MVTecAD"
SAMPLES_DIR = ROOT / "samples"


def collect_samples():
    if not DATA_DIR.exists():
        print(f"Dossier {DATA_DIR} introuvable. Demandez à la Personne A de vous copier 10 images de test.")
        return

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for cat in ["bottle", "cable"]:
        cat_samples = SAMPLES_DIR / cat
        cat_samples.mkdir(exist_ok=True)
        test_dir = DATA_DIR / cat / "test"

        if not test_dir.exists():
            continue

        # Copier 2 images saines
        good_imgs = list((test_dir / "good").glob("*.png"))[:2]
        for idx, img in enumerate(good_imgs):
            shutil.copy(img, cat_samples / f"normal_{idx:02d}.png")

        # Copier 1 image par type de défaut
        for defect_dir in test_dir.iterdir():
            if defect_dir.is_dir() and defect_dir.name != "good":
                defective_imgs = list(defect_dir.glob("*.png"))
                if defective_imgs:
                    shutil.copy(defective_imgs[0], cat_samples / f"{defect_dir.name}_01.png")

    print(f"Échantillons copiés avec succès dans : {SAMPLES_DIR}")


if __name__ == "__main__":
    collect_samples()