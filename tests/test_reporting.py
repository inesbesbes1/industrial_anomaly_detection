"""Test de génération du rapport PDF (Personne B)."""

from pathlib import Path
import numpy as np
from reporting.pdf_generator import generate_inspection_pdf


def test_pdf_generation():
    fake_payload = {
        "category": "cable",
        "status": "anomalous",
        "anomaly_score": 0.87,
        "threshold": 0.5,
        "regions": [
            {"position": "centre", "area_pct": 3.2, "max_intensity": 0.94}
        ]
    }
    fake_text = (
        "**VERDICT : NON CONFORME**\n\n"
        "**DIAGNOSTIC :** Coupure majeure de l'isolant observée en zone centrale.\n\n"
        "**ACTIONS RECOMMANDÉES :**\n"
        "1. Mettre la pièce au rebut.\n"
        "2. Contrôler l'alignement de la machine de sertissage."
    )
    # Image RGB factice (fond gris avec un carré rouge au milieu)
    fake_img = np.full((300, 300, 3), 180, dtype=np.uint8)
    fake_img[100:200, 100:200] = [255, 0, 0]

    out_pdf = Path("tests/_test_output.pdf")
    pdf_path = generate_inspection_pdf(fake_payload, fake_text, fake_img, output_path=str(out_pdf))

    assert Path(pdf_path).exists()
    assert Path(pdf_path).stat().st_size > 1000  # Le fichier fait plus de 1 Ko
    print(f"Test PDF réussi ! Fichier généré : {pdf_path}")

    # Nettoyage
    out_pdf.unlink()


if __name__ == "__main__":
    test_pdf_generation()