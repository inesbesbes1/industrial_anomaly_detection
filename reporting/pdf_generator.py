"""Générateur de rapport d'inspection qualité au format PDF (Personne B)."""

from datetime import datetime
from pathlib import Path
from fpdf import FPDF
import cv2
import numpy as np


class QualityInspectionPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, "FICHE DE CONTRÔLE QUALITÉ INDUSTRIEL", border=False, ln=True, align="C")
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f"Date d'inspection : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Système PatchCore + LLM", ln=True, align="C")
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Système de Contrôle Automatisé - Page {self.page_no()}", align="C")


def generate_inspection_pdf(
    payload: dict,
    llm_analysis: str,
    overlay_img_rgb: np.ndarray,
    output_path: str = "rapport_inspection.pdf"
) -> str:
    """Génère la fiche PDF et la sauvegarde sur le disque."""
    pdf = QualityInspectionPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Tableau synthétique
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, "1. Métriques de Vision", ln=True)

    pdf.set_font("Helvetica", "", 9)
    status_str = "NON CONFORME (DÉFAUT DÉTECTÉ)" if payload["status"] == "anomalous" else "CONFORME"

    pdf.cell(50, 6, "Catégorie :", border=1)
    pdf.cell(0, 6, f" {payload['category'].upper()}", border=1, ln=True)
    pdf.cell(50, 6, "Statut détecté :", border=1)
    pdf.cell(0, 6, f" {status_str}", border=1, ln=True)
    pdf.cell(50, 6, "Score d'anomalie :", border=1)
    pdf.cell(0, 6, f" {payload['anomaly_score']} (Seuil de décision : {payload['threshold']})", border=1, ln=True)
    pdf.cell(50, 6, "Nombre de zones suspectes :", border=1)
    pdf.cell(0, 6, f" {len(payload.get('regions', []))}", border=1, ln=True)
    pdf.ln(5)

    # 2. Visuel de l'anomalie détourée
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. Cartographie de l'Anomalie (Superposition & Contours)", ln=True)

    tmp_path = Path("temp_pdf_overlay.jpg")
    bgr_img = cv2.cvtColor(overlay_img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(tmp_path), bgr_img)
    pdf.image(str(tmp_path), x=55, w=100)
    pdf.ln(5)

    # 3. Rapport d'expertise du LLM
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "3. Rapport d'Expertise & Recommandations Opérateur", ln=True)
    pdf.set_font("Helvetica", "", 9)

    clean_text = llm_analysis.replace("’", "'").replace("—", "-").replace("**", "")
    pdf.multi_cell(0, 5, clean_text.encode("latin-1", "replace").decode("latin-1"))

    # Écriture du fichier et nettoyage
    pdf.output(output_path)
    if tmp_path.exists():
        tmp_path.unlink()

    return output_path