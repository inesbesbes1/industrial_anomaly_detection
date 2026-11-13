"""Générateur de rapport PDF normé Keyence Style - ISO 9001 (Personne B)."""

import io
import uuid
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import qrcode
from fpdf import FPDF
import unicodedata

def clean_pdf_text(text: str) -> str:
    """Convertit les caractères Unicode en caractères compatibles Helvetica."""
    text = str(text)

    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("latin-1", "ignore").decode("latin-1")

    return text

# ─── COULEURS ────────────────────────────────────────────────────────────────
BLUE_DARK   = (13,  49,  97)
BLUE_MID    = (30,  90, 160)
BLUE_LIGHT  = (220, 235, 250)
RED_CRIT    = (180,  20,  20)
ORANGE_MED  = (200, 100,   0)
GREEN_OK    = ( 20, 130,  40)
GREY_BORDER = (180, 180, 180)
WHITE       = (255, 255, 255)
BLACK       = (  0,   0,   0)


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def _severity_color(severity: str) -> tuple:
    return {
        "CRITICAL": RED_CRIT,
        "MEDIUM":   ORANGE_MED,
        "LOW":      ORANGE_MED,
        "NONE":     GREEN_OK,
    }.get(severity, BLACK)


def _decision_color(decision: str) -> tuple:
    return {"PASS": GREEN_OK, "REVIEW": ORANGE_MED, "REJECT": RED_CRIT}.get(decision, BLACK)


def _save_np_to_tmp(img_rgb: np.ndarray, suffix: str = ".jpg") -> Path:
    """Sauvegarde un ndarray RGB en fichier temporaire, retourne le Path."""
    tmp = Path(tempfile.mktemp(suffix=suffix))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return tmp


def _make_qr(data: str, size_px: int = 180) -> Path:
    """Génère un QR code PNG en fichier temporaire."""
    qr = qrcode.QRCode(version=2, box_size=4, border=2,
                       error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").resize((size_px, size_px))
    tmp = Path(tempfile.mktemp(suffix=".png"))
    img.save(str(tmp))
    return tmp


# ─── CLASSE PDF ──────────────────────────────────────────────────────────────
class SentinexPDF(FPDF):

    def __init__(self, inspection_id: str, category: str, decision: str):
        super().__init__()
        self.inspection_id = inspection_id
        self.category = category.upper()
        self.decision = decision
        self.set_margins(14, 14, 14)

    def header(self):
        # Bandeau bleu foncé
        self.set_fill_color(*BLUE_DARK)
        self.rect(0, 0, 210, 20, style="F")

        # Logo texte
        self.set_xy(5, 3)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*WHITE)
        self.cell(60, 7, "SENTINEX VISION QC", ln=False)

        # Titre centré
        self.set_font("Helvetica", "B", 11)
        self.cell(90, 7, "FICHE D'INSPECTION QUALITÉ", align="C", ln=False)

        # Catégorie à droite
        self.set_font("Helvetica", "", 9)
        self.cell(0, 7, f"Ligne : {self.category}", align="R", ln=True)

        # Sous-bandeau bleu clair
        self.set_fill_color(*BLUE_LIGHT)
        self.rect(0, 20, 210, 7, style="F")
        self.set_xy(5, 21)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*BLUE_MID)
        self.cell(0, 5,
                  f"ID Inspection : {self.inspection_id}  |  "
                  f"Système : PatchCore Wide-ResNet50 + LLaMA 3.2  |  "
                  f"Norme : ISO 9001:2015",
                  ln=True)
        self.ln(5)

    def footer(self):
        self.set_y(-13)
        self.set_fill_color(*BLUE_DARK)
        self.rect(0, self.get_y(), 210, 15, style="F")
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*WHITE)
        self.cell(
            0, 13,
            f"Sentinex Industrial Vision QC  |  {self.inspection_id}  |  "
            f"{datetime.now().strftime('%d/%m/%Y %H:%M')}  |  Page {self.page_no()}",
            align="C"
        )

    def section_title(self, text: str):
        self.set_fill_color(*BLUE_MID)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 10)

        text = clean_pdf_text(text)

     # pyrefly: ignore[parse-error]
        self.cell(0, 7, f"  {text}", fill=True, ln=True)
        self.set_text_color(*BLACK)
        self.ln(2)

    def kv_row(self, label: str, value: str, fill: bool = False):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*BLUE_LIGHT if fill else WHITE)

        self.cell(60, 6, clean_pdf_text(label), border=1, fill=fill)

        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, clean_pdf_text(value), border=1, fill=fill, ln=True)


# ─── FONCTION PRINCIPALE ─────────────────────────────────────────────────────
def generate_inspection_pdf(
    payload: dict,
    llm_analysis: str,
    overlay_img_rgb: np.ndarray,
    original_img_rgb: np.ndarray | None = None,
    output_path: str = "rapport_inspection.pdf",
) -> str:
    """Génère la fiche PDF Keyence Style et retourne le chemin du fichier."""

    inspection_id = "QC-" + uuid.uuid4().hex[:10].upper()
    assessment    = payload.get("_assessment", {})
    decision      = assessment.get("decision", "PASS")
    severity      = assessment.get("severity", "NONE")
    regions       = payload.get("regions", [])

    pdf = SentinexPDF(inspection_id, payload.get("category", "?"), decision)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ── 1. BANDEAU VERDICT ───────────────────────────────────────────────────
    pdf.section_title("1. VERDICT DE L'INSPECTION")

    d_color = _decision_color(decision)
    pdf.set_fill_color(*d_color)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, f"  {assessment.get('decision_label', decision)}", fill=True, ln=True)
    pdf.set_text_color(*BLACK)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    score     = payload.get("anomaly_score", 0)
    threshold = payload.get("threshold", 0.5)
    conf      = assessment.get("confidence", "N/A")

    pdf.kv_row("Score d'anomalie optique :", f"{score:.3f}  (Seuil : {threshold})", fill=True)
    pdf.kv_row("Sévérité calculée :", severity)
    pdf.kv_row("Indice de confiance :", str(conf), fill=True)
    pdf.kv_row("Zones suspectes détectées :", str(len(regions)))
    pdf.ln(5)

    # ── 2. DOUBLE VUE AVANT / APRÈS ─────────────────────────────────────────
    pdf.section_title("2. CARTOGRAPHIE VISUELLE — Avant / Après Détection")

    tmp_overlay  = _save_np_to_tmp(overlay_img_rgb)
    tmp_files    = [tmp_overlay]

    if original_img_rgb is not None:
        tmp_original = _save_np_to_tmp(original_img_rgb)
        tmp_files.append(tmp_original)

        # Deux images côte à côte
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(80, 80, 80)
        x_start = pdf.get_x()
        y_start = pdf.get_y()

        pdf.image(str(tmp_original), x=14,  y=y_start, w=86)
        pdf.image(str(tmp_overlay),  x=110, y=y_start, w=86)
        pdf.set_y(y_start + 68)

        # Légendes
        pdf.set_x(14);  pdf.cell(86, 5, "Image Originale (Caméra)", align="C")
        pdf.set_x(110); pdf.cell(86, 5, "Superposition & Contours OpenCV", align="C", ln=True)
    else:
        # Une seule image centrée
        pdf.image(str(tmp_overlay), x=55, w=100)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Superposition & Contours OpenCV", align="C", ln=True)

    pdf.set_text_color(*BLACK)
    pdf.ln(4)

    # ── 3. TABLEAU DES RÉGIONS ───────────────────────────────────────────────
    pdf.section_title("3. RÉGIONS ANOMALES LOCALISÉES")

    if regions:
        # En-tête du tableau
        pdf.set_fill_color(*BLUE_MID)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 9)
        for label, w in [("Région", 20), ("Position", 45), ("Surface (%)", 35), ("Intensité Max", 40), ("Sévérité", 40)]:
            pdf.cell(w, 7, label, border=1, fill=True, align="C")
        pdf.ln()

        pdf.set_text_color(*BLACK)
        pdf.set_font("Helvetica", "", 8)
        for idx, r in enumerate(regions):
            fill = idx % 2 == 0
            pdf.set_fill_color(*BLUE_LIGHT if fill else WHITE)

            sev_region = "HIGH" if r["max_intensity"] >= 0.8 else ("MED" if r["max_intensity"] >= 0.6 else "LOW")
            sev_color  = _severity_color("CRITICAL" if sev_region == "HIGH" else ("MEDIUM" if sev_region == "MED" else "LOW"))

            pdf.cell(20, 6, f"#{r['region_id']}", border=1, fill=fill, align="C")
            pdf.cell(45, 6, r["position"],         border=1, fill=fill, align="C")
            pdf.cell(35, 6, f"{r['area_pct']} %",  border=1, fill=fill, align="C")
            pdf.cell(40, 6, str(r["max_intensity"]), border=1, fill=fill, align="C")

            # Cellule sévérité colorée
            pdf.set_fill_color(*sev_color)
            pdf.set_text_color(*WHITE)
            pdf.cell(40, 6, sev_region, border=1, fill=True, align="C")
            pdf.set_text_color(*BLACK)
            pdf.set_fill_color(*BLUE_LIGHT if fill else WHITE)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, "Aucune région anormale détectée au-dessus du seuil de décision.", ln=True)

    pdf.ln(5)

    # ── 4. DIAGNOSTIC LLM ───────────────────────────────────────────────────
    pdf.section_title("4. DIAGNOSTIC & RECOMMANDATIONS (Moteur LLM)")
    pdf.set_font("Helvetica", "", 9)
    clean = (llm_analysis
             .replace("\u2019", "'").replace("\u2014", "-").replace("**", "")
             .replace("\u2013", "-").replace("\u00e9", "e").replace("\u00e8", "e")
             .replace("\u00ea", "e").replace("\u00e0", "a").replace("\u00f4", "o")
             .replace("\u00ee", "i").replace("\u00fb", "u").replace("\u00e7", "c"))
    clean_latin = clean.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 5, clean_latin)
    pdf.ln(4)

    # ── 5. QR CODE + SIGNATURE ───────────────────────────────────────────────
    pdf.section_title("5. TRAÇABILITÉ & SIGNATURE QUALITÉ")

    qr_path = _make_qr(f"SENTINEX|{inspection_id}|{payload.get('category','?').upper()}|{decision}|{score:.3f}")
    tmp_files.append(qr_path)

    qr_y = pdf.get_y() + 2
    pdf.image(str(qr_path), x=14, y=qr_y, w=30)

    pdf.set_xy(50, qr_y)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(60, 60, 60)
    traceability_text = (
    f"ID Unique        : {inspection_id}\n"
    f"Date             : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
    f"Produit          : {payload.get('category', '?').upper()}\n"
    f"Verdict Final    : {assessment.get('decision_label', decision)}\n"
    f"Valide par       : Sentinex AI Quality Engine v2.0\n"
    f"Norme            : ISO 9001:2015 - Controle Automatise par Vision"
)

    pdf.multi_cell(0, 5, clean_pdf_text(traceability_text))

    # Écriture + nettoyage
    pdf.output(output_path)
    for p in tmp_files:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    return output_path