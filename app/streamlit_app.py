"""Interface Web Streamlit - Industrial Quality Assistant (Personne B)."""

import os
import sys
from pathlib import Path

# Assure la résolution propre des imports quelle que soit la commande de lancement
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
import streamlit as st
import numpy as np

# Imports des modules de la Personne B
from processing.postprocessing import build_llm_payload
from llm.model import query_llm
from reporting.pdf_generator import generate_inspection_pdf

# Imports utilitaires visuels de la Personne A
from detection.visualize import overlay_heatmap, draw_contours, colorize

st.set_page_config(page_title="Quality Assistant AI", layout="wide", page_icon="🏭")


def fake_predict(img_rgb: np.ndarray, cat: str, simulate_defect: bool = True) -> dict:
    """Simulateur d'inférence (génère soit une anomalie, soit une pièce saine)."""
    h, w = img_rgb.shape[:2]
    amap = np.zeros((h, w), np.float32)

    if simulate_defect:
        # Faux défaut au centre
        amap[int(h * 0.35):int(h * 0.50), int(w * 0.40):int(w * 0.60)] = 0.89
        return {
            "category": cat,
            "status": "anomalous",
            "anomaly_score": 0.84,
            "threshold": 0.5,
            "anomaly_map": amap,
            "pixel_threshold": 0.5
        }
    else:
        # Pièce saine conforme (aucun défaut)
        return {
            "category": cat,
            "status": "normal",
            "anomaly_score": 0.12,
            "threshold": 0.5,
            "anomaly_map": amap,
            "pixel_threshold": 0.5
        }


# --- BARRE LATÉRALE ---
st.sidebar.title("🏭 Poste d'Inspection")
category = st.sidebar.selectbox("Catégorie de produit", ["bottle", "cable"])
use_fake = st.sidebar.checkbox("Mode simulation (Mock sans modèle)", value=True)
simulate_defect = True
if use_fake:
    sim_type = st.sidebar.radio("Cas simulé :", ["Pièce Défectueuse (Rebut)", "Pièce Conforme (Saine)"])
    simulate_defect = (sim_type == "Pièce Défectueuse (Rebut)")
model_name = st.sidebar.selectbox("Modèle LLM local (Ollama)", ["qwen2.5:3b", "llama3.2:3b"])

st.sidebar.markdown("---")
st.sidebar.caption("Système couplé PatchCore (Personne A) & Traitement/LLM (Personne B)")


# --- PAGE PRINCIPALE ---
st.title("Système Intelligent de Contrôle Qualité")
st.markdown("Détection d'anomalies non supervisée & Analyse d'incidents par LLM local.")

uploaded_file = st.file_uploader(
    f"Déposer une image de {category.upper()} à contrôler :",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Inférence : vrai modèle ou mock
    with st.spinner("Analyse de la pièce par le modèle de vision..."):
        if use_fake:
            pred_res = fake_predict(img_np, category, simulate_defect=simulate_defect)
        else:
            try:
                from detection.predict import predict
                pred_res = predict(img_np, category=category)
            except Exception as e:
                st.error(
                    f"Impossible d'exécuter le modèle réel : {e}.\n\n"
                    "Si la Personne A n'a pas encore fourni les checkpoints (.ckpt), "
                    "cochez **Mode simulation** dans le panneau de gauche."
                )
                st.stop()

    # 1. Post-traitement OpenCV
    payload = build_llm_payload(pred_res)
    amap = pred_res["anomaly_map"]

    # 2. Préparation des visualisations
    heatmap_colored = colorize(amap)
    overlay_img = overlay_heatmap(img_np, amap)
    overlay_with_contours = draw_contours(overlay_img, amap >= 0.5, color=(255, 255, 255))

    # 3. Affichage du verdict
    st.divider()
    score = payload["anomaly_score"]
    col_v1, col_v2 = st.columns([1, 2])

    with col_v1:
        if payload["status"] == "anomalous":
            st.error(f"### ❌ NON CONFORME\nScore d'anomalie : **{score}** (Seuil : 0.5)")
        else:
            st.success(f"### ✅ CONFORME\nScore d'anomalie : **{score}** (Seuil : 0.5)")

    with col_v2:
        st.progress(min(score, 1.0), text=f"Niveau d'anomalie détecté : {int(score * 100)} %")

    # 4. Affichage visuel des 3 panneaux
    st.subheader("Cartographie Visuelle")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img_np, caption="Image Originale", use_container_width=True)
    with c2:
        st.image(heatmap_colored, caption="Carte de Chaleur Brute", use_container_width=True)
    with c3:
        st.image(overlay_with_contours, caption="Superposition & Contours", use_container_width=True)

    # 5. Appel au LLM
    st.subheader("Rapport d'Expertise Qualité (LLM)")
    with st.spinner("Génération du rapport métier par le LLM..."):
        analysis_text = query_llm(payload, model_name=model_name)
        st.markdown(analysis_text)

    # 6. Export PDF
    st.divider()
    pdf_filename = f"inspection_{category}_{payload['status']}.pdf"
    if st.button("📄 Générer la Fiche d'Inspection Officielle (PDF)"):
        with st.spinner("Compilation du PDF..."):
            pdf_path = generate_inspection_pdf(
                payload,
                analysis_text,
                overlay_with_contours,
                output_path=pdf_filename
            )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Télécharger le Rapport PDF",
                    data=f,
                    file_name=pdf_filename,
                    mime="application/pdf"
                )