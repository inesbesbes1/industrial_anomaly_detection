"""Interface Web Streamlit - Sentinex Industrial Vision QC (Personne B)."""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd

from processing.postprocessing import build_llm_payload, get_binary_mask
from llm.model import stream_llm
from reporting.pdf_generator import generate_inspection_pdf
from detection.visualize import overlay_heatmap, draw_contours, colorize

st.set_page_config(page_title="Industrial Vision QC - Sentinex", layout="wide", page_icon="🏭")

# --- INITIALISATION DE L'HISTORIQUE DE PRODUCTION ---
if "inspections_history" not in st.session_state:
    st.session_state.inspections_history = []


def fake_predict(img_rgb: np.ndarray, cat: str, simulate_defect: bool = True) -> dict:
    h, w = img_rgb.shape[:2]
    amap = np.zeros((h, w), np.float32)
    if simulate_defect:
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
        return {
            "category": cat,
            "status": "normal",
            "anomaly_score": 0.12,
            "threshold": 0.5,
            "anomaly_map": amap,
            "pixel_threshold": 0.5
        }


# --- BARRE LATÉRALE ---
st.sidebar.title("🏭 DefectLens")
st.sidebar.caption("Système Industriel d'Inspection en Temps Réel")

category = st.sidebar.selectbox("Ligne de fabrication", ["bottle", "cable", "transistor"])
use_fake = False
model_name = "llama3.2:3b"
st.sidebar.markdown("---")
sensitivity = st.sidebar.slider("Seuil de décision optique", 0.30, 0.70, 0.50, 0.05)
st.sidebar.caption(f"Seuil actuel : {sensitivity} (Norme MVTec : 0.50)")


# --- NAVIGATION PAR ONGLETS ---
tab_inspect, tab_analytics = st.tabs([
    "🔬 Poste d'Inspection (Live QC)",
    "📊 Statistiques & Tendances (Analytics)"
])

# =========================================================================
# ONGLET 1 : POSTE D'INSPECTION EN DIRECT
# =========================================================================
with tab_inspect:
    st.subheader(f"Station d'Acquisition Optique : Ligne {category.upper()}")

    uploaded_file = st.file_uploader(
        f"Déposer une image de {category.upper()} prise par la caméra :",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        t_start = time.perf_counter()
        with st.spinner("Analyse par le réseau de neurones PatchCore..."):
            if use_fake:
                pred_res = fake_predict(img_np, category, simulate_defect=simulate_defect)
            else:
                try:
                    from detection.predict import predict
                    pred_res = predict(img_np, category=category)
                except Exception as e:
                    st.error(f"Erreur d'inférence : {e}")
                    st.stop()

        latency_ms = (time.perf_counter() - t_start) * 1000

        # --- CONTRÔLE « HORS DOMAINE » (fourni par predict, voir CONTRAT_INTERFACE.md) ---
        # L'image ne ressemble pas aux images d'entraînement (photo web, autre caméra, autre fond) :
        # toute l'image paraît « anormale », ce n'est PAS un défaut. On n'envoie rien au LLM.
        if not pred_res.get("in_domain", True):
            if st.session_state.get("last_scanned") != uploaded_file.name:
                st.session_state.inspections_history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "file_name": uploaded_file.name,
                    "category": category,
                    "score": round(float(pred_res["anomaly_score"]), 3),
                    "decision": "HORS_DOMAINE",
                    "severity": "N/A",
                    "latency_ms": round(latency_ms, 1)
                })
                st.session_state.last_scanned = uploaded_file.name
            st.divider()
            st.warning(
                "### 📷 Image non exploitable : conditions de prise de vue non conformes\n"
                f"Cette image ne ressemble pas aux images de la ligne **{category.upper()}** sur lesquelles le "
                "modèle a été entraîné (cadrage, fond, éclairage ou caméra différents). "
                "Le verdict serait **non fiable** : aucun diagnostic n'est généré.\n\n"
                "**Action :** reprendre la photo sur le poste d'inspection (caméra et éclairage de la ligne)."
            )
            st.caption(f"Indice de dépaysement : {pred_res['domain_score']:.2f} "
                       f"(limite : {pred_res['domain_threshold']:.2f})")
            c1, c2 = st.columns(2)
            c1.image(img_np, caption="Image reçue", use_container_width=True)
            c2.image(colorize(pred_res["anomaly_map"]), caption="Heatmap : anormale presque partout",
                     use_container_width=True)
            st.stop()

        # Post-traitement OpenCV avec sévérité
        pred_res["threshold"] = sensitivity
        pred_res["pixel_threshold"] = sensitivity
        payload = build_llm_payload(pred_res)
        assessment = payload["_assessment"]
        amap = pred_res["anomaly_map"]

        # Enregistrement dans l'historique
        if st.session_state.get("last_scanned") != uploaded_file.name:
            st.session_state.inspections_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "file_name": uploaded_file.name,
                "category": category,
                "score": payload["anomaly_score"],
                "decision": assessment["decision"],
                "severity": assessment["severity"],
                "latency_ms": round(latency_ms, 1)
            })
            st.session_state.last_scanned = uploaded_file.name

        # --- BANDEAU DE DÉCISION ET SÉVÉRITÉ ---
        st.divider()
        col_status, col_sev, col_conf = st.columns([1.5, 1, 1])

        with col_status:
            if assessment["decision"] == "PASS":
                st.success(f"### ✅ {assessment['decision_label']}")
            elif assessment["decision"] == "REVIEW":
                st.warning(f"### ⚠️ {assessment['decision_label']}")
            else:
                st.error(f"### ❌ {assessment['decision_label']}")
            st.caption(f"Score optique : **{payload['anomaly_score']}** | Seuil de rejet : **{sensitivity}**")

        with col_sev:
            st.metric("Niveau de Sévérité", assessment["severity"])

        with col_conf:
            st.metric("Indice de Confiance", assessment["confidence"])

        # --- LES 4 VOLETS VISUELS INDUSTRIELS ---
        st.subheader("Cartographie et Décomposition Visuelle")
        binary_mask = get_binary_mask(amap, threshold=sensitivity)
        heatmap_colored = colorize(amap)
        overlay_img = overlay_heatmap(img_np, amap)
        overlay_with_contours = draw_contours(overlay_img, amap >= sensitivity, color=(255, 255, 255))

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.image(img_np, caption="1. Image Caméra (Originale)", use_container_width=True)
        with v2:
            st.image(heatmap_colored, caption="2. Heatmap Thermique", use_container_width=True)
        with v3:
            st.image(binary_mask, caption="3. Masque Binaire OpenCV", use_container_width=True)
        with v4:
            st.image(overlay_with_contours, caption="4. Superposition & Contours", use_container_width=True)

        # --- DÉTAIL DES RÉGIONS & RAPPORT DU LLM ---
        st.divider()
        col_table, col_llm = st.columns([1, 1.3])

        with col_table:
            st.markdown("#### 📍 Régions Anormales Localisées")
            if payload["regions"]:
                df_regions = pd.DataFrame([
                    {
                        "Région": f"#{r['region_id']}",
                        "Position": r["position"],
                        "Surface (%)": f"{r['area_pct']} %",
                        "Intensité": r["max_intensity"]
                    }
                    for r in payload["regions"]
                ])
                st.dataframe(df_regions, use_container_width=True)
            else:
                st.info("Aucune région suspecte au-dessus du seuil de détection.")

            st.markdown("#### ⚙️ Actions Préconisées par les Règles Métier :")
            for act in assessment["actions"]:
                st.markdown(f"- {act}")

        with col_llm:
            st.markdown("#### 🤖 Diagnostic de l'Ingénieur Qualité (LLM)")
            report_text = st.write_stream(stream_llm(payload, model_name=model_name))

        # --- EXPORT DU RAPPORT PDF ---
        st.divider()
        pdf_name = f"inspection_{category}_{payload['anomaly_score']}.pdf"
        if st.button("📄 Générer la Fiche Officielle d'Inspection (PDF)"):
            with st.spinner("Création du document PDF..."):
                pdf_path = generate_inspection_pdf(payload, report_text, overlay_with_contours, original_img_rgb=img_np, output_path=pdf_name)
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Télécharger le Rapport PDF", data=f, file_name=pdf_name, mime="application/pdf")

# =========================================================================
# ONGLET 2 : ANALYTICS & STATISTIQUES D'ATELIER
# =========================================================================
with tab_analytics:
    st.subheader("📊 Tableau de Bord de Production & Suivi Qualité")

    history = st.session_state.inspections_history

    if not history:
        st.info("Aucune inspection effectuée durant cette session. Effectuez des contrôles dans l'onglet 'Poste d'Inspection' pour alimenter les statistiques.")
    else:
        df_hist = pd.DataFrame(history)

        total_scans = len(df_hist)
        pass_count = len(df_hist[df_hist["decision"] == "PASS"])
        reject_count = len(df_hist[df_hist["decision"] == "REJECT"])
        review_count = len(df_hist[df_hist["decision"] == "REVIEW"])
        # Les images hors domaine ne sont pas des pièces inspectées : exclues du taux de conformité
        n_valid = total_scans - len(df_hist[df_hist["decision"] == "HORS_DOMAINE"])
        yield_rate = (pass_count / n_valid) * 100 if n_valid else 0.0
        avg_lat = df_hist["latency_ms"].mean()

        # KPIs industriels
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Pièces Inspectées", total_scans)
        k2.metric("Taux de Conformité (Yield)", f"{yield_rate:.1f} %")
        k3.metric("Rebuts / Alertes", f"{reject_count + review_count}")
        k4.metric("Temps Moyen de Cycle", f"{avg_lat:.0f} ms")

        st.divider()

        # Graphiques
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("##### 📈 Évolution des Scores d'Anomalie (Session)")
            st.line_chart(df_hist["score"])

        with g2:
            st.markdown("##### 📊 Répartition des Décisions Qualité")
            decision_counts = df_hist["decision"].value_counts()
            st.bar_chart(decision_counts)

        # Tableau d'historique
        st.markdown("##### 📋 Journal d'Audit de Production")
        st.dataframe(df_hist, use_container_width=True)