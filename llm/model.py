"""Interface avec le LLM via Ollama avec RAG et Streaming (Personne B)."""

import ollama
from llm.prompts import SYSTEM_PROMPT, get_user_prompt
from llm.rag import retrieve_norm_context


def stream_llm(payload: dict, model_name: str = "llama3.2:3b"):
    """Générateur qui produit la réponse du LLM mot par mot en streaming.
    
    Intègre le RAG : récupère automatiquement le contexte normatif
    pertinent (norme IPC, ISO, etc.) avant d'appeler le LLM.
    """
    # Nettoyage des clés internes
    safe_payload = {k: v for k, v in payload.items() if not str(k).startswith("_")}

    # ── RAG : récupération du contexte normatif ──────────────────────────────
    norm_context = retrieve_norm_context(
        category=payload.get("category", ""),
        status=payload.get("status", "normal"),
        regions=payload.get("regions", [])
    )

    # Construction du prompt enrichi avec le contexte normatif
    user_prompt = get_user_prompt(safe_payload, norm_context=norm_context)

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            options={"temperature": 0.2},
            stream=True
        )
        for chunk in response:
            yield chunk["message"]["content"]

    except Exception as e:
        yield f"⚠️ Erreur de communication avec le LLM ({e})."


def query_llm(payload: dict, model_name: str = "llama3.2:3b") -> str:
    """Version synchrone standard (pour les tests et le rapport PDF)."""
    return "".join(list(stream_llm(payload, model_name=model_name)))

import io
from PIL import Image


def stream_vlm(payload: dict, img_rgb, model_name: str = "moondream", llm_model: str = "llama3.2:3b"):
    """Pipeline Hybride VLM + LLM."""
    norm_context = retrieve_norm_context(
        category=payload.get("category", ""),
        status=payload.get("status", "normal"),
        regions=payload.get("regions", [])
    )

    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    img_bytes = buf.getvalue()

    cat = payload.get("category", "piece")
    vlm_prompt = (
        f"This is an industrial quality control image of a {cat}. "
        f"An anomaly detection model flagged it with a high anomaly score. "
        "Describe in ONE sentence the visible physical defect you see. "
        "Use only words. Do NOT return numbers, coordinates, or lists."
    )

    try:
        vlm_res = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": vlm_prompt, "images": [img_bytes]}]
        )
        visual_desc = vlm_res["message"]["content"].strip()

        # Nettoyage : si Moondream renvoie quand même des coordonnées,
        # on supprime tout ce qui ressemble à [chiffres] ou (chiffres)
        import re
        visual_desc = re.sub(r"[\[\(][\d\s\.,]+[\]\)]", "", visual_desc).strip()

        # Fallback si vide après nettoyage
        if not visual_desc or len(visual_desc) < 10:
            visual_desc = f"Anomalie visuelle détectée sur la surface de la pièce {cat}."

    except Exception:
        visual_desc = "Anomalie géométrique détectée par la caméra industrielle."

    user_prompt = (
        f"Voici le relevé d'inspection de la pièce :\n"
        f"- Catégorie : {cat}\n"
        f"- Statut : {payload.get('status')} (Score optique : {payload.get('anomaly_score')})\n"
        f"- Décision métier : {payload.get('decision_label')} (Sévérité : {payload.get('severity')})\n"
        f"- Observation visuelle directe du VLM (Pixels caméra) : \"{visual_desc}\"\n"
        f"- Nombre de zones suspectes : {len(payload.get('regions', []))}\n\n"
    )

    if norm_context:
        user_prompt += f"{norm_context}\n\n"

    user_prompt += (
        "En intégrant l'observation visuelle de la caméra et les normes applicables, "
        "rédige le rapport d'expertise qualité en français pour l'opérateur selon la structure obligatoire :\n"
        "- **VERDICT & NIVEAU DE RISQUE**\n"
        "- **DIAGNOSTIC TECHNIQUE**\n"
        "- **ACTIONS OPÉRATEUR RECOMMANDÉES**"
    )

    try:
        response = ollama.chat(
            model=llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.2},
            stream=True
        )
        for chunk in response:
            yield chunk["message"]["content"]
    except Exception as e:
        yield f"⚠️ Erreur de communication ({e})."