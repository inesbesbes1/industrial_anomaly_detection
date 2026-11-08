"""Interface avec le LLM via Ollama (Personne B)."""

import ollama
from llm.prompts import SYSTEM_PROMPT, get_user_prompt


def query_llm(payload: dict, model_name: str = "qwen2.5:3b") -> str:
    """Interroge le LLM local avec le payload d'inspection."""
    user_prompt = get_user_prompt(payload)

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.2}  # Température basse = déterminisme et zéro hallucination
        )
        return response["message"]["content"]
    except Exception as e:
        return (
            f"⚠️ Erreur de communication avec le LLM ({e}).\n\n"
            "Assurez-vous qu'Ollama est démarré et que le modèle est téléchargé "
            f"(`ollama run {model_name}`)."
        )