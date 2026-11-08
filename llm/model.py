"""Interface avec le LLM via Ollama avec support du Streaming (Personne B)."""

import ollama
from llm.prompts import SYSTEM_PROMPT, get_user_prompt


def stream_llm(payload: dict, model_name: str = "llama3.2:3b"):
    """Générateur qui produit la réponse du LLM mot par mot en streaming."""
    # Nettoyage des clés internes si présentes
    safe_payload = {k: v for k, v in payload.items() if not str(k).startswith("_")}
    user_prompt = get_user_prompt(safe_payload)

    try:
        response = ollama.chat(
            model=model_name,
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
        yield f"⚠️ Erreur de communication avec le LLM ({e})."


def query_llm(payload: dict, model_name: str = "llama3.2:3b") -> str:
    """Version synchrone standard (pour les tests et le rapport PDF)."""
    return "".join(list(stream_llm(payload, model_name=model_name)))