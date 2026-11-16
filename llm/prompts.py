"""Définition des prompts système et utilisateur avec RAG (Personne B)."""

import json

SYSTEM_PROMPT = """Tu es un expert senior en contrôle qualité industriel automatisé.
Tu analyses les sorties d'un détecteur de vision par ordinateur (PatchCore) inspectant des pièces (bouteilles, câbles, cuir ou transistors).

CONSIGNES STRICTES :
1. Reste 100% fidèle aux données JSON fournies. N'invente JAMAIS de défaut non signalé par le détecteur.
2. Si status == "normal" : déclare la pièce CONFORME, indique qu'aucun défaut n'a été détecté et autorise la suite du flux.
3. Si status == "anomalous" : analyse chaque région suspecte (position, pourcentage de surface, intensité) et explique le risque industriel.
4. Si le score est très proche du seuil 0.5 (entre 0.45 et 0.55), signale un doute et recommande un contrôle manuel.
5. La décision est DÉJÀ calculée par les règles métier : ton VERDICT doit reprendre exactement "decision_label" du JSON.
6. Si un CONTEXTE NORMATIF est fourni, cite explicitement la norme applicable (ex: "Selon la norme IPC-A-620E...") dans ton diagnostic.

STRUCTURE OBLIGATOIRE DE TA RÉPONSE :
- **VERDICT & NIVEAU DE RISQUE** (valeur de "decision_label" + "severity")
- **DIAGNOSTIC TECHNIQUE** (Description factuelle des zones observées + référence normative si disponible)
- **ACTIONS OPÉRATEUR RECOMMANDÉES** (2 à 3 actions concrètes pour l'atelier)
Sois professionnel, direct et concis."""


def get_user_prompt(payload: dict, norm_context: str = "") -> str:
    """Met en forme le prompt utilisateur avec les données JSON et le contexte RAG."""
    # On retire _assessment du payload affiché (interne, pas utile au LLM)
    display_payload = {k: v for k, v in payload.items() if k != "_assessment"}
    payload_str = json.dumps(display_payload, indent=2, ensure_ascii=False)

    prompt = (
        "Voici le relevé d'inspection de la pièce :\n"
        "```json\n"
        f"{payload_str}\n"
        "```\n"
    )

    # Injection du contexte RAG si disponible
    if norm_context:
        prompt += (
            "\n"
            f"{norm_context}\n"
            "\n"
            "En t'appuyant sur ce contexte normatif, rédige le rapport d'expertise qualité pour l'opérateur de ligne.\n"
            "Cite la norme applicable dans ton diagnostic technique.\n"
        )
    else:
        prompt += "Rédige le rapport d'expertise qualité pour l'opérateur de ligne.\n"

    return prompt