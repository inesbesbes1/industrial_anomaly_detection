"""Module RAG (Retrieval-Augmented Generation) — Sentinex Vision QC (Personne B).

Principe :
  1. Charge la base de connaissances normative (knowledge_base/normes_industrielles.md).
  2. Découpe le document en sections par catégorie et sous-section.
  3. Pour une anomalie donnée, récupère les paragraphes les plus pertinents.
  4. Retourne le contexte normatif à injecter dans le prompt du LLM.
"""

from __future__ import annotations
from pathlib import Path
import re

KB_PATH = Path(__file__).resolve().parent.parent / "knowledge_base" / "normes_industrielles.md"


def _load_sections(path: Path = KB_PATH) -> list[dict]:
    """Charge et découpe la base en sections indexées par catégorie et titre."""
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    sections = []
    current_category = ""
    current_title = ""
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## CATÉGORIE :"):
            # Sauvegarde la section précédente
            if current_lines:
                sections.append({
                    "category": current_category,
                    "title": current_title,
                    "content": "\n".join(current_lines).strip()
                })
            current_category = line.replace("## CATÉGORIE :", "").strip().split("(")[0].strip().lower()
            current_title = line
            current_lines = []
        elif line.startswith("### "):
            if current_lines:
                sections.append({
                    "category": current_category,
                    "title": current_title,
                    "content": "\n".join(current_lines).strip()
                })
            current_title = line.replace("### ", "").strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Dernière section
    if current_lines:
        sections.append({
            "category": current_category,
            "title": current_title,
            "content": "\n".join(current_lines).strip()
        })

    return [s for s in sections if s["content"]]


def _score_section(section: dict, category: str, defect_keywords: list[str]) -> float:
    """Score de pertinence d'une section par rapport à la requête."""
    score = 0.0
    content_lower = section["content"].lower()
    title_lower = section["title"].lower()

    # Correspondance de catégorie (priorité max)
    if section["category"] == category.lower():
        score += 10.0

    # Correspondance mots-clés défaut
    for kw in defect_keywords:
        if kw.lower() in content_lower:
            score += 2.0
        if kw.lower() in title_lower:
            score += 1.0

    return score


def retrieve_norm_context(
    category: str,
    status: str,
    regions: list[dict],
    top_k: int = 2
) -> str:
    """Récupère les paragraphes normatifs les plus pertinents pour l'anomalie.

    Args:
        category:  "bottle", "cable", "leather" ou "transistor"
        status:    "normal" ou "anomalous"
        regions:   liste des régions détectées par OpenCV (avec 'position', 'max_intensity')
        top_k:     nombre de sections à retourner

    Returns:
        Texte de contexte normatif à injecter dans le prompt du LLM.
    """
    if status == "normal" or not regions:
        return ""

    sections = _load_sections()
    if not sections:
        return ""

    # Mots-clés extraits des régions détectées
    defect_keywords = [category, status, "rebut", "review", "defaut", "anomalie"]
    # Ajout de mots-clés spécifiques par catégorie
    category_keywords = {
        "bottle": ["éclat", "fissure", "contamination", "broken", "verre"],
        "cable":  ["coupure", "dénudé", "isolation", "fil", "câble", "court-circuit"],
        "leather": ["coupure", "pli", "tache", "colle", "perforation", "cuir"],
        "transistor": ["broche", "boîtier", "composant", "manquant", "déplacé"],
    }
    defect_keywords += category_keywords.get(category, [])

    # Score et tri des sections
    scored = sorted(
        sections,
        key=lambda s: _score_section(s, category, defect_keywords),
        reverse=True
    )

    top_sections = scored[:top_k]

    if not top_sections:
        return ""

    lines = ["=== CONTEXTE NORMATIF (Base de Connaissances Industrielle) ==="]
    for s in top_sections:
        lines.append(f"\n[{s['title']}]")
        # On garde les lignes les plus informatives (celles avec des défauts/actions)
        relevant_lines = [
            l for l in s["content"].splitlines()
            if any(kw in l.lower() for kw in defect_keywords + ["action", "norme", "tolérance", "risque"])
        ]
        lines.extend(relevant_lines[:8])  # Max 8 lignes par section

    return "\n".join(lines)