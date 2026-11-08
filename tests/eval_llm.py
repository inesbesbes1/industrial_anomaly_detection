"""Benchmark et Évaluation qualitative du LLM (Tâche 6 de Personne B).
Adapté à l'architecture modulaire.
"""

from pathlib import Path
from llm.model import query_llm

BENCHMARK_CASES = [
    # 5 cas conformes (test d'absence totale d'hallucination)
    {"id": "CAS-01", "category": "cable", "status": "normal", "anomaly_score": 0.12, "threshold": 0.5, "regions": []},
    {"id": "CAS-02", "category": "cable", "status": "normal", "anomaly_score": 0.28, "threshold": 0.5, "regions": []},
    {"id": "CAS-03", "category": "bottle", "status": "normal", "anomaly_score": 0.05, "threshold": 0.5, "regions": []},
    {"id": "CAS-04", "category": "bottle", "status": "normal", "anomaly_score": 0.35, "threshold": 0.5, "regions": []},
    {"id": "CAS-05", "category": "bottle", "status": "normal", "anomaly_score": 0.42, "threshold": 0.5, "regions": []},

    # 5 cas limites (score proche du seuil 0.5 : doit conseiller une vérification manuelle)
    {"id": "CAS-06", "category": "cable", "status": "normal", "anomaly_score": 0.48, "threshold": 0.5, "regions": []},
    {"id": "CAS-07", "category": "cable", "status": "anomalous", "anomaly_score": 0.52, "threshold": 0.5, "regions": [{"position": "haut-gauche", "area_pct": 0.3, "max_intensity": 0.55}]},
    {"id": "CAS-08", "category": "bottle", "status": "anomalous", "anomaly_score": 0.54, "threshold": 0.5, "regions": [{"position": "centre", "area_pct": 0.2, "max_intensity": 0.58}]},
    {"id": "CAS-09", "category": "bottle", "status": "normal", "anomaly_score": 0.46, "threshold": 0.5, "regions": []},
    {"id": "CAS-10", "category": "cable", "status": "anomalous", "anomaly_score": 0.55, "threshold": 0.5, "regions": [{"position": "bas", "area_pct": 0.4, "max_intensity": 0.60}]},

    # 10 cas anormaux nets
    {"id": "CAS-11", "category": "cable", "status": "anomalous", "anomaly_score": 0.89, "threshold": 0.5, "regions": [{"position": "centre", "area_pct": 3.8, "max_intensity": 0.95}]},
    {"id": "CAS-12", "category": "cable", "status": "anomalous", "anomaly_score": 0.76, "threshold": 0.5, "regions": [{"position": "haut-droite", "area_pct": 1.5, "max_intensity": 0.82}]},
    {"id": "CAS-13", "category": "cable", "status": "anomalous", "anomaly_score": 0.94, "threshold": 0.5, "regions": [{"position": "gauche", "area_pct": 4.2, "max_intensity": 0.98}, {"position": "droite", "area_pct": 1.1, "max_intensity": 0.74}]},
    {"id": "CAS-14", "category": "bottle", "status": "anomalous", "anomaly_score": 0.91, "threshold": 0.5, "regions": [{"position": "centre", "area_pct": 5.4, "max_intensity": 0.99}]},
    {"id": "CAS-15", "category": "bottle", "status": "anomalous", "anomaly_score": 0.83, "threshold": 0.5, "regions": [{"position": "bas-droite", "area_pct": 2.1, "max_intensity": 0.88}]},
    {"id": "CAS-16", "category": "bottle", "status": "anomalous", "anomaly_score": 0.67, "threshold": 0.5, "regions": [{"position": "haut", "area_pct": 0.9, "max_intensity": 0.71}]},
    {"id": "CAS-17", "category": "cable", "status": "anomalous", "anomaly_score": 0.88, "threshold": 0.5, "regions": [{"position": "bas-gauche", "area_pct": 2.8, "max_intensity": 0.90}]},
    {"id": "CAS-18", "category": "bottle", "status": "anomalous", "anomaly_score": 0.79, "threshold": 0.5, "regions": [{"position": "centre", "area_pct": 1.8, "max_intensity": 0.85}]},
    {"id": "CAS-19", "category": "cable", "status": "anomalous", "anomaly_score": 0.96, "threshold": 0.5, "regions": [{"position": "centre", "area_pct": 6.5, "max_intensity": 0.99}]},
    {"id": "CAS-20", "category": "bottle", "status": "anomalous", "anomaly_score": 0.85, "threshold": 0.5, "regions": [{"position": "haut-droite", "area_pct": 2.4, "max_intensity": 0.89}]}
]


def run_evaluation(model_name: str = "qwen2.5:3b", output_md: str = "rapport/evaluation_llm.md"):
    print(f"Lancement de l'évaluation sur 20 cas avec `{model_name}`...")
    rows = []

    for case in BENCHMARK_CASES:
        print(f"  -> Évaluation {case['id']} ({case['category']} - {case['status']})...")
        response = query_llm(case, model_name=model_name)
        lower_resp = response.lower()

        # 1. Compréhension du statut
        if case["status"] == "normal":
            statut_ok = ("conforme" in lower_resp or "valide" in lower_resp) and ("rebut" not in lower_resp)
        else:
            statut_ok = ("anomalie" in lower_resp or "non conforme" in lower_resp or "défaut" in lower_resp or "rebut" in lower_resp)

        # 2. Absence d'hallucination (si normal, aucune position de défaut citée)
        if case["status"] == "normal":
            hallucination = any(pos in lower_resp for pos in ["haut-gauche", "bas-droite", "centre", "haut-droite"])
        else:
            hallucination = False

        rows.append({
            "id": case["id"],
            "cat": case["category"],
            "status": case["status"],
            "score": case["anomaly_score"],
            "statut_ok": "✅" if statut_ok else "❌",
            "zero_hallu": "✅" if not hallucination else "❌"
        })

    # Construction du tableau Markdown
    md = "# Tableau Qualitatif d'Évaluation du LLM\n\n"
    md += f"Modèle évalué : `{model_name}` (Ollama)\n\n"
    md += "| ID | Catégorie | Statut réel | Score | Statut compris | Zéro hallucination |\n"
    md += "|---|---|---|---|:---:|:---:|\n"
    for r in rows:
        md += f"| {r['id']} | {r['cat']} | {r['status']} | {r['score']} | {r['statut_ok']} | {r['zero_hallu']} |\n"

    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\nÉvaluation terminée avec succès ! Résultat écrit dans : {output_md}")


if __name__ == "__main__":
    run_evaluation()