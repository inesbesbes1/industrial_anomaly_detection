"""Moteur de sévérité déterministe et de règles qualité industrielles (Personne B)."""

from typing import Any


def evaluate_severity_and_decision(predict_result: dict, regions: list[dict]) -> dict[str, Any]:
    """Calcule mathématiquement la sévérité et la décision sans intervention du LLM."""
    score = float(predict_result.get("anomaly_score", 0.0))
    threshold = float(predict_result.get("threshold", 0.5))
    status = predict_result.get("status", "normal")
    category = predict_result.get("category", "inconnue")

    margin = abs(score - threshold)
    if margin >= 0.25:
        confidence = "HIGH (98 %)"
    elif margin >= 0.10:
        confidence = "MEDIUM (85 %)"
    else:
        confidence = "LOW (Zone de doute)"

    # 1. Pièce conforme
    if status == "normal" or score < threshold:
        return {
            "severity": "NONE",
            "decision": "PASS",
            "decision_label": "PIÈCE CONFORME (PASS)",
            "confidence": confidence,
            "badge_color": "green",
            "uncertainty_flag": margin < 0.10,
            "actions": [
                "Autoriser le transfert vers le poste suivant.",
                "Enregistrer le matricule de contrôle conforme."
            ]
        }

    # 2. Pièce défectueuse : calcul de sévérité
    total_area_pct = sum(r.get("area_pct", 0.0) for r in regions)
    max_intensity = max((r.get("max_intensity", 0.0) for r in regions), default=score)
    regions_count = len(regions)

    # CRITIQUE / HIGH
    if total_area_pct >= 3.0 or max_intensity >= 0.85 or (total_area_pct >= 1.5 and regions_count >= 2):
        severity = "CRITICAL"
        decision = "REJECT"
        decision_label = "REBUT IMMÉDIAT (REJECT)"
        badge_color = "red"
        if category == "cable":
            actions = [
                "Éjecter immédiatement le faisceau (risque majeur de court-circuit).",
                "Contrôler la station de sertissage et l'outil de dénudage.",
                "Consigner le défaut dans le registre de non-conformité."
            ]
        else:
            actions = [
                "Éjecter la bouteille (risque critique de bris / éclat de verre).",
                "Vérifier le poste d'embouteillage amont.",
                "Inspecter les 5 pièces précédentes pour détecter une dérive."
            ]

    # MOYEN / MEDIUM
    elif total_area_pct >= 0.8 or max_intensity >= 0.65:
        severity = "MEDIUM"
        decision = "REVIEW"
        decision_label = "CONTRÔLE OPÉRATEUR (REVIEW)"
        badge_color = "orange"
        actions = [
            "Placer la pièce en bac de contrôle pour contre-expertise.",
            "Inspecter la zone suspecte sous loupe binoculaire.",
            "Effectuer une nouvelle prise de vue après nettoyage de l'optique."
        ]

    # FAIBLE / LOW
    else:
        severity = "LOW"
        decision = "REVIEW"
        decision_label = "CONTRÔLE RAPIDE (LOW RISK)"
        badge_color = "yellow"
        actions = [
            "Vérification visuelle rapide de la surface.",
            "Contrôler s'il s'agit d'une poussière amovible ou d'un reflet.",
            "Valider sous dérogation si la zone fonctionnelle est intacte."
        ]

    return {
        "severity": severity,
        "decision": decision,
        "decision_label": decision_label,
        "badge_color": badge_color,
        "confidence": confidence,
        "uncertainty_flag": margin < 0.10,
        "actions": actions,
        "metrics_summary": {
            "total_defect_area_pct": round(total_area_pct, 2),
            "max_defect_intensity": round(max_intensity, 2),
            "regions_count": regions_count
        }
    }