from __future__ import annotations

from collections import Counter


def _safe_percent(value: float) -> str:
    return f"{max(0.0, value) * 100:.1f}%"


def build_agent_assessment(potholes, road_condition, summary):
    """Create a professional decision layer over raw detection results."""
    counts = Counter(p.get("severity", "Low") for p in potholes)
    pothole_count = int(summary.get("pothole_count", len(potholes)))
    average_area_ratio = float(summary.get("average_area_ratio", 0.0))
    total_area_ratio = float(summary.get("total_area_ratio", 0.0))
    high_count = int(summary.get("high_severity_count", counts.get("High", 0)))
    medium_count = int(counts.get("Medium", 0))
    top_score = max((float(p.get("severity_score", 0.0)) for p in potholes), default=0.0)
    avg_conf = (
        sum(float(p.get("confidence", 0.0)) for p in potholes) / len(potholes)
        if potholes
        else 0.0
    )

    emergency_alert = False
    priority = "Low"
    recommended_action = "No immediate maintenance required"
    risk_band = "Low"
    estimated_sla = "Monitor during routine inspection cycle"

    if (
        high_count >= 2
        or top_score >= 0.85
        or total_area_ratio >= 0.08
        or (road_condition == "Poor" and pothole_count >= 3)
    ):
        emergency_alert = True
        priority = "Critical"
        recommended_action = "Escalate for immediate repair intervention"
        risk_band = "Critical"
        estimated_sla = "Dispatch emergency maintenance within 24 hours"
    elif high_count >= 1 or road_condition == "Poor" or total_area_ratio >= 0.04:
        priority = "High"
        recommended_action = "Schedule urgent maintenance and traffic safety review"
        risk_band = "High"
        estimated_sla = "Repair planning recommended within 48 hours"
    elif medium_count >= 1 or pothole_count >= 2:
        priority = "Medium"
        recommended_action = "Add to maintenance queue and monitor progression"
        risk_band = "Moderate"
        estimated_sla = "Inspect and schedule repairs within 7 days"

    reasoning = []
    if pothole_count == 0:
        reasoning.append("No potholes were detected in the uploaded road image.")
    else:
        reasoning.append(
            f"{pothole_count} pothole(s) were detected with {high_count} high-severity and {medium_count} medium-severity case(s)."
        )
        reasoning.append(
            f"The detected potholes affect {_safe_percent(total_area_ratio)} of the visible road surface, with an average impact of {_safe_percent(average_area_ratio)} per pothole."
        )
        reasoning.append(
            f"Model confidence across detections averages {avg_conf:.2f}, supporting a {risk_band.lower()} operational risk assessment."
        )
        if emergency_alert:
            reasoning.append(
                "Emergency escalation was triggered because the severity pattern indicates immediate roadway safety risk."
            )

    next_steps = []
    if pothole_count == 0:
        next_steps.extend(
            [
                "Keep the road segment under routine observation.",
                "Capture another inspection image if visual road damage increases.",
            ]
        )
    elif emergency_alert:
        next_steps.extend(
            [
                "Notify maintenance control and field operations immediately.",
                "Place temporary hazard signage or access control if the segment is active.",
                "Log the segment for post-repair verification imaging.",
            ]
        )
    elif priority == "High":
        next_steps.extend(
            [
                "Create a high-priority repair ticket for the affected road section.",
                "Plan a field verification visit to validate surface deterioration extent.",
                "Review adjacent segments for progressive failure.",
            ]
        )
    else:
        next_steps.extend(
            [
                "Add the defect to the next scheduled maintenance cycle.",
                "Track the segment for growth in pothole size or count.",
                "Retain the uploaded image as a baseline inspection record.",
            ]
        )

    techniques = {
        "cnn": "Active in project as image-based severity modelling support.",
        "rf": "Active in project as feature-based severity analysis support.",
        "ann": "Active in project as nonlinear feature analysis support.",
        "generative_ai": "Used offline during training to improve robustness under lighting and weather variation.",
        "agentic_ai": "Active in deployment as the maintenance decision and escalation layer.",
    }

    summary_text = (
        "Road status is stable with no actionable pothole risk."
        if pothole_count == 0
        else (
            f"The inspection indicates a {road_condition.lower()} road condition with {priority.lower()} maintenance priority."
        )
    )

    return {
        "agent_name": "Road Maintenance Decision Agent",
        "agent_version": "1.0",
        "summary": summary_text,
        "recommended_action": recommended_action,
        "priority": priority,
        "risk_band": risk_band,
        "emergency_alert": emergency_alert,
        "estimated_sla": estimated_sla,
        "reasoning": reasoning,
        "next_steps": next_steps,
        "techniques": techniques,
    }
