from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Intervention:
    title: str
    description: str


_REASON_TO_ACTIONS = {
    "Owner Surrender": [
        Intervention("Post-adoption check-in", "Schedule follow-ups at 7/30 days to identify emerging issues early."),
        Intervention("Behaviour/training support", "Provide subsidised training resources and behaviour advice."),
        Intervention("Practical support", "Share support options for cost, housing, or temporary fostering where available."),
    ],
    "Stray": [
        Intervention("ID and containment check", "Verify microchip/ID tags and advise on secure containment."),
        Intervention("Lost-pet protocol", "Provide guidance for lost-dog prevention and rapid recovery steps."),
        Intervention("Follow-up", "Contact adopter to assess escape risk and offer support."),
    ],
    "Public Assist": [
        Intervention("Targeted follow-up", "Clarify why assistance was needed and provide tailored support."),
        Intervention("Resource pack", "Provide local support contacts and practical guidance."),
    ],
    "Abandoned": [
        Intervention("High-touch follow-up", "Initiate frequent welfare check-ins and risk triage."),
        Intervention("Support escalation", "Offer emergency support and consider foster placement if needed."),
    ],
    "Other": [
        Intervention("Case review", "Flag for manual review; reason category was rare/merged."),
        Intervention("Follow-up", "Perform adopter contact and identify specific needs."),
    ],
    "No Return": [
        Intervention("No action", "No return predicted within the horizon based on current threshold."),
    ],
}


def recommend_interventions(reason: str) -> List[Intervention]:
    """Return a list of recommended interventions for a predicted return reason."""
    return _REASON_TO_ACTIONS.get(reason, _REASON_TO_ACTIONS["Other"])
