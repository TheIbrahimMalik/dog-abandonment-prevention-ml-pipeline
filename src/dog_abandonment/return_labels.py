from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from dog_abandonment.preprocessing import (
    load_intakes_csv,
    load_outcomes_csv,
    filter_dogs,
    nearest_outcome_after_intake,
)

ADOPTION_OUTCOME_TYPES = {"Adoption", "Rto-Adopt"}


@dataclass(frozen=True)
class ReturnLabelConfig:
    return_days: Sequence[int] = (30, 60, 90)


def build_adoption_episodes(
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    max_days_to_outcome: int | None = 3650,
) -> pd.DataFrame:
    enc = nearest_outcome_after_intake(intakes, outcomes, max_days=max_days_to_outcome)

    ad = enc[enc["outcome_type_outcome"].isin(ADOPTION_OUTCOME_TYPES)].copy()
    ad["adoption_time"] = ad["match_time_outcome"]
    ad["intake_time"] = ad["match_time_intake"]
    ad["days_to_adoption"] = (ad["adoption_time"] - ad["intake_time"]).dt.total_seconds() / 86400.0

    keep = [
        "animal_id",
        "intake_time",
        "adoption_time",
        "days_to_adoption",
        "intake_type_intake",
        "intake_condition_intake",
        "sex_upon_intake_intake",
        "age_upon_intake_intake",
        "breed_intake",
        "color_intake",
        "found_location_intake",
        "outcome_subtype_outcome",
    ]
    keep = [c for c in keep if c in ad.columns]
    return ad[keep].reset_index(drop=True)


def attach_next_intake_after_adoption(episodes: pd.DataFrame, intakes: pd.DataFrame) -> pd.DataFrame:
    i = intakes.copy()
    i["intake_time_evt"] = i["intake_datetime"].fillna(i["intake_monthyear"])
    i = i.dropna(subset=["intake_time_evt"])

    # IMPORTANT: sort primarily by time for merge_asof compatibility (pandas 3.x)
    i = i.sort_values(["intake_time_evt", "animal_id"]).reset_index(drop=True)

    left = episodes.dropna(subset=["adoption_time"]).copy()
    left = left.sort_values(["adoption_time", "animal_id"]).reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        i[["animal_id", "intake_time_evt", "intake_type", "intake_condition"]],
        by="animal_id",
        left_on="adoption_time",
        right_on="intake_time_evt",
        direction="forward",
        allow_exact_matches=False,
    )

    return merged.rename(
        columns={
            "intake_time_evt": "next_intake_time",
            "intake_type": "next_intake_type",
            "intake_condition": "next_intake_condition",
        }
    )


def add_return_labels(df: pd.DataFrame, cfg: ReturnLabelConfig) -> pd.DataFrame:
    out = df.copy()
    out["days_to_return"] = (out["next_intake_time"] - out["adoption_time"]).dt.total_seconds() / 86400.0
    out["returned"] = out["next_intake_time"].notna().astype(int)

    for d in cfg.return_days:
        out[f"return_within_{d}d"] = ((out["returned"] == 1) & (out["days_to_return"] <= d)).astype(int)

    out["return_reason"] = out["next_intake_type"].where(out["returned"] == 1, "No Return")
    return out


def build_return_risk_dataset(intakes_csv: str, outcomes_csv: str, cfg: ReturnLabelConfig) -> pd.DataFrame:
    intakes = filter_dogs(load_intakes_csv(intakes_csv))
    outcomes = filter_dogs(load_outcomes_csv(outcomes_csv))

    episodes = build_adoption_episodes(intakes, outcomes)
    episodes = attach_next_intake_after_adoption(episodes, intakes)
    labeled = add_return_labels(episodes, cfg)
    return labeled
