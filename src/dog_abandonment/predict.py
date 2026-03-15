from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from dog_abandonment.interventions import recommend_interventions


def _parse_age_to_days(age_str: str) -> float | None:
    if age_str is None or (isinstance(age_str, float) and np.isnan(age_str)):
        return None
    s = str(age_str).strip().lower()
    if not s or s in {"unknown", "unk", "nan"}:
        return None
    m = re.search(r"([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit.startswith("year") or unit in {"yr", "yrs"}:
        return val * 365
    if unit.startswith("month") or unit == "mo":
        return val * 30
    if unit.startswith("week") or unit in {"wk", "wks"}:
        return val * 7
    if unit.startswith("day") or unit == "d":
        return val
    return None


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features expected by the trained pipelines."""
    out = df.copy()

    # Datetime-derived features
    adt = pd.to_datetime(out.get("adoption_time"), errors="coerce")
    out["adoption_month"] = adt.dt.month
    out["adoption_dow"] = adt.dt.dayofweek

    # Age numeric
    if "age_days" not in out.columns:
        out["age_days"] = out.get("age_upon_intake_intake", pd.Series([None] * len(out))).apply(_parse_age_to_days)

    # Sex / altered from strings like "Neutered Male"
    s = out.get("sex_upon_intake_intake", pd.Series(["Unknown"] * len(out))).fillna("Unknown").astype(str).str.lower()
    out["sex"] = np.select([s.str.contains("male"), s.str.contains("female")], ["Male", "Female"], default="Unknown")
    out["altered"] = np.select(
        [s.str.contains("neuter") | s.str.contains("spay"), s.str.contains("intact")],
        ["Altered", "Intact"],
        default="Unknown",
    )

    # Breed / colour (primary tokens)
    b = out.get("breed_intake", pd.Series(["Unknown"] * len(out))).fillna("Unknown").astype(str)
    out["is_mix"] = b.str.contains("mix", case=False, na=False).astype(int)
    out["breed_primary"] = (
        b.str.split("/", n=1).str[0]
         .str.replace(" Mix", "", regex=False)
         .str.strip()
    )

    c = out.get("color_intake", pd.Series(["Unknown"] * len(out))).fillna("Unknown").astype(str)
    out["colour_primary"] = c.str.split("/", n=1).str[0].str.strip()

    return out


def _extract_model_columns(pipe) -> Tuple[List[str], List[str], Dict[str, set]]:
    """Get expected cat/num columns and categorical allowed values (per feature) from fitted pipeline."""
    pre = pipe.named_steps["preprocess"]

    cat_cols, num_cols = [], []
    for name, _trans, cols in pre.transformers_:
        if name == "cat":
            cat_cols = list(cols)
        elif name == "num":
            num_cols = list(cols)

    # OneHotEncoder categories for each categorical feature
    cat_pipe = pre.named_transformers_["cat"]
    onehot = cat_pipe.named_steps["onehot"]
    categories = onehot.categories_

    allowed: Dict[str, set] = {}
    for col, cats in zip(cat_cols, categories):
        allowed[col] = set(cats.tolist())

    return cat_cols, num_cols, allowed


def _normalise_categoricals(df: pd.DataFrame, allowed: Dict[str, set]) -> pd.DataFrame:
    """Map unknown categories to 'Other' when the trained model supports it."""
    out = df.copy()
    for col, allowed_vals in allowed.items():
        if col not in out.columns:
            continue
        # If model was trained with 'Other', map unseen to 'Other' for consistency
        if "Other" in allowed_vals:
            out[col] = out[col].where(out[col].isin(allowed_vals), "Other")
    return out


def _load_default_threshold(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return 0.5
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    # Use validation best-F1 threshold by default (avoids peeking at test)
    try:
        return float(m["val"]["best_f1"]["threshold"])
    except Exception:
        return 0.5


def predict_batch(
    input_csv: str,
    *,
    horizon_days: int = 90,
    threshold: Optional[float] = None,
    binary_model_path: Optional[str] = None,
    reason_model_path: Optional[str] = None,
    binary_metrics_path: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    binary_model_path = binary_model_path or f"artifacts/models/binary_lr_return_within_{horizon_days}d.joblib"
    reason_model_path = reason_model_path or f"artifacts/models/reason_lr_return_within_{horizon_days}d.joblib"
    binary_metrics_path = binary_metrics_path or f"artifacts/metrics/binary_metrics_return_within_{horizon_days}d.json"

    bin_pipe = joblib.load(binary_model_path)
    reason_pipe = joblib.load(reason_model_path)

    df = pd.read_csv(input_path, low_memory=False)
    df = _build_features(df)

    # Extract expected columns + align unknown categories
    cat_cols, num_cols, allowed = _extract_model_columns(bin_pipe)
    df = _normalise_categoricals(df, allowed)

    # Ensure required columns exist
    needed = cat_cols + num_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    X = df[needed].copy()

    # Default threshold from metrics if not specified
    thr = threshold if threshold is not None else _load_default_threshold(Path(binary_metrics_path))

    risk_prob = bin_pipe.predict_proba(X)[:, 1]
    risk_flag = (risk_prob >= thr).astype(int)

    # Reason predicted only for flagged rows (triage)
    reason_pred = np.array(["No Return"] * len(df), dtype=object)
    if risk_flag.sum() > 0:
        # Normalise categoricals for reason model too (use its encoder categories)
        r_cat_cols, r_num_cols, r_allowed = _extract_model_columns(reason_pipe)
        Xr = df[r_cat_cols + r_num_cols].copy()
        Xr = _normalise_categoricals(Xr, r_allowed)

        reason_pred[risk_flag == 1] = reason_pipe.predict(Xr[risk_flag == 1])

    # Interventions
    interventions = []
    for r in reason_pred:
        recs = recommend_interventions(str(r))
        interventions.append(" | ".join([f"{x.title}: {x.description}" for x in recs]))

    out = pd.DataFrame({
        "animal_id": df.get("animal_id", pd.Series([None] * len(df))),
        "adoption_time": df.get("adoption_time", pd.Series([None] * len(df))),
        f"risk_return_within_{horizon_days}d": risk_prob,
        f"predicted_return_within_{horizon_days}d": risk_flag,
        "predicted_return_reason": reason_pred,
        "recommended_interventions": interventions,
        "threshold_used": thr,
    })

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)

    return out
