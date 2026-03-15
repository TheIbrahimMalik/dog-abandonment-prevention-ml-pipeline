from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class TrainConfig:
    dataset_csv: str = "data/processed/return_risk_dataset.csv"
    horizon_days: int = 90
    out_dir: str = "artifacts"
    val_frac: float = 0.10
    test_frac: float = 0.20
    min_reason_count: int = 50  # collapse rare reason classes to "Other"
    top_n_breeds: int = 50
    top_n_colours: int = 20


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _time_split(df: pd.DataFrame, val_frac: float, test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("adoption_time").reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start = int(n * (1 - test_frac - val_frac))
    train = df.iloc[:val_start].copy()
    val = df.iloc[val_start:test_start].copy()
    test = df.iloc[test_start:].copy()
    return train, val, test


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
    """Feature set available at adoption time (or earlier)."""
    out = df.copy()

    # Datetime-derived features
    adt = pd.to_datetime(out["adoption_time"], errors="coerce")
    out["adoption_month"] = adt.dt.month
    out["adoption_dow"] = adt.dt.dayofweek

    # Age numeric
    if "age_upon_intake_intake" in out.columns:
        out["age_days"] = out["age_upon_intake_intake"].apply(_parse_age_to_days)
    else:
        out["age_days"] = np.nan

    # Sex / altered from strings like "Neutered Male"
    s = out.get("sex_upon_intake_intake", pd.Series(["Unknown"] * len(out))).fillna("Unknown").astype(str).str.lower()
    out["sex"] = np.select([s.str.contains("male"), s.str.contains("female")], ["Male", "Female"], default="Unknown")
    out["altered"] = np.select(
        [s.str.contains("neuter") | s.str.contains("spay"), s.str.contains("intact")],
        ["Altered", "Intact"],
        default="Unknown",
    )

    # Breed / colour (reduce cardinality)
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


def _bucket_top_n(train_series: pd.Series, series: pd.Series, top_n: int, other: str = "Other") -> pd.Series:
    top = train_series.value_counts().head(top_n).index
    return series.where(series.isin(top), other)


def _make_pipeline(cat_cols, num_cols, *, balanced: bool) -> Pipeline:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ]
    )

    clf = LogisticRegression(
        max_iter=10000,
        solver="saga",
        class_weight=("balanced" if balanced else None),
    )

    return Pipeline(steps=[("preprocess", pre), ("model", clf)])


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    return {
        "threshold": float(thr),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    candidates = np.unique(np.clip(y_prob, 0.0, 1.0))
    if len(candidates) > 2000:
        idx = np.linspace(0, len(candidates) - 1, 2000).astype(int)
        candidates = candidates[idx]

    best_thr, best_f1 = 0.5, -1.0
    for thr in candidates:
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def train(cfg: TrainConfig = TrainConfig()) -> None:
    ds_path = Path(cfg.dataset_csv)
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}. Run `dog-abandonment preprocess` first.")

    df = pd.read_csv(
        ds_path,
        parse_dates=["intake_time", "adoption_time", "next_intake_time"],
        low_memory=False,
    )
    df = _build_features(df)

    target_col = f"return_within_{cfg.horizon_days}d"
    if target_col not in df.columns:
        raise ValueError(f"Target column missing: {target_col}")

    # Split first (so bucketing uses TRAIN only)
    train_df, val_df, test_df = _time_split(df, cfg.val_frac, cfg.test_frac)

    # Bucket high-cardinality cats using train distribution
    for col, topn in [("breed_primary", cfg.top_n_breeds), ("colour_primary", cfg.top_n_colours)]:
        train_df[col] = _bucket_top_n(train_df[col], train_df[col], topn)
        val_df[col] = _bucket_top_n(train_df[col], val_df[col], topn)
        test_df[col] = _bucket_top_n(train_df[col], test_df[col], topn)

    # Feature columns
    cat_cols = [
        "intake_type_intake",
        "intake_condition_intake",
        "sex",
        "altered",
        "breed_primary",
        "colour_primary",
        "outcome_subtype_outcome",
    ]
    num_cols = [
        "days_to_adoption",
        "age_days",
        "adoption_month",
        "adoption_dow",
        "is_mix",
    ]

    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]

    X_train, y_train = train_df[cat_cols + num_cols], train_df[target_col].astype(int).to_numpy()
    X_val, y_val = val_df[cat_cols + num_cols], val_df[target_col].astype(int).to_numpy()
    X_test, y_test = test_df[cat_cols + num_cols], test_df[target_col].astype(int).to_numpy()

    # Binary model (balanced is usually helpful for PR-AUC in imbalanced setting)
    bin_pipe = _make_pipeline(cat_cols, num_cols, balanced=True)
    bin_pipe.fit(X_train, y_train)

    val_prob = bin_pipe.predict_proba(X_val)[:, 1]
    best_thr = _best_f1_threshold(y_val, val_prob)
    test_prob = bin_pipe.predict_proba(X_test)[:, 1]

    binary_metrics = {
        "run_utc": _utc_now_iso(),
        "dataset_rows": int(len(df)),
        "horizon_days": int(cfg.horizon_days),
        "split": {"val_frac": cfg.val_frac, "test_frac": cfg.test_frac},
        "val": {"at_0.5": _metrics_at_threshold(y_val, val_prob, 0.5), "best_f1": _metrics_at_threshold(y_val, val_prob, best_thr)},
        "test": {"at_0.5": _metrics_at_threshold(y_test, test_prob, 0.5), "best_f1": _metrics_at_threshold(y_test, test_prob, best_thr)},
        "features": {"categorical": cat_cols, "numeric": num_cols},
        "model": "LogisticRegression(saga, class_weight=balanced, onehot)",
        "bucketing": {"top_n_breeds": cfg.top_n_breeds, "top_n_colours": cfg.top_n_colours},
    }

    # Reason model on returns within horizon only (unweighted default: matches operational majority)
    reason_df = df[df[target_col].astype(int) == 1].copy()
    reason_df = reason_df[reason_df["return_reason"].ne("No Return")].copy()

    vc = reason_df["return_reason"].value_counts()
    rare = set(vc[vc < cfg.min_reason_count].index.tolist())
    if rare:
        reason_df["return_reason"] = reason_df["return_reason"].apply(lambda x: "Other" if x in rare else x)

    r_train, r_val, r_test = _time_split(reason_df, cfg.val_frac, cfg.test_frac)

    # Apply same bucketing for reason model
    for col, topn in [("breed_primary", cfg.top_n_breeds), ("colour_primary", cfg.top_n_colours)]:
        r_train[col] = _bucket_top_n(r_train[col], r_train[col], topn)
        r_val[col] = _bucket_top_n(r_train[col], r_val[col], topn)
        r_test[col] = _bucket_top_n(r_train[col], r_test[col], topn)

    Xr_train = r_train[cat_cols + num_cols]
    yr_train = r_train["return_reason"].astype(str).to_numpy()
    Xr_val = r_val[cat_cols + num_cols]
    yr_val = r_val["return_reason"].astype(str).to_numpy()
    Xr_test = r_test[cat_cols + num_cols]
    yr_test = r_test["return_reason"].astype(str).to_numpy()

    reason_pipe = _make_pipeline(cat_cols, num_cols, balanced=False)
    # swap estimator for multiclass-friendly solver
    reason_pipe.set_params(
        model=LogisticRegression(max_iter=12000, solver="lbfgs")
    )
    reason_pipe.fit(Xr_train, yr_train)

    pred_val = reason_pipe.predict(Xr_val)
    pred_test = reason_pipe.predict(Xr_test)

    reason_metrics = {
        "run_utc": _utc_now_iso(),
        "horizon_days": int(cfg.horizon_days),
        "min_reason_count": int(cfg.min_reason_count),
        "classes": sorted(list(set(yr_train))),
        "val": {"accuracy": float(accuracy_score(yr_val, pred_val)), "macro_f1": float(f1_score(yr_val, pred_val, average="macro", zero_division=0))},
        "test": {"accuracy": float(accuracy_score(yr_test, pred_test)), "macro_f1": float(f1_score(yr_test, pred_test, average="macro", zero_division=0))},
        "model": "LogisticRegression(lbfgs, onehot)",
        "bucketing": {"top_n_breeds": cfg.top_n_breeds, "top_n_colours": cfg.top_n_colours},
    }

    out_dir = Path(cfg.out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    joblib.dump(bin_pipe, out_dir / "models" / f"binary_lr_return_within_{cfg.horizon_days}d.joblib")
    joblib.dump(reason_pipe, out_dir / "models" / f"reason_lr_return_within_{cfg.horizon_days}d.joblib")

    with open(out_dir / "metrics" / f"binary_metrics_return_within_{cfg.horizon_days}d.json", "w") as f:
        json.dump(binary_metrics, f, indent=2)

    with open(out_dir / "metrics" / f"reason_metrics_return_within_{cfg.horizon_days}d.json", "w") as f:
        json.dump(reason_metrics, f, indent=2)

    print("Wrote models to:", out_dir / "models")
    print("Wrote metrics to:", out_dir / "metrics")
    print("\nBinary (test @ 0.5):", binary_metrics["test"]["at_0.5"])
    print("Binary (test @ best-F1 threshold):", binary_metrics["test"]["best_f1"])
    print("Reason (test):", reason_metrics["test"])
