"""
src/dog_abandonment/preprocessing.py

Ibrahim Malik (initial version September 2025; refactored 2026)

"""

import re
from datetime import datetime
import numpy as np
import pandas as pd

# PEP 8: https://peps.python.org/pep-0008/
# pandas docs: https://pandas.pydata.org/docs/

# =========================
# Column maps (rename to snake_case)
# https://www.w3schools.com/python/gloss_python_snakecase.asp
# https://www.geeksforgeeks.org/python/python-constant/
# =========================
INTAKES_RENAME = {
    "Animal ID": "animal_id",
    "Name": "name",
    "DateTime": "intake_datetime",
    "MonthYear": "intake_monthyear",
    "Found Location": "found_location",
    "Intake Type": "intake_type",
    "Intake Condition": "intake_condition",
    "Animal Type": "animal_type",
    "Sex upon Intake": "sex_upon_intake",
    "Age upon Intake": "age_upon_intake",
    "Breed": "breed",
    "Color": "color",
}

OUTCOMES_RENAME = {
    "Animal ID": "animal_id",
    "Date of Birth": "date_of_birth",
    "Name": "name",
    "DateTime": "outcome_datetime",
    "MonthYear": "outcome_monthyear",
    "Outcome Type": "outcome_type",
    "Outcome Subtype": "outcome_subtype",
    "Animal Type": "animal_type",
    "Sex upon Outcome": "sex_upon_outcome",
    "Age upon Outcome": "age_upon_outcome",
    "Breed": "breed",
    "Color": "color",
}

_AUSTIN_TZ = "America/Chicago"

# =========================
# String cleaners
# =========================
def _clean_str_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.normalize("NFKC")                 # Unicode normalize
    s = s.str.replace("\u00A0", " ", regex=False)  # NBSP -> space
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def _clean_animal_id(s: pd.Series) -> pd.Series:
    return _clean_str_series(s).str.upper()

def _clean_name(s: pd.Series) -> pd.Series:
    return _clean_str_series(s).str.lstrip("*").str.strip()

# =========================
# Robust datetime parsing
# =========================
def _clean_datetime_text(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.normalize("NFKC")
    s = s.str.replace("\u00A0", " ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    # normalize am/pm
    s = s.str.replace(r"\b(a\.?m\.?)\b", "AM", regex=True, case=False)
    s = s.str.replace(r"\b(p\.?m\.?)\b", "PM", regex=True, case=False)
    # ISO T -> space
    s = s.str.replace("T", " ", regex=False)
    # ensure seconds exist (hh:mm -> hh:mm:00)
    s = s.str.replace(r"(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2})(?!:)", r"\1:00", regex=True)
    # collapse commas
    s = s.str.replace(r",\s*", " ", regex=True)
    return s

def _to_uniform_utc_naive(series: pd.Series, assume_local_tz: str = _AUSTIN_TZ) -> pd.Series:
    """
    Parse mixed datetime strings:
      1) parse with utc=True (handles ISO + offsets) -> UTC -> naive
      2) parse naive -> localize America/Chicago (DST robust) -> UTC -> naive
      3) retry with dayfirst
      4) tiny per-row set of formats
    """
    if series is None:
        return series
    raw = _clean_datetime_text(series)

    # Pass 1: utc=True (handles ISO + offsets); format="mixed" silences inference warnings
    dt1 = pd.to_datetime(raw, errors="coerce", utc=True, format="ISO8601")
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    m1 = dt1.notna()
    if m1.any():
        out.loc[m1] = dt1.loc[m1].dt.tz_convert("UTC").dt.tz_localize(None)

    # Pass 2: naive -> local tz -> UTC
    m2 = out.isna()
    if m2.any():
        # parse naive strings without inferring (mixed formats are allowed)
        dt2 = pd.to_datetime(raw[m2], errors="coerce", format="mixed")
        dt2 = (
            dt2.dt.tz_localize(assume_local_tz, nonexistent="shift_forward", ambiguous="NaT")
               .dt.tz_convert("UTC")
               .dt.tz_localize(None)
        )
        out.loc[m2] = dt2

    # Pass 3: dayfirst
    m3 = out.isna()
    if m3.any():
        dt3 = pd.to_datetime(raw[m3], errors="coerce", dayfirst=True, format="mixed")
        dt3 = (
            dt3.dt.tz_localize(assume_local_tz, nonexistent="shift_forward", ambiguous="NaT")
               .dt.tz_convert("UTC")
               .dt.tz_localize(None)
        )
        out.loc[m3] = dt3

    # Pass 4: explicit fmts (rare)
    m4 = out.isna()
    if m4.any():
        fmts = [
            "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S",
            "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]
        def _try_formats(x):
            for f in fmts:
                try:
                    dt = datetime.strptime(x, f)
                    return (pd.Timestamp(dt, tz=_AUSTIN_TZ).tz_convert("UTC").tz_localize(None))
                except Exception:
                    continue
            return pd.NaT
        out.loc[m4] = raw[m4].apply(_try_formats)

    return out

def _to_monthyear(s: pd.Series) -> pd.Series:
    """
    Parse strings like 'Oct-13' reliably to Timestamp at month start.
    """
    if s is None:
        return s
    s = _clean_str_series(s)

    # fast path for patterns like 'Oct-13'
    dt = pd.to_datetime(s, format="%b-%y", errors="coerce")

    # fallback (handles '2014-10', '10/2014', etc.)
    missing = dt.isna()
    if missing.any():
        dt.loc[missing] = pd.to_datetime(s[missing], errors="coerce", format="mixed")

    # normalize to month start (00:00)
    dt = dt.dt.to_period("M").dt.to_timestamp()
    assert dt.dt.day.eq(1).all() and dt.dt.hour.eq(0).all()
    return dt

# =========================
# Loaders
# =========================
def load_intakes_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # Normalize column names so rename works even with stray spaces
    df.columns = df.columns.str.normalize("NFKC").str.strip()
    df = df.rename(columns=INTAKES_RENAME)

    if "animal_id" in df:        df["animal_id"] = _clean_animal_id(df["animal_id"])
    if "name" in df:             df["name"] = _clean_name(df["name"])
    if "intake_datetime" in df:  df["intake_datetime"]  = _to_uniform_utc_naive(df["intake_datetime"])
    if "intake_monthyear" in df: df["intake_monthyear"] = _to_monthyear(df["intake_monthyear"])
    if "animal_type" in df:      df["animal_type"] = _clean_str_series(df["animal_type"]).str.title()
    return df

def load_outcomes_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.normalize("NFKC").str.strip()
    df = df.rename(columns=OUTCOMES_RENAME)

    if "animal_id" in df:         df["animal_id"] = _clean_animal_id(df["animal_id"])
    if "name" in df:              df["name"] = _clean_name(df["name"])
    if "outcome_datetime" in df:  df["outcome_datetime"]  = _to_uniform_utc_naive(df["outcome_datetime"])
    if "outcome_monthyear" in df: df["outcome_monthyear"] = _to_monthyear(df["outcome_monthyear"])
    if "date_of_birth" in df:     df["date_of_birth"]     = _to_uniform_utc_naive(df["date_of_birth"])
    if "animal_type" in df:       df["animal_type"] = _clean_str_series(df["animal_type"]).str.title()
    return df

# =========================
# Summaries & filters
# =========================
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().rename("missing_frac")
    dtypes = df.dtypes.rename("dtype")
    out = pd.concat([miss, dtypes], axis=1).reset_index().rename(columns={"index": "column"})
    out["missing_pct"] = (out["missing_frac"] * 100).round(2)
    return out.sort_values(["missing_frac", "column"], ascending=[False, True])

def filter_dogs(df: pd.DataFrame, animal_type_col: str = "animal_type") -> pd.DataFrame:
    if animal_type_col not in df.columns:
        return df
    mask = df[animal_type_col].str.lower() == "dog"
    return df[mask.fillna(False)].copy()

# =========================
# Age parsing (e.g., "2 years", "6 mo", "10 days")
# =========================
_AGE_UNITS_IN_DAYS = {
    "year": 365, "years": 365, "yr": 365, "yrs": 365,
    "month": 30, "months": 30, "mo": 30,
    "week": 7, "weeks": 7, "wk": 7, "wks": 7,
    "day": 1, "days": 1, "d": 1,
}

def parse_age_to_days(age_str):
    if pd.isna(age_str):
        return None
    s = str(age_str).strip().lower()
    if not s or s in {"unknown", "unk", "nan"}:
        return None
    m = re.search(r"([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).rstrip("s")
    for k, days in _AGE_UNITS_IN_DAYS.items():
        if unit == k or unit + "s" == k:
            return val * days
    return None

# =========================
# Merge: intake -> first outcome >= intake (per animal)
# =========================
def nearest_outcome_after_intake(
    intakes: pd.DataFrame,
    outcomes: pd.DataFrame,
    max_days: int | None = None
) -> pd.DataFrame:
    """
    Vectorized as-of merge (fast):
      - Intake time:  DateTime -> fallback to MonthYear anchored to 5th @ 12:00
      - Outcome time: DateTime -> fallback to MonthYear anchored to 25th @ 12:00
                      -> fallback to (date_of_birth + age_upon_outcome)
      - Merges the first outcome >= intake for the same animal_id.
      - Returns all intake cols as *_intake, all outcome cols as *_outcome,
        plus match_time_intake, match_time_outcome, time_source_intake,
        time_source_outcome, and days_to_outcome.
    """
    by = "animal_id"
    t_in, m_in   = "intake_datetime",   "intake_monthyear"
    t_out, m_out = "outcome_datetime",  "outcome_monthyear"

    def _anchor_day(s: pd.Series, day: int) -> pd.Series:
        """Place month-only timestamps on a specific day @ 12:00 for ordering."""
        mm = pd.to_datetime(s, errors="coerce")
        return (mm.dt.to_period("M").dt.to_timestamp()
                + pd.to_timedelta(day - 1, unit="D")
                + pd.Timedelta(hours=12))

    def _approx_from_dob_age(df: pd.DataFrame, dob_col: str, age_col: str) -> pd.Series:
        """Approximate outcome time from DOB + age_upon_outcome (in days)."""
        if dob_col not in df.columns or age_col not in df.columns:
            return pd.Series(pd.NaT, index=df.index)
        dob = pd.to_datetime(df[dob_col], errors="coerce")
        age_days = df[age_col].apply(parse_age_to_days)
        age_td = pd.to_timedelta(age_days, unit="D", errors="coerce")
        return dob + age_td

    # ---------- Intake side ----------
    i0 = intakes.copy()
    i0[by] = i0[by].astype(str).str.strip().str.upper()

    dt_i = pd.to_datetime(i0.get(t_in), errors="coerce")
    my_i = _anchor_day(i0.get(m_in), 5) if m_in in i0.columns else pd.NaT
    match_i = dt_i.fillna(my_i)

    src_i = np.where(dt_i.notna(), "datetime",
             np.where(my_i.notna(), "monthyear_d5", "missing"))

    i_payload = i0.rename(columns={c: (c if c == by else f"{c}_intake") for c in i0.columns})
    i_payload["match_time_intake"]  = pd.to_datetime(match_i, errors="coerce")
    i_payload["time_source_intake"] = pd.Series(src_i, index=i0.index, dtype="object")
    i_payload = i_payload.dropna(subset=["match_time_intake", by]).copy()

    # ---------- Outcome side ----------
    o0 = outcomes.copy()
    o0[by] = o0[by].astype(str).str.strip().str.upper()

    dt_o = pd.to_datetime(o0.get(t_out), errors="coerce")
    my_o = _anchor_day(o0.get(m_out), 25) if m_out in o0.columns else pd.NaT
    ap_o = _approx_from_dob_age(o0, "date_of_birth", "age_upon_outcome")

    match_o = dt_o.copy()
    match_o = match_o.where(match_o.notna(), my_o)
    match_o = match_o.where(match_o.notna(), ap_o)

    src_o = np.where(dt_o.notna(), "datetime",
             np.where(my_o.notna(), "monthyear_d25",
             np.where(ap_o.notna(), "dob_plus_age", "missing")))

    o_payload = o0.rename(columns={c: (c if c == by else f"{c}_outcome") for c in o0.columns})
    o_payload["match_time_outcome"]  = pd.to_datetime(match_o, errors="coerce")
    o_payload["time_source_outcome"] = pd.Series(src_o, index=o0.index, dtype="object")
    o_payload = o_payload.dropna(subset=["match_time_outcome", by]).copy()

    # ---------- FINAL sort (global, by time first) ----------
    # enforce dtype and drop any lingering NaT
    i_payload["match_time_intake"]  = pd.to_datetime(i_payload["match_time_intake"],  errors="coerce")
    o_payload["match_time_outcome"] = pd.to_datetime(o_payload["match_time_outcome"], errors="coerce")
    i_payload = i_payload.dropna(subset=["match_time_intake", by])
    o_payload = o_payload.dropna(subset=["match_time_outcome", by])

    # IMPORTANT: sort by time first, then by id (merge_asof expects global sort on the key)
    i_sorted = i_payload.sort_values(["match_time_intake", by], kind="mergesort").reset_index(drop=True)
    o_sorted = o_payload.sort_values(["match_time_outcome", by], kind="mergesort").reset_index(drop=True)

    # Optional debug assertions (safe to leave in)
    if not i_sorted["match_time_intake"].is_monotonic_increasing:
        raise ValueError("Left 'match_time_intake' not globally sorted.")
    if not o_sorted["match_time_outcome"].is_monotonic_increasing:
        raise ValueError("Right 'match_time_outcome' not globally sorted.")

    # ---------- As-of merge ----------
    tol = pd.Timedelta(days=max_days) if max_days is not None else None
    merged = pd.merge_asof(
        i_sorted, o_sorted,
        left_on="match_time_intake",
        right_on="match_time_outcome",
        by=by,
        direction="forward",
        allow_exact_matches=True,
        tolerance=tol,
    )

    # ---------- Delta in days ----------
    merged["days_to_outcome"] = (
        pd.to_datetime(merged["match_time_outcome"], errors="coerce")
        - pd.to_datetime(merged["match_time_intake"], errors="coerce")
    ).dt.total_seconds() / 86400.0

    return merged.reset_index(drop=True)