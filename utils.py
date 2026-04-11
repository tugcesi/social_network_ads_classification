"""
Helper functions for feature engineering and predictions.
Mirrors the preprocessing pipeline used in the training notebook.
"""

import pickle
import joblib

import numpy as np
import pandas as pd
import streamlit as st

from app_config import (
    AGE_BINS,
    AGE_LABELS,
    DATASET_PATH,
    MODEL_FEATURES,
    MODEL_JOBLIB_PATH,
    MODEL_PKL_PATH,
    SALARY_BINS,
    SALARY_LABELS,
    YOUNG_RICH_AGE_THRESHOLD,
    YOUNG_RICH_SALARY_THRESHOLD,
)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load and return the raw Social Network Ads dataset."""
    df = pd.read_csv(DATASET_PATH)
    return df


@st.cache_resource
def load_model():
    """
    Load the trained classification model.

    Tries Purchased.pkl first (bundle with feature list), then falls back to
    Purchased.joblib (bare model).

    Returns
    -------
    model : fitted sklearn estimator
    features : list[str]
    """
    try:
        with open(MODEL_PKL_PATH, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict) and "model" in bundle:
            return bundle["model"], bundle.get("features", MODEL_FEATURES)
        # Bare model saved as pkl
        return bundle, MODEL_FEATURES
    except Exception:
        pass

    try:
        model = joblib.load(MODEL_JOBLIB_PATH)
        return model, MODEL_FEATURES
    except Exception as exc:
        raise RuntimeError(
            "Model yüklenemedi. Purchased.pkl veya Purchased.joblib "
            "dosyasının mevcut olduğundan emin olun."
        ) from exc


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering pipeline used during model training.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least: Gender (str or int), Age (int), EstimatedSalary (int).

    Returns
    -------
    pd.DataFrame with engineered features ready for model inference.
    """
    df = df.copy()

    # 1. Encode gender ─ Female → 0, Male → 1
    if not pd.api.types.is_numeric_dtype(df["Gender"]):
        df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    # 2. Salary levels
    df["SalaryLevel"] = pd.cut(
        df["EstimatedSalary"],
        bins=SALARY_BINS,
        labels=SALARY_LABELS,
    )

    # 3. Age groups
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
    )

    # 4. Young-rich flag
    df["IsYoungRich"] = (
        (df["Age"] < YOUNG_RICH_AGE_THRESHOLD)
        & (df["EstimatedSalary"] > YOUNG_RICH_SALARY_THRESHOLD)
    ).astype(int)

    # 5. One-hot encode (drop_first=True mirrors pd.get_dummies(drop_first=True))
    df = pd.get_dummies(df, columns=["SalaryLevel", "AgeGroup"], drop_first=True)

    # Ensure all expected columns are present (fill missing dummies with 0)
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[MODEL_FEATURES].astype(float)


def predict_single(model, gender: str, age: int, salary: int):
    """
    Run inference for a single user.

    Parameters
    ----------
    model : fitted sklearn estimator
    gender : "Male" or "Female"
    age : integer age
    salary : integer estimated salary

    Returns
    -------
    prediction : int  (0 or 1)
    probability : float  (probability of class 1)
    probabilities : np.ndarray  ([prob_0, prob_1])
    """
    row = pd.DataFrame(
        [{"Gender": gender, "Age": age, "EstimatedSalary": salary}]
    )
    X = engineer_features(row)
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    return prediction, float(probabilities[1]), probabilities


# ── Dataset-level metrics ─────────────────────────────────────────────────────

@st.cache_data
def compute_dataset_stats(df: pd.DataFrame) -> dict:
    """Return a dict of useful statistics computed from the raw dataset."""
    total = len(df)
    purchased = int(df["Purchased"].sum())
    not_purchased = total - purchased
    purchase_rate = purchased / total

    male_rate = df[df["Gender"] == "Male"]["Purchased"].mean()
    female_rate = df[df["Gender"] == "Female"]["Purchased"].mean()

    # Engineered features for deeper stats
    df_eng = df.copy()
    df_eng["Gender_enc"] = df_eng["Gender"].map({"Female": 0, "Male": 1})
    df_eng["SalaryLevel"] = pd.cut(
        df_eng["EstimatedSalary"], bins=SALARY_BINS, labels=SALARY_LABELS
    )
    df_eng["AgeGroup"] = pd.cut(
        df_eng["Age"], bins=AGE_BINS, labels=AGE_LABELS
    )

    age_group_purchase = (
        df_eng.groupby("AgeGroup", observed=False)["Purchased"].mean() * 100
    ).round(1)

    salary_purchase = (
        df_eng.groupby("SalaryLevel", observed=False)["Purchased"].mean() * 100
    ).round(1)

    return {
        "total": total,
        "purchased": purchased,
        "not_purchased": not_purchased,
        "purchase_rate": purchase_rate,
        "male_purchase_rate": male_rate,
        "female_purchase_rate": female_rate,
        "age_group_purchase": age_group_purchase,
        "salary_purchase": salary_purchase,
        "avg_age": df["Age"].mean(),
        "avg_salary": df["EstimatedSalary"].mean(),
        "age_purchased_mean": df[df["Purchased"] == 1]["Age"].mean(),
        "age_not_purchased_mean": df[df["Purchased"] == 0]["Age"].mean(),
        "salary_purchased_mean": df[df["Purchased"] == 1]["EstimatedSalary"].mean(),
        "salary_not_purchased_mean": df[df["Purchased"] == 0]["EstimatedSalary"].mean(),
    }
