"""
Feature Engineering Script for Social Network Ads Classification
=================================================================
This script performs comprehensive feature engineering on the Social Network Ads
dataset to build a classification model that predicts whether users will purchase
a product after clicking on a social network ad.

Dataset columns:
    - User ID          : Unique identifier (dropped — not predictive)
    - Gender           : Male / Female  (categorical)
    - Age              : User age       (numerical)
    - EstimatedSalary  : Salary in USD  (numerical)
    - Purchased        : Target (0 = no purchase, 1 = purchase)

Usage:
    python feature_engineering.py
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

# Use non-interactive backend so the script runs without a display
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(os.path.dirname(__file__), "Social_Network_Ads.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
RANDOM_STATE = 42

os.makedirs(FIGURES_DIR, exist_ok=True)


# ===========================================================================
# 1. DATA LOADING & QUALITY CHECKS
# ===========================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load the Social Network Ads dataset and run basic quality checks."""
    df = pd.read_csv(path)
    print("=" * 65)
    print("1. DATA LOADING & QUALITY CHECKS")
    print("=" * 65)
    print(f"\nDataset shape : {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nColumn dtypes:\n", df.dtypes.to_string())
    print("\nFirst 5 rows:\n", df.head().to_string())

    # Missing values
    missing = df.isnull().sum()
    print("\nMissing values per column:\n", missing.to_string())

    # Duplicate rows
    n_dup = df.duplicated().sum()
    print(f"\nDuplicate rows : {n_dup}")

    # Class balance
    purchase_counts = df["Purchased"].value_counts()
    purchase_pct = df["Purchased"].value_counts(normalize=True) * 100
    print("\nTarget distribution:")
    for label in [0, 1]:
        print(f"  {label} → {purchase_counts[label]:>4d}  ({purchase_pct[label]:.1f} %)")

    print("\nNumerical statistics:\n", df.describe().to_string())
    return df


# ===========================================================================
# 2. CATEGORICAL ENCODING
# ===========================================================================

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the Gender column using Label Encoding and One-Hot Encoding."""
    print("\n" + "=" * 65)
    print("2. CATEGORICAL ENCODING")
    print("=" * 65)

    # Label Encoding: Male → 1, Female → 0
    le = LabelEncoder()
    df["Gender_Label"] = le.fit_transform(df["Gender"])
    print(f"\nLabel encoding mapping : {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # One-Hot Encoding (produces Gender_Male / Gender_Female)
    ohe = pd.get_dummies(df["Gender"], prefix="Gender")
    df = pd.concat([df, ohe], axis=1)
    print(f"One-Hot Encoding columns created : {list(ohe.columns)}")

    return df


# ===========================================================================
# 3. NUMERICAL FEATURE SCALING
# ===========================================================================

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply StandardScaler, MinMaxScaler, and RobustScaler to Age and EstimatedSalary."""
    print("\n" + "=" * 65)
    print("3. NUMERICAL FEATURE SCALING")
    print("=" * 65)

    num_cols = ["Age", "EstimatedSalary"]

    # Standard scaling (mean=0, std=1)
    scaler_std = StandardScaler()
    scaled_std = scaler_std.fit_transform(df[num_cols])
    df["Age_Standardized"] = scaled_std[:, 0]
    df["Salary_Standardized"] = scaled_std[:, 1]

    # Min-Max normalization → [0, 1]
    scaler_mm = MinMaxScaler()
    scaled_mm = scaler_mm.fit_transform(df[num_cols])
    df["Age_Normalized"] = scaled_mm[:, 0]
    df["Salary_Normalized"] = scaled_mm[:, 1]

    # Robust scaling (median / IQR — resistant to outliers)
    scaler_rb = RobustScaler()
    scaled_rb = scaler_rb.fit_transform(df[num_cols])
    df["Age_Robust"] = scaled_rb[:, 0]
    df["Salary_Robust"] = scaled_rb[:, 1]

    print("\nStandardized Age   : mean={:.4f}, std={:.4f}".format(
        df["Age_Standardized"].mean(), df["Age_Standardized"].std()))
    print("Standardized Salary: mean={:.4f}, std={:.4f}".format(
        df["Salary_Standardized"].mean(), df["Salary_Standardized"].std()))
    print("Normalized Age     : min={:.4f}, max={:.4f}".format(
        df["Age_Normalized"].min(), df["Age_Normalized"].max()))
    print("Normalized Salary  : min={:.4f}, max={:.4f}".format(
        df["Salary_Normalized"].min(), df["Salary_Normalized"].max()))

    return df


# ===========================================================================
# 4. FEATURE BINNING / BUCKETING
# ===========================================================================

def bin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create age groups and salary brackets."""
    print("\n" + "=" * 65)
    print("4. FEATURE BINNING / BUCKETING")
    print("=" * 65)

    # Age groups
    age_bins = [17, 25, 35, 45, 55, 100]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
    df["Age_Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)

    # Salary brackets (quantile-based → equal-frequency bins)
    df["Salary_Bracket"] = pd.qcut(
        df["EstimatedSalary"],
        q=3,
        labels=["Low", "Medium", "High"],
    )

    print("\nAge group distribution:\n", df["Age_Group"].value_counts().sort_index().to_string())
    print("\nSalary bracket distribution:\n", df["Salary_Bracket"].value_counts().sort_index().to_string())

    # Numeric codes for downstream use
    age_group_order = {"18-25": 0, "26-35": 1, "36-45": 2, "46-55": 3, "56+": 4}
    salary_bracket_order = {"Low": 0, "Medium": 1, "High": 2}
    df["Age_Group_Code"] = df["Age_Group"].map(age_group_order).astype(int)
    df["Salary_Bracket_Code"] = df["Salary_Bracket"].map(salary_bracket_order).astype(int)

    return df


# ===========================================================================
# 5. INTERACTION FEATURES
# ===========================================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction terms between Age, EstimatedSalary, Gender, and group columns."""
    print("\n" + "=" * 65)
    print("5. INTERACTION FEATURES")
    print("=" * 65)

    # Numerical interactions
    df["Age_x_Salary"] = df["Age"] * df["EstimatedSalary"]
    df["Age_div_Salary"] = df["Age"] / df["EstimatedSalary"]

    # Age group × Gender (string label for analysis)
    df["AgeGroup_x_Gender"] = (
        df["Age_Group"].astype(str) + "_" + df["Gender"]
    )

    # Salary bracket × Gender (string label for analysis)
    df["SalaryBracket_x_Gender"] = (
        df["Salary_Bracket"].astype(str) + "_" + df["Gender"]
    )

    # Numeric cross-feature
    df["AgeGroupCode_x_GenderLabel"] = (
        df["Age_Group_Code"] * df["Gender_Label"]
    )
    df["SalaryBracketCode_x_GenderLabel"] = (
        df["Salary_Bracket_Code"] * df["Gender_Label"]
    )

    print("\nNew interaction features created:")
    interaction_cols = [
        "Age_x_Salary", "Age_div_Salary",
        "AgeGroup_x_Gender", "SalaryBracket_x_Gender",
        "AgeGroupCode_x_GenderLabel", "SalaryBracketCode_x_GenderLabel",
    ]
    print(df[interaction_cols].describe(include="all").to_string())

    return df


# ===========================================================================
# 6. POLYNOMIAL FEATURES
# ===========================================================================

def create_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add quadratic and cubic terms for Age and EstimatedSalary."""
    print("\n" + "=" * 65)
    print("6. POLYNOMIAL FEATURES")
    print("=" * 65)

    df["Age_Sq"] = df["Age"] ** 2
    df["Age_Cu"] = df["Age"] ** 3
    df["Salary_Sq"] = df["EstimatedSalary"] ** 2
    df["Salary_Cu"] = df["EstimatedSalary"] ** 3

    poly_cols = ["Age_Sq", "Age_Cu", "Salary_Sq", "Salary_Cu"]
    print("\nPolynomial features sample (first 5 rows):\n",
          df[poly_cols].head().to_string())

    return df


# ===========================================================================
# 7. STATISTICAL FEATURES
# ===========================================================================

def create_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute group-level statistical aggregations and merge them back."""
    print("\n" + "=" * 65)
    print("7. STATISTICAL FEATURES")
    print("=" * 65)

    # --- Age statistics per gender ---
    age_gender_stats = df.groupby("Gender")["Age"].agg(["mean", "std"]).rename(
        columns={"mean": "Age_Mean_by_Gender", "std": "Age_Std_by_Gender"}
    )
    df = df.merge(age_gender_stats, on="Gender", how="left")

    # --- Salary statistics per gender ---
    salary_gender_stats = df.groupby("Gender")["EstimatedSalary"].agg(["mean", "std"]).rename(
        columns={"mean": "Salary_Mean_by_Gender", "std": "Salary_Std_by_Gender"}
    )
    df = df.merge(salary_gender_stats, on="Gender", how="left")

    # --- Purchase rate by age group ---
    purchase_rate_age = (
        df.groupby("Age_Group")["Purchased"]
        .mean()
        .rename("Purchase_Rate_AgeGroup")
    )
    df = df.merge(purchase_rate_age, on="Age_Group", how="left")

    # --- Purchase rate by salary bracket ---
    purchase_rate_salary = (
        df.groupby("Salary_Bracket")["Purchased"]
        .mean()
        .rename("Purchase_Rate_SalaryBracket")
    )
    df = df.merge(purchase_rate_salary, on="Salary_Bracket", how="left")

    # Deviation from group mean (how far is this user from their gender average?)
    df["Age_Dev_from_Gender_Mean"] = df["Age"] - df["Age_Mean_by_Gender"]
    df["Salary_Dev_from_Gender_Mean"] = (
        df["EstimatedSalary"] - df["Salary_Mean_by_Gender"]
    )

    print("\nAge statistics by gender:\n", age_gender_stats.to_string())
    print("\nSalary statistics by gender:\n", salary_gender_stats.to_string())
    print("\nPurchase rate by age group:\n", purchase_rate_age.to_string())
    print("\nPurchase rate by salary bracket:\n", purchase_rate_salary.to_string())

    return df


# ===========================================================================
# 8. OUTLIER DETECTION & HANDLING
# ===========================================================================

def detect_and_handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Identify outliers via Z-score and IQR, then document the handling strategy."""
    print("\n" + "=" * 65)
    print("8. OUTLIER DETECTION & HANDLING")
    print("=" * 65)

    num_cols = ["Age", "EstimatedSalary"]

    # --- Z-score method ---
    print("\n--- Z-score method (threshold = 3) ---")
    z_scores = np.abs(stats.zscore(df[num_cols]))
    z_outlier_mask = (z_scores > 3).any(axis=1)
    print(f"Outliers detected (Z-score) : {z_outlier_mask.sum()}")
    if z_outlier_mask.sum() > 0:
        print(df[z_outlier_mask][["Age", "EstimatedSalary", "Purchased"]].to_string())

    # Flag column
    df["Is_Outlier_Zscore"] = z_outlier_mask.astype(int)

    # --- IQR method ---
    print("\n--- IQR method ---")
    iqr_mask = pd.Series(False, index=df.index)
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        col_mask = (df[col] < lower) | (df[col] > upper)
        print(f"  {col}: Q1={q1:.0f}, Q3={q3:.0f}, IQR={iqr:.0f}, "
              f"bounds=[{lower:.0f}, {upper:.0f}], outliers={col_mask.sum()}")
        iqr_mask = iqr_mask | col_mask

    print(f"Outliers detected (IQR)    : {iqr_mask.sum()}")
    df["Is_Outlier_IQR"] = iqr_mask.astype(int)

    # --- Handling strategy ---
    print("\nOutlier handling strategy:")
    print("  • Flag outliers with binary indicator columns (Is_Outlier_Zscore, Is_Outlier_IQR).")
    print("  • Keep outlier rows in the dataset — the dataset is small (400 rows).")
    print("  • For tree-based models (Random Forest, XGBoost) outliers have minimal impact.")
    print("  • For distance-based models (KNN, SVM) use Robust-scaled features.")
    print("  • Use RobustScaler columns (Age_Robust, Salary_Robust) for sensitive algorithms.")

    return df


# ===========================================================================
# 9. FEATURE SELECTION
# ===========================================================================

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove User ID, low-variance features, and analyse correlations."""
    print("\n" + "=" * 65)
    print("9. FEATURE SELECTION")
    print("=" * 65)

    # Drop User ID — not predictive
    if "User ID" in df.columns:
        df = df.drop(columns=["User ID"])
        print("\nDropped: User ID")

    # Drop string / categorical columns that are not encoded yet
    drop_str_cols = [
        col for col in ["Gender", "Age_Group", "Salary_Bracket",
                        "AgeGroup_x_Gender", "SalaryBracket_x_Gender"]
        if col in df.columns
    ]
    df = df.drop(columns=drop_str_cols)
    print(f"Dropped string/categorical label columns: {drop_str_cols}")

    # Low-variance filter on numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(numeric_df)
    low_var_cols = numeric_df.columns[~selector.get_support()].tolist()
    if low_var_cols:
        print(f"\nLow-variance columns removed: {low_var_cols}")
        df = df.drop(columns=low_var_cols)
    else:
        print("\nNo low-variance columns found.")

    # Correlation with target
    numeric_df = df.select_dtypes(include=[np.number])
    corr_with_target = (
        numeric_df.corr()["Purchased"]
        .drop("Purchased")
        .abs()
        .sort_values(ascending=False)
    )
    print("\nFeature correlations with target (absolute value, top 20):")
    print(corr_with_target.head(20).to_string())

    return df


# ===========================================================================
# 10. VISUALIZATIONS
# ===========================================================================

def visualize(df_original: pd.DataFrame, df_engineered: pd.DataFrame) -> None:
    """Generate and save key visualisation plots."""
    print("\n" + "=" * 65)
    print("10. VISUALIZATIONS")
    print("=" * 65)

    # ── Figure 1: Original distributions ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Original Feature Distributions", fontsize=14, fontweight="bold")

    axes[0].hist(df_original["Age"], bins=20, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")

    axes[1].hist(df_original["EstimatedSalary"], bins=20, color="#55A868", edgecolor="white")
    axes[1].set_title("EstimatedSalary Distribution")
    axes[1].set_xlabel("Estimated Salary (USD)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "01_original_distributions.png"), dpi=150)
    plt.close(fig)

    # ── Figure 2: Scatter – Age vs Salary coloured by Purchased ─────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, marker in [(0, "#4C72B0", "o"), (1, "#C44E52", "^")]:
        subset = df_original[df_original["Purchased"] == label]
        ax.scatter(subset["Age"], subset["EstimatedSalary"],
                   c=color, marker=marker, alpha=0.6,
                   label=f"Purchased={label}", s=30)
    ax.set_title("Age vs Estimated Salary (colored by Purchased)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Estimated Salary (USD)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "02_age_vs_salary_scatter.png"), dpi=150)
    plt.close(fig)

    # ── Figure 3: Purchase rate by Age Group ────────────────────────────────
    if "Age_Group" in df_original.columns:
        age_purchase = (
            df_original.groupby("Age_Group")["Purchased"].mean() * 100
        ).reset_index()
        age_purchase.columns = ["Age_Group", "Purchase_Rate_%"]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(age_purchase["Age_Group"].astype(str),
                      age_purchase["Purchase_Rate_%"],
                      color=sns.color_palette("Blues_d", len(age_purchase)))
        ax.set_title("Purchase Rate by Age Group")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Purchase Rate (%)")
        for bar, val in zip(bars, age_purchase["Purchase_Rate_%"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "03_purchase_rate_by_age_group.png"), dpi=150)
        plt.close(fig)

    # ── Figure 4: Purchase rate by Salary Bracket ────────────────────────────
    if "Salary_Bracket" in df_original.columns:
        sal_purchase = (
            df_original.groupby("Salary_Bracket")["Purchased"].mean() * 100
        ).reset_index()
        sal_purchase.columns = ["Salary_Bracket", "Purchase_Rate_%"]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(sal_purchase["Salary_Bracket"].astype(str),
                      sal_purchase["Purchase_Rate_%"],
                      color=sns.color_palette("Greens_d", len(sal_purchase)))
        ax.set_title("Purchase Rate by Salary Bracket")
        ax.set_xlabel("Salary Bracket")
        ax.set_ylabel("Purchase Rate (%)")
        for bar, val in zip(bars, sal_purchase["Purchase_Rate_%"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "04_purchase_rate_by_salary_bracket.png"), dpi=150)
        plt.close(fig)

    # ── Figure 5: Correlation heat-map (engineered features) ─────────────────
    numeric_df = df_engineered.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=False, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=0.3, ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Correlation Matrix — Engineered Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "05_correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    # ── Figure 6: Feature importance via RandomForest ────────────────────────
    numeric_df = df_engineered.select_dtypes(include=[np.number])
    if "Purchased" in numeric_df.columns:
        X = numeric_df.drop(columns=["Purchased"])
        y = numeric_df["Purchased"]
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        rf.fit(X, y)
        importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
        top_n = 20
        importance_top = importance.tail(top_n)

        fig, ax = plt.subplots(figsize=(8, 8))
        importance_top.plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "06_feature_importance.png"), dpi=150)
        plt.close(fig)

    # ── Figure 7: Gender breakdown of purchase rate ──────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    gender_purchase = (
        df_original.groupby("Gender")["Purchased"].mean() * 100
    ).reset_index()
    gender_purchase.columns = ["Gender", "Purchase_Rate_%"]
    bars = ax.bar(gender_purchase["Gender"],
                  gender_purchase["Purchase_Rate_%"],
                  color=["#4C72B0", "#C44E52"])
    ax.set_title("Purchase Rate by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Purchase Rate (%)")
    for bar, val in zip(bars, gender_purchase["Purchase_Rate_%"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "07_purchase_rate_by_gender.png"), dpi=150)
    plt.close(fig)

    print(f"\nAll plots saved to: {FIGURES_DIR}/")


# ===========================================================================
# 11. BEFORE / AFTER COMPARISON
# ===========================================================================

def before_after_comparison(df_original: pd.DataFrame, df_final: pd.DataFrame) -> None:
    """Print a summary comparing the original and engineered feature sets."""
    print("\n" + "=" * 65)
    print("11. BEFORE / AFTER COMPARISON")
    print("=" * 65)

    orig_features = [c for c in df_original.columns if c != "Purchased"]
    final_features = [c for c in df_final.columns if c != "Purchased"]

    print(f"\nOriginal features ({len(orig_features)}): {orig_features}")
    print(f"\nEngineered features ({len(final_features)}):")
    for f in sorted(final_features):
        print(f"  {f}")

    print(f"\nFeatures added : {len(final_features) - len(orig_features)}")


# ===========================================================================
# 12. MODEL SELECTION RECOMMENDATIONS
# ===========================================================================

def model_recommendations() -> None:
    """Print model selection recommendations based on the engineered feature set."""
    print("\n" + "=" * 65)
    print("12. MODEL SELECTION RECOMMENDATIONS")
    print("=" * 65)

    recommendations = """
Based on the engineered features, the following models are recommended:

┌──────────────────────────────┬─────────────────────────────────────────────────┐
│ Model                        │ Why suitable                                    │
├──────────────────────────────┼─────────────────────────────────────────────────┤
│ Logistic Regression          │ Good baseline; benefits from standardised /     │
│                              │ normalised features (Age_Standardized,           │
│                              │ Salary_Standardized) and polynomial terms.      │
├──────────────────────────────┼─────────────────────────────────────────────────┤
│ Random Forest                │ Handles mixed feature types, non-linear          │
│                              │ relationships, and outliers robustly.           │
│                              │ Directly outputs feature importance.            │
├──────────────────────────────┼─────────────────────────────────────────────────┤
│ Gradient Boosting /          │ Strong performance on tabular data; handles     │
│ XGBoost / LightGBM           │ interactions and non-linearities natively.      │
├──────────────────────────────┼─────────────────────────────────────────────────┤
│ Support Vector Machine (SVM) │ Effective in moderate-sized datasets;           │
│                              │ use RobustScaler columns for best results.      │
├──────────────────────────────┼─────────────────────────────────────────────────┤
│ K-Nearest Neighbours (KNN)   │ Simple and effective; use Normalized or         │
│                              │ Robust-scaled features to ensure equal weight.  │
└──────────────────────────────┴─────────────────────────────────────────────────┘

Recommended feature subsets per model type:
  • Linear models   → Age_Standardized, Salary_Standardized, Gender_Label,
                       Age_Sq, Age_Cu, Salary_Sq, Salary_Cu, Age_x_Salary
  • Tree-based      → Age, EstimatedSalary, Gender_Label, Age_Group_Code,
                       Salary_Bracket_Code, all interaction features
  • Distance-based  → Age_Normalized, Salary_Normalized, Gender_Label,
                       or Age_Robust, Salary_Robust

Tips:
  • Always evaluate with cross-validation (StratifiedKFold recommended).
  • Use class_weight='balanced' if the class imbalance is significant.
  • Consider SMOTE for oversampling the minority class.
  • Tune hyperparameters with GridSearchCV or Optuna.
"""
    print(recommendations)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """Run the full feature engineering pipeline."""
    # --- Load raw data ---
    df = load_data(DATASET_PATH)
    df_original = df.copy()

    # --- Step 2: Categorical encoding ---
    df = encode_categorical(df)

    # --- Step 3: Scaling ---
    df = scale_features(df)

    # --- Step 4: Binning ---
    df = bin_features(df)

    # --- Step 5: Interaction features ---
    df = create_interaction_features(df)

    # --- Step 6: Polynomial features ---
    df = create_polynomial_features(df)

    # --- Step 7: Statistical features ---
    df_with_bins = df.copy()          # keep Age_Group / Salary_Bracket for stats
    df = create_statistical_features(df)

    # --- Step 8: Outlier detection ---
    df = detect_and_handle_outliers(df)

    # --- Visualise (pass df with categorical columns still present) ---
    visualize(df_with_bins, df.select_dtypes(include=[np.number, "category"]))

    # --- Step 9: Feature selection (cleans up string columns) ---
    df_final = select_features(df.copy())

    # --- Before/After comparison ---
    before_after_comparison(df_original, df_final)

    # --- Model recommendations ---
    model_recommendations()

    # --- Save engineered dataset ---
    output_path = os.path.join(os.path.dirname(__file__), "Social_Network_Ads_Engineered.csv")
    df_final.to_csv(output_path, index=False)
    print(f"\nEngineered dataset saved to: {output_path}")
    print(f"Final dataset shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print("\nFeature engineering pipeline complete ✓")


if __name__ == "__main__":
    main()
