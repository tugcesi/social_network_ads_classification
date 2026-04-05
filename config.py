"""
Configuration constants for the Social Network Ads Classification Streamlit app.
"""

# ── File paths ────────────────────────────────────────────────────────────────
MODEL_PKL_PATH = "Purchased.pkl"
MODEL_JOBLIB_PATH = "Purchased.joblib"
DATASET_PATH = "Social_Network_Ads.csv"

# ── Feature engineering bins ─────────────────────────────────────────────────
SALARY_BINS = [15000, 43000, 70000, 88000, 150000]
SALARY_LABELS = [1, 2, 3, 4]
SALARY_LEVEL_NAMES = {
    1: "Düşük (15K–43K)",
    2: "Orta-Düşük (43K–70K)",
    3: "Orta-Yüksek (70K–88K)",
    4: "Yüksek (88K–150K)",
}

AGE_BINS = [18, 30, 40, 50, 60]
AGE_LABELS = [1, 2, 3, 4]
AGE_GROUP_NAMES = {
    1: "Genç (18–30)",
    2: "Orta Yaş (30–40)",
    3: "Olgun (40–50)",
    4: "Kıdemli (50–60)",
}

YOUNG_RICH_AGE_THRESHOLD = 30
YOUNG_RICH_SALARY_THRESHOLD = 50000

# ── Model features (must match training order) ───────────────────────────────
MODEL_FEATURES = [
    "Gender",
    "Age",
    "EstimatedSalary",
    "IsYoungRich",
    "SalaryLevel_2",
    "SalaryLevel_3",
    "SalaryLevel_4",
    "AgeGroup_2",
    "AgeGroup_3",
    "AgeGroup_4",
]

# ── UI input bounds ───────────────────────────────────────────────────────────
AGE_MIN = 18
AGE_MAX = 60
AGE_DEFAULT = 30

SALARY_MIN = 15000
SALARY_MAX = 150000
SALARY_DEFAULT = 50000
SALARY_STEP = 1000

# ── App metadata ─────────────────────────────────────────────────────────────
APP_TITLE = "Social Network Ads Classifier"
APP_ICON = "📊"
APP_DESCRIPTION = (
    "Sosyal medya reklamlarına tıklayan kullanıcıların ürün satın alıp "
    "almayacağını tahmin eden makine öğrenmesi uygulaması."
)

# ── Colour palette ────────────────────────────────────────────────────────────
COLOR_POSITIVE = "#2ecc71"   # purchase → yes
COLOR_NEGATIVE = "#e74c3c"   # purchase → no
COLOR_PRIMARY = "#3498db"
COLOR_SECONDARY = "#9b59b6"
COLOR_ACCENT = "#f39c12"

# ── Gender display labels ─────────────────────────────────────────────────────
GENDER_LABELS = {"Female": "Kadın", "Male": "Erkek"}
