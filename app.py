"""
Social Network Ads Classification – Streamlit Web Application
=============================================================
Four pages:
  1. 🏠 Ana Sayfa     – Dashboard, overview, statistics
  2. 🔮 Tahmin        – Interactive prediction form
  3. 📊 Veri Analizi  – Dataset exploration and visualisations
  4. 🤖 Model Bilgisi – Model details and performance metrics
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from config import (
    AGE_BINS,
    AGE_DEFAULT,
    AGE_GROUP_NAMES,
    AGE_LABELS,
    AGE_MAX,
    AGE_MIN,
    APP_DESCRIPTION,
    APP_ICON,
    APP_TITLE,
    COLOR_ACCENT,
    COLOR_NEGATIVE,
    COLOR_POSITIVE,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    GENDER_LABELS,
    SALARY_BINS,
    SALARY_DEFAULT,
    SALARY_LABELS,
    SALARY_LEVEL_NAMES,
    SALARY_MAX,
    SALARY_MIN,
    SALARY_STEP,
    YOUNG_RICH_AGE_THRESHOLD,
    YOUNG_RICH_SALARY_THRESHOLD,
)
from utils import (
    compute_dataset_stats,
    engineer_features,
    load_dataset,
    load_model,
    predict_single,
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.9rem; }

    .result-box-yes {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 1.5rem; border-radius: 14px;
        color: white; text-align: center;
        box-shadow: 0 6px 20px rgba(56,239,125,0.35);
    }
    .result-box-no {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        padding: 1.5rem; border-radius: 14px;
        color: white; text-align: center;
        box-shadow: 0 6px 20px rgba(231,76,60,0.35);
    }
    .result-box-yes h1, .result-box-no h1 { margin: 0; font-size: 2.2rem; }
    .result-box-yes p,  .result-box-no p  { margin: 0.4rem 0 0; font-size: 1.1rem; opacity: 0.92; }

    .sidebar-info {
        background: rgba(255,255,255,0.07);
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.82rem; color: #ccc; margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("---")
    page = st.radio(
        "Sayfa Seçin",
        options=["🏠 Ana Sayfa", "🔮 Tahmin", "📊 Veri Analizi", "🤖 Model Bilgisi"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="sidebar-info">'
        "Model: BernoulliNB<br>"
        "Veri Seti: Social Network Ads<br>"
        "Gözlem Sayısı: 400"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Load shared resources ─────────────────────────────────────────────────────
df_raw = load_dataset()
model, features = load_model()
stats = compute_dataset_stats(df_raw)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME / DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Ana Sayfa":
    st.title("📊 Social Network Ads – Sınıflandırma Uygulaması")
    st.markdown(f"**{APP_DESCRIPTION}**")
    st.markdown("---")

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h2>{stats["total"]}</h2>'
            "<p>Toplam Kullanıcı</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><h2>{stats["purchased"]}</h2>'
            "<p>Satın Alan</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><h2>{stats["not_purchased"]}</h2>'
            "<p>Satın Almayan</p></div>",
            unsafe_allow_html=True,
        )
    with c4:
        pct = f"{stats['purchase_rate']*100:.1f}%"
        st.markdown(
            f'<div class="metric-card"><h2>{pct}</h2>'
            "<p>Satın Alma Oranı</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### ")

    # ── Charts ───────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sınıf Dağılımı")
        labels = ["Satın Almadı", "Satın Aldı"]
        sizes = [stats["not_purchased"], stats["purchased"]]
        colors = [COLOR_NEGATIVE, COLOR_POSITIVE]

        fig = plt.figure(figsize=(5, 4))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        plt.title("Satın Alma Dağılımı", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Cinsiyete Göre Satın Alma Oranı")
        genders = ["Erkek", "Kadın"]
        rates = [
            stats["male_purchase_rate"] * 100,
            stats["female_purchase_rate"] * 100,
        ]
        bar_colors = [COLOR_PRIMARY, COLOR_SECONDARY]

        fig = plt.figure(figsize=(5, 4))
        bars = plt.bar(genders, rates, color=bar_colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, rates):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )
        plt.ylabel("Satın Alma Oranı (%)", fontsize=11)
        plt.title("Cinsiyete Göre Satın Alma", fontsize=13, fontweight="bold")
        plt.ylim(0, max(rates) + 12)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Model accuracy summary ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Model Özeti")

    df_eng = engineer_features(df_raw.copy())
    y = df_raw["Purchased"]
    _, X_test, _, y_test = train_test_split(
        df_eng, y, test_size=0.20, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Doğruluk (Accuracy)", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
    m2.metric("Kesinlik (Precision)", f"{precision_score(y_test, y_pred)*100:.1f}%")
    m3.metric("Duyarlılık (Recall)", f"{recall_score(y_test, y_pred)*100:.1f}%")
    m4.metric("F1-Score", f"{f1_score(y_test, y_pred)*100:.1f}%")

    st.info(
        "Model: **BernoulliNB (Bernoulli Naive Bayes)** – "
        "Veri setindeki en iyi performansı sergileyen makine öğrenmesi algoritması."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 – PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Tahmin":
    st.title("🔮 Satın Alma Tahmini")
    st.markdown(
        "Kullanıcı bilgilerini girerek modelin satın alma tahminini görüntüleyin."
    )
    st.markdown("---")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("Kullanıcı Bilgileri")
        with st.form("prediction_form"):
            gender = st.selectbox("Cinsiyet", options=["Female", "Male"],
                                  format_func=lambda x: GENDER_LABELS[x])
            age = st.slider("Yaş", min_value=AGE_MIN, max_value=AGE_MAX,
                            value=AGE_DEFAULT, step=1)
            salary = st.slider(
                "Tahmini Maaş ($)",
                min_value=SALARY_MIN, max_value=SALARY_MAX,
                value=SALARY_DEFAULT, step=SALARY_STEP,
                format="$%d",
            )
            submitted = st.form_submit_button("🔍 Tahmin Et", use_container_width=True)

        # Feature info
        st.markdown("#### Hesaplanan Özellikler")
        salary_level = pd.cut(
            [salary], bins=SALARY_BINS, labels=SALARY_LABELS
        )[0]
        age_group = pd.cut([age], bins=AGE_BINS, labels=AGE_LABELS)[0]
        is_young_rich = int(age < YOUNG_RICH_AGE_THRESHOLD and salary > YOUNG_RICH_SALARY_THRESHOLD)

        feat_data = {
            "Özellik": ["Yaş Grubu", "Maaş Seviyesi", "Genç-Zengin"],
            "Değer": [
                AGE_GROUP_NAMES.get(int(age_group), "—") if not pd.isna(age_group) else "—",
                SALARY_LEVEL_NAMES.get(int(salary_level), "—") if not pd.isna(salary_level) else "—",
                "✅ Evet" if is_young_rich else "❌ Hayır",
            ],
        }
        st.dataframe(pd.DataFrame(feat_data), hide_index=True, use_container_width=True)

    with col_result:
        st.subheader("Tahmin Sonucu")

        if submitted:
            try:
                pred, prob_yes, probs = predict_single(model, gender, age, salary)

                if pred == 1:
                    st.markdown(
                        '<div class="result-box-yes">'
                        "<h1>✅ Satın Alacak</h1>"
                        f"<p>Güven Skoru: <strong>{prob_yes*100:.1f}%</strong></p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-box-no">'
                        "<h1>❌ Satın Almayacak</h1>"
                        f"<p>Güven Skoru: <strong>{(1-prob_yes)*100:.1f}%</strong></p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Olasılık Dağılımı")
                fig = plt.figure(figsize=(6, 3))
                categories = ["Satın Almayacak", "Satın Alacak"]
                bar_colors = [COLOR_NEGATIVE, COLOR_POSITIVE]
                bars = plt.barh(
                    categories, [probs[0] * 100, probs[1] * 100],
                    color=bar_colors, edgecolor="white", height=0.5,
                )
                for bar, val in zip(bars, [probs[0] * 100, probs[1] * 100]):
                    plt.text(
                        val + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontweight="bold", fontsize=11,
                    )
                plt.xlim(0, 115)
                plt.xlabel("Olasılık (%)", fontsize=11)
                plt.title("Model Olasılıkları", fontsize=13, fontweight="bold")
                plt.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Summary
                st.markdown("#### Özet")
                summary = pd.DataFrame(
                    {
                        "Parametre": ["Cinsiyet", "Yaş", "Maaş", "Tahmin", "Güven"],
                        "Değer": [
                            "Erkek" if gender == "Male" else "Kadın",
                            age,
                            f"${salary:,}",
                            "Satın Alacak" if pred == 1 else "Satın Almayacak",
                            f"{max(probs)*100:.1f}%",
                        ],
                    }
                )
                st.dataframe(summary, hide_index=True, use_container_width=True)

            except Exception as exc:
                st.error(f"Tahmin sırasında hata oluştu: {exc}")
        else:
            st.info("👈 Sol taraftaki formu doldurup **Tahmin Et** butonuna tıklayın.")
            # Placeholder visual
            fig = plt.figure(figsize=(6, 3))
            plt.barh(
                ["Satın Almayacak", "Satın Alacak"],
                [50, 50],
                color=["#bdc3c7", "#bdc3c7"],
                edgecolor="white",
                height=0.5,
            )
            plt.xlim(0, 115)
            plt.xlabel("Olasılık (%)", fontsize=11)
            plt.title("Model Olasılıkları (Bekleniyor…)", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 – DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Veri Analizi":
    st.title("📊 Veri Analizi")
    st.markdown("---")

    # ── Basic stats ───────────────────────────────────────────────────────────
    st.subheader("Veri Seti İstatistikleri")
    st.dataframe(
        df_raw[["Age", "EstimatedSalary", "Purchased"]].describe().round(2).T,
        use_container_width=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Dağılımlar", "Korelasyon", "Yaş & Maaş Analizi", "Demografik Analiz"]
    )

    # ── Tab 1: Distributions ─────────────────────────────────────────────────
    with tab1:
        st.subheader("Özellik Dağılımları")
        col1, col2 = st.columns(2)

        with col1:
            fig = plt.figure(figsize=(5.5, 4))
            colors_dist = [COLOR_NEGATIVE, COLOR_POSITIVE]
            for val, color, label in zip(
                [0, 1], colors_dist, ["Satın Almadı", "Satın Aldı"]
            ):
                subset = df_raw[df_raw["Purchased"] == val]["Age"]
                plt.hist(
                    subset, bins=20, alpha=0.65, color=color,
                    edgecolor="white", label=label,
                )
            plt.xlabel("Yaş", fontsize=11)
            plt.ylabel("Frekans", fontsize=11)
            plt.title("Yaş Dağılımı (Satın Alma Durumuna Göre)", fontsize=12, fontweight="bold")
            plt.legend()
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            fig = plt.figure(figsize=(5.5, 4))
            for val, color, label in zip(
                [0, 1], colors_dist, ["Satın Almadı", "Satın Aldı"]
            ):
                subset = df_raw[df_raw["Purchased"] == val]["EstimatedSalary"]
                plt.hist(
                    subset, bins=20, alpha=0.65, color=color,
                    edgecolor="white", label=label,
                )
            plt.xlabel("Tahmini Maaş ($)", fontsize=11)
            plt.ylabel("Frekans", fontsize=11)
            plt.title("Maaş Dağılımı (Satın Alma Durumuna Göre)", fontsize=12, fontweight="bold")
            plt.legend()
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Gender distribution
        st.subheader("Cinsiyet Dağılımı")
        col3, col4 = st.columns(2)

        gender_counts = df_raw["Gender"].value_counts()
        with col3:
            fig = plt.figure(figsize=(4.5, 4))
            plt.bar(
                ["Erkek", "Kadın"],
                [gender_counts.get("Male", 0), gender_counts.get("Female", 0)],
                color=[COLOR_PRIMARY, COLOR_SECONDARY],
                edgecolor="white",
                width=0.5,
            )
            plt.ylabel("Kullanıcı Sayısı", fontsize=11)
            plt.title("Cinsiyete Göre Dağılım", fontsize=12, fontweight="bold")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col4:
            purchased_by_gender = df_raw.groupby("Gender")["Purchased"].sum()
            fig = plt.figure(figsize=(4.5, 4))
            plt.bar(
                ["Erkek", "Kadın"],
                [
                    purchased_by_gender.get("Male", 0),
                    purchased_by_gender.get("Female", 0),
                ],
                color=[COLOR_PRIMARY, COLOR_SECONDARY],
                edgecolor="white",
                width=0.5,
            )
            plt.ylabel("Satın Alan Sayısı", fontsize=11)
            plt.title("Cinsiyete Göre Satın Alan", fontsize=12, fontweight="bold")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Tab 2: Correlation ────────────────────────────────────────────────────
    with tab2:
        st.subheader("Korelasyon Analizi")
        df_corr = df_raw.copy()
        df_corr["Gender"] = df_corr["Gender"].map({"Female": 0, "Male": 1})
        corr_matrix = df_corr[["Gender", "Age", "EstimatedSalary", "Purchased"]].corr()

        fig = plt.figure(figsize=(7, 5.5))
        cols = corr_matrix.columns
        n = len(cols)
        im = plt.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(n), cols, rotation=30, ha="right", fontsize=11)
        plt.yticks(range(n), cols, fontsize=11)
        for i in range(n):
            for j in range(n):
                plt.text(
                    j, i,
                    f"{corr_matrix.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if abs(corr_matrix.values[i, j]) > 0.5 else "black",
                )
        plt.title("Korelasyon Matrisi", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            "**Önemli bulgular:**\n"
            "- Yaş ve Satın Alma arasında **pozitif** korelasyon (yaş arttıkça satın alma artıyor).\n"
            "- Maaş ve Satın Alma arasında **pozitif** korelasyon.\n"
            "- Cinsiyet ile Satın Alma arasındaki korelasyon zayıf."
        )

    # ── Tab 3: Age & Salary Analysis ─────────────────────────────────────────
    with tab3:
        st.subheader("Yaş Grubu & Maaş Seviyesi Analizi")

        df_bins = df_raw.copy()
        df_bins["AgeGroup"] = pd.cut(
            df_bins["Age"], bins=AGE_BINS,
            labels=["18-30", "30-40", "40-50", "50-60"],
        )
        df_bins["SalaryLevel"] = pd.cut(
            df_bins["EstimatedSalary"],
            bins=SALARY_BINS,
            labels=["Düşük", "Orta-Düşük", "Orta-Yüksek", "Yüksek"],
        )

        col1, col2 = st.columns(2)

        with col1:
            age_purchase = (
                df_bins.groupby("AgeGroup", observed=False)["Purchased"]
                .mean() * 100
            )
            fig = plt.figure(figsize=(5.5, 4))
            bar_col = [COLOR_PRIMARY] * len(age_purchase)
            bars = plt.bar(
                age_purchase.index.astype(str),
                age_purchase.values,
                color=bar_col,
                edgecolor="white",
                width=0.6,
            )
            for bar, val in zip(bars, age_purchase.values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontweight="bold", fontsize=10,
                )
            plt.xlabel("Yaş Grubu", fontsize=11)
            plt.ylabel("Satın Alma Oranı (%)", fontsize=11)
            plt.title("Yaş Grubuna Göre Satın Alma Oranı", fontsize=12, fontweight="bold")
            plt.ylim(0, max(age_purchase.values) + 15)
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            salary_purchase = (
                df_bins.groupby("SalaryLevel", observed=False)["Purchased"]
                .mean() * 100
            )
            fig = plt.figure(figsize=(5.5, 4))
            bars = plt.bar(
                salary_purchase.index.astype(str),
                salary_purchase.values,
                color=COLOR_ACCENT,
                edgecolor="white",
                width=0.6,
            )
            for bar, val in zip(bars, salary_purchase.values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontweight="bold", fontsize=10,
                )
            plt.xlabel("Maaş Seviyesi", fontsize=11)
            plt.ylabel("Satın Alma Oranı (%)", fontsize=11)
            plt.title("Maaş Seviyesine Göre Satın Alma Oranı", fontsize=12, fontweight="bold")
            plt.ylim(0, max(salary_purchase.values) + 15)
            plt.xticks(rotation=15, ha="right")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Young-Rich flag
        st.subheader("Genç-Zengin Segmenti")
        yr_counts = df_raw.copy()
        yr_counts["IsYoungRich"] = (
            (yr_counts["Age"] < YOUNG_RICH_AGE_THRESHOLD)
            & (yr_counts["EstimatedSalary"] > YOUNG_RICH_SALARY_THRESHOLD)
        ).astype(int)
        yr_purchase = yr_counts.groupby("IsYoungRich")["Purchased"].mean() * 100

        col3, col4 = st.columns(2)
        with col3:
            labels_yr = ["Normal", "Genç-Zengin"]
            values_yr = [yr_purchase.get(0, 0), yr_purchase.get(1, 0)]
            fig = plt.figure(figsize=(4.5, 3.5))
            bars = plt.bar(labels_yr, values_yr, color=[COLOR_PRIMARY, COLOR_ACCENT], edgecolor="white", width=0.5)
            for bar, val in zip(bars, values_yr):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontweight="bold", fontsize=11,
                )
            plt.ylabel("Satın Alma Oranı (%)", fontsize=11)
            plt.title("Genç-Zengin Satın Alma Oranı", fontsize=12, fontweight="bold")
            plt.ylim(0, max(values_yr) + 15)
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col4:
            yr_n = yr_counts["IsYoungRich"].value_counts()
            fig = plt.figure(figsize=(4.5, 3.5))
            plt.pie(
                [yr_n.get(0, 0), yr_n.get(1, 0)],
                labels=["Normal", "Genç-Zengin"],
                colors=[COLOR_PRIMARY, COLOR_ACCENT],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"edgecolor": "white"},
            )
            plt.title("Genç-Zengin Segment Dağılımı", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Tab 4: Demographics ───────────────────────────────────────────────────
    with tab4:
        st.subheader("Demografik Satın Alma Analizi")

        col1, col2 = st.columns(2)

        with col1:
            # Scatter: Age vs Salary coloured by Purchased
            df_scatter = df_raw.copy()
            bought = df_scatter[df_scatter["Purchased"] == 1]
            not_bought = df_scatter[df_scatter["Purchased"] == 0]

            fig = plt.figure(figsize=(5.5, 4.5))
            plt.scatter(
                not_bought["Age"], not_bought["EstimatedSalary"],
                color=COLOR_NEGATIVE, alpha=0.5, s=30, label="Satın Almadı",
            )
            plt.scatter(
                bought["Age"], bought["EstimatedSalary"],
                color=COLOR_POSITIVE, alpha=0.7, s=30, label="Satın Aldı",
            )
            plt.xlabel("Yaş", fontsize=11)
            plt.ylabel("Tahmini Maaş ($)", fontsize=11)
            plt.title("Yaş vs Maaş (Satın Alma)", fontsize=12, fontweight="bold")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            # Mean age and salary comparison
            categories = ["Satın Almadı\n(Yaş)", "Satın Aldı\n(Yaş)"]
            values = [
                stats["age_not_purchased_mean"],
                stats["age_purchased_mean"],
            ]
            fig = plt.figure(figsize=(5.5, 4.5))
            bars = plt.bar(categories, values, color=[COLOR_NEGATIVE, COLOR_POSITIVE],
                           edgecolor="white", width=0.5)
            for bar, val in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontweight="bold", fontsize=11,
                )
            plt.ylabel("Ortalama Yaş", fontsize=11)
            plt.title("Ortalama Yaş Karşılaştırması", fontsize=12, fontweight="bold")
            plt.ylim(0, max(values) + 6)
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Raw data preview
        st.subheader("Ham Veri Önizlemesi")
        st.dataframe(df_raw.head(20), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 – MODEL INFORMATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Bilgisi":
    st.title("🤖 Model Bilgisi")
    st.markdown("---")

    col_info, col_perf = st.columns([1, 1], gap="large")

    with col_info:
        st.subheader("Model Türü & Parametreler")
        model_params = {k: str(v) for k, v in model.get_params().items()}
        params_df = pd.DataFrame(
            list(model_params.items()), columns=["Parametre", "Değer"]
        )
        st.dataframe(params_df, hide_index=True, use_container_width=True)

        st.subheader("Kullanılan Özellikler")
        feat_df = pd.DataFrame(
            {"#": range(1, len(features) + 1), "Özellik": features}
        )
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

    with col_perf:
        st.subheader("Performans Metrikleri")

        df_eng = engineer_features(df_raw.copy())
        y = df_raw["Purchased"]
        X_train, X_test, y_train, y_test = train_test_split(
            df_eng, y, test_size=0.20, random_state=42, stratify=y
        )
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics_df = pd.DataFrame(
            {
                "Metrik": ["Doğruluk", "Kesinlik", "Duyarlılık", "F1-Score"],
                "Değer": [f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"],
                "Yüzde": [
                    f"{acc*100:.1f}%", f"{prec*100:.1f}%",
                    f"{rec*100:.1f}%", f"{f1*100:.1f}%",
                ],
            }
        )
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        # Metric bar chart
        fig = plt.figure(figsize=(5.5, 3.5))
        metric_names = ["Accuracy", "Precision", "Recall", "F1"]
        metric_vals = [acc, prec, rec, f1]
        bar_colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_POSITIVE]
        bars = plt.bar(metric_names, metric_vals, color=bar_colors, edgecolor="white", width=0.6)
        for bar, val in zip(bars, metric_vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val*100:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=10,
            )
        plt.ylim(0, 1.12)
        plt.ylabel("Skor", fontsize=11)
        plt.title("Model Performans Metrikleri", fontsize=12, fontweight="bold")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Karışıklık Matrisi (Confusion Matrix)")
    cm = confusion_matrix(y_test, y_pred)
    col_cm, col_report = st.columns(2)

    with col_cm:
        fig = plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        labels_cm = ["Satın Almadı (0)", "Satın Aldı (1)"]
        plt.xticks([0, 1], labels_cm, fontsize=9)
        plt.yticks([0, 1], labels_cm, fontsize=9, rotation=90, va="center")
        plt.xlabel("Tahmin Edilen", fontsize=11)
        plt.ylabel("Gerçek", fontsize=11)
        plt.title("Confusion Matrix", fontsize=13, fontweight="bold")
        for i in range(2):
            for j in range(2):
                plt.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_report:
        st.subheader("Sınıflandırma Raporu")
        report_dict = classification_report(
            y_test, y_pred, output_dict=True,
            target_names=["Satın Almadı", "Satın Aldı"],
        )
        report_df = pd.DataFrame(report_dict).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    # ── Feature log-probability plot (BernoulliNB) ────────────────────────────
    if hasattr(model, "feature_log_prob_"):
        st.markdown("---")
        st.subheader("Özellik Log-Olasılıkları (BernoulliNB)")

        log_probs = model.feature_log_prob_
        importance = np.abs(log_probs[1] - log_probs[0])
        sorted_idx = np.argsort(importance)[::-1]

        fig = plt.figure(figsize=(8, 4))
        bars = plt.bar(
            range(len(features)),
            importance[sorted_idx],
            color=COLOR_PRIMARY,
            edgecolor="white",
        )
        plt.xticks(
            range(len(features)),
            [features[i] for i in sorted_idx],
            rotation=35, ha="right", fontsize=9,
        )
        plt.ylabel("|log P(x|1) – log P(x|0)|", fontsize=10)
        plt.title("BernoulliNB – Özellik Ayrımcılık Gücü", fontsize=12, fontweight="bold")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Yüksek değer → özellik sınıflar arasında daha güçlü ayırım yapıyor."
        )

    # ── Training info ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Eğitim Bilgileri")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(
            """
| Bilgi | Değer |
|---|---|
| Algoritma | BernoulliNB |
| Test Oranı | %20 |
| Random State | 42 |
| Stratified Split | Evet |
"""
        )
    with info_col2:
        st.markdown(
            """
| Veri | Değer |
|---|---|
| Toplam Gözlem | 400 |
| Eğitim Seti | 320 |
| Test Seti | 80 |
| Özellik Sayısı | 10 |
"""
        )
