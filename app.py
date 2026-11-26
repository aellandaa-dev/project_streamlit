from __future__ import annotations

from datetime import date
from pathlib import Path
import pickle
import warnings
import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_core import (
    BASE_FEATURES,
    TARGET,
    engineer_prediction_features,
    get_feature_importance,
    normalize_columns,
    prepare_dataframe,
    run_kmeans_analysis,
    train_classification_models,
    train_regression_models,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "PRSA_data_2010.1.1-2014.12.31.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="AERIS · Beijing PM2.5 Intelligence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --ink: #17212f;
  --muted: #687789;
  --blue: #2677e8;
  --line: rgba(125, 145, 170, .18);
  --glass: rgba(255,255,255,.78);
}
html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Inter, sans-serif;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 12% 3%, rgba(95, 166, 255, .13), transparent 26%),
    radial-gradient(circle at 88% 12%, rgba(144, 183, 255, .15), transparent 28%),
    linear-gradient(155deg, #f7faff 0%, #f5f7fb 46%, #f2f6fc 100%);
}
[data-testid="stHeader"] { background: rgba(247,250,255,.62); backdrop-filter: blur(18px); }
[data-testid="stSidebar"] {
  background: rgba(249, 251, 255, .78);
  border-right: 1px solid rgba(128,148,176,.16);
  backdrop-filter: blur(22px);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
.block-container { max-width: 1450px; padding-top: 1.7rem; padding-bottom: 3rem; }
[data-testid="stMetric"] {
  background: rgba(255,255,255,.76);
  border: 1px solid var(--line);
  border-radius: 19px;
  padding: 18px 19px;
  box-shadow: 0 8px 25px rgba(44, 67, 99, .055);
  backdrop-filter: blur(16px);
  transition: transform .22s ease, box-shadow .22s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 12px 30px rgba(44,67,99,.09); }
[data-testid="stMetricLabel"] { color: #718096; font-weight: 600; }
[data-testid="stMetricValue"] { color: #17212f; font-weight: 750; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 13px !important;
  border: 1px solid rgba(38,119,232,.17) !important;
  box-shadow: 0 5px 16px rgba(38,119,232,.10);
  transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
  font-weight: 650;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-1px); box-shadow: 0 9px 22px rgba(38,119,232,.16);
}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input, [data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div, [data-testid="stSlider"] {
  border-radius: 12px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid rgba(125,145,170,.17);
  border-radius: 15px;
  overflow: hidden;
}
[data-testid="stTabs"] button { border-radius: 11px 11px 0 0; font-weight: 600; }
hr { border-color: rgba(125,145,170,.18); }
.aeris-hero {
  position: relative; overflow: hidden; padding: 30px 34px; border-radius: 26px;
  color: #fff; background: linear-gradient(120deg, rgba(23,81,163,.96), rgba(47,130,226,.91));
  box-shadow: 0 18px 48px rgba(30,92,170,.18); margin-bottom: 24px;
  animation: fadeUp .55s ease both;
}
.aeris-hero:after {
  content: ""; position: absolute; right: -55px; top: -115px; width: 330px; height: 330px;
  border: 1px solid rgba(255,255,255,.22); border-radius: 50%;
  box-shadow: 0 0 0 35px rgba(255,255,255,.055), 0 0 0 75px rgba(255,255,255,.035);
}
.aeris-eyebrow { letter-spacing: .13em; text-transform: uppercase; font-size: 11px; font-weight: 750; opacity: .78; }
.aeris-title { font-size: clamp(28px, 3.8vw, 43px); letter-spacing: -.04em; font-weight: 800; margin: 8px 0 7px; line-height: 1.08; white-space: pre-line; }
.aeris-subtitle { max-width: 760px; font-size: 14px; line-height: 1.65; color: rgba(255,255,255,.83); }
.glass-card {
  background: rgba(255,255,255,.74); border: 1px solid rgba(125,145,170,.17);
  border-radius: 20px; padding: 20px 22px; box-shadow: 0 9px 27px rgba(42,65,97,.055);
  backdrop-filter: blur(18px); transition: transform .2s ease, box-shadow .2s ease;
  animation: fadeUp .42s ease both;
}
.glass-card:hover { transform: translateY(-2px); box-shadow: 0 14px 32px rgba(42,65,97,.085); }
.glass-card h3 { margin: 0 0 8px; font-size: 16px; color: #1b2b40; }
.glass-card p { margin: 0; color: #718096; font-size: 13px; line-height: 1.6; }
.section-kicker { color: #728197; font-size: 11px; font-weight: 750; text-transform: uppercase; letter-spacing: .13em; margin-bottom: 5px; }
.section-title { color: #1b2b40; font-size: 25px; font-weight: 780; letter-spacing: -.035em; margin: 0 0 6px; }
.section-desc { color: #728197; font-size: 13px; margin-bottom: 18px; }
.sidebar-brand { padding: 5px 4px 17px; }
.sidebar-brand-mark { width: 39px; height: 39px; border-radius: 13px; display: inline-flex; align-items: center; justify-content: center; background: linear-gradient(145deg,#2379e8,#75b4ff); color: white; font-size: 22px; font-weight: 800; box-shadow: 0 8px 18px rgba(38,119,232,.22); }
.sidebar-brand-name { display: inline-block; margin-left: 9px; vertical-align: middle; font-weight: 800; letter-spacing: -.03em; color: #1b2b40; font-size: 19px; }
.sidebar-caption { color: #8390a0; font-size: 11px; margin: 4px 0 0 2px; }
.member-list { background: rgba(255,255,255,.64); padding: 13px 14px; border-radius: 15px; border: 1px solid rgba(125,145,170,.16); color: #617185; font-size: 11px; line-height: 1.55; }
.member-list strong { color: #26374b; }
.small-note { color: #7a8797; font-size: 11px; line-height: 1.55; }
@keyframes fadeUp { from {opacity:0; transform:translateY(8px)} to {opacity:1; transform:translateY(0)} }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_header(kicker: str, title: str, description: str = ""):
    st.markdown(
        f'<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div>'
        f'<div class="section-desc">{description}</div>',
        unsafe_allow_html=True,
    )


def hero(eyebrow: str, title: str, subtitle: str):
    st.markdown(
        f'<div class="aeris-hero"><div class="aeris-eyebrow">{eyebrow}</div>'
        f'<div class="aeris-title">{title}</div><div class="aeris-subtitle">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def dataframe_download(df: pd.DataFrame, label: str, filename: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def current_dataset_key(df: pd.DataFrame) -> str:
    cols = "|".join(map(str, df.columns))
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]
    return f"{len(df)}::{cols}::{digest}"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


# Sidebar: data source and navigation
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><span class="sidebar-brand-mark">◉</span>'
        '<span class="sidebar-brand-name">AERIS</span>'
        '<div class="sidebar-caption">Beijing air quality intelligence</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    uploaded_file = st.file_uploader("Gunakan dataset lain (CSV)", type=["csv"], help="File perlu memiliki kolom waktu, pm2.5, cuaca, dan arah angin seperti dataset proyek.")
    st.markdown("<div class='small-note'>Dataset bawaan: PRSA Beijing · 2010–2014</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "NAVIGASI",
        ["Overview", "Dataset & EDA", "Preprocessing", "Regression Lab", "Klasifikasi", "K-Means Clustering", "Prediksi"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown(
        "<div class='member-list'><strong>Anggota Kelompok</strong><br>"
        "Farel Nofri Landa · 241011401676<br>"
        "Septian Cahaya Irwanto · 241011402116<br>"
        "Muhammad Rojak Ardiansyah · 241011400688</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='small-note' style='margin-top:12px'>Machine Learning · Streamlit</div>", unsafe_allow_html=True)

try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        source_label = f"Upload: {uploaded_file.name}"
    else:
        raw_df = load_default_data()
        source_label = "Dataset bawaan Beijing PM2.5"
    normalized_raw = normalize_columns(raw_df)
    clean_df, cleaning_report = prepare_dataframe(normalized_raw)
except Exception as exc:
    st.error(f"Dataset tidak dapat diproses: {exc}")
    st.info("Gunakan CSV dengan kolom year, month, day, hour, pm2.5, DEWP, TEMP, PRES, cbwd, Iws, Is, dan Ir.")
    st.stop()

data_key = current_dataset_key(normalized_raw)
if st.session_state.get("active_data_key") != data_key:
    st.session_state["active_data_key"] = data_key
    for key in ["regression_bundle", "classification_bundle", "cluster_bundle", "selected_regression_model", "reg_detail_choice"]:
        st.session_state.pop(key, None)


# -----------------------------------------------------------------------------
# OVERVIEW
# -----------------------------------------------------------------------------
if page == "Overview":
    hero(
        "AIR QUALITY · MACHINE LEARNING",
        "Understand the air.\nPredict what comes next.",
        "Dashboard analisis dan prediksi konsentrasi PM2.5 di Beijing menggunakan data meteorologi, waktu, dan arah angin. Dibangun dengan alur machine learning yang dapat dijelajahi tahap demi tahap.",
    )
    st.caption(f"Sumber data aktif: {source_label}")
    valid_count = len(clean_df)
    missing_target = int(cleaning_report["missing_target_removed"])
    date_min = clean_df["timestamp"].min().strftime("%d %b %Y")
    date_max = clean_df["timestamp"].max().strftime("%d %b %Y")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total observasi mentah", f"{len(normalized_raw):,}")
    c2.metric("Target PM2.5 tersedia", f"{valid_count:,}")
    c3.metric("Target kosong dibuang", f"{missing_target:,}")
    c4.metric("Jumlah fitur model", "16")

    st.markdown("\n")
    left, right = st.columns([1.35, 1])
    with left:
        section_header("RINGKASAN DATA", "Perjalanan data Beijing", f"Periode observasi valid: {date_min} — {date_max}")
        monthly = clean_df.set_index("timestamp")[TARGET].resample("MS").mean().rename("Rata-rata PM2.5").reset_index()
        fig = px.area(monthly, x="timestamp", y="Rata-rata PM2.5", template="plotly_white")
        fig.update_traces(line=dict(width=2.2), fillcolor="rgba(53,130,224,.14)")
        fig.update_layout(height=315, margin=dict(l=8, r=8, t=12, b=8), xaxis_title="Waktu", yaxis_title="PM2.5 (µg/m³)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        section_header("GAMBARAN TARGET", "Sebaran konsentrasi PM2.5", "Distribusi nilai target setelah observasi tanpa label dikeluarkan.")
        hist = px.histogram(clean_df, x=TARGET, nbins=48, template="plotly_white", color_discrete_sequence=["#3d8be9"])
        hist.update_layout(height=315, margin=dict(l=8, r=8, t=12, b=8), xaxis_title="PM2.5 (µg/m³)", yaxis_title="Frekuensi", showlegend=False)
        st.plotly_chart(hist, use_container_width=True)

    st.markdown("### Alur implementasi")
    cards = st.columns(4)
    workflow = [
        ("01", "Eksplorasi", "Memahami struktur, kualitas, distribusi, dan hubungan antarvariabel."),
        ("02", "Preprocessing", "Membersihkan target kosong, menyusun fitur waktu, imputasi, dan encoding."),
        ("03", "Model & evaluasi", "Membandingkan regresi, klasifikasi relatif, dan clustering."),
        ("04", "Prediksi", "Memakai model terlatih untuk input manual atau file prediksi batch."),
    ]
    for col, (num, title, desc) in zip(cards, workflow):
        with col:
            st.markdown(f'<div class="glass-card"><div class="section-kicker">STEP {num}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("\n")
    st.info("Fokus utama proyek adalah regresi PM2.5. Klasifikasi menggunakan tiga kelas kuantil relatif dari data training (bukan kategori ISPU/AQI), sedangkan K-Means dipakai untuk eksplorasi pola kelompok.")


# -----------------------------------------------------------------------------
# DATASET & EDA
# -----------------------------------------------------------------------------
elif page == "Dataset & EDA":
    section_header("01 · DATA UNDERSTANDING", "Dataset & Exploratory Data Analysis", "Kenali struktur observasi, tipe data, nilai kosong, distribusi target, dan pola lingkungan.")
    tabs = st.tabs(["Preview", "Kualitas data", "Distribusi & hubungan"])
    with tabs[0]:
        a, b, c = st.columns(3)
        a.metric("Baris", f"{len(normalized_raw):,}")
        b.metric("Kolom", f"{len(normalized_raw.columns)}")
        c.metric("Duplikasi", f"{int(normalized_raw.duplicated().sum()):,}")
        st.dataframe(normalized_raw.head(30), use_container_width=True, hide_index=True)
        dataframe_download(normalized_raw.head(5000), "Unduh contoh data (maks. 5.000 baris)", "beijing_pm25_preview.csv")
        with st.expander("Lihat tipe data setiap kolom"):
            info = pd.DataFrame({"Kolom": normalized_raw.columns, "Tipe data": normalized_raw.dtypes.astype(str).values, "Nilai unik": [normalized_raw[c].nunique(dropna=True) for c in normalized_raw.columns]})
            st.dataframe(info, use_container_width=True, hide_index=True)
    with tabs[1]:
        missing = normalized_raw.isna().sum().rename("Nilai kosong").reset_index().rename(columns={"index": "Kolom"})
        missing["Persentase"] = (missing["Nilai kosong"] / len(normalized_raw) * 100).round(2)
        st.dataframe(missing.sort_values("Nilai kosong", ascending=False), use_container_width=True, hide_index=True)
        fig = px.bar(missing.sort_values("Nilai kosong", ascending=True), x="Nilai kosong", y="Kolom", orientation="h", template="plotly_white", color_discrete_sequence=["#6aa6ef"])
        fig.update_layout(height=330, margin=dict(l=8, r=8, t=12, b=8), xaxis_title="Jumlah nilai kosong", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Nilai kosong pada target tidak dapat dipakai untuk pembelajaran supervised regression, sehingga observasi tersebut dikeluarkan dari data latih/evaluasi. Nilai kosong pada fitur ditangani oleh SimpleImputer di dalam pipeline.")
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            hist = px.histogram(clean_df, x=TARGET, nbins=55, marginal="box", template="plotly_white", color_discrete_sequence=["#4c96ed"])
            hist.update_layout(height=330, margin=dict(l=8, r=8, t=12, b=8), xaxis_title="PM2.5 (µg/m³)", yaxis_title="Frekuensi")
            st.plotly_chart(hist, use_container_width=True)
        with col2:
            scatter_data = clean_df.sample(min(5000, len(clean_df)), random_state=42)
            scatter = px.scatter(scatter_data, x="TEMP", y=TARGET, color="cbwd", opacity=.48, template="plotly_white", labels={"TEMP": "Temperatur (°C)", TARGET: "PM2.5 (µg/m³)", "cbwd": "Arah angin"})
            scatter.update_layout(height=330, margin=dict(l=8, r=8, t=12, b=8), legend_title_text="Arah angin")
            st.plotly_chart(scatter, use_container_width=True)
        numeric = clean_df.select_dtypes(include=np.number).drop(columns=["No"], errors="ignore")
        corr = numeric.corr(numeric_only=True)
        heat = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template="plotly_white")
        heat.update_layout(height=440, margin=dict(l=8, r=8, t=12, b=8), coloraxis_colorbar_title="Korelasi")
        st.plotly_chart(heat, use_container_width=True)


# -----------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------
elif page == "Preprocessing":
    section_header("02 · DATA PREPARATION", "Preprocessing pipeline", "Setiap proses ditampilkan agar keputusan pembersihan data dapat ditelusuri.")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Observasi awal", f"{cleaning_report['rows_before']:,}")
    r2.metric("Observasi siap model", f"{cleaning_report['rows_after']:,}")
    r3.metric("Target kosong dihapus", f"{cleaning_report['missing_target_removed']:,}")
    r4.metric("Duplikasi dihapus", f"{cleaning_report['duplicate_rows_removed']:,}")

    steps = [
        ("1. Validasi skema", "Memastikan kolom waktu, target PM2.5, variabel cuaca, dan arah angin tersedia."),
        ("2. Konversi tipe data", "Kolom numerik dikonversi ke tipe numerik. Tanggal dan jam digabungkan menjadi timestamp."),
        ("3. Pembersihan baris", f"Menghapus {cleaning_report['invalid_timestamp_removed']:,} timestamp tidak valid, {cleaning_report['duplicate_rows_removed']:,} duplikasi, serta {cleaning_report['missing_target_removed']:,} baris tanpa target PM2.5."),
        ("4. Rekayasa fitur waktu", "Membuat sin/cos jam dan bulan agar pola siklik lebih mudah dipelajari, serta fitur hari dalam minggu."),
        ("5. Encoding & imputasi", "Arah angin (cbwd) di-One-Hot Encode. Fitur numerik diimputasi median dan kategori dengan modus."),
        ("6. Scaling", "StandardScaler diterapkan pada model linear/SVM/Naive Bayes; model berbasis pohon menggunakan fitur tanpa scaling."),
        ("7. Pembagian data", "Train/test disusun kronologis. Bagian waktu paling akhir dipertahankan sebagai data uji."),
    ]
    for title, desc in steps:
        st.markdown(f'<div class="glass-card" style="margin-bottom:10px;padding:15px 18px"><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

    with st.expander("Lihat ringkasan missing value setelah pembersihan target"):
        miss_after = pd.Series(cleaning_report["missing_after_predictors"], name="Missing pada fitur").rename_axis("Fitur").reset_index()
        st.dataframe(miss_after, use_container_width=True, hide_index=True)
    with st.expander("Lihat fitur yang akan digunakan model"):
        feature_table = pd.DataFrame({
            "Fitur": ["year", "month", "day", "hour", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir", "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek"],
            "Deskripsi": ["Tahun pengamatan", "Bulan pengamatan", "Tanggal dalam bulan", "Jam pengamatan", "Dew point", "Temperatur", "Tekanan udara", "Arah angin", "Kecepatan angin kumulatif", "Akumulasi jam salju", "Akumulasi jam hujan", "Komponen siklik jam (sin)", "Komponen siklik jam (cos)", "Komponen siklik bulan (sin)", "Komponen siklik bulan (cos)", "Hari dalam minggu"],
        })
        st.dataframe(feature_table, use_container_width=True, hide_index=True)
    st.markdown("### Data hasil preprocessing")
    st.dataframe(clean_df[["timestamp", TARGET] + [c for c in BASE_FEATURES if c not in ["year", "month", "day", "hour", TARGET]]].head(25), use_container_width=True, hide_index=True)
    dataframe_download(clean_df, "Unduh data hasil preprocessing", "beijing_pm25_preprocessed.csv")


# -----------------------------------------------------------------------------
# REGRESSION LAB
# -----------------------------------------------------------------------------
elif page == "Regression Lab":
    section_header("03 · SUPERVISED LEARNING", "Regression model lab", "Prediksi nilai numerik PM2.5. Metrik evaluasi: MAE, RMSE, dan R².")
    model_names = [
        "Linear Regression",
        "Support Vector Regression (LinearSVR)",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "Extra Trees Regressor",
        "Gradient Boosting Regressor",
    ]
    c1, c2 = st.columns([1.35, .85])
    with c1:
        chosen_models = st.multiselect(
            "Algoritma yang akan dibandingkan",
            model_names,
            default=["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor"],
        )
    with c2:
        test_percent = st.slider("Proporsi data testing", min_value=10, max_value=35, value=20, step=5)
    st.caption("Data dibagi secara kronologis (tanpa shuffle): bagian awal untuk training dan bagian akhir untuk testing.")
    if st.button("Train & evaluasi model regresi", type="primary", disabled=not chosen_models, use_container_width=True):
        with st.spinner("Melatih model regresi dan menghitung metrik..."):
            try:
                bundle = train_regression_models(clean_df, chosen_models, test_fraction=test_percent / 100)
                bundle["data_key"] = data_key
                st.session_state.pop("reg_detail_choice", None)
                st.session_state["regression_bundle"] = bundle
                st.session_state["selected_regression_model"] = bundle["best_name"]
            except Exception as exc:
                st.error(f"Training gagal: {exc}")

    bundle = st.session_state.get("regression_bundle")
    if bundle and bundle.get("data_key") == data_key:
        results = bundle["results"]
        metric_df = pd.DataFrame([{"Model": name, **res["metrics"], "Durasi training (detik)": round(res["training_seconds"], 2)} for name, res in results.items()])
        metric_df = metric_df.sort_values("RMSE", ascending=True).reset_index(drop=True)
        best_name = bundle["best_name"]
        best_metrics = results[best_name]["metrics"]
        st.markdown("\n")
        st.success(f"Model terbaik berdasarkan RMSE pada data testing: {best_name}")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{best_metrics['MAE']:.2f} µg/m³")
        m2.metric("RMSE", f"{best_metrics['RMSE']:.2f} µg/m³")
        m3.metric("R²", f"{best_metrics['R²']:.4f}")
        st.markdown("### Perbandingan performa")
        st.dataframe(metric_df, use_container_width=True, hide_index=True)
        fig = px.bar(metric_df.sort_values("RMSE", ascending=True), x="Model", y="RMSE", color="Model", template="plotly_white", text_auto=".2f")
        fig.update_layout(height=340, margin=dict(l=8, r=8, t=12, b=8), showlegend=False, xaxis_title="", yaxis_title="RMSE (lebih rendah lebih baik)")
        st.plotly_chart(fig, use_container_width=True)

        model_choice = st.selectbox("Lihat detail model", list(results.keys()), index=list(results.keys()).index(best_name), key="reg_detail_choice")
        detail = results[model_choice]
        actual = np.asarray(detail["actual"])
        pred = np.asarray(detail["predictions"])
        detail_tabs = st.tabs(["Actual vs Predicted", "Residual", "Feature importance"])
        with detail_tabs[0]:
            sample_n = min(3500, len(actual))
            indices = np.linspace(0, len(actual) - 1, sample_n, dtype=int)
            comp = pd.DataFrame({"Aktual": actual[indices], "Prediksi": pred[indices]})
            fig = px.scatter(comp, x="Aktual", y="Prediksi", opacity=.48, template="plotly_white", color_discrete_sequence=["#3c8ce8"])
            low = float(min(comp.min()))
            high = float(max(comp.max()))
            fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(color="#ef8a64", dash="dash", width=2))
            fig.update_layout(height=390, margin=dict(l=8, r=8, t=12, b=8), xaxis_title="PM2.5 aktual (µg/m³)", yaxis_title="PM2.5 prediksi (µg/m³)")
            st.plotly_chart(fig, use_container_width=True)
        with detail_tabs[1]:
            residuals = actual - pred
            residual_fig = px.histogram(x=residuals, nbins=50, template="plotly_white", color_discrete_sequence=["#6d9de5"])
            residual_fig.update_layout(height=320, margin=dict(l=8, r=8, t=12, b=8), xaxis_title="Residual = aktual - prediksi", yaxis_title="Frekuensi")
            st.plotly_chart(residual_fig, use_container_width=True)
            st.caption("Residual positif berarti model memprediksi lebih rendah daripada nilai aktual; residual negatif berarti model memprediksi lebih tinggi.")
        with detail_tabs[2]:
            importance = get_feature_importance(detail["pipeline"], "regression")
            if importance.empty:
                st.info("Model ini tidak menyediakan feature_importances_ secara langsung.")
            else:
                imp = importance.head(15).sort_values("Importance")
                fig = px.bar(imp, x="Importance", y="Fitur", orientation="h", template="plotly_white", color_discrete_sequence=["#4d91e5"])
                fig.update_layout(height=420, margin=dict(l=8, r=8, t=12, b=8), xaxis_title="Feature importance", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Simpan model")
        save_col, download_col = st.columns(2)
        model_payload = {
            "task": "regression", "model_name": best_name,
            "pipeline": results[best_name]["pipeline"],
            "feature_columns": bundle["feature_columns"],
            "metrics": best_metrics,
            "training_period": [clean_df["timestamp"].min(), clean_df["timestamp"].iloc[bundle["split_at"] - 1]],
            "test_period": [clean_df["timestamp"].iloc[bundle["split_at"]], clean_df["timestamp"].max()],
        }
        model_bytes = pickle.dumps(model_payload, protocol=pickle.HIGHEST_PROTOCOL)
        with save_col:
            if st.button("Simpan model terbaik ke folder models/", use_container_width=True):
                import joblib
                path = MODEL_DIR / "beijing_pm25_regressor.joblib"
                joblib.dump(model_payload, path)
                st.success(f"Model tersimpan: {path.name}")
        with download_col:
            st.download_button("Download model (.pkl)", data=model_bytes, file_name="beijing_pm25_regressor.pkl", mime="application/octet-stream", use_container_width=True)
    else:
        st.markdown('<div class="glass-card"><h3>Belum ada model regresi</h3><p>Pilih algoritma lalu tekan tombol training. Hasil evaluasi dan model terbaik akan muncul di bagian ini.</p></div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CLASSIFICATION
# -----------------------------------------------------------------------------
elif page == "Klasifikasi":
    section_header("04 · SUPERVISED CLASSIFICATION", "Klasifikasi relatif PM2.5", "Bagian klasifikasi untuk mempraktikkan SVM, Decision Tree, Random Forest, dan Naive Bayes pada label kategori turunan.")
    st.warning("Label kelas dibentuk dari kuantil data training (relatif rendah/sedang/tinggi), bukan batas regulasi ISPU atau AQI. Prediksi regresi tetap menjadi hasil utama proyek.")
    clf_names = ["Support Vector Classifier (LinearSVC)", "Decision Tree Classifier", "Random Forest Classifier", "Gaussian Naive Bayes"]
    col1, col2 = st.columns([1.35, .85])
    with col1:
        chosen_clf = st.multiselect("Algoritma klasifikasi", clf_names, default=clf_names)
    with col2:
        clf_test = st.slider("Proporsi data testing klasifikasi", 10, 35, 20, 5)
    if st.button("Train & evaluasi klasifikasi", type="primary", disabled=not chosen_clf, use_container_width=True):
        with st.spinner("Melatih model klasifikasi..."):
            try:
                clf_bundle = train_classification_models(clean_df, chosen_clf, test_fraction=clf_test / 100)
                clf_bundle["data_key"] = data_key
                st.session_state["classification_bundle"] = clf_bundle
            except Exception as exc:
                st.error(f"Training klasifikasi gagal: {exc}")
    clf_bundle = st.session_state.get("classification_bundle")
    if clf_bundle and clf_bundle.get("data_key") == data_key:
        st.caption(f"Batas kuantil dari data training: {clf_bundle['class_thresholds'][0]:.2f} dan {clf_bundle['class_thresholds'][1]:.2f} µg/m³.")
        rows = [{"Model": n, **r["metrics"], "Durasi training (detik)": round(r["training_seconds"], 2)} for n, r in clf_bundle["results"].items()]
        score_df = pd.DataFrame(rows).sort_values("F1 macro", ascending=False)
        st.dataframe(score_df, use_container_width=True, hide_index=True)
        selected_clf_name = st.selectbox("Detail algoritma", list(clf_bundle["results"].keys()), index=list(clf_bundle["results"].keys()).index(clf_bundle["best_name"]))
        clf_detail = clf_bundle["results"][selected_clf_name]
        labels = clf_bundle["class_labels"]
        cm = clf_detail["confusion_matrix"]
        cm_fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues", aspect="auto", template="plotly_white")
        cm_fig.update_layout(height=360, margin=dict(l=8, r=8, t=12, b=8), xaxis_title="Prediksi", yaxis_title="Aktual", coloraxis_showscale=False)
        st.plotly_chart(cm_fig, use_container_width=True)
        report = clf_detail["report"]
        report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Kelas / metrik"})
        st.dataframe(report_df.round(3), use_container_width=True, hide_index=True)
        best_clf_name = clf_bundle["best_name"]
        best_clf_detail = clf_bundle["results"][best_clf_name]
        clf_payload = {
            "task": "classification", "model_name": best_clf_name,
            "pipeline": best_clf_detail["pipeline"], "feature_columns": clf_bundle["feature_columns"],
            "class_labels": labels, "class_thresholds": clf_bundle["class_thresholds"],
            "class_definition": clf_bundle["class_definition"], "metrics": best_clf_detail["metrics"],
        }
        save_clf_col, download_clf_col = st.columns(2)
        with save_clf_col:
            if st.button("Simpan klasifikasi terbaik ke models/", use_container_width=True):
                import joblib
                path = MODEL_DIR / "beijing_pm25_classifier.joblib"
                joblib.dump(clf_payload, path)
                st.success(f"Model tersimpan: {path.name}")
        with download_clf_col:
            st.download_button("Download model klasifikasi (.pkl)", data=pickle.dumps(clf_payload, protocol=pickle.HIGHEST_PROTOCOL), file_name="beijing_pm25_classifier.pkl", mime="application/octet-stream", use_container_width=True)
    else:
        st.markdown('<div class="glass-card"><h3>Belum ada model klasifikasi</h3><p>Bagian ini mengubah target numerik menjadi tiga label relatif agar algoritma klasifikasi pada modul dapat dipraktikkan tanpa menggantikan tujuan utama regresi.</p></div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# K-MEANS CLUSTERING
# -----------------------------------------------------------------------------
elif page == "K-Means Clustering":
    section_header("05 · UNSUPERVISED LEARNING", "K-Means clustering", "Kelompokkan kondisi udara berdasarkan pola PM2.5 dan variabel meteorologi.")
    default_cluster_features = [TARGET, "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    cluster_features = st.multiselect("Fitur untuk clustering", default_cluster_features, default=default_cluster_features)
    selected_k = st.slider("Jumlah cluster (K)", 2, 7, 3)
    st.caption("Silhouette score dihitung untuk rentang K=2 sampai K=7 pada sampel maksimal 5.000 baris; model akhir dipasang ke seluruh data valid.")
    if st.button("Jalankan analisis K-Means", type="primary", disabled=len(cluster_features) < 2, use_container_width=True):
        with st.spinner("Menghitung silhouette score dan membentuk cluster..."):
            try:
                cluster_bundle = run_kmeans_analysis(clean_df, selected_k=selected_k, feature_columns=cluster_features)
                cluster_bundle["data_key"] = data_key
                st.session_state["cluster_bundle"] = cluster_bundle
            except Exception as exc:
                st.error(f"Clustering gagal: {exc}")
    cluster_bundle = st.session_state.get("cluster_bundle")
    if cluster_bundle and cluster_bundle.get("data_key") == data_key:
        scores = cluster_bundle["scores"]
        sc1, sc2 = st.columns([1.1, .9])
        with sc1:
            fig = px.line(scores, x="K", y="Silhouette", markers=True, template="plotly_white", color_discrete_sequence=["#387fd7"])
            fig.update_layout(height=310, margin=dict(l=8, r=8, t=12, b=8), xaxis_title="Jumlah cluster (K)", yaxis_title="Silhouette score")
            st.plotly_chart(fig, use_container_width=True)
        with sc2:
            st.metric("K yang dipilih", cluster_bundle["selected_k"])
            st.metric("Silhouette score K terpilih", f"{cluster_bundle['silhouette_selected']:.4f}")
            st.caption("Silhouette yang lebih tinggi umumnya menunjukkan pemisahan cluster yang lebih baik pada ruang fitur yang digunakan.")
        st.markdown("### Profil setiap cluster")
        st.dataframe(cluster_bundle["profile"].reset_index(), use_container_width=True, hide_index=True)
        cluster_data = cluster_bundle["clustered"]
        scatter_x = "TEMP" if "TEMP" in cluster_data.columns else cluster_bundle["features"][0]
        scatter_y = TARGET if TARGET in cluster_data.columns else cluster_bundle["features"][1]
        sample = cluster_data.sample(min(5000, len(cluster_data)), random_state=42)
        scatter = px.scatter(sample, x=scatter_x, y=scatter_y, color="Cluster", opacity=.55, template="plotly_white", labels={scatter_x: scatter_x, scatter_y: scatter_y})
        scatter.update_layout(height=390, margin=dict(l=8, r=8, t=12, b=8))
        st.plotly_chart(scatter, use_container_width=True)
        dataframe_download(cluster_data, "Download hasil cluster", "beijing_pm25_kmeans_clusters.csv")
        payload = {"model": cluster_bundle["model"], "scaler": cluster_bundle["scaler"], "features": cluster_bundle["features"], "selected_k": cluster_bundle["selected_k"]}
        save_km_col, download_km_col = st.columns(2)
        with save_km_col:
            if st.button("Simpan model K-Means ke models/", use_container_width=True):
                import joblib
                path = MODEL_DIR / "beijing_pm25_kmeans.joblib"
                joblib.dump(payload, path)
                st.success(f"Model tersimpan: {path.name}")
        with download_km_col:
            st.download_button("Download model K-Means (.pkl)", data=pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), file_name="beijing_pm25_kmeans.pkl", mime="application/octet-stream", use_container_width=True)
    else:
        st.markdown('<div class="glass-card"><h3>Cluster belum dibentuk</h3><p>Pilih fitur yang ingin dianalisis, tentukan nilai K, lalu jalankan analisis untuk melihat silhouette score dan profil cluster.</p></div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------------------------
elif page == "Prediksi":
    section_header("06 · MODEL INFERENCE", "Prediksi PM2.5", "Gunakan model regresi yang sudah dilatih untuk memprediksi konsentrasi PM2.5 dari kondisi waktu dan cuaca.")
    reg_bundle = st.session_state.get("regression_bundle")
    if not reg_bundle or reg_bundle.get("data_key") != data_key:
        st.warning("Belum ada model regresi untuk dataset aktif. Buka halaman Regression Lab dan latih minimal satu algoritma terlebih dahulu.")
    else:
        available_models = list(reg_bundle["results"].keys())
        default_model = st.session_state.get("selected_regression_model", reg_bundle["best_name"])
        model_name = st.selectbox("Pilih model", available_models, index=available_models.index(default_model) if default_model in available_models else 0)
        st.session_state["selected_regression_model"] = model_name
        pipeline = reg_bundle["results"][model_name]["pipeline"]
        st.markdown("### Input kondisi pengamatan")
        default_date = clean_df["timestamp"].max().date()
        with st.form("manual_prediction_form"):
            date_value = st.date_input("Tanggal", value=default_date, min_value=date(2000, 1, 1), max_value=date(2035, 12, 31))
            hour_value = st.slider("Jam pengamatan", 0, 23, 12)
            a, b, c = st.columns(3)
            with a:
                dewp = st.number_input("Dew point · DEWP (°C)", min_value=-50.0, max_value=40.0, value=1.8, step=0.5)
                temp = st.number_input("Temperatur · TEMP (°C)", min_value=-30.0, max_value=50.0, value=12.4, step=0.5)
            with b:
                pres = st.number_input("Tekanan · PRES (hPa)", min_value=950.0, max_value=1080.0, value=1016.4, step=0.5)
                wind_speed = st.number_input("Kecepatan angin · Iws", min_value=0.0, max_value=700.0, value=23.9, step=0.5)
            with c:
                wind_options = sorted(clean_df["cbwd"].dropna().astype(str).unique().tolist())
                wind_dir = st.selectbox("Arah angin · cbwd", wind_options, index=wind_options.index("SE") if "SE" in wind_options else 0)
                snow = st.number_input("Jam salju · Is", min_value=0, max_value=100, value=0, step=1)
                rain = st.number_input("Jam hujan · Ir", min_value=0, max_value=100, value=0, step=1)
            submit_prediction = st.form_submit_button("Prediksi PM2.5", type="primary", use_container_width=True)
        if submit_prediction:
            input_row = pd.DataFrame([{
                "year": date_value.year, "month": date_value.month, "day": date_value.day, "hour": hour_value,
                "DEWP": dewp, "TEMP": temp, "PRES": pres, "cbwd": wind_dir,
                "Iws": wind_speed, "Is": snow, "Ir": rain,
            }])
            try:
                X_one = engineer_prediction_features(input_row)
                pred_value = float(pipeline.predict(X_one)[0])
                st.markdown("\n")
                st.markdown('<div class="glass-card"><div class="section-kicker">HASIL PREDIKSI</div><div style="font-size:42px;font-weight:800;letter-spacing:-.04em;color:#2476d9">'
                            + f"{pred_value:,.2f}" + '</div><div style="font-size:13px;color:#6e7d90">µg/m³ PM2.5 · ' + model_name + '</div></div>', unsafe_allow_html=True)
                st.caption("Nilai ini merupakan estimasi model berdasarkan fitur yang dimasukkan, bukan pengukuran sensor aktual.")
            except Exception as exc:
                st.error(f"Prediksi gagal: {exc}")

        st.markdown("---")
        st.markdown("### Prediksi batch dari CSV")
        st.caption("Unggah file CSV yang berisi kolom year, month, day, hour, DEWP, TEMP, PRES, cbwd, Iws, Is, dan Ir. Kolom pm2.5 tidak diperlukan.")
        template = pd.DataFrame([{
            "year": 2015, "month": 1, "day": 1, "hour": 0, "DEWP": -5,
            "TEMP": 2, "PRES": 1020, "cbwd": "NW", "Iws": 10.0, "Is": 0, "Ir": 0,
        }])
        dataframe_download(template, "Download template input prediksi", "template_input_pm25.csv")
        batch_file = st.file_uploader("Upload input batch (CSV)", type=["csv"], key="batch_prediction_upload")
        if batch_file is not None:
            try:
                batch_raw = pd.read_csv(batch_file)
                X_batch = engineer_prediction_features(batch_raw)
                batch_predictions = pipeline.predict(X_batch)
                output = normalize_columns(batch_raw).copy()
                output["prediksi_pm2.5"] = np.round(batch_predictions, 3)
                st.success(f"Berhasil membuat {len(output):,} prediksi.")
                st.dataframe(output.head(50), use_container_width=True, hide_index=True)
                dataframe_download(output, "Download seluruh hasil prediksi", "hasil_prediksi_pm25.csv")
            except Exception as exc:
                st.error(f"File batch belum sesuai format: {exc}")


# Footer shared across pages
st.markdown("\n---")
st.markdown(
    "<div style='display:flex;justify-content:space-between;gap:18px;flex-wrap:wrap;color:#8995a4;font-size:11px'>"
    "<span>AERIS · Beijing PM2.5 Machine Learning Web App</span>"
    "<span>Kelompok: Farel Nofri Landa · Septian Cahaya Irwanto · Muhammad Rojak Ardiansyah</span>"
    "</div>",
    unsafe_allow_html=True,
)
