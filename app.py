from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_core import (
    MODEL_LABELS,
    build_supervised_frame,
    load_dataset,
    prepare_prediction_input,
    run_kmeans,
    train_classifier,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "PRSA_data_2010.1.1-2014.12.31.csv"

st.set_page_config(
    page_title="AirScope · Beijing PM2.5 Lab",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root { --ink:#14233B; --muted:#718096; --blue:#267DFF; --line:rgba(137,158,190,.20); }
html, body, [class*="css"] { font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Helvetica Neue","Segoe UI",Arial,sans-serif; }
.stApp {
  background:
    radial-gradient(circle at 7% 0%, rgba(105,169,255,.17), transparent 31%),
    radial-gradient(circle at 94% 12%, rgba(153,207,255,.18), transparent 25%),
    linear-gradient(135deg,#F4F7FC 0%,#F8FAFE 52%,#EEF4FC 100%);
  color:var(--ink);
}
[data-testid="stHeader"] { background:rgba(248,250,254,.70); backdrop-filter:blur(18px); }
[data-testid="stSidebar"] { background:rgba(248,251,255,.84); border-right:1px solid rgba(138,159,190,.18); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
.block-container { max-width:1440px; padding-top:2rem; padding-bottom:3rem; }
section[data-testid="stSidebar"] .stRadio label { font-weight:560; color:#30435F; }
.nav-group-title { color:#8290A4; font-size:.72rem; font-weight:760; letter-spacing:.08em; text-transform:uppercase; margin:15px 5px 5px; }
section[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap:2px; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius:10px; padding:5px 8px; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:rgba(38,125,255,.10); }
.stButton > button, .stDownloadButton > button, .stLinkButton > a {
  border-radius:14px !important; border:1px solid rgba(54,123,230,.18) !important;
  background:linear-gradient(180deg,#FFFFFF 0%,#F4F8FF 100%) !important;
  color:#1C5EC5 !important; font-weight:650 !important; min-height:2.7rem;
  box-shadow:0 6px 18px rgba(44,89,153,.08); transition:all .22s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-1px); box-shadow:0 10px 24px rgba(44,89,153,.14); border-color:rgba(54,123,230,.35) !important; }
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"] {
  border-radius:12px !important;
}
[data-testid="stMetric"] {
  background:rgba(255,255,255,.68); border:1px solid rgba(255,255,255,.88);
  border-radius:20px; padding:18px 20px; box-shadow:0 10px 28px rgba(33,65,112,.055);
  backdrop-filter:blur(16px); animation:rise .45s ease both;
}
[data-testid="stMetricLabel"] { color:#75849A !important; font-size:.82rem !important; }
[data-testid="stMetricValue"] { color:#152641 !important; font-weight:760 !important; letter-spacing:-.04em; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius:16px; overflow:hidden; }
[data-testid="stExpander"] { background:rgba(255,255,255,.55); border:1px solid var(--line); border-radius:16px; }
[data-testid="stTabs"] button { border-radius:12px 12px 0 0; font-weight:650; }
hr { border-color:rgba(125,149,182,.20); }
.hero {
  position:relative; overflow:hidden; border-radius:28px; padding:34px 38px;
  background:linear-gradient(118deg,rgba(255,255,255,.82),rgba(236,245,255,.72));
  border:1px solid rgba(255,255,255,.96); box-shadow:0 18px 46px rgba(43,78,128,.09);
  backdrop-filter:blur(22px); margin-bottom:22px; animation:rise .48s ease both;
}
.hero:after { content:""; position:absolute; width:260px; height:260px; border-radius:50%; right:-60px; top:-115px; background:radial-gradient(circle,rgba(90,157,255,.20),rgba(90,157,255,0)); }
.eyebrow { color:#367FE1; font-weight:750; letter-spacing:.12em; text-transform:uppercase; font-size:.72rem; margin-bottom:10px; }
.hero-title { color:#12233D; font-weight:800; font-size:clamp(2rem,4vw,3.2rem); line-height:1.05; letter-spacing:-.055em; margin:0 0 14px 0; }
.hero-copy { color:#687B96; font-size:1.02rem; line-height:1.65; max-width:760px; margin:0; }
.glass-card { background:rgba(255,255,255,.70); border:1px solid rgba(255,255,255,.94); border-radius:22px; padding:22px 24px; box-shadow:0 12px 32px rgba(32,67,120,.06); backdrop-filter:blur(18px); animation:rise .42s ease both; }
.glass-card h3 { margin:0 0 7px 0; color:#1C2F4D; font-size:1.08rem; font-weight:750; }
.glass-card p { margin:0; color:#718096; line-height:1.6; font-size:.92rem; }
.section-kicker { color:#4285E7; text-transform:uppercase; letter-spacing:.1em; font-weight:750; font-size:.72rem; margin:24px 0 6px; }
.section-title { color:#172A45; font-size:1.55rem; font-weight:790; letter-spacing:-.04em; margin:0 0 8px; }
.muted { color:#718096; }
.pill { display:inline-block; padding:6px 11px; border-radius:99px; font-size:.76rem; font-weight:700; color:#2C6DC7; background:rgba(55,130,232,.10); border:1px solid rgba(55,130,232,.13); }
.result-chip { display:inline-block; padding:9px 15px; border-radius:14px; font-size:1rem; font-weight:750; background:rgba(38,125,255,.10); color:#1C65CA; border:1px solid rgba(38,125,255,.18); }
.small-note { color:#8290A4; font-size:.80rem; }
[data-testid="stAlert"] { border-radius:16px; }
@keyframes rise { from {opacity:0; transform:translateY(9px)} to {opacity:1; transform:translateY(0)} }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner="Memuat dataset Beijing...")
def get_raw_data() -> pd.DataFrame:
    return load_dataset(str(DATA_PATH))


raw = get_raw_data()

if "model_results" not in st.session_state:
    st.session_state.model_results = {}
if "cluster_result" not in st.session_state:
    st.session_state.cluster_result = None


# --- Sidebar navigation ---
def choose_page(group_key: str) -> None:
    selected = st.session_state.get(group_key)
    if selected:
        st.session_state.active_page = selected
        for key in ["nav_intro", "nav_model", "nav_apply"]:
            if key != group_key:
                st.session_state[key] = None

if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"

with st.sidebar:
    st.markdown("""
    <div style="padding:4px 5px 20px 5px">
      <div style="font-size:1.7rem;font-weight:850;letter-spacing:-.06em;color:#172A45">Air<span style="color:#267DFF">Scope</span></div>
      <div style="font-size:.76rem;color:#8391A5;margin-top:2px">BEIJING PM2.5 · ML LAB</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='nav-group-title'>Pengantar</div>", unsafe_allow_html=True)
    intro_pages = ["Overview", "About Dataset", "Data Explorer"]
    st.radio("Pengantar", intro_pages, index=intro_pages.index(st.session_state.active_page) if st.session_state.active_page in intro_pages else None,
             format_func=lambda p: {"Overview":"⌂  Beranda", "About Dataset":"▤  Tentang Dataset", "Data Explorer":"▥  Eksplorasi Data (EDA)"}[p],
             label_visibility="collapsed", key="nav_intro", on_change=choose_page, args=("nav_intro",))

    st.markdown("<div class='nav-group-title'>Pengolahan &amp; Model</div>", unsafe_allow_html=True)
    model_pages = ["Data Preparation", "Model Lab", "K-Means Clustering"]
    st.radio("Pengolahan dan Model", model_pages, index=model_pages.index(st.session_state.active_page) if st.session_state.active_page in model_pages else None,
             format_func=lambda p: {"Data Preparation":"☷  Preprocessing", "Model Lab":"◉  Modelling & Evaluasi", "K-Means Clustering":"◌  K-Means Clustering"}[p],
             label_visibility="collapsed", key="nav_model", on_change=choose_page, args=("nav_model",))

    st.markdown("<div class='nav-group-title'>Penerapan</div>", unsafe_allow_html=True)
    apply_pages = ["Live Prediction", "Chapter Guide"]
    st.radio("Penerapan", apply_pages, index=apply_pages.index(st.session_state.active_page) if st.session_state.active_page in apply_pages else None,
             format_func=lambda p: {"Live Prediction":"♙  Prediksi PM2.5", "Chapter Guide":"✓  Kesimpulan & Saran"}[p],
             label_visibility="collapsed", key="nav_apply", on_change=choose_page, args=("nav_apply",))

    page = st.session_state.active_page
    st.divider()
    st.markdown("<div class='nav-group-title'>Unggah dataset (opsional)</div>", unsafe_allow_html=True)
    custom_dataset = st.file_uploader("Pilih file CSV", type=["csv"], key="optional_dataset", label_visibility="collapsed")
    if custom_dataset is not None:
        try:
            candidate = pd.read_csv(custom_dataset)
            required_columns = {"year", "month", "day", "hour", "pm2.5", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"}
            missing_columns = sorted(required_columns - set(candidate.columns))
            if missing_columns:
                st.error("Format CSV belum sesuai. Kolom yang belum ada: " + ", ".join(missing_columns))
            else:
                raw = candidate
                st.success(f"Dataset kustom aktif · {len(raw):,} baris")
        except Exception as exc:
            st.error(f"CSV tidak dapat dibaca: {exc}")

    with st.expander("Pengaturan label target", expanded=False):
        low_threshold = st.number_input("Batas kelas Rendah (µg/m³)", min_value=1.0, max_value=200.0, value=35.0, step=1.0, key="target_low")
        high_threshold = st.number_input("Batas kelas Sedang (µg/m³)", min_value=2.0, max_value=500.0, value=75.0, step=1.0, key="target_high")
        st.caption("Rendah ≤ batas 1 · Sedang di atas batas 1 sampai batas 2 · Tinggi di atas batas 2.")
    if high_threshold <= low_threshold:
        st.warning("Batas kelas Sedang harus lebih besar daripada batas kelas Rendah. Nilai default 35/75 dipakai sementara.")
        low_threshold, high_threshold = 35.0, 75.0
    st.markdown("<div class='small-note'>DATASET</div>", unsafe_allow_html=True)
    st.caption("Beijing · 2010–2014 · Data per jam")
    st.caption(f"{len(raw):,} baris mentah")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='small-note'>ANGGOTA KELOMPOK</div>", unsafe_allow_html=True)
    st.markdown(
        """<div style='font-size:0.82rem; line-height:1.55; color:var(--text-color);'>
        <strong>Farel Nofri Landa</strong><br><span style='opacity:.72'>241011401676</span><br>
        <strong>Septian Cahaya Irwanto</strong><br><span style='opacity:.72'>241011402116</span><br>
        <strong>Muhammad Rojak Ardiansyah</strong><br><span style='opacity:.72'>241011400688</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='small-note'>Learning project · Streamlit + scikit-learn</div>", unsafe_allow_html=True)

# Rebuild target using the sidebar thresholds. Changing target boundaries clears
# stale supervised models, while unsupervised K-Means can remain cached in-session.
threshold_pair = (float(low_threshold), float(high_threshold))
if st.session_state.get("active_thresholds") != threshold_pair:
    st.session_state.model_results = {}
    st.session_state.active_thresholds = threshold_pair
X, y, cleaned = build_supervised_frame(raw, low_threshold, high_threshold)


def hero(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>{kicker}</div><div class='hero-title'>{title}</div><p class='hero-copy'>{copy}</p></div>",
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='muted' style='margin-bottom:14px'>{subtitle}</div>", unsafe_allow_html=True)


def glass(title: str, body: str) -> None:
    st.markdown(f"<div class='glass-card'><h3>{title}</h3><p>{body}</p></div>", unsafe_allow_html=True)


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


# --- OVERVIEW ---
if page == "Overview":
    hero(
        "BEIJING AIR QUALITY · MACHINE LEARNING",
        "Understand the air.\nExplore the patterns.",
        "Laboratorium interaktif untuk mempelajari alur machine learning dari data mentah hingga prediksi PM2.5. Dibangun mengikuti praktik Streamlit pada BAB 1–7, dengan tampilan minimalis bergaya Cupertino.",
    )
    missing_pm = int(raw["pm2.5"].isna().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observasi mentah", f"{len(raw):,}", "2010–2014")
    c2.metric("Data PM2.5 tersedia", f"{raw['pm2.5'].notna().sum():,}", f"{missing_pm:,} target kosong")
    c3.metric("Rata-rata PM2.5", f"{raw['pm2.5'].mean():.1f}", "µg/m³ · dataset")
    c4.metric("Fitur prediksi", f"{len(X.columns)}", "waktu + cuaca")

    st.markdown("<div class='section-kicker'>THE WORKFLOW</div>", unsafe_allow_html=True)
    section("Satu dataset, lima pendekatan", "Bandingkan empat model supervised learning dan satu algoritma clustering pada data Beijing yang sama.")
    feature_cols = st.columns(3)
    with feature_cols[0]: glass("01 · Explore", "Lihat struktur data, statistik deskriptif, tren PM2.5, distribusi kelas, dan korelasi antarvariabel.")
    with feature_cols[1]: glass("02 · Prepare", "Periksa missing value, buang target yang belum tersedia, bentuk label kelas, encoding fitur kategori, dan scaling numerik.")
    with feature_cols[2]: glass("03 · Train", "Latih SVM, Decision Tree, Random Forest, dan Gaussian Naïve Bayes dengan train-test split yang konsisten.")
    feature_cols2 = st.columns(2)
    with feature_cols2[0]: glass("04 · Cluster", "Uji jumlah cluster K=2–8, normalisasi fitur numerik, lalu evaluasi menggunakan inertia dan silhouette score.")
    with feature_cols2[1]: glass("05 · Predict", "Prediksi satu kondisi cuaca melalui form interaktif atau unggah CSV untuk prediksi batch.")

    col_a, col_b = st.columns([1.25, 1])
    with col_a:
        section("Rata-rata PM2.5 per tahun")
        yearly = raw.groupby("year", as_index=False)["pm2.5"].mean()
        fig = px.line(yearly, x="year", y="pm2.5", markers=True, labels={"year":"Tahun", "pm2.5":"PM2.5 (µg/m³)"})
        fig.update_traces(line=dict(width=3), marker=dict(size=9))
        fig.update_layout(height=310, margin=dict(l=12,r=12,t=12,b=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.45)")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        section("Proporsi kelas target")
        class_counts = y.value_counts().reindex(["Rendah", "Sedang", "Tinggi"]).rename_axis("Kategori").reset_index(name="Jumlah")
        fig = px.pie(class_counts, names="Kategori", values="Jumlah", hole=.66)
        fig.update_layout(height=310, margin=dict(l=12,r=12,t=12,b=12), paper_bgcolor="rgba(0,0,0,0)", legend_title_text="Kategori")
        fig.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"<div class='small-note'>Label pembelajaran: Rendah ≤ {low_threshold:g}, Sedang ({low_threshold:g}, {high_threshold:g}], Tinggi > {high_threshold:g} µg/m³. Ini adalah pengelompokan untuk praktik klasifikasi, bukan label AQI resmi.</div>", unsafe_allow_html=True)

# --- ABOUT DATASET ---
elif page == "About Dataset":
    hero("DATASET PROFILE · BEIJING 2010–2014", "Get to know the data.", "Ringkasan sumber data, cakupan observasi, variabel, dan target yang digunakan pada eksperimen kualitas udara Beijing PM2.5.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah observasi", f"{len(raw):,}")
    c2.metric("Rentang tahun", f"{int(raw['year'].min())}–{int(raw['year'].max())}")
    c3.metric("Jumlah variabel", f"{raw.shape[1]}")
    c4.metric("PM2.5 tersedia", f"{raw['pm2.5'].notna().sum():,}")
    section("Deskripsi dataset")
    st.markdown("Dataset Beijing PM2.5 berisi pengukuran kualitas udara dan kondisi meteorologi per jam. Kolom `pm2.5` menjadi dasar pembentukan label klasifikasi Rendah, Sedang, dan Tinggi. Fitur PM2.5 tidak dimasukkan sebagai input prediksi agar tidak terjadi target leakage.")
    section("Kamus variabel")
    descriptions = {
        "No":"Nomor observasi / identitas baris", "year":"Tahun pengamatan", "month":"Bulan pengamatan", "day":"Tanggal dalam bulan", "hour":"Jam pengamatan", "pm2.5":"Konsentrasi partikel PM2.5 (µg/m³)", "DEWP":"Titik embun (°C)", "TEMP":"Temperatur (°C)", "PRES":"Tekanan atmosfer (hPa)", "cbwd":"Arah angin", "Iws":"Kecepatan angin kumulatif", "Is":"Durasi akumulasi salju", "Ir":"Durasi akumulasi hujan"}
    dictionary = pd.DataFrame({"Kolom": list(raw.columns), "Keterangan": [descriptions.get(col, "Variabel pada dataset") for col in raw.columns], "Tipe data": [str(raw[col].dtype) for col in raw.columns]})
    st.dataframe(dictionary, use_container_width=True, hide_index=True)
    section("Cuplikan data")
    st.dataframe(raw.head(20), use_container_width=True)
    st.download_button("Download dataset aktif · CSV", raw.to_csv(index=False).encode("utf-8"), "beijing_pm25_dataset.csv", "text/csv")

# --- DATA EXPLORER ---
elif page == "Data Explorer":
    hero("CHAPTER 2 · DATA DISPLAY & VISUALIZATION", "Explore the dataset.", "Kenali bentuk data sebelum membangun model. Pilih periode, telusuri observasi, dan lihat bagaimana pola PM2.5 berubah menurut waktu dan kondisi cuaca.")
    years = sorted(raw["year"].unique().tolist())
    selected_years = st.multiselect("Pilih tahun", years, default=years)
    if not selected_years:
        st.warning("Pilih minimal satu tahun. Seluruh tahun ditampilkan sementara.")
        selected_years = years
    view = raw[raw["year"].isin(selected_years)].copy()
    st.caption(f"Menampilkan {len(view):,} dari {len(raw):,} observasi")
    c1,c2,c3 = st.columns(3)
    c1.metric("Rata-rata PM2.5", f"{view['pm2.5'].mean():.1f} µg/m³")
    c2.metric("Median PM2.5", f"{view['pm2.5'].median():.1f} µg/m³")
    c3.metric("Nilai maksimum", f"{view['pm2.5'].max():.0f} µg/m³")

    tab1, tab2, tab3 = st.tabs(["Data table", "Distribution", "Relationships"])
    with tab1:
        st.dataframe(view.head(500), use_container_width=True, height=400)
        st.download_button("Download data yang difilter · CSV", view.to_csv(index=False).encode("utf-8"), "beijing_pm25_filtered.csv", "text/csv")
    with tab2:
        left,right = st.columns(2)
        with left:
            hist = px.histogram(view.dropna(subset=["pm2.5"]), x="pm2.5", nbins=60, labels={"pm2.5":"PM2.5 (µg/m³)"}, title="Distribusi PM2.5")
            hist.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.4)")
            st.plotly_chart(hist, use_container_width=True)
        with right:
            by_hour = view.groupby("hour", as_index=False)["pm2.5"].mean()
            hourly = px.line(by_hour, x="hour", y="pm2.5", markers=True, title="Rata-rata PM2.5 menurut jam", labels={"hour":"Jam", "pm2.5":"PM2.5 (µg/m³)"})
            hourly.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.4)")
            st.plotly_chart(hourly, use_container_width=True)
    with tab3:
        numeric = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
        corr = view[numeric].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Korelasi fitur numerik")
        fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Korelasi menunjukkan hubungan linear, bukan bukti sebab-akibat.")

    with st.expander("Media playground · praktik BAB 2.6", expanded=False):
        st.markdown("Unggah gambar, audio, atau video untuk mencoba komponen media Streamlit: `st.image()`, `st.audio()`, dan `st.video()`.")
        media_file = st.file_uploader("Pilih media", type=["png","jpg","jpeg","webp","mp3","wav","ogg","mp4","mov"], key="media_playground")
        if media_file is not None:
            suffix = Path(media_file.name).suffix.lower()
            if suffix in [".png", ".jpg", ".jpeg", ".webp"]:
                st.image(media_file, caption=media_file.name, use_column_width=True)
            elif suffix in [".mp3", ".wav", ".ogg"]:
                st.audio(media_file)
            elif suffix in [".mp4", ".mov"]:
                st.video(media_file)

# --- DATA PREPARATION ---
elif page == "Data Preparation":
    hero("CHAPTER 3–7 · DATA CLEANING", "Prepare reliable inputs.", "Tahap prapemrosesan mengadaptasi pola di buku: cek informasi kolom, missing value, lakukan data reduction, pisahkan fitur dan target, lalu siapkan data untuk training.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Jumlah baris awal", f"{len(raw):,}")
    c2.metric("Target PM2.5 kosong", f"{raw['pm2.5'].isna().sum():,}")
    c3.metric("Baris untuk supervised ML", f"{len(cleaned):,}")
    tab1,tab2,tab3 = st.tabs(["Data quality", "Feature engineering", "Clean dataset"])
    with tab1:
        section("Missing value per kolom")
        missing = raw.isna().sum().rename("missing").to_frame()
        missing["missing_%"] = (missing["missing"] / len(raw) * 100).round(2)
        st.dataframe(missing, use_container_width=True)
        st.markdown("**Keputusan pembersihan:** baris tanpa nilai `pm2.5` tidak dapat diberi label target, sehingga dikeluarkan dari data supervised. Fitur cuaca yang kosong (jika ada pada data unggahan) akan diimputasi median/modus di dalam pipeline.")
        st.markdown("**Data reduction:** kolom `No` diperlakukan sebagai ID baris dan tidak digunakan. Kolom `day` tidak dipakai sebagai fitur langsung; pola waktu direpresentasikan dengan encoding siklik untuk jam dan bulan.")
    with tab2:
        section("Fitur input dan target")
        st.markdown("**Target (y):** kelas PM2.5 yang diturunkan dari kolom `pm2.5`.\n\n**Input (X):** tahun, titik embun, temperatur, tekanan, kecepatan angin, akumulasi salju/hujan, jam dan bulan (encoding siklik), serta arah angin.")
        st.dataframe(pd.DataFrame({"feature":X.columns, "type":["kategori" if c=="cbwd" else "numerik" for c in X.columns], "role":["Input" for _ in X.columns]}), use_container_width=True)
        st.markdown("**Encoding kategori:** One-Hot Encoding untuk `cbwd`. **Scaling:** StandardScaler untuk fitur numerik. Keduanya dipasang dalam `Pipeline` agar preprocessing hanya dipelajari dari data training dan tidak bocor ke data test.")
        dist = y.value_counts().reindex(["Rendah","Sedang","Tinggi"]).rename_axis("kelas").reset_index(name="jumlah")
        dist["persentase"] = (dist["jumlah"] / dist["jumlah"].sum() * 100).round(2)
        st.dataframe(dist, use_container_width=True)
    with tab3:
        st.dataframe(cleaned[["year","month","day","hour","pm2.5","DEWP","TEMP","PRES","cbwd","Iws","Is","Ir"]].head(1000), use_container_width=True, height=420)
        cleaned_export = cleaned.copy()
        cleaned_export["pm25_class"] = y.values
        st.download_button("Download cleaned dataset · CSV", cleaned_export.to_csv(index=False).encode("utf-8"), "beijing_pm25_cleaned.csv", "text/csv")

# --- MODEL LAB ---
elif page == "Model Lab":
    hero("CHAPTER 3 · 4 · 5 · 7 · SUPERVISED LEARNING", "Train. Compare. Evaluate.", "Latih empat algoritma klasifikasi dengan pembagian train-test yang sama. Evaluasi dilakukan menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.")
    names = list(MODEL_LABELS.keys())
    defaults = names
    selected_models = st.multiselect("Pilih algoritma", names, default=defaults)
    a,b,c,d = st.columns(4)
    with a: sample_size = st.slider("Ukuran sampel", 2000, min(20000, len(X)), 8000, step=1000)
    with b: test_size_pct = st.select_slider("Data testing", options=[20,25,30], value=30, format_func=lambda v:f"{v}%")
    with c: rf_trees = st.slider("Jumlah pohon RF", 40, 200, 100, step=20)
    with d: tree_depth = st.slider("Kedalaman pohon", 3, 30, 12)
    svm_c = st.select_slider("Regularisasi SVM (C)", options=[0.1,0.5,1.0,2.0,5.0], value=1.0)
    st.caption("Pembagian 70:30 mengikuti contoh Random Forest di BAB 5 dan SVM pada BAB 3. BAB 7 juga mencontohkan split 80:20; pilih 20% di atas untuk mencoba variasi tersebut.")
    if st.button("Train & evaluate selected models", type="primary", disabled=(not selected_models)):
        progress = st.progress(0, text="Menyiapkan data...")
        results = {}
        for idx, name in enumerate(selected_models):
            progress.progress(idx / len(selected_models), text=f"Training {MODEL_LABELS[name]}...")
            results[name] = train_classifier(
                name, X, y, test_size=test_size_pct/100,
                sample_size=sample_size, rf_trees=rf_trees,
                tree_depth=tree_depth, svm_c=svm_c,
            )
        st.session_state.model_results = results
        progress.progress(1.0, text="Semua model selesai dilatih.")
        st.success("Training dan evaluasi selesai. Hasil tersimpan selama sesi aplikasi aktif.")

    results = st.session_state.model_results
    if results:
        section("Model comparison", "Semua metrik dihitung pada test split dari sampel yang sama untuk perbandingan yang lebih adil.")
        rows = []
        for name, result in results.items():
            m = result["metrics"]
            rows.append({"Model": MODEL_LABELS[name], "Accuracy":m["accuracy"], "Precision":m["precision_weighted"], "Recall":m["recall_weighted"], "F1-score":m["f1_weighted"], "Train rows":m["train_rows"], "Test rows":m["test_rows"]})
        metric_df = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
        cols = st.columns(4)
        best = metric_df.iloc[0]
        for col, key, label in zip(cols, ["Accuracy","Precision","Recall","F1-score"], ["Top accuracy","Top precision","Top recall","Top F1-score"]):
            col.metric(label, f"{best[key]*100:.1f}%", best["Model"])
        st.dataframe(metric_df.style.format({"Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}","F1-score":"{:.3f}"}), use_container_width=True)
        melted = metric_df.melt(id_vars="Model", value_vars=["Accuracy","Precision","Recall","F1-score"], var_name="Metric", value_name="Score")
        fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group", range_y=[0,1], title="Perbandingan metrik test")
        fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.45)")
        st.plotly_chart(fig, use_container_width=True)

        chosen = st.selectbox("Lihat detail evaluasi", list(results.keys()), format_func=lambda n:MODEL_LABELS[n])
        selected = results[chosen]
        left,right = st.columns([1,1.15])
        with left:
            section("Confusion matrix")
            cm_fig = px.imshow(selected["confusion_matrix"], x=selected["labels"], y=selected["labels"], text_auto=True, color_continuous_scale="Blues", labels={"x":"Prediksi", "y":"Aktual", "color":"Jumlah"})
            cm_fig.update_layout(height=390, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(cm_fig, use_container_width=True)
        with right:
            section("Classification report")
            report = selected["report"]
            report_df = pd.DataFrame({k:{"precision":v.get("precision",0),"recall":v.get("recall",0),"f1-score":v.get("f1-score",0),"support":v.get("support",0)} for k,v in report.items() if isinstance(v,dict)}).T
            report_df = report_df.loc[[r for r in ["Rendah","Sedang","Tinggi","weighted avg","macro avg"] if r in report_df.index]]
            st.dataframe(report_df.style.format({"precision":"{:.3f}","recall":"{:.3f}","f1-score":"{:.3f}","support":"{:.0f}"}), use_container_width=True)
        importance = selected.get("feature_importance")
        if importance is not None:
            section("Feature importance", "Tersedia pada Decision Tree dan Random Forest berdasarkan feature_importances_ dari model.")
            imp = importance.head(12).sort_values("importance")
            fig = px.bar(imp, x="importance", y="feature", orientation="h", title=f"Top feature · {MODEL_LABELS[chosen]}")
            fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.45)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Algoritma ini tidak menyediakan feature_importances_ bawaan. Gunakan metrik evaluasi dan confusion matrix untuk membaca performanya.")

        st.divider()
        section("Export model", "Pipeline yang diunduh sudah mencakup preprocessing dan estimator, sehingga transformasi input saat prediksi tetap konsisten.")
        buffer = io.BytesIO()
        pickle.dump({"pipeline":selected["pipeline"], "algorithm":chosen, "feature_columns":list(X.columns), "classes":selected["labels"], "thresholds":{"low":low_threshold,"high":high_threshold}}, buffer)
        st.download_button(f"Download {chosen} pipeline · PKL", buffer.getvalue(), f"beijing_pm25_{chosen.lower().replace(' ','_')}.pkl", "application/octet-stream")
        st.download_button("Download comparison · CSV", metric_df.to_csv(index=False).encode("utf-8"), "model_comparison.csv", "text/csv")
    else:
        glass("Belum ada model yang dilatih", "Pilih algoritma dan parameter di atas, lalu tekan Train & evaluate. Hasil evaluasi akan muncul di halaman ini dan dapat digunakan pada halaman Live Prediction.")

# --- KMEANS ---
elif page == "K-Means Clustering":
    hero("CHAPTER 6 · UNSUPERVISED LEARNING", "Find hidden patterns.", "K-Means mengelompokkan kondisi meteorologi tanpa menggunakan label kelas PM2.5. Fitur numerik dinormalisasi terlebih dahulu, lalu jumlah cluster dievaluasi pada rentang K=2 sampai 8.")
    c1,c2,c3 = st.columns(3)
    with c1: k_value = st.slider("Jumlah cluster (K)", 2, 8, 4)
    with c2: cluster_sample = st.slider("Sampel clustering", 1000, 8000, 3500, step=500)
    with c3: max_k = st.select_slider("Rentang uji K", options=[5,6,7,8], value=8)
    st.caption("K-Means hanya memakai fitur cuaca/waktu yang telah distandardisasi; kolom pm2.5 tidak digunakan untuk membentuk cluster.")
    if st.button("Run K-Means experiment", type="primary"):
        with st.spinner("Menghitung elbow dan silhouette score..."):
            st.session_state.cluster_result = run_kmeans(raw, k=k_value, sample_size=cluster_sample, max_k=max(max_k, k_value))
        st.success("Clustering selesai.")
    # Ensure that the selected K is included in the K range used for validation.
    max_k = max(max_k, k_value)
    result = st.session_state.cluster_result
    if result:
        c1,c2,c3 = st.columns(3)
        c1.metric("Silhouette score", f"{result['silhouette']:.3f}")
        c2.metric("Inertia", f"{result['inertia']:,.0f}")
        c3.metric("Jumlah data", f"{result['sample_rows']:,}")
        left,right = st.columns(2)
        with left:
            section("Elbow method", "Inertia yang makin kecil berarti cluster makin rapat; nilai K dipilih dengan mempertimbangkan titik tekuk dan interpretasi.")
            fig = px.line(result["elbow"], x="k", y="inertia", markers=True, labels={"k":"Jumlah cluster (K)","inertia":"Inertia"})
            fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.45)")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            section("Silhouette score", "Nilai lebih tinggi (mendekati 1) umumnya menunjukkan cluster yang lebih kompak dan terpisah.")
            fig = px.line(result["elbow"], x="k", y="silhouette", markers=True, labels={"k":"Jumlah cluster (K)","silhouette":"Silhouette"})
            fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.45)")
            st.plotly_chart(fig, use_container_width=True)
        section("Cluster map", "Pilih dua dimensi numerik untuk melihat sebaran anggota cluster. Sumbu menggunakan nilai asli agar mudah ditafsirkan.")
        numeric_options = ["DEWP","TEMP","PRES","Iws","Is","Ir","hour","month"]
        default_x = "TEMP"; default_y = "DEWP"
        ca,cb = st.columns(2)
        with ca: x_axis = st.selectbox("Sumbu X", numeric_options, index=numeric_options.index(default_x))
        with cb: y_axis = st.selectbox("Sumbu Y", numeric_options, index=numeric_options.index(default_y))
        plot_data = result["data"].copy()
        fig = px.scatter(plot_data, x=x_axis, y=y_axis, color=plot_data["cluster"].astype(str), opacity=.60, labels={"color":"Cluster"}, title=f"Clustering by {x_axis} × {y_axis}")
        fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.45)")
        st.plotly_chart(fig, use_container_width=True)
        section("Cluster profile")
        st.dataframe(result["profile"].round(2), use_container_width=True)
        payload = io.BytesIO()
        pickle.dump({"model":result["model"],"scaler":result["scaler"],"features":result["features"]}, payload)
        st.download_button("Download K-Means model + scaler · PKL", payload.getvalue(), "beijing_pm25_kmeans.pkl", "application/octet-stream")
        st.download_button("Download cluster assignments · CSV", plot_data.to_csv(index=False).encode("utf-8"), "beijing_pm25_clusters.csv", "text/csv")
    else:
        glass("Mulai eksperimen clustering", "Atur K dan ukuran sampel, jalankan eksperimen, lalu bandingkan inertia dan silhouette score untuk menentukan jumlah cluster yang masuk akal.")

# --- LIVE PREDICTION ---
elif page == "Live Prediction":
    hero("MODEL INFERENCE · INTERACTIVE FORM", "Predict a PM2.5 class.", "Masukkan kondisi waktu dan cuaca untuk mendapatkan kelas PM2.5 hasil prediksi model yang sudah dilatih. Gunakan satu baris data atau unggah beberapa baris sekaligus.")
    results = st.session_state.model_results
    if not results:
        st.warning("Latih model terlebih dahulu di halaman Model Lab. Setelah training selesai, model akan tersedia di sini.")
    else:
        options = list(results.keys())
        model_name = st.selectbox("Model prediksi", options, format_func=lambda n:MODEL_LABELS[n])
        chosen = results[model_name]["pipeline"]
        tab1,tab2 = st.tabs(["Single prediction", "Batch prediction"])
        with tab1:
            with st.form("single_prediction"):
                st.markdown("**Waktu observasi**")
                c1,c2,c3 = st.columns(3)
                year = c1.selectbox("Tahun", [2010,2011,2012,2013,2014], index=4)
                month = c2.selectbox("Bulan", list(range(1,13)), index=5)
                hour = c3.slider("Jam", 0, 23, 8)
                st.markdown("**Kondisi meteorologi**")
                a,b,c = st.columns(3)
                dewp = a.number_input("DEWP · titik embun (°C)", -40.0, 30.0, -5.0, step=0.5)
                temp = b.number_input("TEMP · suhu (°C)", -30.0, 45.0, 10.0, step=0.5)
                pres = c.number_input("PRES · tekanan (hPa)", 980.0, 1050.0, 1015.0, step=0.5)
                d,e,f = st.columns(3)
                cbwd = d.selectbox("Arah angin (cbwd)", ["NW","NE","SE","cv"])
                iws = e.number_input("Iws · kecepatan angin", 0.0, 500.0, 5.0, step=0.5)
                snow = f.number_input("Is · akumulasi salju", 0.0, 100.0, 0.0, step=1.0)
                rain = st.number_input("Ir · curah hujan", 0.0, 100.0, 0.0, step=1.0)
                submitted = st.form_submit_button("Predict PM2.5 class", type="primary", use_container_width=True)
            if submitted:
                row = {"year":year,"month":month,"hour":hour,"DEWP":dewp,"TEMP":temp,"PRES":pres,"cbwd":cbwd,"Iws":iws,"Is":snow,"Ir":rain}
                inp = prepare_prediction_input(row)
                prediction = chosen.predict(inp)[0]
                st.markdown(f"<div class='glass-card'><div class='eyebrow'>PREDICTION RESULT</div><div style='font-size:1.65rem;font-weight:820;color:#172A45;margin-bottom:8px'>{prediction}</div><p>Model: {MODEL_LABELS[model_name]}. Keluaran merupakan kelas pembelajaran dari batas target yang ditetapkan pada proses pelatihan.</p></div>", unsafe_allow_html=True)
                if hasattr(chosen, "predict_proba"):
                    proba = chosen.predict_proba(inp)[0]
                    prob_df = pd.DataFrame({"Kelas":chosen.classes_,"Probabilitas":proba}).sort_values("Probabilitas",ascending=False)
                    fig = px.bar(prob_df, x="Kelas", y="Probabilitas", range_y=[0,1], text=prob_df["Probabilitas"].map(lambda v:f"{v:.1%}"), title="Probabilitas tiap kelas")
                    fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.45)")
                    st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.markdown("Unggah CSV yang memiliki kolom `year, month, hour, DEWP, TEMP, PRES, cbwd, Iws, Is, Ir`. Baris tambahan lain akan dipertahankan pada file hasil.")
            uploaded = st.file_uploader("Upload CSV untuk batch prediction", type=["csv"])
            if uploaded is not None:
                batch = pd.read_csv(uploaded)
                try:
                    batch_features = prepare_prediction_input(batch)
                    batch["predicted_pm25_class"] = chosen.predict(batch_features)
                    if hasattr(chosen,"predict_proba"):
                        probs = chosen.predict_proba(batch_features)
                        for idx, class_name in enumerate(chosen.classes_):
                            batch[f"prob_{class_name.lower()}"] = probs[:,idx]
                    st.dataframe(batch.head(100), use_container_width=True)
                    st.download_button("Download hasil prediksi batch · CSV", batch.to_csv(index=False).encode("utf-8"), "beijing_pm25_batch_predictions.csv", "text/csv")
                except Exception as exc:
                    st.error(f"Tidak dapat memproses file: {exc}")

# --- CHAPTER GUIDE ---
elif page == "Chapter Guide":
    hero("LEARNING ROADMAP · BUKU BAB 1–7", "From first app to ML models.", "Peta praktik yang menghubungkan tahapan pada buku *Implementasi Algoritma Machine Learning Berbasis Web dengan Framework Streamlit* dengan implementasi AirScope.")
    chapters = [
        ("BAB 1", "Pengenalan Framework Web Streamlit", ["Tentang Streamlit dan perannya untuk aplikasi data science/ML", "Instalasi Python, virtual environment, Streamlit, Matplotlib, scikit-learn", "Aplikasi awal melalui `streamlit run app.py`"], "Diaktifkan oleh struktur proyek, requirements.txt, dan aplikasi AirScope."),
        ("BAB 2", "Membuat Aplikasi Dasar Streamlit", ["Elemen teks: title, header, caption, markdown", "Menampilkan data: dataframe, tabel, metric", "Grafik: line, histogram, pie, heatmap, scatter", "Input form: selectbox, slider, multiselect, number_input, file_uploader", "Media: image, audio, dan video melalui uploader", "Layout: sidebar, columns, tabs, container, expander", "Control flow: navigasi halaman, tombol, kondisi, dan status training"], "Dipakai di seluruh halaman dashboard."),
        ("BAB 3", "Support Vector Machine (SVM)", ["Memuat dataset dan melihat head / struktur", "Cek missing values dan data reduction", "Menentukan X dan y, split 70:30", "Preprocessing numerik/kategorikal dan scaling", "Training SVM linear, evaluasi classification report + confusion matrix", "Menyimpan pipeline model dengan pickle"], "Model SVM tersedia di Model Lab dan Live Prediction."),
        ("BAB 4", "Decision Tree (C4.5)", ["Pemeriksaan data dan pembersihan fitur", "Pemisahan data input (X) dan target (y)", "Training Decision Tree dengan entropy", "Evaluasi accuracy, precision, recall, F1, confusion matrix", "Menyimpan model dan menggunakannya melalui form prediksi"], "Decision Tree entropy dan feature importance tersedia."),
        ("BAB 5", "Random Forest", ["Pembersihan dan data reduction", "Train-test split 70:30", "Training ensemble pohon keputusan", "Membaca metrik evaluasi dan feature importance", "Menyimpan pipeline Random Forest"], "Jumlah pohon dan kedalaman dapat diatur di Model Lab."),
        ("BAB 6", "K-Means Clustering", ["Memilih fitur numerik dan melakukan normalisasi", "Training K-Means dengan nilai K tertentu", "Membandingkan rentang K=2–8", "Evaluasi inertia dan silhouette score", "Visualisasi cluster dan menyimpan model + scaler"], "Eksperimen cluster dan ekspor hasil tersedia di K-Means Clustering."),
        ("BAB 7", "Naïve Bayes", ["Memuat dan memeriksa dataset", "Menangani missing values dan menentukan target", "Split train-test (contoh buku 80:20)", "Training Gaussian Naive Bayes", "Evaluasi classification report dan confusion matrix", "Menyimpan model dengan pickle"], "GaussianNB tersedia; rasio test 20% dapat dipilih di Model Lab."),
    ]
    for chapter, title, tasks, note in chapters:
        with st.expander(f"{chapter} · {title}", expanded=(chapter in ["BAB 1","BAB 2"])):
            for task in tasks:
                st.markdown(f"- ✓ {task}")
            st.caption(note)
    st.divider()
    section("Catatan adaptasi dataset")
    st.markdown("Buku menggunakan dataset yang berbeda untuk tiap algoritma (kanker, diabetes, Iris, perumahan, dan penyakit jantung). Pada proyek ini seluruh algoritma dipraktikkan pada satu dataset Beijing agar hasil klasifikasi dapat dibandingkan secara konsisten. Target PM2.5 dikonversi menjadi tiga kelas untuk algoritma supervised, sedangkan K-Means dijalankan tanpa target.")


# Common footer
st.markdown("<div style='height:22px'></div><hr><div class='small-note'>AirScope · Beijing PM2.5 Machine Learning Lab · Educational use only · Pipeline trained in-session, not a regulatory air-quality service.</div>", unsafe_allow_html=True)
