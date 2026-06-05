import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
import io

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SPK Pemilihan RAM Terbaik",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace;
    }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 1.8rem;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0;
        opacity: 0.75;
        font-size: 0.9rem;
    }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .metric-card.gold  { border-left-color: #f59e0b; }
    .metric-card.silver{ border-left-color: #94a3b8; }
    .metric-card.bronze{ border-left-color: #d97706; }

    .rank-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
    }
    .badge-smart  { background:#dbeafe; color:#1d4ed8; }
    .badge-saw    { background:#dcfce7; color:#16a34a; }
    .badge-topsis { background:#fce7f3; color:#be185d; }

    .section-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.4rem;
    }

    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.87rem;
        color: #1e40af;
    }
    .warn-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.87rem;
        color: #92400e;
    }
    div[data-testid="stSidebar"] {
        background: #0f172a;
        color: white;
    }
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label {
        color: #cbd5e1 !important;
    }
    div[data-testid="stSidebar"] h3 {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>SPK Pemilihan RAM Terbaik</h1>
    <p>Sistem Penunjang Keputusan &nbsp;|&nbsp; Metode SMART, SAW & TOPSIS &nbsp;|&nbsp; Dataset: DDR2/DDR3/DDR4/DDR5 RAM Benchmarks</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CORE SPK FUNCTIONS
# ─────────────────────────────────────────────

def smart_method(df, criteria, weights, criteria_types):
    X = df[criteria].values.astype(float)
    utility_matrix = np.zeros_like(X)
    for j, col in enumerate(criteria):
        col_vals = X[:, j]
        min_val, max_val = col_vals.min(), col_vals.max()
        if max_val == min_val:
            utility_matrix[:, j] = 100
        elif criteria_types[col] == "benefit":
            utility_matrix[:, j] = ((col_vals - min_val) / (max_val - min_val)) * 100
        else:
            utility_matrix[:, j] = ((max_val - col_vals) / (max_val - min_val)) * 100
    smart_scores = utility_matrix @ weights
    df_result = pd.DataFrame({"RAM": df["RAM"].values, "SMART_Score": smart_scores})
    df_result["SMART_Rank"] = df_result["SMART_Score"].rank(ascending=False).astype(int)
    df_result = df_result.sort_values("SMART_Score", ascending=False).reset_index(drop=True)
    df_utility = pd.DataFrame(utility_matrix, columns=criteria, index=df["RAM"].values)
    return df_result, df_utility


def saw_method(df, criteria, weights, criteria_types):
    X = df[criteria].values.astype(float)
    R = np.zeros_like(X)
    for j, col in enumerate(criteria):
        col_vals = X[:, j]
        if criteria_types[col] == "benefit":
            max_val = col_vals.max()
            R[:, j] = col_vals / max_val if max_val != 0 else 1
        else:
            min_val = col_vals.min()
            with np.errstate(divide="ignore", invalid="ignore"):
                R[:, j] = np.where(col_vals != 0, min_val / col_vals, 1)
    Vi = (R * weights).sum(axis=1)
    df_result = pd.DataFrame({"RAM": df["RAM"].values, "SAW_Score": Vi})
    df_result["SAW_Rank"] = df_result["SAW_Score"].rank(ascending=False).astype(int)
    df_result = df_result.sort_values("SAW_Score", ascending=False).reset_index(drop=True)
    df_R = pd.DataFrame(R, columns=criteria, index=df["RAM"].values)
    return df_result, df_R


def topsis_method(df, criteria, weights, criteria_types):
    X = df[criteria].values.astype(float)
    R = np.zeros_like(X)
    for j in range(len(criteria)):
        norm = np.sqrt(np.sum(X[:, j] ** 2))
        R[:, j] = X[:, j] / norm if norm != 0 else X[:, j]
    V = R * weights
    A_plus  = np.array([V[:, j].max() if criteria_types[criteria[j]] == "benefit" else V[:, j].min() for j in range(len(criteria))])
    A_minus = np.array([V[:, j].min() if criteria_types[criteria[j]] == "benefit" else V[:, j].max() for j in range(len(criteria))])
    D_plus  = np.sqrt(np.sum((V - A_plus)  ** 2, axis=1))
    D_minus = np.sqrt(np.sum((V - A_minus) ** 2, axis=1))
    denom = D_plus + D_minus
    Ci = np.where(denom != 0, D_minus / denom, 0)
    df_result = pd.DataFrame({"RAM": df["RAM"].values, "D_plus": D_plus, "D_minus": D_minus, "TOPSIS_Score": Ci})
    df_result["TOPSIS_Rank"] = df_result["TOPSIS_Score"].rank(ascending=False).astype(int)
    df_result = df_result.sort_values("TOPSIS_Score", ascending=False).reset_index(drop=True)
    details = {"D_plus": D_plus, "D_minus": D_minus, "A_plus": A_plus, "A_minus": A_minus, "V": V}
    return df_result, details


def run_saw_fast(X, weights, criteria_types_list):
    R = np.zeros_like(X)
    for j, ctype in enumerate(criteria_types_list):
        col_vals = X[:, j]
        if ctype == "benefit":
            max_val = col_vals.max()
            R[:, j] = col_vals / max_val if max_val != 0 else 1
        else:
            min_val = col_vals.min()
            with np.errstate(divide="ignore", invalid="ignore"):
                R[:, j] = np.where(col_vals != 0, min_val / col_vals, 1)
    return (R * weights).sum(axis=1)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"],
        help="Upload file RAM_Benchmarks_megalist.csv dari Kaggle",
    )

    st.markdown("---")
    st.markdown("### Filter Data")
    gen_filter = st.multiselect(
        "Generasi RAM",
        options=["DDR2", "DDR3", "DDR4", "DDR5"],
        default=["DDR4", "DDR5"],
    )
    top_n = st.slider("Jumlah Alternatif (Top N per generasi)", 5, 30, 20)

    st.markdown("---")
    st.markdown("### Bobot Kriteria")
    st.caption("Total bobot harus = 1.0")

    w_latency     = st.slider("Latency (Cost)",      0.0, 1.0, 0.25, 0.05)
    w_readUncached= st.slider("Read Uncached (Benefit)", 0.0, 1.0, 0.35, 0.05)
    w_write       = st.slider("Write (Benefit)",     0.0, 1.0, 0.25, 0.05)
    w_price       = st.slider("Price (Cost)",        0.0, 1.0, 0.15, 0.05)

    total_w = w_latency + w_readUncached + w_write + w_price
    if abs(total_w - 1.0) > 0.001:
        st.warning(f"Total bobot: {total_w:.2f} (harus 1.00)")
    else:
        st.success(f"Total bobot: {total_w:.2f} ✓")

    st.markdown("---")
    st.markdown("### Tipe Kriteria")
    st.caption("Ditentukan otomatis berdasarkan nama kolom")

    st.markdown("---")
    st.markdown("### Tentang Aplikasi")
    st.markdown(
        """
        <div style='color:#94a3b8;font-size:0.8rem;line-height:1.6'>
        Dataset: DDR2/DDR3/DDR4/DDR5 RAM Benchmarks<br>
        Sumber: Kaggle<br><br>
        Metode SPK:<br>
        • SMART<br>
        • SAW<br>
        • TOPSIS
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


if uploaded is None:
    st.markdown("""
    <div class="warn-box">
        <b>Belum ada dataset.</b> Upload file <code>RAM_Benchmarks_megalist.csv</code> dari Kaggle di sidebar kiri.<br><br>
        Download dataset di: <a href="https://www.kaggle.com/datasets/alanjo/ddr2ddr3ddr4ddr5-ram-benchmarks" target="_blank">
        kaggle.com/datasets/alanjo/ddr2ddr3ddr4ddr5-ram-benchmarks</a>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df_raw = load_data(uploaded.read())

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
CRITERIA      = ["latency", "readUncached", "write", "price"]
CRITERIA_TYPES = {
    "latency":      "cost",
    "readUncached": "benefit",
    "write":        "benefit",
    "price":        "cost",
}

# Validasi kolom
missing_cols = [c for c in CRITERIA if c not in df_raw.columns]
if missing_cols:
    st.error(f"Kolom tidak ditemukan di dataset: {missing_cols}")
    st.stop()

# Filter generasi
if gen_filter:
    df_filtered = df_raw[df_raw["gen"].isin(gen_filter)].copy()
else:
    df_filtered = df_raw.copy()

# Drop baris yang tidak punya data di semua kriteria (hapus NaN di price juga ok)
df_filtered = df_filtered.dropna(subset=["latency", "readUncached", "write"])
df_filtered = df_filtered.reset_index(drop=True)

# Ambil top N per generasi berdasarkan readUncached
frames = []
for g in (gen_filter if gen_filter else df_filtered["gen"].unique()):
    sub = df_filtered[df_filtered["gen"] == g].nlargest(top_n, "readUncached")
    frames.append(sub)
df_work = pd.concat(frames).drop_duplicates().reset_index(drop=True)

# Jika price kosong, isi dengan median per generasi (agar price tetap bisa jadi kriteria)
for g in df_work["gen"].unique():
    mask = (df_work["gen"] == g) & (df_work["price"].isna())
    med  = df_work.loc[df_work["gen"] == g, "price"].median()
    df_work.loc[mask, "price"] = med if not np.isnan(med) else df_work["price"].median()

df_work = df_work.dropna(subset=CRITERIA).reset_index(drop=True)
df_work.rename(columns={"memoryName": "RAM"}, inplace=True)

df_spk = df_work[["RAM", "gen"] + CRITERIA].copy()

# Normalisasi bobot
raw_weights = np.array([w_latency, w_readUncached, w_write, w_price])
if raw_weights.sum() == 0:
    raw_weights = np.ones(4)
weights = raw_weights / raw_weights.sum()

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dataset",
    "SMART",
    "SAW",
    "TOPSIS",
    "📈 Perbandingan",
    "🔍 Sensitivitas",
])


# ══════════════════════════════════════════════
# TAB 1 — DATASET
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Overview Dataset</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data Asli",    f"{len(df_raw):,}")
    c2.metric("Setelah Filter",     f"{len(df_work):,}")
    c3.metric("Generasi Dipilih",   ", ".join(gen_filter) if gen_filter else "Semua")
    c4.metric("Alternatif SPK",     len(df_spk))

    st.markdown("---")

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.markdown('<p class="section-title">Preview Data Alternatif</p>', unsafe_allow_html=True)
        st.dataframe(
            df_spk.style.format({
                "latency":      "{:.0f} ns",
                "readUncached": "{:,.0f} MB/s",
                "write":        "{:,.0f} MB/s",
                "price":        "${:.2f}",
            }),
            use_container_width=True, height=380,
        )

    with col_right:
        st.markdown('<p class="section-title">Statistik Kriteria</p>', unsafe_allow_html=True)
        desc = df_spk[CRITERIA].describe().round(2)
        st.dataframe(desc, use_container_width=True)

        st.markdown('<p class="section-title" style="margin-top:1rem">Bobot Kriteria</p>', unsafe_allow_html=True)
        wdf = pd.DataFrame({
            "Kriteria": ["Latency", "Read Uncached", "Write", "Price"],
            "Tipe":     ["Cost", "Benefit", "Benefit", "Cost"],
            "Bobot":    weights.round(4),
        })
        st.dataframe(wdf, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Distribusi Nilai Kriteria</p>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b"]
    labels_map = {
        "latency":      "Latency (ns)",
        "readUncached": "Read Uncached (MB/s)",
        "write":        "Write (MB/s)",
        "price":        "Price (USD)",
    }
    for ax, col, color in zip(axes, CRITERIA, colors):
        data = df_spk[col].dropna()
        ax.hist(data, bins=15, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Mean={data.mean():.0f}")
        ax.set_title(labels_map[col], fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylabel("Frekuensi")
        sns.despine(ax=ax)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<p class="section-title">Heatmap Korelasi</p>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    corr = df_spk[CRITERIA].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, square=True, linewidths=1, ax=ax2,
                cbar_kws={"shrink": 0.8})
    ax2.set_title("Korelasi Antar Kriteria", fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════
# RUN SPK
# ══════════════════════════════════════════════
smart_result, smart_utility = smart_method(df_spk, CRITERIA, weights, CRITERIA_TYPES)
saw_result,   saw_R         = saw_method(  df_spk, CRITERIA, weights, CRITERIA_TYPES)
topsis_result, topsis_det   = topsis_method(df_spk, CRITERIA, weights, CRITERIA_TYPES)

# Gabungkan
df_comp = df_spk[["RAM", "gen"]].copy()
df_comp = (df_comp
    .merge(smart_result[["RAM","SMART_Score","SMART_Rank"]], on="RAM")
    .merge(saw_result[["RAM","SAW_Score","SAW_Rank"]], on="RAM")
    .merge(topsis_result[["RAM","TOPSIS_Score","TOPSIS_Rank"]], on="RAM")
)
df_comp["Avg_Rank"]   = (df_comp["SMART_Rank"] + df_comp["SAW_Rank"] + df_comp["TOPSIS_Rank"]) / 3
df_comp["Final_Rank"] = df_comp["Avg_Rank"].rank().astype(int)
df_comp = df_comp.sort_values("Final_Rank").reset_index(drop=True)


def show_method_tab(tab, method_name, score_col, rank_col, df_result, extra_cols=None):
    with tab:
        st.markdown(f'<p class="section-title">Hasil Ranking — Metode {method_name}</p>', unsafe_allow_html=True)

        # Top 3 cards
        top3 = df_result.head(3)
        card_styles = ["gold", "silver", "bronze"]
        rank_labels = ["#1 Terbaik", "#2 Runner-up", "#3 Peringkat 3"]
        cols = st.columns(3)
        for idx, (_, row) in enumerate(top3.iterrows()):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card {card_styles[idx]}">
                    <div style="font-size:0.7rem;color:#64748b;font-family:IBM Plex Mono,monospace;
                                letter-spacing:1px;text-transform:uppercase">{rank_labels[idx]}</div>
                    <div style="font-size:1rem;font-weight:700;margin:4px 0">{row['RAM'][:35]}</div>
                    <div style="font-family:IBM Plex Mono,monospace;font-size:1.3rem;color:#0f172a">
                        {row[score_col]:.4f}
                    </div>
                    <div style="font-size:0.75rem;color:#64748b">Skor {method_name}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Full table
        display_cols = ["RAM", score_col, rank_col]
        if extra_cols:
            display_cols = ["RAM"] + extra_cols + [score_col, rank_col]

        merged = df_result[display_cols].merge(
            df_spk[["RAM","gen"]], on="RAM", how="left"
        )
        display_cols_final = ["RAM","gen"] + [c for c in display_cols if c != "RAM"]
        st.dataframe(
            merged[display_cols_final].style.format(
                {score_col: "{:.4f}", **({c: "{:.4f}" for c in (extra_cols or [])})}
            ).background_gradient(subset=[score_col], cmap="Blues"),
            use_container_width=True, hide_index=True,
        )

        st.markdown("---")

        # Bar chart skor
        st.markdown('<p class="section-title">Distribusi Skor</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, max(4, len(df_result) * 0.35)))
        sorted_df = df_result.sort_values(score_col, ascending=True)
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(sorted_df[score_col].min(), sorted_df[score_col].max())
        bars = ax.barh(
            sorted_df["RAM"].str[:30],
            sorted_df[score_col],
            color=cmap(norm(sorted_df[score_col].values)),
            edgecolor="white", linewidth=0.4,
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.7, label=f"Skor {method_name}")
        ax.set_xlabel(f"Skor {method_name}", fontsize=10)
        ax.set_title(f"Ranking Semua Alternatif — Metode {method_name}", fontsize=12, fontweight="bold")
        ax.axvline(sorted_df[score_col].mean(), color="navy", linestyle="--", linewidth=1, alpha=0.6, label="Mean")
        ax.legend(fontsize=8)
        sns.despine(ax=ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Matriks Normalisasi
        st.markdown("---")
        st.markdown('<p class="section-title">Detail Matriks</p>', unsafe_allow_html=True)

        if method_name == "SMART":
            st.caption("Matriks Utilitas (0–100)")
            st.dataframe(
                smart_utility.style.format("{:.2f}").background_gradient(cmap="Blues"),
                use_container_width=True,
            )
        elif method_name == "SAW":
            st.caption("Matriks Normalisasi R")
            st.dataframe(
                saw_R.style.format("{:.4f}").background_gradient(cmap="Blues"),
                use_container_width=True,
            )
        elif method_name == "TOPSIS":
            st.caption("Matriks Terbobot V + Jarak Ideal")
            V_df = pd.DataFrame(topsis_det["V"], columns=CRITERIA, index=df_spk["RAM"].values)
            dist_df = pd.DataFrame({
                "RAM": df_spk["RAM"].values,
                "D+ (ke A+)": topsis_det["D_plus"].round(6),
                "D- (ke A-)": topsis_det["D_minus"].round(6),
            }).set_index("RAM")
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Matriks Terbobot V")
                st.dataframe(V_df.style.format("{:.6f}").background_gradient(cmap="Blues"), use_container_width=True)
            with col_b:
                st.caption("Jarak ke Solusi Ideal")
                st.dataframe(dist_df.style.format("{:.6f}").background_gradient(cmap="Oranges"), use_container_width=True)


show_method_tab(tab2, "SMART",  "SMART_Score",  "SMART_Rank",  smart_result)
show_method_tab(tab3, "SAW",    "SAW_Score",    "SAW_Rank",    saw_result)
show_method_tab(tab4, "TOPSIS", "TOPSIS_Score", "TOPSIS_Rank", topsis_result,
                extra_cols=["D_plus", "D_minus"])


# ══════════════════════════════════════════════
# TAB 5 — PERBANDINGAN
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">Konsensus Ketiga Metode</p>', unsafe_allow_html=True)

    # Winner announcement
    winner = df_comp.iloc[0]
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#f59e0b,#d97706);color:white;
                border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem">
        <div style="font-size:0.75rem;letter-spacing:2px;opacity:0.85;font-family:IBM Plex Mono,monospace">
            RAM TERPILIH — PERINGKAT #1 KONSENSUS
        </div>
        <div style="font-size:1.6rem;font-weight:700;margin:0.3rem 0">{winner['RAM']}</div>
        <div style="opacity:0.9;font-size:0.9rem">
            SMART #{int(winner['SMART_Rank'])} &nbsp;|&nbsp;
            SAW #{int(winner['SAW_Rank'])} &nbsp;|&nbsp;
            TOPSIS #{int(winner['TOPSIS_Rank'])} &nbsp;|&nbsp;
            Generasi: {winner['gen']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top 5 podium
    st.markdown('<p class="section-title">Top 5 RAM Terbaik</p>', unsafe_allow_html=True)
    top5 = df_comp.head(5)
    for _, row in top5.iterrows():
        rank = int(row["Final_Rank"])
        bg   = ["#fef9c3","#f1f5f9","#fef3c7","#f8fafc","#f8fafc"][rank-1]
        st.markdown(f"""
        <div style="background:{bg};border:1px solid #e2e8f0;border-radius:8px;
                    padding:0.8rem 1.2rem;margin-bottom:0.5rem;display:flex;
                    align-items:center;gap:1rem">
            <div style="font-family:IBM Plex Mono,monospace;font-size:1.4rem;
                        font-weight:700;min-width:40px;color:#0f172a">#{rank}</div>
            <div style="flex:1">
                <div style="font-weight:600">{row['RAM']}</div>
                <div style="font-size:0.78rem;color:#64748b">{row['gen']}</div>
            </div>
            <div style="display:flex;gap:0.5rem">
                <span class="rank-badge badge-smart">SMART #{int(row['SMART_Rank'])}</span>
                <span class="rank-badge badge-saw">SAW #{int(row['SAW_Rank'])}</span>
                <span class="rank-badge badge-topsis">TOPSIS #{int(row['TOPSIS_Rank'])}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Full comparison table
    st.markdown('<p class="section-title">Tabel Perbandingan Lengkap</p>', unsafe_allow_html=True)
    disp = df_comp[["Final_Rank","RAM","gen",
                    "SMART_Score","SMART_Rank",
                    "SAW_Score","SAW_Rank",
                    "TOPSIS_Score","TOPSIS_Rank"]].copy()
    disp.columns = ["Final Rank","RAM","Gen",
                    "SMART Score","SMART Rank",
                    "SAW Score","SAW Rank",
                    "TOPSIS Score","TOPSIS Rank"]
    st.dataframe(
        disp.style.format({
            "SMART Score":  "{:.4f}",
            "SAW Score":    "{:.4f}",
            "TOPSIS Score": "{:.4f}",
        }).background_gradient(subset=["Final Rank"], cmap="YlOrRd_r"),
        hide_index=True, use_container_width=True,
    )

    st.markdown("---")

    # Grouped bar chart
    st.markdown('<p class="section-title">Perbandingan Skor per Metode</p>', unsafe_allow_html=True)
    top_n_chart = min(15, len(df_comp))
    chart_df = df_comp.head(top_n_chart).set_index("RAM")

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(top_n_chart)
    w = 0.25
    smart_n = (chart_df["SMART_Score"] - chart_df["SMART_Score"].min()) / (chart_df["SMART_Score"].max() - chart_df["SMART_Score"].min() + 1e-10)
    saw_n   = (chart_df["SAW_Score"]   - chart_df["SAW_Score"].min())   / (chart_df["SAW_Score"].max()   - chart_df["SAW_Score"].min()   + 1e-10)
    topsis_n= (chart_df["TOPSIS_Score"]- chart_df["TOPSIS_Score"].min())/ (chart_df["TOPSIS_Score"].max()- chart_df["TOPSIS_Score"].min()+ 1e-10)
    ax.bar(x - w, smart_n,  w, label="SMART",  color="#3b82f6", alpha=0.85)
    ax.bar(x,     saw_n,    w, label="SAW",    color="#10b981", alpha=0.85)
    ax.bar(x + w, topsis_n, w, label="TOPSIS", color="#f59e0b", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([r[:20] for r in chart_df.index], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Skor Ternormalisasi (0–1)")
    ax.set_title(f"Perbandingan Skor Ternormalisasi — Top {top_n_chart} RAM", fontsize=12, fontweight="bold")
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Korelasi Spearman
    st.markdown("---")
    st.markdown('<p class="section-title">Konsistensi Antar Metode — Korelasi Spearman</p>', unsafe_allow_html=True)

    methods_rank = ["SMART_Rank", "SAW_Rank", "TOPSIS_Rank"]
    corr_matrix  = np.zeros((3, 3))
    for i, m1 in enumerate(methods_rank):
        for j, m2 in enumerate(methods_rank):
            r, _ = stats.spearmanr(df_comp[m1], df_comp[m2])
            corr_matrix[i, j] = r

    avg_corr = (corr_matrix[0,1] + corr_matrix[0,2] + corr_matrix[1,2]) / 3
    col_corr1, col_corr2 = st.columns([1, 1.5])

    with col_corr1:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        labels = ["SMART", "SAW", "TOPSIS"]
        sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap="RdYlGn",
                    xticklabels=labels, yticklabels=labels,
                    vmin=-1, vmax=1, center=0, linewidths=2, ax=ax3,
                    annot_kws={"size": 13, "weight": "bold"})
        ax3.set_title("Korelasi Spearman Antar Metode", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    with col_corr2:
        interpretation = (
            "Ketiga metode memberikan hasil yang **SANGAT KONSISTEN**"
            if avg_corr > 0.8
            else "Ketiga metode memberikan hasil yang **CUKUP KONSISTEN**"
            if avg_corr > 0.6
            else "Terdapat **PERBEDAAN SIGNIFIKAN** antar metode"
        )
        st.markdown(f"""
        <div class="info-box" style="margin-top:1rem">
            <b>Rata-rata Korelasi Spearman: {avg_corr:.4f}</b><br><br>
            {interpretation.replace("**","<b>").replace("**","</b>")}<br><br>
            Nilai korelasi mendekati 1.0 berarti ketiga metode menghasilkan
            urutan ranking yang serupa, sehingga keputusan dapat diandalkan.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Korelasi per pasang metode:**")
        pairs = [("SMART","SAW",0,1), ("SMART","TOPSIS",0,2), ("SAW","TOPSIS",1,2)]
        for m1, m2, i, j in pairs:
            r = corr_matrix[i, j]
            color = "#16a34a" if r > 0.8 else "#ca8a04" if r > 0.6 else "#dc2626"
            st.markdown(
                f"<span style='color:{color};font-weight:600'>{m1} vs {m2}: r = {r:.4f}</span>",
                unsafe_allow_html=True
            )

    # Radar chart top 5
    st.markdown("---")
    st.markdown('<p class="section-title">Radar Chart — Top 5 RAM</p>', unsafe_allow_html=True)
    top5 = df_comp.head(5)
    methods_s = ["SMART_Score","SAW_Score","TOPSIS_Score"]
    scores_n  = {}
    for m in methods_s:
        col = df_comp[m]
        scores_n[m] = (col - col.min()) / (col.max() - col.min() + 1e-10)

    N      = 3
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    radar_colors = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6"]

    fig4, axes4 = plt.subplots(1, min(5, len(top5)), figsize=(4 * min(5, len(top5)), 4.5),
                               subplot_kw=dict(polar=True))
    if len(top5) == 1:
        axes4 = [axes4]

    for ax_idx, (_, row) in enumerate(top5.iterrows()):
        ax = axes4[ax_idx]
        ram_name = row["RAM"]
        vals  = [scores_n[m][df_comp["RAM"] == ram_name].values[0] for m in methods_s]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=radar_colors[ax_idx])
        ax.fill(angles, vals, alpha=0.2, color=radar_colors[ax_idx])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["SMART","SAW","TOPSIS"], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"#{int(row['Final_Rank'])}: {ram_name[:18]}", fontsize=8, fontweight="bold", pad=15)

    plt.suptitle("Radar Chart: Top 5 RAM — Konsistensi Skor per Metode", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════
# TAB 6 — SENSITIVITAS
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<p class="section-title">Analisis Sensitivitas Bobot</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="margin-bottom:1rem">
        Pengujian stabilitas ranking dengan <b>200 skenario bobot acak</b> menggunakan distribusi Dirichlet.
        RAM yang memiliki range ranking kecil berarti rekomendasinya stabil terhadap perubahan bobot.
    </div>
    """, unsafe_allow_html=True)

    n_scenarios = 200
    X_mat = df_spk[CRITERIA].values.astype(float)
    ctypes_list = [CRITERIA_TYPES[c] for c in CRITERIA]

    with st.spinner("Menjalankan analisis sensitivitas..."):
        rank_stability = np.zeros((len(df_spk), n_scenarios))
        np.random.seed(42)
        for s in range(n_scenarios):
            w_rand  = np.random.dirichlet(np.ones(len(CRITERIA)))
            scores  = run_saw_fast(X_mat, w_rand, ctypes_list)
            rank_stability[:, s] = pd.Series(scores).rank(ascending=False).values

    rank_mean = rank_stability.mean(axis=1)
    rank_std  = rank_stability.std(axis=1)
    rank_min  = rank_stability.min(axis=1)
    rank_max  = rank_stability.max(axis=1)

    df_sens = pd.DataFrame({
        "RAM":       df_spk["RAM"].values,
        "Gen":       df_spk["gen"].values,
        "Rank Rata-rata": rank_mean.round(2),
        "Std Dev":   rank_std.round(2),
        "Rank Min":  rank_min.astype(int),
        "Rank Max":  rank_max.astype(int),
        "Range":     (rank_max - rank_min).astype(int),
        "Stabilitas": ["Sangat Stabil" if s < 2 else "Stabil" if s < 4 else "Kurang Stabil"
                       for s in rank_std],
    }).sort_values("Rank Rata-rata").reset_index(drop=True)

    # Summary metrics
    n_stabil   = (df_sens["Stabilitas"] == "Sangat Stabil").sum()
    n_mid      = (df_sens["Stabilitas"] == "Stabil").sum()
    n_unstabil = (df_sens["Stabilitas"] == "Kurang Stabil").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Sangat Stabil",   n_stabil,   delta=None)
    c2.metric("Stabil",          n_mid,      delta=None)
    c3.metric("Kurang Stabil",   n_unstabil, delta=None)

    # Bar chart sensitivitas
    fig5, ax5 = plt.subplots(figsize=(13, max(5, len(df_sens) * 0.45)))
    y_pos = np.arange(len(df_sens))
    color_map = {"Sangat Stabil": "#2ecc71", "Stabil": "#f39c12", "Kurang Stabil": "#e74c3c"}
    colors_stab = [color_map[s] for s in df_sens["Stabilitas"]]

    ax5.barh(y_pos,
             df_sens["Range"],
             left=df_sens["Rank Min"],
             color=colors_stab, alpha=0.75, edgecolor="gray", linewidth=0.4)
    ax5.scatter(df_sens["Rank Rata-rata"], y_pos, color="navy", zorder=5, s=50, label="Mean Rank")
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(df_sens["RAM"].str[:28], fontsize=7)
    ax5.set_xlabel("Ranking (lebih kecil = lebih baik)", fontsize=10)
    ax5.set_title(
        f"Analisis Sensitivitas Bobot ({n_scenarios} Skenario Random)\nRange Ranking per Alternatif",
        fontsize=12, fontweight="bold"
    )
    ax5.invert_yaxis()
    ax5.grid(axis="x", alpha=0.3)

    patches = [
        mpatches.Patch(color="#2ecc71", label="Sangat Stabil (std < 2)"),
        mpatches.Patch(color="#f39c12", label="Stabil (std 2–4)"),
        mpatches.Patch(color="#e74c3c", label="Kurang Stabil (std > 4)"),
    ]
    ax5.legend(handles=patches + [
        plt.Line2D([0],[0], marker="o", color="navy", label="Mean Rank", linestyle="")
    ], loc="lower right", fontsize=8)
    sns.despine(ax=ax5)
    plt.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown('<p class="section-title">Tabel Detail Sensitivitas</p>', unsafe_allow_html=True)

    def color_stab(val):
        colors_s = {"Sangat Stabil": "background-color:#d1fae5;color:#065f46",
                    "Stabil":        "background-color:#fef3c7;color:#92400e",
                    "Kurang Stabil": "background-color:#fee2e2;color:#991b1b"}
        return colors_s.get(val, "")

    st.dataframe(
        df_sens.style
            .applymap(color_stab, subset=["Stabilitas"])
            .format({"Rank Rata-rata": "{:.2f}", "Std Dev": "{:.2f}"}),
        hide_index=True, use_container_width=True,
    )

    # Download button
    st.markdown("---")
    csv_out = df_comp.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Hasil SPK (CSV)",
        data=csv_out,
        file_name="hasil_spk_ram.csv",
        mime="text/csv",
    )
