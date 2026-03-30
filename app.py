"""
app.py
------
Streamlit EDA Dashboard — entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

from data_loader import load_file, supported_extensions
from data_cleaner import DataCleaner
from eda_engine import EDAEngine
import visualizer as viz

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Agent · Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0F111A;
    color: #E8EAF6;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13151F;
    border-right: 1px solid #1E2133;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #1A1D2E;
    border: 1px solid #2A2D3E;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; color: #7C83FD; }
[data-testid="stMetricLabel"] { color: #9E9EBE; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #13151F; padding: 4px; border-radius: 8px; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 6px; color: #9E9EBE;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
    padding: 6px 16px;
}
.stTabs [aria-selected="true"] { background: #7C83FD !important; color: white !important; }

/* Dividers */
hr { border-color: #1E2133; }

/* Buttons */
.stButton button {
    background: #7C83FD; color: white; border: none; border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    padding: 8px 24px;
}
.stButton button:hover { background: #5C63DD; }

/* Warning / success boxes */
.success-box {
    background: #0D2818; border: 1px solid #27AE60;
    border-radius: 8px; padding: 12px 16px; color: #69F0AE;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
}
.warning-box {
    background: #2A1F08; border: 1px solid #F39C12;
    border-radius: 8px; padding: 12px 16px; color: #FFD740;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
}
.info-box {
    background: #0D1A2E; border: 1px solid #2980B9;
    border-radius: 8px; padding: 12px 16px; color: #82B1FF;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session-state helpers
# ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "raw_df": None, "clean_df": None, "eda": None,
        "cleaning_report": None, "file_name": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🔬 EDA Agent")
    st.markdown("---")

    # ── Data source ──────────────────────────
    st.markdown("### 📂 Data Source")
    source = st.radio("Choose source", ["Upload file", "Use sample dataset"], label_visibility="collapsed")

    if source == "Upload file":
        uploaded = st.file_uploader(
            "Drop your dataset",
            type=[ext.lstrip(".") for ext in supported_extensions() if ext not in (".db", ".sqlite", ".sqlite3")],
            help=f"Supported: {', '.join(supported_extensions())}",
        )
        if uploaded and st.button("Load dataset", use_container_width=True):
            with st.spinner("Loading…"):
                df = load_file(file_obj=uploaded, file_name=uploaded.name)
                st.session_state.raw_df = df
                st.session_state.clean_df = None
                st.session_state.eda = None
                st.session_state.file_name = uploaded.name
            st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
    else:
        sample_name = st.selectbox("Sample dataset", ["Titanic", "Iris", "Housing prices", "E-commerce orders"])
        if st.button("Load sample", use_container_width=True):
            df = _load_sample(sample_name)
            st.session_state.raw_df = df
            st.session_state.clean_df = None
            st.session_state.eda = None
            st.session_state.file_name = sample_name

    st.markdown("---")

    # ── Cleaning options ─────────────────────
    if st.session_state.raw_df is not None:
        st.markdown("### 🧹 Cleaning Options")
        missing_thresh = st.slider("Drop columns if missing% ≥", 10, 95, 60, 5,
                                   help="Columns with more missing values than this threshold are dropped.")
        outlier_method = st.selectbox("Outlier removal", ["iqr", "zscore", "none"])
        outlier_thresh = st.slider("Outlier threshold", 1.5, 5.0, 3.0, 0.5)
        encode_cats = st.checkbox("Label-encode categoricals", value=True)

        if st.button("🚀 Run Cleaning + EDA", use_container_width=True):
            with st.spinner("Cleaning dataset…"):
                cleaner = DataCleaner(
                    missing_threshold=missing_thresh / 100,
                    outlier_method=None if outlier_method == "none" else outlier_method,
                    outlier_threshold=outlier_thresh,
                    encode_categoricals=encode_cats,
                )
                clean_df = cleaner.clean(st.session_state.raw_df)
                st.session_state.clean_df = clean_df
                st.session_state.cleaning_report = cleaner.report

            with st.spinner("Running EDA…"):
                engine = EDAEngine(clean_df)
                engine.run_all()
                st.session_state.eda = engine
            st.success("Done! Explore the tabs →")

        st.markdown("---")

        # ── Target column ────────────────────
        if st.session_state.clean_df is not None:
            st.markdown("### 🎯 Target Column")
            cols = ["(none)"] + st.session_state.clean_df.columns.tolist()
            target = st.selectbox("Select target (optional)", cols)
            if target != "(none)" and st.button("Analyse target", use_container_width=True):
                engine = st.session_state.eda
                engine.target = target
                engine.results["target_analysis"] = engine._target_analysis()
                st.success(f"Target set to: {target}")


# ──────────────────────────────────────────────
# Sample datasets
# ──────────────────────────────────────────────
def _load_sample(name: str) -> pd.DataFrame:
    np.random.seed(42)
    n = 800
    if name == "Titanic":
        try:
            return pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        except Exception:
            pass
    if name == "Iris":
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        return iris.frame
    if name == "Housing prices":
        try:
            from sklearn.datasets import fetch_california_housing
            h = fetch_california_housing(as_frame=True)
            return h.frame.sample(n, random_state=42)
        except Exception:
            pass
    # E-commerce fallback / default synthetic
    return pd.DataFrame({
        "order_id": range(1, n + 1),
        "customer_age": np.random.normal(35, 10, n).clip(18, 80).astype(int),
        "purchase_amount": np.random.exponential(120, n).round(2),
        "items_bought": np.random.poisson(3, n),
        "discount_pct": np.random.choice([0, 5, 10, 15, 20, np.nan], n),
        "category": np.random.choice(["Electronics", "Clothing", "Books", "Home", "Sports"], n),
        "region": np.random.choice(["North", "South", "East", "West", None], n, p=[0.25, 0.25, 0.24, 0.24, 0.02]),
        "returned": np.random.choice([0, 1], n, p=[0.88, 0.12]),
        "rating": np.random.choice([1, 2, 3, 4, 5, None], n, p=[0.05, 0.08, 0.17, 0.35, 0.33, 0.02]),
        "signup_date": pd.date_range("2021-01-01", periods=n, freq="10h"),
    })


# ══════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════

if st.session_state.raw_df is None:
    # ── Landing screen ────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 80px 40px;">
        <div style="font-size:4rem">🔬</div>
        <h1 style="font-size:2.6rem; margin:16px 0 8px">EDA Agent Dashboard</h1>
        <p style="color:#9E9EBE; font-size:1.1rem; max-width:580px; margin:0 auto 32px">
            Upload any tabular dataset or pick a sample from the sidebar.
            The agent will auto-clean your data and run a full exploratory analysis.
        </p>
        <div style="display:flex; gap:16px; justify-content:center; flex-wrap:wrap; margin-top:24px">
            <div class="info-box" style="width:200px">📊 Distribution analysis</div>
            <div class="info-box" style="width:200px">🔗 Correlation matrices</div>
            <div class="info-box" style="width:200px">🎯 Target analysis</div>
            <div class="info-box" style="width:200px">⚠️ Outlier detection</div>
            <div class="info-box" style="width:200px">🧹 Auto data cleaning</div>
            <div class="info-box" style="width:200px">📐 Normality tests</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
raw_df = st.session_state.raw_df
clean_df = st.session_state.clean_df
eda: EDAEngine = st.session_state.eda

st.markdown(f"## 📁 `{st.session_state.file_name}`")

if clean_df is None:
    st.markdown('<div class="warning-box">⚡ Dataset loaded. Configure cleaning options in the sidebar and click <strong>Run Cleaning + EDA</strong>.</div>', unsafe_allow_html=True)
    st.markdown("### Raw data preview")
    st.dataframe(raw_df.head(100), use_container_width=True)
    st.stop()


# ──────────────────────────────────────────────
# KPI row
# ──────────────────────────────────────────────
ov = eda.results["overview"]
r = st.session_state.cleaning_report

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Rows", f"{ov['n_rows']:,}")
col2.metric("Columns", ov["n_cols"])
col3.metric("Numeric", ov["n_numeric"])
col4.metric("Categorical", ov["n_categorical"])
col5.metric("Missing %", f"{ov['missing_pct']}%")
col6.metric("Duplicates removed", r.dropped_rows_duplicates)

st.markdown("---")

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tabs = st.tabs([
    "🧹 Cleaning Report",
    "📊 Distributions",
    "🔗 Correlations",
    "📦 Outliers",
    "🐱 Categoricals",
    "🔬 Normality",
    "⚡ Scatter / Pair",
    "🎯 Target",
    "🗂️ Data",
])

tab_clean, tab_dist, tab_corr, tab_out, tab_cat, tab_norm, tab_scat, tab_tgt, tab_data = tabs


# ══════════════════════════════════════════════
# TAB 1 — Cleaning Report
# ══════════════════════════════════════════════
with tab_clean:
    st.markdown("### 🧹 Data Cleaning Summary")
    r = st.session_state.cleaning_report

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before vs After**")
        before_after = pd.DataFrame({
            "Metric": ["Rows", "Columns"],
            "Before": [r.original_shape[0], r.original_shape[1]],
            "After": [r.final_shape[0], r.final_shape[1]],
        })
        st.dataframe(before_after, use_container_width=True, hide_index=True)

        if r.type_conversions:
            st.markdown("**Type conversions**")
            tc = pd.DataFrame(r.type_conversions.items(), columns=["Column", "New type"])
            st.dataframe(tc, use_container_width=True, hide_index=True)

    with c2:
        if r.imputed_columns:
            st.markdown("**Imputation strategies**")
            imp = pd.DataFrame(r.imputed_columns.items(), columns=["Column", "Strategy"])
            st.dataframe(imp, use_container_width=True, hide_index=True)

        if r.dropped_columns:
            st.markdown("**Dropped columns**")
            st.markdown(f'<div class="warning-box">{", ".join(r.dropped_columns)}</div>', unsafe_allow_html=True)

        if r.warnings:
            st.markdown("**Warnings**")
            for w in r.warnings:
                st.markdown(f'<div class="warning-box">⚠ {w}</div>', unsafe_allow_html=True)

    # Missing chart
    st.markdown("### Missing Values (original data)")
    st.plotly_chart(viz.plot_missing_heatmap(raw_df), use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — Distributions
# ══════════════════════════════════════════════
with tab_dist:
    st.markdown("### 📊 Univariate Distributions")
    numeric_cols = eda._numeric_cols

    if not numeric_cols:
        st.info("No numeric columns detected.")
    else:
        selected_col = st.selectbox("Select feature", numeric_cols, key="dist_col")
        st.plotly_chart(viz.plot_distribution(clean_df, selected_col), use_container_width=True)

        st.markdown("---")
        st.markdown("### Box Plots — All Numeric Features")
        max_cols = min(12, len(numeric_cols))
        cols_to_plot = st.multiselect("Columns to plot", numeric_cols, default=numeric_cols[:max_cols])
        if cols_to_plot:
            st.plotly_chart(viz.plot_box_plots(clean_df, cols_to_plot), use_container_width=True)

        st.markdown("---")
        st.markdown("### Numeric Summary Statistics")
        stats_df = viz.styled_stats_table(eda.results["univariate_numeric"])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 3 — Correlations
# ══════════════════════════════════════════════
with tab_corr:
    st.markdown("### 🔗 Correlation Analysis")
    corr_data = eda.results["correlation"]
    method = st.radio("Method", ["Pearson", "Spearman"], horizontal=True)
    matrix = corr_data["pearson"] if method == "Pearson" else corr_data["spearman"]
    st.plotly_chart(viz.plot_correlation_heatmap(matrix, method), use_container_width=True)

    if not matrix.empty:
        st.markdown("#### Top correlated pairs")
        pairs = (
            matrix.where(np.tril(np.ones(matrix.shape), k=-1).astype(bool))
            .stack()
            .reset_index()
        )
        pairs.columns = ["Feature A", "Feature B", "Correlation"]
        pairs["abs_corr"] = pairs["Correlation"].abs()
        pairs = pairs.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")
        st.dataframe(pairs.head(20).reset_index(drop=True), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 4 — Outliers
# ══════════════════════════════════════════════
with tab_out:
    st.markdown("### 📦 Outlier Analysis")
    out_df = eda.results["outlier_summary"]
    if out_df.empty:
        st.info("No numeric columns for outlier analysis.")
    else:
        st.dataframe(out_df, use_container_width=True, hide_index=True)
        col_out = st.selectbox("Inspect column", eda._numeric_cols, key="out_col")
        st.plotly_chart(viz.plot_outlier_strip(clean_df, col_out), use_container_width=True)

    st.markdown("---")
    st.markdown("### Skewness Overview")
    skew_df = eda.results["skewness_kurtosis"]
    if not skew_df.empty:
        st.plotly_chart(viz.plot_skewness(skew_df), use_container_width=True)
        st.dataframe(skew_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 5 — Categoricals
# ══════════════════════════════════════════════
with tab_cat:
    st.markdown("### 🐱 Categorical Features")
    cat_data = eda.results["univariate_categorical"]

    if not cat_data:
        st.info("No categorical columns detected after cleaning.")
    else:
        # Summary table
        summary_rows = []
        for col, info in cat_data.items():
            summary_rows.append({
                "Column": col,
                "Unique values": info["n_unique"],
                "Top value": info["top_value"],
                "Top freq %": info["top_freq_pct"],
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        selected_cat = st.selectbox("Inspect column", list(cat_data.keys()), key="cat_col")
        # Use original raw_df for richer labels before encoding
        plot_df = raw_df if selected_cat in raw_df.columns else clean_df
        st.plotly_chart(viz.plot_categorical_bar(plot_df, selected_cat), use_container_width=True)


# ══════════════════════════════════════════════
# TAB 6 — Normality
# ══════════════════════════════════════════════
with tab_norm:
    st.markdown("### 🔬 Normality Tests")
    st.markdown('<div class="info-box">Shapiro-Wilk (n ≤ 5000) or Kolmogorov-Smirnov test. H₀: data is normally distributed. Green = fail to reject H₀ (p > 0.05).</div>', unsafe_allow_html=True)
    norm_df = eda.results["normality_tests"]
    if norm_df.empty:
        st.info("Not enough data to run normality tests.")
    else:
        st.plotly_chart(viz.plot_normality_results(norm_df), use_container_width=True)
        st.dataframe(norm_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 7 — Scatter / Pair
# ══════════════════════════════════════════════
with tab_scat:
    st.markdown("### ⚡ Scatter & Pair Plots")
    numeric_cols = eda._numeric_cols
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for scatter plots.")
    else:
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X axis", numeric_cols, key="scat_x")
        y_col = c2.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="scat_y")
        color_options = ["(none)"] + clean_df.columns.tolist()
        color_col = c3.selectbox("Color by", color_options, key="scat_c")
        cc = None if color_col == "(none)" else color_col
        st.plotly_chart(viz.plot_scatter(clean_df, x_col, y_col, cc), use_container_width=True)

        st.markdown("---")
        st.markdown("### Pair Plot")
        pair_cols = st.multiselect("Select features (2–6)", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
        if len(pair_cols) >= 2:
            st.plotly_chart(viz.plot_pairplot(clean_df, pair_cols, cc), use_container_width=True)


# ══════════════════════════════════════════════
# TAB 8 — Target
# ══════════════════════════════════════════════
with tab_tgt:
    st.markdown("### 🎯 Target Column Analysis")
    tgt_data = eda.results.get("target_analysis")
    if tgt_data is None:
        st.markdown('<div class="info-box">Set a target column in the sidebar to enable target analysis.</div>', unsafe_allow_html=True)
    else:
        t = tgt_data["target"]
        st.markdown(f"**Target:** `{t}` — **Type:** `{tgt_data['type']}`")
        st.plotly_chart(viz.plot_target_distribution(clean_df, t), use_container_width=True)

        if tgt_data["type"] == "regression" and "feature_correlations" in tgt_data:
            st.markdown("#### Feature correlations with target")
            fc = tgt_data["feature_correlations"]
            if fc:
                fc_df = pd.DataFrame(fc).T.reset_index().rename(columns={"index": "feature"})
                fc_df = fc_df.sort_values("pearson_r", key=abs, ascending=False)
                st.dataframe(fc_df, use_container_width=True, hide_index=True)

        if tgt_data["type"] == "classification":
            st.markdown("#### Class distribution")
            cd = tgt_data["class_distribution"]
            cd_df = pd.DataFrame(cd.items(), columns=["Class", "Proportion"])
            st.dataframe(cd_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 9 — Data
# ══════════════════════════════════════════════
with tab_data:
    st.markdown("### 🗂️ Cleaned Dataset")
    st.markdown(f"`{clean_df.shape[0]:,}` rows × `{clean_df.shape[1]}` columns")

    # Download button
    csv_buf = io.BytesIO()
    clean_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download cleaned CSV",
        data=csv_buf.getvalue(),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )
    st.dataframe(clean_df, use_container_width=True)
    st.markdown("#### Data types")
    dtypes_df = clean_df.dtypes.reset_index()
    dtypes_df.columns = ["Column", "Dtype"]
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
