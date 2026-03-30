"""
visualizer.py
-------------
Plotly-based chart factory consumed by the Streamlit dashboard.
All functions receive pre-computed data from EDAEngine and return
plotly.graph_objects.Figure objects.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional

# ── Brand palette ────────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold
SEQUENTIAL = px.colors.sequential.Plasma
DIVERGING = px.colors.diverging.RdBu

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15,17,26,0)",
    plot_bgcolor="rgba(15,17,26,0)",
    font=dict(family="IBM Plex Mono, monospace", size=12, color="#E8EAF6"),
    margin=dict(l=40, r=20, t=50, b=40),
)


def _base_layout(**kwargs) -> dict:
    d = LAYOUT_DEFAULTS.copy()
    d.update(kwargs)
    return d


# ──────────────────────────────────────────────────────────────────────────────
# 1. Missing values heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_missing_heatmap(df: pd.DataFrame) -> go.Figure:
    """Bar chart of missing % per column."""
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="✓ No missing values", showarrow=False,
                           font=dict(size=16, color="#69F0AE"))
        fig.update_layout(**_base_layout(title="Missing Values"))
        return fig

    fig = go.Figure(go.Bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        marker=dict(
            color=missing.values,
            colorscale=SEQUENTIAL,
            showscale=True,
            colorbar=dict(title="% Missing"),
        ),
        text=[f"{v:.1f}%" for v in missing.values],
        textposition="outside",
    ))
    fig.update_layout(**_base_layout(
        title="Missing Values by Column",
        xaxis_title="Missing %",
        yaxis_title="",
        height=max(300, 30 * len(missing) + 80),
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. Distribution histogram + KDE
# ──────────────────────────────────────────────────────────────────────────────

def plot_distribution(df: pd.DataFrame, col: str) -> go.Figure:
    s = df[col].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=s, name="Histogram", nbinsx=40,
        marker_color=PALETTE[0], opacity=0.65,
        histnorm="probability density",
    ))
    # KDE
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(s)
        x_kde = np.linspace(s.min(), s.max(), 300)
        fig.add_trace(go.Scatter(
            x=x_kde, y=kde(x_kde), mode="lines", name="KDE",
            line=dict(color=PALETTE[2], width=2.5),
        ))
    except Exception:
        pass
    # Mean & Median lines
    fig.add_vline(x=float(s.mean()), line_dash="dash", line_color="#FFD740",
                  annotation_text="Mean", annotation_position="top right")
    fig.add_vline(x=float(s.median()), line_dash="dot", line_color="#69F0AE",
                  annotation_text="Median", annotation_position="top left")
    fig.update_layout(**_base_layout(title=f"Distribution: {col}", xaxis_title=col, yaxis_title="Density"))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. Box plots (multiple columns)
# ──────────────────────────────────────────────────────────────────────────────

def plot_box_plots(df: pd.DataFrame, cols: list) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col,
            marker_color=PALETTE[i % len(PALETTE)],
            boxmean="sd",
            jitter=0.3,
            pointpos=-1.8,
        ))
    fig.update_layout(**_base_layout(
        title="Box Plots — Numeric Features",
        yaxis_title="Value",
        height=450,
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. Correlation heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, method: str = "Pearson") -> go.Figure:
    if corr_matrix.empty:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numeric columns", showarrow=False)
        return fig
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    z = corr_matrix.values.copy()
    z[mask] = np.nan
    labels = corr_matrix.columns.tolist()
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1,
        text=np.round(z, 2),
        texttemplate="%{text}",
        hoverongaps=False,
        colorbar=dict(title="r"),
    ))
    fig.update_layout(**_base_layout(
        title=f"{method} Correlation Matrix",
        height=max(400, 40 * len(labels) + 100),
        xaxis=dict(tickangle=-45),
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. Scatter plot with optional colour by target
# ──────────────────────────────────────────────────────────────────────────────

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                 color_col: Optional[str] = None) -> go.Figure:
    sub = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna()
    if color_col:
        fig = px.scatter(sub, x=x_col, y=y_col, color=color_col,
                         color_continuous_scale=SEQUENTIAL, opacity=0.7,
                         trendline="ols", template="plotly_dark")
    else:
        fig = px.scatter(sub, x=x_col, y=y_col, opacity=0.6,
                         trendline="ols", template="plotly_dark")
        fig.update_traces(marker=dict(color=PALETTE[0]))
    fig.update_layout(**_base_layout(title=f"Scatter: {x_col} vs {y_col}"))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6. Categorical bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_categorical_bar(df: pd.DataFrame, col: str, top_n: int = 15) -> go.Figure:
    vc = df[col].value_counts().head(top_n)
    fig = go.Figure(go.Bar(
        x=vc.index.astype(str),
        y=vc.values,
        marker_color=PALETTE[:len(vc)],
        text=vc.values,
        textposition="outside",
    ))
    fig.update_layout(**_base_layout(
        title=f"Value Counts: {col} (top {top_n})",
        xaxis_title=col,
        yaxis_title="Count",
        xaxis=dict(tickangle=-30),
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7. Outlier visualisation (strip + box combined)
# ──────────────────────────────────────────────────────────────────────────────

def plot_outlier_strip(df: pd.DataFrame, col: str) -> go.Figure:
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_outlier = (s < lower) | (s > upper)

    fig = go.Figure()
    fig.add_trace(go.Box(y=s, name="Box", marker_color=PALETTE[1], boxmean=True))
    fig.add_trace(go.Scatter(
        y=s[is_outlier], x=["Box"] * is_outlier.sum(),
        mode="markers", name="Outliers",
        marker=dict(color="#FF5252", size=8, symbol="x"),
    ))
    fig.add_hline(y=upper, line_dash="dash", line_color="#FFD740",
                  annotation_text=f"Upper fence ({upper:.2f})")
    fig.add_hline(y=lower, line_dash="dash", line_color="#69F0AE",
                  annotation_text=f"Lower fence ({lower:.2f})")
    fig.update_layout(**_base_layout(title=f"Outlier Analysis: {col}", showlegend=True))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 8. Skewness bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_skewness(skew_df: pd.DataFrame) -> go.Figure:
    df = skew_df.sort_values("skewness")
    colors = ["#FF5252" if abs(v) > 1 else "#FFD740" if abs(v) > 0.5 else "#69F0AE"
              for v in df["skewness"]]
    fig = go.Figure(go.Bar(
        x=df["feature"], y=df["skewness"],
        marker_color=colors,
        text=df["skewness"].round(2),
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="white", line_width=1)
    fig.add_hline(y=1, line_dash="dot", line_color="#FFD740", opacity=0.5)
    fig.add_hline(y=-1, line_dash="dot", line_color="#FFD740", opacity=0.5)
    fig.update_layout(**_base_layout(
        title="Skewness per Feature  (|skew|>1 = high skew)",
        xaxis_title="", yaxis_title="Skewness",
        xaxis=dict(tickangle=-30),
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 9. Pair-plot (scatter matrix)
# ──────────────────────────────────────────────────────────────────────────────

def plot_pairplot(df: pd.DataFrame, cols: list,
                  color_col: Optional[str] = None) -> go.Figure:
    sub = df[cols + ([color_col] if color_col and color_col not in cols else [])].dropna()
    fig = px.scatter_matrix(
        sub, dimensions=cols,
        color=color_col,
        color_continuous_scale=SEQUENTIAL,
        template="plotly_dark",
        opacity=0.5,
    )
    fig.update_traces(diagonal_visible=True, showupperhalf=False)
    fig.update_layout(**_base_layout(title="Pair Plot", height=700))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 10. Normality test results
# ──────────────────────────────────────────────────────────────────────────────

def plot_normality_results(norm_df: pd.DataFrame) -> go.Figure:
    if norm_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for normality tests", showarrow=False)
        return fig
    colors = ["#69F0AE" if v else "#FF5252" for v in norm_df["is_normal"]]
    fig = go.Figure(go.Bar(
        x=norm_df["feature"],
        y=norm_df["p_value"],
        marker_color=colors,
        text=[f"p={v:.4f}" for v in norm_df["p_value"]],
        textposition="outside",
    ))
    fig.add_hline(y=0.05, line_dash="dash", line_color="#FFD740",
                  annotation_text="α = 0.05")
    fig.update_layout(**_base_layout(
        title="Normality Test p-values (green = normal)",
        xaxis_title="", yaxis_title="p-value",
        xaxis=dict(tickangle=-30),
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 11. Target distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame, target: str) -> go.Figure:
    s = df[target].dropna()
    if pd.api.types.is_numeric_dtype(s):
        return plot_distribution(df, target)
    vc = s.value_counts()
    fig = go.Figure(go.Pie(
        labels=vc.index.astype(str), values=vc.values,
        hole=0.4,
        marker=dict(colors=PALETTE[:len(vc)]),
    ))
    fig.update_layout(**_base_layout(title=f"Target Distribution: {target}"))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 12. Numeric stats table helper (returns styled df for st.dataframe)
# ──────────────────────────────────────────────────────────────────────────────

def styled_stats_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    display_cols = ["feature", "mean", "std", "min", "25%", "50%", "75%", "max", "skewness", "kurtosis"]
    available = [c for c in display_cols if c in stats_df.columns]
    return stats_df[available].round(4)
