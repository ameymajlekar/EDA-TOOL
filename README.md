# 🔬 EDA Agent Dashboard

An autonomous data cleaning + exploratory data analysis agent powered by Python and Streamlit.

---

## Project Structure

```
eda_agent/
├── app.py              # Streamlit dashboard (entry point)
├── data_loader.py      # Multi-format dataset loader
├── data_cleaner.py     # Autonomous data cleaning agent
├── eda_engine.py       # EDA computation engine
├── visualizer.py       # Plotly chart factory
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run app.py
```

---

## Module Overview

### `data_loader.py`
Detects and loads datasets from CSV, TSV, Excel, JSON, Parquet, Feather, and SQLite.
Auto-detects delimiters for text files and JSON orientation.

| Method | Description |
|--------|-------------|
| `load_file(path=...)` | Load from disk path |
| `load_file(file_obj=...)` | Load from Streamlit uploader |
| `supported_extensions()` | List supported file types |

---

### `data_cleaner.py` — `DataCleaner`
Runs a configurable multi-step cleaning pipeline.

**Pipeline steps (in order):**
1. **Column name sanitisation** — lowercase, strip whitespace, replace special chars with `_`
2. **Drop high-missing columns** — configurable threshold (default 60%)
3. **Type inference & casting** — auto-detect numeric, datetime, and boolean columns
4. **Drop constant columns** — columns with a single unique value
5. **Duplicate removal** — exact row deduplication
6. **Missing value imputation** — mean/median (numeric), mode (categorical), ffill (datetime)
7. **Outlier removal** — IQR multiplier or Z-score threshold
8. **Categorical encoding** — label-encode low-cardinality columns

**Configuration:**
```python
cleaner = DataCleaner(
    missing_threshold=0.60,   # drop columns with >60% missing
    outlier_method="iqr",     # "iqr" | "zscore" | None
    outlier_threshold=3.0,    # IQR multiplier or Z-score cutoff
    cardinality_limit=50,     # skip encoding if > 50 unique values
    drop_constant=True,
    encode_categoricals=True,
)
clean_df = cleaner.clean(raw_df)
print(cleaner.report.summary())
```

---

### `eda_engine.py` — `EDAEngine`
Computes all EDA metrics from a cleaned DataFrame.

**Analyses performed:**
| Key | Description |
|-----|-------------|
| `overview` | Shape, dtype breakdown, memory usage |
| `missing` | Missing count/% per column |
| `univariate_numeric` | Describe + skewness, kurtosis, CV |
| `univariate_categorical` | Value counts, top freq |
| `correlation` | Pearson + Spearman matrices |
| `outlier_summary` | IQR & Z-score outlier counts per feature |
| `skewness_kurtosis` | Skew label classification |
| `normality_tests` | Shapiro-Wilk / KS test per feature |
| `target_analysis` | Regression or classification target stats |

```python
engine = EDAEngine(clean_df, target="price")
results = engine.run_all()
```

---

### `visualizer.py`
Plotly chart factory. Dark-themed, monospace typography.

| Function | Chart |
|----------|-------|
| `plot_missing_heatmap(df)` | Horizontal bar chart of missing % |
| `plot_distribution(df, col)` | Histogram + KDE + mean/median lines |
| `plot_box_plots(df, cols)` | Multi-column box plots |
| `plot_correlation_heatmap(matrix)` | Lower-triangle heatmap |
| `plot_scatter(df, x, y, color)` | Scatter with OLS trendline |
| `plot_categorical_bar(df, col)` | Value count bar chart |
| `plot_outlier_strip(df, col)` | Box + outlier strip with fences |
| `plot_skewness(skew_df)` | Skewness colour-coded bar chart |
| `plot_pairplot(df, cols)` | Scatter matrix |
| `plot_normality_results(norm_df)` | p-value bar chart |
| `plot_target_distribution(df, t)` | Histogram or pie chart |

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| 🧹 Cleaning Report | Before/after metrics, imputation log, type conversions |
| 📊 Distributions | Histogram + KDE per feature, box plots, summary stats |
| 🔗 Correlations | Pearson/Spearman heatmap + top correlated pairs |
| 📦 Outliers | IQR/Z-score outlier counts, strip plots, skewness chart |
| 🐱 Categoricals | Value counts, top-N bar charts |
| 🔬 Normality | Shapiro-Wilk / KS test p-values |
| ⚡ Scatter / Pair | Interactive scatter + pair plot matrix |
| 🎯 Target | Target distribution + feature correlations |
| 🗂️ Data | Cleaned data table + CSV download |

---

## Extending the Agent

**Add a new cleaning step:**
```python
# In data_cleaner.py, add a method and call it in clean()
def _my_custom_step(self, df):
    ...
    return df
```

**Add a new EDA analysis:**
```python
# In eda_engine.py, add a method and register in run_all()
def _my_analysis(self):
    ...
    return result
```

**Add a new chart:**
```python
# In visualizer.py
def plot_my_chart(df, col):
    fig = go.Figure(...)
    fig.update_layout(**_base_layout(title="My Chart"))
    return fig
```
