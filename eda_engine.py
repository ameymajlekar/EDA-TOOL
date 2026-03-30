"""
eda_engine.py
-------------
Performs comprehensive exploratory data analysis on a cleaned DataFrame.
Returns structured results (dicts / DataFrames) consumed by the dashboard.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


class EDAEngine:
    """
    Runs a full EDA suite and stores results as attributes for easy access.

    Usage
    -----
        engine = EDAEngine(df)
        engine.run_all()
        print(engine.results["univariate_numeric"])
    """

    def __init__(self, df: pd.DataFrame, target: Optional[str] = None):
        self.df = df.copy()
        self.target = target
        self.results: dict = {}
        self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # ──────────────────────────────────────────────
    # Orchestrator
    # ──────────────────────────────────────────────

    def run_all(self) -> dict:
        self.results["overview"] = self._overview()
        self.results["missing"] = self._missing_analysis()
        self.results["univariate_numeric"] = self._univariate_numeric()
        self.results["univariate_categorical"] = self._univariate_categorical()
        self.results["correlation"] = self._correlation()
        self.results["outlier_summary"] = self._outlier_summary()
        self.results["skewness_kurtosis"] = self._skewness_kurtosis()
        self.results["normality_tests"] = self._normality_tests()
        if self.target and self.target in self.df.columns:
            self.results["target_analysis"] = self._target_analysis()
        return self.results

    # ──────────────────────────────────────────────
    # Individual analyses
    # ──────────────────────────────────────────────

    def _overview(self) -> dict:
        df = self.df
        dtypes_count = df.dtypes.astype(str).value_counts().to_dict()
        return {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "n_numeric": len(self._numeric_cols),
            "n_categorical": len(self._cat_cols),
            "total_missing": int(df.isnull().sum().sum()),
            "missing_pct": round(df.isnull().mean().mean() * 100, 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
            "dtypes_breakdown": dtypes_count,
            "columns": df.columns.tolist(),
        }

    def _missing_analysis(self) -> pd.DataFrame:
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        result = pd.DataFrame({
            "column": missing.index,
            "missing_count": missing.values,
            "missing_pct": missing_pct.values,
        })
        return result[result["missing_count"] > 0].sort_values("missing_pct", ascending=False).reset_index(drop=True)

    def _univariate_numeric(self) -> pd.DataFrame:
        if not self._numeric_cols:
            return pd.DataFrame()
        desc = self.df[self._numeric_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
        desc["skewness"] = self.df[self._numeric_cols].skew()
        desc["kurtosis"] = self.df[self._numeric_cols].kurt()
        desc["cv"] = (desc["std"] / desc["mean"].replace(0, np.nan)).round(4)
        desc["zeros_pct"] = ((self.df[self._numeric_cols] == 0).mean() * 100).round(2)
        return desc.reset_index().rename(columns={"index": "feature"})

    def _univariate_categorical(self) -> dict:
        result = {}
        for col in self._cat_cols:
            vc = self.df[col].value_counts(dropna=False)
            result[col] = {
                "n_unique": int(self.df[col].nunique()),
                "top_values": vc.head(10).to_dict(),
                "top_value": str(vc.index[0]) if len(vc) > 0 else None,
                "top_freq_pct": round(vc.iloc[0] / len(self.df) * 100, 2) if len(vc) > 0 else 0,
            }
        return result

    def _correlation(self) -> dict:
        if len(self._numeric_cols) < 2:
            return {"pearson": pd.DataFrame(), "spearman": pd.DataFrame()}
        pearson = self.df[self._numeric_cols].corr(method="pearson")
        spearman = self.df[self._numeric_cols].corr(method="spearman")
        return {"pearson": pearson, "spearman": spearman}

    def _outlier_summary(self) -> pd.DataFrame:
        records = []
        for col in self._numeric_cols:
            series = self.df[col].dropna()
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((series < lower) | (series > upper)).sum())
            z = np.abs(stats.zscore(series))
            n_z = int((z > 3).sum())
            records.append({
                "feature": col,
                "iqr_outliers": n_out,
                "iqr_outlier_pct": round(n_out / len(self.df) * 100, 2),
                "zscore_outliers_gt3": n_z,
                "lower_fence": round(lower, 4),
                "upper_fence": round(upper, 4),
            })
        return pd.DataFrame(records)

    def _skewness_kurtosis(self) -> pd.DataFrame:
        records = []
        for col in self._numeric_cols:
            s = self.df[col].dropna()
            skew = round(float(s.skew()), 4)
            kurt = round(float(s.kurt()), 4)
            skew_label = (
                "highly right-skewed" if skew > 1 else
                "right-skewed" if skew > 0.5 else
                "highly left-skewed" if skew < -1 else
                "left-skewed" if skew < -0.5 else
                "approximately symmetric"
            )
            records.append({"feature": col, "skewness": skew,
                            "kurtosis": kurt, "skew_label": skew_label})
        return pd.DataFrame(records)

    def _normality_tests(self) -> pd.DataFrame:
        records = []
        for col in self._numeric_cols:
            s = self.df[col].dropna()
            if len(s) < 8:
                continue
            sample = s.sample(min(5000, len(s)), random_state=42)
            try:
                stat, p = stats.shapiro(sample) if len(sample) <= 5000 else stats.kstest(sample, "norm", args=(sample.mean(), sample.std()))
                records.append({
                    "feature": col,
                    "test": "shapiro" if len(sample) <= 5000 else "ks",
                    "statistic": round(float(stat), 4),
                    "p_value": round(float(p), 6),
                    "is_normal": p > 0.05,
                })
            except Exception:
                pass
        return pd.DataFrame(records)

    def _target_analysis(self) -> dict:
        t = self.target
        target_series = self.df[t]
        result = {"target": t, "dtype": str(target_series.dtype)}

        if pd.api.types.is_numeric_dtype(target_series):
            result["type"] = "regression"
            result["describe"] = target_series.describe().to_dict()
            corr = {}
            for col in self._numeric_cols:
                if col == t:
                    continue
                valid = self.df[[col, t]].dropna()
                if len(valid) > 2:
                    r, p = stats.pearsonr(valid[col], valid[t])
                    corr[col] = {"pearson_r": round(r, 4), "p_value": round(p, 6)}
            result["feature_correlations"] = corr
        else:
            result["type"] = "classification"
            result["class_distribution"] = target_series.value_counts(normalize=True).round(4).to_dict()
            result["n_classes"] = int(target_series.nunique())

        return result

    # ──────────────────────────────────────────────
    # Helper: get data for a specific chart
    # ──────────────────────────────────────────────

    def get_distribution_data(self, col: str) -> dict:
        """Returns histogram bins + KDE values for a numeric column."""
        s = self.df[col].dropna()
        counts, bin_edges = np.histogram(s, bins="auto")
        kde_x = np.linspace(s.min(), s.max(), 200)
        try:
            kde = stats.gaussian_kde(s)
            kde_y = kde(kde_x)
        except Exception:
            kde_y = np.zeros_like(kde_x)
        return {
            "bins": bin_edges.tolist(),
            "counts": counts.tolist(),
            "kde_x": kde_x.tolist(),
            "kde_y": kde_y.tolist(),
        }

    def get_scatter_data(self, x_col: str, y_col: str) -> dict:
        """Returns x, y values for a scatter plot."""
        sub = self.df[[x_col, y_col]].dropna()
        return {"x": sub[x_col].tolist(), "y": sub[y_col].tolist()}

    def pairwise_scatter_matrix_data(self, max_cols: int = 5) -> dict:
        cols = self._numeric_cols[:max_cols]
        return {
            "columns": cols,
            "data": self.df[cols].dropna().to_dict(orient="list"),
        }
