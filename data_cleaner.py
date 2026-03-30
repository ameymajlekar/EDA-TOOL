"""
data_cleaner.py
---------------
Autonomous data cleaning agent that preprocesses any tabular dataset
into a format suitable for EDA. Handles missing values, type inference,
outlier detection, duplicates, and encoding.
"""

import pandas as pd
import numpy as np
from scipy import stats
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Report dataclass
# ──────────────────────────────────────────────

@dataclass
class CleaningReport:
    original_shape: tuple = (0, 0)
    final_shape: tuple = (0, 0)
    dropped_columns: list = field(default_factory=list)
    dropped_rows_duplicates: int = 0
    dropped_rows_outliers: int = 0
    imputed_columns: dict = field(default_factory=dict)   # col -> strategy
    type_conversions: dict = field(default_factory=dict)  # col -> new_dtype
    encoded_columns: list = field(default_factory=list)
    constant_columns_dropped: list = field(default_factory=list)
    high_cardinality_columns: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  DATA CLEANING REPORT",
            "=" * 60,
            f"  Original shape : {self.original_shape}",
            f"  Final shape    : {self.final_shape}",
            f"  Rows removed   : duplicates={self.dropped_rows_duplicates}, "
            f"outliers={self.dropped_rows_outliers}",
            f"  Columns dropped: {len(self.dropped_columns)} "
            f"({self.dropped_columns[:5]}{'...' if len(self.dropped_columns)>5 else ''})",
            f"  Type conversions: {self.type_conversions}",
            f"  Imputation     : {self.imputed_columns}",
            f"  Encoded cols   : {self.encoded_columns}",
        ]
        if self.warnings:
            lines.append("  ⚠ Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Main cleaner class
# ──────────────────────────────────────────────

class DataCleaner:
    """
    Cleans a raw DataFrame through a configurable pipeline.

    Parameters
    ----------
    missing_threshold   : Drop columns with > this fraction of NaN (default 0.6)
    outlier_method      : 'iqr' | 'zscore' | None
    outlier_threshold   : IQR multiplier or Z-score cut-off
    cardinality_limit   : Max unique values before a column is flagged as
                          high-cardinality (not one-hot encoded)
    drop_constant       : Drop columns with a single unique value
    encode_categoricals : Label-encode low-cardinality object columns
    """

    def __init__(
        self,
        missing_threshold: float = 0.60,
        outlier_method: Optional[str] = "iqr",
        outlier_threshold: float = 3.0,
        cardinality_limit: int = 50,
        drop_constant: bool = True,
        encode_categoricals: bool = True,
    ):
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.cardinality_limit = cardinality_limit
        self.drop_constant = drop_constant
        self.encode_categoricals = encode_categoricals
        self.report = CleaningReport()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline and return a cleaned DataFrame."""
        self.report = CleaningReport()
        self.report.original_shape = df.shape
        logger.info(f"Starting cleaning → shape {df.shape}")

        df = df.copy()
        df = self._sanitise_column_names(df)
        df = self._drop_high_missing(df)
        df = self._infer_and_cast_types(df)
        df = self._drop_constant_columns(df)
        df = self._drop_duplicates(df)
        df = self._impute_missing(df)
        df = self._remove_outliers(df)
        if self.encode_categoricals:
            df = self._encode_categoricals(df)

        self.report.final_shape = df.shape
        logger.info(f"Cleaning complete → shape {df.shape}")
        print(self.report.summary())
        return df

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _sanitise_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase, strip whitespace, replace spaces/special chars with _."""
        new_cols = []
        for c in df.columns:
            c2 = str(c).strip().lower()
            c2 = re.sub(r"[^a-z0-9]+", "_", c2).strip("_")
            new_cols.append(c2 or "unnamed")
        df.columns = new_cols
        return df

    def _drop_high_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_frac = df.isnull().mean()
        to_drop = missing_frac[missing_frac > self.missing_threshold].index.tolist()
        if to_drop:
            logger.info(f"Dropping {len(to_drop)} high-missing columns: {to_drop}")
            self.report.dropped_columns.extend(to_drop)
            df = df.drop(columns=to_drop)
        return df

    def _infer_and_cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            original_dtype = str(df[col].dtype)
            if df[col].dtype == object:
                # Try numeric
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notnull().mean() > 0.85:
                    df[col] = converted
                    self.report.type_conversions[col] = "float64"
                    continue
                # Try datetime
                try:
                    converted_dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    if converted_dt.notnull().mean() > 0.85:
                        df[col] = converted_dt
                        self.report.type_conversions[col] = "datetime64"
                        continue
                except Exception:
                    pass
                # Try boolean
                bool_map = {"true": True, "false": False, "yes": True, "no": False,
                            "1": True, "0": False}
                lower_vals = df[col].dropna().str.lower().unique()
                if set(lower_vals).issubset(bool_map.keys()):
                    df[col] = df[col].str.lower().map(bool_map)
                    self.report.type_conversions[col] = "bool"
        return df

    def _drop_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.drop_constant:
            return df
        constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant:
            logger.info(f"Dropping constant columns: {constant}")
            self.report.constant_columns_dropped.extend(constant)
            self.report.dropped_columns.extend(constant)
            df = df.drop(columns=constant)
        return df

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        self.report.dropped_rows_duplicates = removed
        if removed:
            logger.info(f"Dropped {removed} duplicate rows")
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    fill_val = df[col].median()
                    strategy = "median"
                else:
                    fill_val = df[col].mean()
                    strategy = "mean"
                df[col] = df[col].fillna(fill_val)
                self.report.imputed_columns[col] = strategy
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
                self.report.imputed_columns[col] = "forward/backward fill"
            else:
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "unknown"
                df[col] = df[col].fillna(fill_val)
                self.report.imputed_columns[col] = f"mode ({fill_val})"
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.outlier_method is None:
            return df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df

        before = len(df)
        mask = pd.Series([True] * len(df), index=df.index)

        if self.outlier_method == "iqr":
            for col in numeric_cols:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.outlier_threshold * iqr
                upper = q3 + self.outlier_threshold * iqr
                mask &= df[col].between(lower, upper)
        elif self.outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(df[numeric_cols].dropna()))
            mask_arr = (z_scores < self.outlier_threshold).all(axis=1)
            mask = pd.Series(mask_arr, index=df[numeric_cols].dropna().index)
            mask = mask.reindex(df.index, fill_value=True)

        df = df[mask]
        removed = before - len(df)
        self.report.dropped_rows_outliers = removed
        if removed:
            logger.info(f"Removed {removed} outlier rows via {self.outlier_method}")
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique > self.cardinality_limit:
                self.report.high_cardinality_columns.append(col)
                self.report.warnings.append(
                    f"'{col}' has {n_unique} unique values — skipped encoding"
                )
                continue
            df[col] = df[col].astype("category").cat.codes
            self.report.encoded_columns.append(col)
        return df
