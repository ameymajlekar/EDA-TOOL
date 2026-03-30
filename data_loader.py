"""
data_loader.py
--------------
Utility to load datasets from various file formats into a pandas DataFrame.
Supports: CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet, Feather, SQLite.
"""

import pandas as pd
import os
import sqlite3
from pathlib import Path
from typing import Optional, Union


SUPPORTED_FORMATS = {
    ".csv": "CSV",
    ".tsv": "TSV",
    ".txt": "CSV/TSV",
    ".xlsx": "Excel",
    ".xls": "Excel (legacy)",
    ".json": "JSON",
    ".parquet": "Parquet",
    ".feather": "Feather",
    ".db": "SQLite",
    ".sqlite": "SQLite",
    ".sqlite3": "SQLite",
}


def load_file(
    path: Union[str, Path, None] = None,
    file_obj=None,
    file_name: Optional[str] = None,
    sql_table: Optional[str] = None,
    **read_kwargs,
) -> pd.DataFrame:
    """
    Load a dataset from a file path or file-like object.

    Parameters
    ----------
    path        : Disk path to the file (string or Path).
    file_obj    : File-like object (e.g., from st.file_uploader).
    file_name   : Original filename — used to infer format when file_obj is given.
    sql_table   : Table name for SQLite files (optional; loads first table if None).
    **read_kwargs: Extra kwargs forwarded to the pandas reader.

    Returns
    -------
    pd.DataFrame
    """
    if file_obj is not None:
        name = file_name or getattr(file_obj, "name", "upload.csv")
        ext = Path(name).suffix.lower()
        return _read_from_obj(file_obj, ext, sql_table=sql_table, **read_kwargs)

    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        ext = path.suffix.lower()
        return _read_from_path(path, ext, sql_table=sql_table, **read_kwargs)

    raise ValueError("Provide either `path` or `file_obj`.")


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _read_from_path(path: Path, ext: str, sql_table=None, **kw) -> pd.DataFrame:
    if ext in (".csv", ".txt"):
        return _try_csv(path, **kw)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t", **kw)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, **kw)
    if ext == ".json":
        return _read_json(path, **kw)
    if ext == ".parquet":
        return pd.read_parquet(path, **kw)
    if ext == ".feather":
        return pd.read_feather(path, **kw)
    if ext in (".db", ".sqlite", ".sqlite3"):
        return _read_sqlite(str(path), sql_table, **kw)
    raise ValueError(f"Unsupported format: {ext}. Supported: {list(SUPPORTED_FORMATS.keys())}")


def _read_from_obj(obj, ext: str, sql_table=None, **kw) -> pd.DataFrame:
    if ext in (".csv", ".txt"):
        return _try_csv(obj, **kw)
    if ext == ".tsv":
        return pd.read_csv(obj, sep="\t", **kw)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(obj, **kw)
    if ext == ".json":
        return _read_json(obj, **kw)
    if ext == ".parquet":
        return pd.read_parquet(obj, **kw)
    if ext == ".feather":
        return pd.read_feather(obj, **kw)
    raise ValueError(f"Unsupported file object format: {ext}")


def _try_csv(source, **kw) -> pd.DataFrame:
    """Attempt to auto-detect delimiter."""
    # Try comma first, then semicolon, then pipe
    for sep in [",", ";", "|", "\t"]:
        try:
            df = pd.read_csv(source, sep=sep, **kw)
            if df.shape[1] > 1:
                return df
            if hasattr(source, "seek"):
                source.seek(0)
        except Exception:
            if hasattr(source, "seek"):
                source.seek(0)
    return pd.read_csv(source, **kw)


def _read_json(source, **kw) -> pd.DataFrame:
    for orient in ["records", "split", "index", "columns", "values"]:
        try:
            df = pd.read_json(source, orient=orient, **kw)
            if not df.empty:
                return df
            if hasattr(source, "seek"):
                source.seek(0)
        except Exception:
            if hasattr(source, "seek"):
                source.seek(0)
    return pd.read_json(source, **kw)


def _read_sqlite(db_path: str, table: Optional[str], **kw) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        if table is None:
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if tables.empty:
                raise ValueError("SQLite database contains no tables.")
            table = tables["name"].iloc[0]
        df = pd.read_sql(f"SELECT * FROM {table}", conn, **kw)
    finally:
        conn.close()
    return df


def supported_extensions() -> list:
    return list(SUPPORTED_FORMATS.keys())


def format_info() -> pd.DataFrame:
    return pd.DataFrame(
        [(ext, fmt) for ext, fmt in SUPPORTED_FORMATS.items()],
        columns=["Extension", "Format"],
    )
