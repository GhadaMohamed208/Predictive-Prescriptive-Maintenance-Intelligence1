"""Loading of NASA C-MAPSS files and generic CSV trajectories."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


class DataFormatError(ValueError):
    """Raised when a file cannot be interpreted as the expected format."""


def read_cmapss(path: str | Path) -> pd.DataFrame:
    """Read a whitespace separated, header-less C-MAPSS trajectory file (26 columns)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")  # trailing whitespace can create an empty column
    if df.shape[1] != C.N_RAW_COLUMNS:
        raise DataFormatError(
            f"{Path(path).name}: expected {C.N_RAW_COLUMNS} columns, found {df.shape[1]}."
        )
    df.columns = C.RAW_COLUMNS
    df[C.ID_COL] = df[C.ID_COL].astype(int)
    df[C.CYCLE_COL] = df[C.CYCLE_COL].astype(int)
    return df


def read_rul_file(path: str | Path) -> pd.Series:
    """One true RUL per test engine, ordered by engine id 1..N."""
    s = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].astype(float)
    s.index = np.arange(1, len(s) + 1)
    s.name = "rul_true"
    return s


def load_fd001(data_dir: str | Path = C.DATA_DIR, subset: str = "FD001"):
    data_dir = Path(data_dir)
    files = {k: data_dir / f"{k}_{subset}.txt" for k in ("train", "test", "RUL")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset file(s): {missing}")
    return read_cmapss(files["train"]), read_cmapss(files["test"]), read_rul_file(files["RUL"])


def verify_subset_identity(train: pd.DataFrame, test: pd.DataFrame, rul: pd.Series) -> dict:
    """Check the *contents* against the documented FD001 properties (do not trust file names).

    Documented FD001 (readme.txt): 100 train / 100 test trajectories, ONE operating
    condition, ONE fault mode (HPC degradation)."""
    n_cond = int(train[C.SETTING_COLS].round(1).drop_duplicates().shape[0])
    info = {
        "n_train_engines": int(train[C.ID_COL].nunique()),
        "n_test_engines": int(test[C.ID_COL].nunique()),
        "n_rul_values": int(len(rul)),
        "op3_unique_values": int(train["op3"].nunique()),
        "operating_condition_clusters_rounded_1dp": n_cond,
        "columns": int(train.shape[1]),
    }
    checks = {
        "100 train engines": info["n_train_engines"] == 100,
        "100 test engines": info["n_test_engines"] == 100,
        "one RUL value per test engine": info["n_rul_values"] == info["n_test_engines"],
        "single operating condition (op3 constant)": info["op3_unique_values"] == 1,
        "26 columns": info["columns"] == 26,
    }
    info["checks"] = checks
    info["matches_FD001"] = bool(all(checks.values()))
    return info


# --------------------------- generic CSV reading (upload) ---------------------------
def read_uploaded_bytes(raw: bytes, max_rows: int = C.MAX_UPLOAD_ROWS) -> pd.DataFrame:
    """Parse uploaded bytes as CSV (comma/semicolon/tab) or header-less NASA whitespace text.

    The bytes are only ever parsed as data; nothing is executed or evaluated."""
    if len(raw) > C.MAX_UPLOAD_MB * 1024 * 1024:
        raise DataFormatError(f"File exceeds the {C.MAX_UPLOAD_MB} MB limit.")
    if not raw.strip():
        raise DataFormatError("The file is empty.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except Exception as exc:  # pragma: no cover
            raise DataFormatError("Could not decode the file as text.") from exc
    first = text.splitlines()[0]
    try:
        if "," in first or ";" in first or "\t" in first:
            sep = max([",", ";", "\t"], key=first.count)
            df = pd.read_csv(io.StringIO(text), sep=sep, nrows=max_rows + 1)
        else:  # whitespace separated (NASA style); header only if first token is non-numeric
            tok = first.split()[0] if first.split() else ""
            try:
                float(tok)
                header = None
            except ValueError:
                header = 0
            df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=header, nrows=max_rows + 1,
                             engine="python")
            df = df.dropna(axis=1, how="all")
            if header is None and df.shape[1] == C.N_RAW_COLUMNS:
                df.columns = C.RAW_COLUMNS
    except Exception as exc:
        raise DataFormatError(f"The file could not be parsed as CSV: {exc}") from exc
    if len(df) > max_rows:
        raise DataFormatError(f"File has more than {max_rows:,} rows (limit).")
    if df.shape[1] < 3:
        raise DataFormatError("The file has fewer than 3 columns; need engine id, cycle and sensors.")
    df.columns = [str(c).strip() for c in df.columns]
    return df
