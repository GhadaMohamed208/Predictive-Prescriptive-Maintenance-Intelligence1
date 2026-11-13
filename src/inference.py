"""Inference service: loads the saved pipeline once and predicts with the training schema."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .evaluate_models import risk_state
from .feature_engineering import FeatureConfig, build_features


class ArtifactsMissingError(RuntimeError):
    pass


def artifacts_available() -> bool:
    return C.MODEL_PATH.exists() and C.METADATA_PATH.exists()


@dataclass
class Predictor:
    bundle: dict
    metadata: dict
    cfg: FeatureConfig
    feature_names: list[str]
    reference: dict | None

    @property
    def cap(self) -> float:
        return float(self.bundle["rul_cap"])

    @property
    def sensors(self) -> list[str]:
        return list(self.bundle["sensors"])

    @property
    def model(self):
        return self.bundle["model"]

    @classmethod
    def load(cls, model_path: Path = C.MODEL_PATH, metadata_path: Path = C.METADATA_PATH) -> "Predictor":
        if not Path(model_path).exists() or not Path(metadata_path).exists():
            raise ArtifactsMissingError(
                "Trained model artifacts not found. Run:  python -m src.train_models")
        bundle = joblib.load(model_path)
        meta = json.loads(Path(metadata_path).read_text())
        cfg = FeatureConfig.from_dict(bundle["feature_config"])
        ref = None
        if C.REFERENCE_STATS_PATH.exists():
            ref = json.loads(C.REFERENCE_STATS_PATH.read_text())
        return cls(bundle=bundle, metadata=meta, cfg=cfg, feature_names=list(bundle["feature_names"]),
                   reference=ref)

    # ---------------------------------------------------------------- features / predictions
    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature frame [unit, cycle, *features] in the exact training order (causal)."""
        F = build_features(df, self.cfg)
        return F

    def predict_features(self, F: pd.DataFrame) -> np.ndarray:
        X = F[self.feature_names]  # enforces training feature order
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Non-finite engineered features; check the input data.")
        return np.clip(self.model.predict(X), 0.0, self.cap)

    def predict_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prediction at every historical cycle (each uses only data up to that cycle)."""
        F = self.features(df)
        out = F[[C.ID_COL, C.CYCLE_COL]].copy()
        out["rul_pred"] = self.predict_features(F)
        return out

    def predict_latest(self, df: pd.DataFrame, thresholds: dict | None = None
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """One row per engine at its latest valid cycle. Returns (summary, feature rows)."""
        thresholds = thresholds or C.DEFAULT_THRESHOLDS
        F = self.features(df)
        last = F.groupby(C.ID_COL).tail(1).reset_index(drop=True)
        pred = self.predict_features(last)
        res = last[[C.ID_COL, C.CYCLE_COL]].rename(columns={C.CYCLE_COL: "last_cycle"}).copy()
        res["rul_pred"] = pred
        res["state"] = [risk_state(p, thresholds) for p in pred]
        res["at_cap"] = pred >= self.cap - 1e-9
        return res, last

    # ---------------------------------------------------------------- OOD checks
    def ood_report(self, feature_row: pd.DataFrame) -> dict:
        """Compare engineered features of a row with the training min/max ranges."""
        if not self.reference:
            return {"available": False}
        fr = self.reference["features"]
        out = []
        r = feature_row.iloc[0]
        for f in self.feature_names:
            lo, hi = fr[f]["min"], fr[f]["max"]
            if r[f] < lo or r[f] > hi:
                out.append(f)
        frac = len(out) / len(self.feature_names)
        return {"available": True, "n_outside": len(out), "fraction": frac, "features": out,
                "flag": frac > 0.05}


def state_from_thresholds(rul: float, thresholds: dict) -> str:
    return risk_state(rul, thresholds)
