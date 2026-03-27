"""Load trained linear models and produce predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from .features import FEATURE_COLS


class LinearPredictor:
    """Logistic Regression (win probability) + Ridge (margin) predictor."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.scaler = joblib.load(model_dir / "scaler.joblib")
        self.logreg = joblib.load(model_dir / "logreg.joblib")
        self.ridge  = joblib.load(model_dir / "ridge.joblib")

        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {}

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Return per-game predictions.

        Parameters
        ----------
        features_df : DataFrame whose columns include ``FEATURE_COLS``.

        Returns
        -------
        DataFrame with columns ``pHomeWin_ml`` and ``margin_ml``.
        """
        X = features_df[FEATURE_COLS].values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)

        proba  = self.logreg.predict_proba(X_scaled)[:, 1]
        margin = self.ridge.predict(X_scaled)

        return pd.DataFrame({
            "pHomeWin_ml": proba,
            "margin_ml":   margin,
        })


def load_predictor(model_dir: Path) -> Optional[LinearPredictor]:
    """Load the linear predictor if artefacts are present."""
    required_files = [
        "scaler.joblib",
        "logreg.joblib",
        "ridge.joblib",
    ]
    if all((model_dir / f).exists() for f in required_files):
        return LinearPredictor(model_dir)
    return None
