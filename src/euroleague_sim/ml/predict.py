"""Load trained ML models and produce ensemble predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from .features import FEATURE_COLS


class EnsemblePredictor:
    """Combines Random Forest + Neural Network for win-probability and margin."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.scaler  = joblib.load(model_dir / "scaler.joblib")
        self.rf_clf  = joblib.load(model_dir / "rf_classifier.joblib")
        self.rf_reg  = joblib.load(model_dir / "rf_regressor.joblib")
        self.nn_clf  = joblib.load(model_dir / "nn_classifier.joblib")
        self.nn_reg  = joblib.load(model_dir / "nn_regressor.joblib")

        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {}

    # ------------------------------------------------------------------ #
    def predict(
        self,
        features_df: pd.DataFrame,
        rf_weight: float = 0.5,
        nn_weight: float = 0.5,
    ) -> pd.DataFrame:
        """Return per-game predictions.

        Parameters
        ----------
        features_df : DataFrame whose columns include ``FEATURE_COLS``.
        rf_weight, nn_weight : ensemble weights (normalised internally).

        Returns
        -------
        DataFrame with columns:
            pHomeWin_rf, pHomeWin_nn, pHomeWin_ml,
            margin_rf, margin_nn, margin_ml
        """
        X = features_df[FEATURE_COLS].values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)

        rf_proba  = self.rf_clf.predict_proba(X_scaled)[:, 1]
        nn_proba  = self.nn_clf.predict_proba(X_scaled)[:, 1]

        rf_margin = self.rf_reg.predict(X_scaled)
        nn_margin = self.nn_reg.predict(X_scaled)

        w_total = rf_weight + nn_weight
        ens_proba  = (rf_weight * rf_proba  + nn_weight * nn_proba)  / w_total
        ens_margin = (rf_weight * rf_margin + nn_weight * nn_margin) / w_total

        return pd.DataFrame({
            "pHomeWin_rf":  rf_proba,
            "pHomeWin_nn":  nn_proba,
            "pHomeWin_ml":  ens_proba,
            "margin_rf":    rf_margin,
            "margin_nn":    nn_margin,
            "margin_ml":    ens_margin,
        })


def load_predictor(model_dir: Path) -> Optional[EnsemblePredictor]:
    """Load the ensemble predictor if all artefacts are present."""
    required_files = [
        "scaler.joblib",
        "rf_classifier.joblib",
        "rf_regressor.joblib",
        "nn_classifier.joblib",
        "nn_regressor.joblib",
    ]
    if all((model_dir / f).exists() for f in required_files):
        return EnsemblePredictor(model_dir)
    return None
