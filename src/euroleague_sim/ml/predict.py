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
    """Combines Random Forest + XGBoost + Neural Network for win-probability and margin."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.scaler  = joblib.load(model_dir / "scaler.joblib")
        self.rf_clf  = joblib.load(model_dir / "rf_classifier.joblib")
        self.rf_reg  = joblib.load(model_dir / "rf_regressor.joblib")
        self.nn_clf  = joblib.load(model_dir / "nn_classifier.joblib")
        self.nn_reg  = joblib.load(model_dir / "nn_regressor.joblib")

        xgb_clf_path = model_dir / "xgb_classifier.joblib"
        xgb_reg_path = model_dir / "xgb_regressor.joblib"
        self.has_xgb = xgb_clf_path.exists() and xgb_reg_path.exists()
        if self.has_xgb:
            self.xgb_clf = joblib.load(xgb_clf_path)
            self.xgb_reg = joblib.load(xgb_reg_path)

        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {}

    # ------------------------------------------------------------------ #
    def predict(
        self,
        features_df: pd.DataFrame,
        rf_weight: float = 0.35,
        nn_weight: float = 0.30,
        xgb_weight: float = 0.35,
    ) -> pd.DataFrame:
        """Return per-game predictions.

        Parameters
        ----------
        features_df : DataFrame whose columns include ``FEATURE_COLS``.
        rf_weight, nn_weight, xgb_weight : ensemble weights (normalised internally).

        Returns
        -------
        DataFrame with columns:
            pHomeWin_rf, pHomeWin_xgb, pHomeWin_nn, pHomeWin_ml,
            margin_rf, margin_xgb, margin_nn, margin_ml
        """
        X = features_df[FEATURE_COLS].values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)

        rf_proba  = self.rf_clf.predict_proba(X_scaled)[:, 1]
        nn_proba  = self.nn_clf.predict_proba(X_scaled)[:, 1]
        rf_margin = self.rf_reg.predict(X_scaled)
        nn_margin = self.nn_reg.predict(X_scaled)

        if self.has_xgb:
            xgb_proba  = self.xgb_clf.predict_proba(X_scaled)[:, 1]
            xgb_margin = self.xgb_reg.predict(X_scaled)

            w_total = rf_weight + xgb_weight + nn_weight
            ens_proba  = (rf_weight * rf_proba + xgb_weight * xgb_proba +
                          nn_weight * nn_proba) / w_total
            ens_margin = (rf_weight * rf_margin + xgb_weight * xgb_margin +
                          nn_weight * nn_margin) / w_total

            return pd.DataFrame({
                "pHomeWin_rf":   rf_proba,
                "pHomeWin_xgb":  xgb_proba,
                "pHomeWin_nn":   nn_proba,
                "pHomeWin_ml":   ens_proba,
                "margin_rf":     rf_margin,
                "margin_xgb":    xgb_margin,
                "margin_nn":     nn_margin,
                "margin_ml":     ens_margin,
            })

        # Fallback: no XGBoost models available (backward compat)
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
    """Load the ensemble predictor if core artefacts are present.

    XGBoost models are optional — the predictor gracefully degrades to
    RF + NN if they're missing (e.g. models trained before XGBoost was added).
    """
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
