"""Load trained models and produce predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from .features import FEATURE_COLS


class ModelPredictor:
    """Predictor using trained win and margin models."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        
        meta_path = model_dir / "metadata.json"
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {}
            
        self.requires_scaling = self.metadata.get("requires_scaling", False)
        
        if self.requires_scaling:
            self.scaler = joblib.load(model_dir / "scaler.joblib")
        else:
            self.scaler = None
            
        self.win_model = joblib.load(model_dir / "win_model.joblib")
        self.margin_model = joblib.load(model_dir / "margin_model.joblib")

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
        
        if self.requires_scaling and self.scaler is not None:
            X = self.scaler.transform(X)

        proba = self.win_model.predict_proba(X)[:, 1]
        margin = self.margin_model.predict(X)

        return pd.DataFrame({
            "pHomeWin_ml": proba,
            "margin_ml":   margin,
        })


def load_predictor(base_model_dir: Path, model_name: str = "baseline") -> Optional[ModelPredictor]:
    """Load the predictor if artefacts are present."""
    model_dir = base_model_dir / model_name
    
    required_files = [
        "win_model.joblib",
        "margin_model.joblib",
        "metadata.json",
    ]
    
    if not all((model_dir / f).exists() for f in required_files):
        return None
        
    # Check if scaling is required to ensure scaler exists
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("requires_scaling", False):
        if not (model_dir / "scaler.joblib").exists():
            return None
            
    return ModelPredictor(model_dir)
