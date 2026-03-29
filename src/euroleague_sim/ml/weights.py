"""Helper for extracting feature weights from trained estimators."""

from typing import Any, Dict, List, Tuple
import numpy as np


def get_weights_from_estimator(est: Any, feature_cols: List[str]) -> Tuple[Dict[str, float], str]:
    """Extract and optionally average feature weights from an estimator.
    
    Returns a tuple of (weights_dict, weight_type).
    weight_type is typically "coefficients" or "importances".
    """
    # 1. Handle CalibratedClassifierCV (Average all internal folds)
    if hasattr(est, "calibrated_classifiers_"):
        all_weights = []
        weight_type = ""
        
        for clf in est.calibrated_classifiers_:
            inner_est = getattr(clf, "estimator", getattr(clf, "base_estimator", None))
            if hasattr(inner_est, "coef_"):
                all_weights.append(inner_est.coef_)
                weight_type = "coefficients"
            elif hasattr(inner_est, "feature_importances_"):
                all_weights.append(inner_est.feature_importances_)
                weight_type = "importances"
                
        if all_weights:
            # Stack and mean to handle potential shape differences like (n_features,) vs (1, n_features)
            stacked = np.stack(all_weights)
            avg_weights = np.mean(stacked, axis=0)
            if avg_weights.ndim > 1:
                avg_weights = avg_weights[0]
            return dict(zip(feature_cols, avg_weights.tolist())), weight_type
            
    # 2. Handle Standard Linear Models
    if hasattr(est, "coef_"):
        coefs = est.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]
        return dict(zip(feature_cols, coefs.tolist())), "coefficients"
        
    # 3. Handle Standard Tree Models
    if hasattr(est, "feature_importances_"):
        return dict(zip(feature_cols, est.feature_importances_.tolist())), "importances"
        
    return {}, ""
