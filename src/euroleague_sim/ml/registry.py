"""Model Registry for Euroleague predictions."""

from typing import Any, Callable, Dict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge


def get_baseline_model() -> Dict[str, Any]:
    """Return the optimized Logistic Regression + Ridge baseline models.
    
    The win estimator is wrapped in CalibratedClassifierCV to ensure
    probabilities are calibrated during cross-validation and final fit.
    """
    return {
        "name": "Baseline (LogReg + Ridge)",
        "win": CalibratedClassifierCV(
            LogisticRegression(C=0.3, max_iter=1000, solver="lbfgs", random_state=42),
            method="sigmoid",
            cv=3,
        ),
        "margin": Ridge(alpha=75.0, random_state=42),
        "requires_scaling": True,
    }


MODEL_REGISTRY: Dict[str, Callable[[], Dict[str, Any]]] = {
    "baseline": get_baseline_model,
}


def get_model(name: str) -> Dict[str, Any]:
    """Get a model dictionary from the registry by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
