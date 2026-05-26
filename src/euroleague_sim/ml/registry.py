"""Model Registry for Euroleague predictions."""

from typing import Any, Callable, Dict

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge

from .calibration import BetaCalibratedClassifier


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
        "margin": Ridge(alpha=106.18, random_state=42),
        "requires_scaling": True,
        "walk_forward": False,
    }


def get_catboost_model() -> Dict[str, Any]:
    """Return the CatBoost win/margin models, evaluated via walk-forward optimisation.

    CatBoost's symmetric (oblivious) trees act as a strong native regulariser on
    small datasets, so the config favours shallow trees, heavy L2 regularisation
    and a slow learning rate to avoid overfitting ~1000 rows. The win classifier
    is wrapped in :class:`BetaCalibratedClassifier` to correct tail miscalibration
    out-of-sample. Trees are scale-invariant, so no ``StandardScaler`` is needed.
    """
    win_clf = CatBoostClassifier(
        iterations=400,
        depth=3,
        learning_rate=0.03,
        l2_leaf_reg=10.0,
        loss_function="Logloss",
        random_seed=42,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        early_stopping_rounds=50,
        allow_writing_files=False,
        verbose=False,
    )
    margin_reg = CatBoostRegressor(
        iterations=400,
        depth=3,
        learning_rate=0.03,
        l2_leaf_reg=10.0,
        loss_function="RMSE",
        random_seed=42,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        early_stopping_rounds=50,
        allow_writing_files=False,
        verbose=False,
    )
    return {
        "name": "CatBoost (Beta-cal + WFO)",
        "win": BetaCalibratedClassifier(win_clf, calib_fraction=0.25, min_calib_samples=30),
        "margin": margin_reg,
        "requires_scaling": False,
        "walk_forward": True,
    }


MODEL_REGISTRY: Dict[str, Callable[[], Dict[str, Any]]] = {
    "baseline": get_baseline_model,
    "catboost": get_catboost_model,
}


def get_model(name: str) -> Dict[str, Any]:
    """Get a model dictionary from the registry by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
