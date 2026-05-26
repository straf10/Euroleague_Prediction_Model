"""Beta-calibration wrapper for probabilistic classifiers.

Replaces sklearn's ``CalibratedClassifierCV`` (Platt/sigmoid scaling). Beta
calibration fits a three-parameter Beta distribution to the classifier scores,
which corrects miscalibration at the *tails* (very confident predictions) far
better than the logistic curve that Platt scaling assumes — the failure mode we
observed on this small, high-variance EuroLeague dataset.

The wrapper is fully scikit-learn compatible (``BaseEstimator`` /
``ClassifierMixin``) so it works with ``clone`` inside the walk-forward and
time-series CV loops and serialises cleanly through ``joblib``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from betacal import BetaCalibration
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def _has_early_stopping(est: Any) -> bool:
    """Duck-type check for a CatBoost-style estimator that supports early stopping
    via an ``eval_set`` and exposes ``get_best_iteration``.
    """
    return hasattr(est, "get_best_iteration")


def fit_catboost_es(
    est: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    es_fraction: float = 0.15,
    min_es_samples: int = 40,
    refit_on_full: bool = True,
) -> Any:
    """Fit an estimator with chronological-tail early stopping, then refit on full.

    For CatBoost estimators with enough data, the last ``es_fraction`` of the
    (already time-ordered) window is held out as an ``eval_set`` so the
    configured ``early_stopping_rounds`` halts training once the validation
    metric plateaus. The estimator is then refit on the *full* window capped at
    the discovered ``best_iteration`` — no validation data is wasted at inference
    time and no compute is spent on plateaued iterations.

    Non-CatBoost estimators (e.g. ``Ridge``) and tiny windows fall back to a
    plain ``fit`` so the baseline path is unchanged.
    """
    n = len(y)
    n_val = int(round(n * es_fraction))
    split = n - n_val

    if not _has_early_stopping(est) or n_val < min_es_samples or split <= 0:
        est.fit(X, y)
        return est

    est.fit(X[:split], y[:split], eval_set=(X[split:], y[split:]))
    if not refit_on_full:
        return est

    # CatBoost forbids set_params on a fitted model, so refit a fresh clone capped
    # at the best iteration found during early stopping (no eval_set needed).
    best_it = est.get_best_iteration()
    refit = clone(est)
    if best_it and best_it > 0:
        refit.set_params(iterations=int(best_it) + 1)
    refit.fit(X, y)
    return refit


class BetaCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a binary classifier and calibrate its probabilities with Beta calibration.

    On ``fit`` the (already chronologically ordered) training window is split
    into a *proper-train* portion and a *calibration* tail. The base estimator
    is fit on the proper-train portion, its scores on the held-out tail are used
    to fit the Beta calibrator, and — when ``refit_on_full`` is set — the base
    estimator is finally refit on the full window so no data is wasted at
    inference time while the calibrator remains learned out-of-sample.

    Parameters
    ----------
    base_estimator : a binary classifier exposing ``predict_proba``.
    calib_fraction : fraction of the (ordered) training window reserved as the
        calibration tail.
    min_calib_samples : if the calibration tail is smaller than this, calibration
        is skipped and raw base probabilities are returned (keeps early
        walk-forward folds functional rather than fitting an unstable calibrator).
    parameters : Beta-calibration parameterisation passed to ``betacal``
        (``"abm"`` is the full three-parameter map).
    refit_on_full : refit the base estimator on the full window after the
        calibrator has been learned on the held-out tail.
    """

    def __init__(
        self,
        base_estimator: Any,
        *,
        calib_fraction: float = 0.25,
        min_calib_samples: int = 30,
        parameters: str = "abm",
        refit_on_full: bool = True,
    ) -> None:
        self.base_estimator = base_estimator
        self.calib_fraction = calib_fraction
        self.min_calib_samples = min_calib_samples
        self.parameters = parameters
        self.refit_on_full = refit_on_full

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BetaCalibratedClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        n = len(y)
        n_calib = int(round(n * self.calib_fraction))
        split = n - n_calib

        calibrator: Optional[BetaCalibration] = None
        base = clone(self.base_estimator)

        # Need a usable calibration tail with both classes present to fit Beta.
        can_calibrate = (
            n_calib >= self.min_calib_samples
            and split > 0
            and len(np.unique(y[:split])) == 2
            and len(np.unique(y[split:])) == 2
        )

        if can_calibrate:
            # The calibration tail doubles as the early-stopping eval_set: the base
            # never trains on it, so using it for both is leakage-free.
            if _has_early_stopping(base):
                base.fit(X[:split], y[:split], eval_set=(X[split:], y[split:]))
            else:
                base.fit(X[:split], y[:split])
            calib_scores = base.predict_proba(X[split:])[:, 1]
            calibrator = BetaCalibration(parameters=self.parameters)
            calibrator.fit(calib_scores, y[split:])
            if self.refit_on_full:
                best_it = base.get_best_iteration() if _has_early_stopping(base) else None
                base = clone(self.base_estimator)
                if best_it and best_it > 0:
                    base.set_params(iterations=int(best_it) + 1)
                base.fit(X, y)
        else:
            base.fit(X, y)

        self.base_estimator_ = base
        self.calibrator_ = calibrator
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        raw = self.base_estimator_.predict_proba(X)[:, 1]
        if self.calibrator_ is not None:
            p1 = np.clip(self.calibrator_.predict(raw), 0.0, 1.0)
        else:
            p1 = raw
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = (self.predict_proba(X)[:, 1] > 0.5).astype(int)
        return self.classes_[idx]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Expose the underlying estimator's importances for diagnostics."""
        return self.base_estimator_.feature_importances_
