"""Swappable probability-calibration wrapper for binary classifiers.

We compared three calibration families on this small, high-variance EuroLeague
dataset:

* ``"beta"``     — three-parameter Beta calibration (``betacal``). Corrects
  tail miscalibration well; the default we shipped originally.
* ``"sigmoid"``  — one-parameter logistic (Platt) scaling via sklearn's
  ``_SigmoidCalibration``. Cheap; assumes a logistic-shaped reliability curve.
* ``"isotonic"`` — non-parametric monotonic regression via
  ``IsotonicRegression``. Most flexible but needs more calibration samples.

All three are wrapped the same way: the (already chronologically ordered)
training window is split into a *proper-train* prefix and a *calibration tail*.
The base classifier is fit on the prefix, its scores on the tail are used to
fit the chosen calibrator, and — when ``refit_on_full`` is set — the base
classifier is finally refit on the full window so no data is wasted at
inference while the calibrator remains learned strictly out-of-sample.

For CatBoost the tail also doubles as ``eval_set``, so ``early_stopping_rounds``
fires on the same held-out data the calibrator sees (still leakage-free: the
base never trains on those rows). The best iteration is then carried into the
refit on the full window.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from betacal import BetaCalibration
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression


CALIB_METHODS = ("beta", "sigmoid", "isotonic")


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

    best_it = est.get_best_iteration()
    refit = clone(est)
    if best_it and best_it > 0:
        refit.set_params(iterations=int(best_it) + 1)
    refit.fit(X, y)
    return refit


def _build_calibrator(method: str, parameters: str = "abm") -> Any:
    """Instantiate a 1-D probability calibrator by name."""
    if method == "beta":
        return BetaCalibration(parameters=parameters)
    if method == "sigmoid":
        return _SigmoidCalibration()
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    raise ValueError(f"Unknown calibration method '{method}'. Use one of {CALIB_METHODS}.")


def _calibrator_predict(calibrator: Any, scores: np.ndarray) -> np.ndarray:
    """Uniform predict interface across betacal / sklearn calibrators."""
    return np.clip(calibrator.predict(scores), 0.0, 1.0)


class CalibratedWinClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a binary classifier and calibrate its probabilities via Beta, Platt
    (sigmoid), or isotonic regression — selected by the ``method`` parameter.

    The fit protocol is identical across methods so the only thing that varies
    in benchmarks is the calibrator family itself, not the data-splitting
    strategy:

    1. Split the time-ordered training window into proper-train (head) and
       calibration tail.
    2. Fit the base estimator on the proper-train. For CatBoost, pass the tail
       as ``eval_set`` to drive early stopping (leakage-free — the base never
       trains on those rows).
    3. Score the tail with the proper-train base and fit the chosen calibrator
       on (scores, y_tail).
    4. If ``refit_on_full`` is set, refit a fresh clone of the base estimator on
       the *full* window — capped at ``best_iteration`` for CatBoost — so no
       data is wasted at inference time.

    Parameters
    ----------
    base_estimator : a binary classifier exposing ``predict_proba``.
    method : ``"beta"``, ``"sigmoid"``, or ``"isotonic"``.
    calib_fraction : fraction of the (ordered) training window reserved as the
        calibration tail.
    min_calib_samples : if the calibration tail is smaller than this, calibration
        is skipped and raw base probabilities are returned (keeps early
        walk-forward folds functional rather than fitting an unstable calibrator).
    parameters : Beta-calibration parameterisation passed to ``betacal``
        (``"abm"`` is the full three-parameter map). Ignored for sigmoid/isotonic.
    refit_on_full : refit the base estimator on the full window after the
        calibrator has been learned on the held-out tail.
    """

    def __init__(
        self,
        base_estimator: Any,
        *,
        method: str = "beta",
        calib_fraction: float = 0.25,
        min_calib_samples: int = 30,
        parameters: str = "abm",
        refit_on_full: bool = True,
    ) -> None:
        self.base_estimator = base_estimator
        self.method = method
        self.calib_fraction = calib_fraction
        self.min_calib_samples = min_calib_samples
        self.parameters = parameters
        self.refit_on_full = refit_on_full

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibratedWinClassifier":
        if self.method not in CALIB_METHODS:
            raise ValueError(
                f"Unknown calibration method '{self.method}'. Use one of {CALIB_METHODS}."
            )

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        n = len(y)
        n_calib = int(round(n * self.calib_fraction))
        split = n - n_calib

        calibrator: Optional[Any] = None
        base = clone(self.base_estimator)

        can_calibrate = (
            n_calib >= self.min_calib_samples
            and split > 0
            and len(np.unique(y[:split])) == 2
            and len(np.unique(y[split:])) == 2
        )

        if can_calibrate:
            if _has_early_stopping(base):
                base.fit(X[:split], y[:split], eval_set=(X[split:], y[split:]))
            else:
                base.fit(X[:split], y[:split])
            calib_scores = base.predict_proba(X[split:])[:, 1]
            calibrator = _build_calibrator(self.method, parameters=self.parameters)
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
            p1 = _calibrator_predict(self.calibrator_, raw)
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


class BetaCalibratedClassifier(CalibratedWinClassifier):
    """Backwards-compatible alias: ``CalibratedWinClassifier(method="beta")``.

    Kept so existing pickled artefacts and imports continue to load.
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
        super().__init__(
            base_estimator,
            method="beta",
            calib_fraction=calib_fraction,
            min_calib_samples=min_calib_samples,
            parameters=parameters,
            refit_on_full=refit_on_full,
        )
