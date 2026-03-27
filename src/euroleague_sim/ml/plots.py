"""Training diagnostic figures (coefficients, correlation, calibration, margin fit)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit

from .features import FEATURE_COLS


def save_training_diagnostics(
    ordered_df: pd.DataFrame,
    X_scaled: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    logreg: LogisticRegression,
    ridge: Ridge,
    output_path: Path,
    *,
    feature_cols: Sequence[str] | None = None,
    cv_folds: int = 5,
) -> None:
    """Write a 2×2 figure: coef tornado, feature correlation, calibration, margin scatter.

    Calibration uses time-series cross-validated predicted probabilities so the
    reliability curve reflects out-of-fold behaviour (not training-set overfit).
    """
    names: List[str] = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    # --- 1) Logistic coefficients (horizontal tornado) ---
    ax1 = axes[0, 0]
    coefs = logreg.coef_[0]
    order = np.argsort(np.abs(coefs))
    names_s = [names[i] for i in order]
    vals_s = coefs[order]
    colors = np.where(vals_s >= 0, "#2E7D32", "#C62828")
    ax1.barh(names_s, vals_s, color=colors, edgecolor="white", linewidth=0.4)
    ax1.axvline(0.0, color="black", linewidth=0.8)
    ax1.set_xlabel("Coefficient (standardized features)")
    ax1.set_title(
        "Logistic regression coefficients\n"
        "(+ → higher P(home win); − → higher P(away win))"
    )

    # --- 2) Feature correlation heatmap ---
    ax2 = axes[0, 1]
    X_raw = ordered_df[names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    corr = X_raw.corr().fillna(0.0)
    im = ax2.imshow(corr.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_yticks(np.arange(len(names)))
    ax2.set_xticklabels(names, rotation=55, ha="right", fontsize=7)
    ax2.set_yticklabels(names, fontsize=7)
    ax2.set_title("Feature correlation\n(dark |r| → redundant / shared variance)")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Pearson r")

    # --- 3) Calibration (reliability) curve, CV probabilities ---
    ax3 = axes[1, 0]
    n_splits = min(cv_folds, max(2, len(y_cls) // 10))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    # TimeSeriesSplit is not a full partition of indices, so we cannot use
    # cross_val_predict; accumulate out-of-fold probabilities on test folds only.
    proba_oof = np.full(len(y_cls), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X_scaled):
        lr_fold = clone(logreg)
        lr_fold.fit(X_scaled[train_idx], y_cls[train_idx])
        proba_oof[test_idx] = lr_fold.predict_proba(X_scaled[test_idx])[:, 1]
    mask = np.isfinite(proba_oof)
    prob_true, prob_pred = calibration_curve(
        y_cls[mask],
        proba_oof[mask],
        n_bins=10,
        strategy="uniform",
    )
    ax3.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration")
    ax3.plot(prob_pred, prob_true, "o-", color="#1565C0", linewidth=1.5, markersize=6, label="Model (time-series CV)")
    ax3.set_xlabel("Mean predicted P(home win)")
    ax3.set_ylabel("Fraction of home wins")
    ax3.set_title("Probability calibration\n(out-of-sample via time-series CV)")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_aspect("equal", adjustable="box")
    ax3.grid(True, alpha=0.3)

    # --- 4) Actual vs predicted margin (Ridge) ---
    ax4 = axes[1, 1]
    pred_m = ridge.predict(X_scaled)
    resid = y_reg - pred_m
    sc = ax4.scatter(
        pred_m,
        y_reg,
        c=resid,
        cmap="coolwarm",
        alpha=0.55,
        s=22,
        edgecolors="none",
    )
    lo = float(min(pred_m.min(), y_reg.min()))
    hi = float(max(pred_m.max(), y_reg.max()))
    pad = 0.05 * (hi - lo + 1e-6)
    lim = (lo - pad, hi + pad)
    ax4.plot(lim, lim, "k--", linewidth=1.0, label="Perfect prediction (y = x)")
    ax4.set_xlim(lim)
    ax4.set_ylim(lim)
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_xlabel("Predicted margin (home − away)")
    ax4.set_ylabel("Actual margin (home − away)")
    ax4.set_title("Ridge: actual vs predicted margin\n(color = residual, in-sample)")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Residual (actual − pred)")

    fig.suptitle("Euroleague ML training diagnostics", fontsize=14, y=1.02)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
