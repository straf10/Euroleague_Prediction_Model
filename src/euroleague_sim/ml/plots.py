"""Training diagnostic figures (coefficients, correlation, calibration, margin fit)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .features import FEATURE_COLS
from .evaluate import EvalResult


def save_training_diagnostics(
    ordered_df: pd.DataFrame,
    X_fit: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    win_est: Any,
    margin_est: Any,
    eval_res: EvalResult,
    output_path: Path,
    *,
    feature_cols: Sequence[str] | None = None,
) -> None:
    """Write a 2×2 figure: importance tornado/bar, feature correlation, calibration, margin scatter."""
    names: List[str] = list(feature_cols) if feature_cols is not None else list(FEATURE_COLS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    # --- 1) Feature importance (tornado or bar) ---
    ax1 = axes[0, 0]
    
    from .weights import get_weights_from_estimator
    weights_dict, weight_type = get_weights_from_estimator(win_est, names)
    
    if weight_type == "coefficients":
        coefs = np.array([weights_dict[n] for n in names])
        order = np.argsort(np.abs(coefs))
        names_s = [names[i] for i in order]
        vals_s = coefs[order]
        colors = np.where(vals_s >= 0, "#2E7D32", "#C62828")
        ax1.barh(names_s, vals_s, color=colors, edgecolor="white", linewidth=0.4)
        ax1.axvline(0.0, color="black", linewidth=0.8)
        ax1.set_xlabel("Coefficient")
        ax1.set_title(
            "Linear coefficients\n"
            "(+ → higher P(home win); − → higher P(away win))"
        )
    elif weight_type == "importances":
        importances = np.array([weights_dict[n] for n in names])
        order = np.argsort(importances)
        names_s = [names[i] for i in order]
        vals_s = importances[order]
        ax1.barh(names_s, vals_s, color="#1565C0", edgecolor="white", linewidth=0.4)
        ax1.set_xlabel("Feature Importance")
        ax1.set_title("Tree Feature Importances\n(magnitude only, no direction)")
    else:
        ax1.text(0.5, 0.5, "No coef_ or feature_importances_ found", 
                 ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Feature Importance")

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
    mask = np.isfinite(eval_res.oof_proba)
    if mask.any():
        prob_true, prob_pred = calibration_curve(
            eval_res.y_win[mask],
            eval_res.oof_proba[mask],
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

    # --- 4) Actual vs predicted margin ---
    ax4 = axes[1, 1]
    pred_m = margin_est.predict(X_fit)
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
    ax4.set_title("Margin: actual vs predicted\n(color = residual, in-sample)")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label("Residual (actual − pred)")

    fig.suptitle("Euroleague ML training diagnostics", fontsize=14, y=1.02)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
