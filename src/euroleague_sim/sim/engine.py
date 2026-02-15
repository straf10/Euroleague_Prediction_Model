from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np


def simulate_next_round(
    matchup_df: pd.DataFrame,
    alpha1: float = 0.9,
    alpha2: float = 1.0,
    alpha3: float = 1.5,
    sigma: float = 11.5,
    n_sims: int = 20_000,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Monte Carlo simulation for margin distribution (spec section 8).

    matchup_df must contain columns A and B (from compute_matchup_features).

    For each game:
        mu = alpha1 * A + alpha2 * B + alpha3
        margin ~ Normal(mu, sigma)
        home win iff margin > 0

    Stores: pHomeWin (MC), muMargin, sigmaMargin, q10, q50, q90.
    """
    required = ["home_team", "away_team", "A", "B"]
    for c in required:
        if c not in matchup_df.columns:
            raise KeyError(f"matchup_df missing column: {c}")

    df = matchup_df.copy().reset_index(drop=True)
    n_games = len(df)
    if n_games == 0:
        return df

    rng = np.random.default_rng(seed)

    a_vals = df["A"].to_numpy(dtype=float)
    b_vals = df["B"].to_numpy(dtype=float)

    mu = alpha1 * a_vals + alpha2 * b_vals + alpha3  # shape (n_games,)

    margins = rng.normal(loc=mu, scale=sigma, size=(int(n_sims), n_games))  # (n_sims, n_games)

    home_win_pct = (margins > 0).mean(axis=0)
    mean_margin = margins.mean(axis=0)
    q10 = np.percentile(margins, 10, axis=0)
    q50 = np.percentile(margins, 50, axis=0)
    q90 = np.percentile(margins, 90, axis=0)

    df["muMargin"] = mu
    df["sigmaMargin"] = sigma
    df["pHomeWin"] = home_win_pct
    df["meanMargin"] = mean_margin
    df["q10"] = q10
    df["q50"] = q50
    df["q90"] = q90
    df["n_sims"] = int(n_sims)

    cols_front = [c for c in ["Round", "Gamecode", "home_team", "away_team"] if c in df.columns]
    rest = [c for c in df.columns if c not in cols_front]
    return df[cols_front + rest]
