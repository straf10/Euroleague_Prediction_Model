"""Expanded feature engineering for ML models.

Supports two modes:
1. Training mode  – point-in-time features from historical games (no lookahead).
2. Prediction mode – features for upcoming games using current cumulative stats.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


# ---- Ordered list of feature columns consumed by the ML models ----
FEATURE_COLS: List[str] = [
    "elo_diff_scaled",           # (elo_home − elo_away) / 25
    "net_efg_wma5",              # 5-game WMA recent form: eFG%
    "net_tov_wma5",              # 5-game WMA recent form: TOV%
    "net_orb_wma5",              # 5-game WMA recent form: ORB%
    "net_ftr_wma5",              # 5-game WMA recent form: FT rate
    "round_progress",            # round / max_rounds
    "el_rest_days_diff",         # EL schedule density: capped home rest − away rest
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wma5_rolling(arr: np.ndarray) -> float:
    """Linearly-weighted average over *arr*, most recent weighted highest.

    NaN values (e.g. from ``shift(1)`` at position 0) are dropped before
    weighting so that partial windows produce usable values.

    The newest element always receives weight 5.  For *n* < 5 the lightest
    weights are dropped: e.g. n=3 → weights ``[3, 4, 5]``, sum = 12.
    A full window (n=5) uses ``[1, 2, 3, 4, 5]``, sum = 15.
    """
    clean = arr[np.isfinite(arr)]
    n = len(clean)
    if n == 0:
        return np.nan
    w = np.arange(5 - n + 1, 6, dtype=float)
    return np.dot(clean, w) / w.sum()


# ---------------------------------------------------------------------------
# Cumulative (expanding-window) stats for training
# ---------------------------------------------------------------------------

def compute_cumulative_features(team_game_df: pd.DataFrame) -> pd.DataFrame:
    """Add expanding cumulative stats to each team-game row.

    For team T's N-th game the cumulative columns reflect games 1 .. N-1
    (shift-by-one to prevent lookahead bias).  Both overall and home/away
    split statistics are computed.
    """
    df = team_game_df.sort_values(["Team", "Round", "Gamecode"]).copy()
    df["Win"] = (df["PointsFor"] > df["PointsAgainst"]).astype(float)

    # ---------- overall ----------
    g = df.groupby("Team")
    df["cum_pf"]   = g["PointsFor"].transform(lambda x: x.shift(1).expanding().sum())
    df["cum_pa"]   = g["PointsAgainst"].transform(lambda x: x.shift(1).expanding().sum())
    df["cum_poss"] = g["Possessions"].transform(lambda x: x.shift(1).expanding().sum())
    df["cum_n"]    = g["PointsFor"].transform(lambda x: x.shift(1).expanding().count())
    df["cum_wins"] = g["Win"].transform(lambda x: x.shift(1).expanding().sum())
    df["form"]     = g["NetPer100"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["form_wma5"] = g["NetPer100"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).apply(
            _wma5_rolling, raw=True,
        )
    )

    df["cum_off_rtg"] = np.where(
        df["cum_poss"] > 0, 100.0 * df["cum_pf"] / df["cum_poss"], np.nan,
    )
    df["cum_def_rtg"] = np.where(
        df["cum_poss"] > 0, 100.0 * df["cum_pa"] / df["cum_poss"], np.nan,
    )
    df["cum_net_rtg"] = df["cum_off_rtg"] - df["cum_def_rtg"]
    df["cum_win_pct"] = np.where(
        df["cum_n"] > 0, df["cum_wins"] / df["cum_n"], np.nan,
    )
    df["cum_pace"] = np.where(
        df["cum_n"] > 0, df["cum_poss"] / df["cum_n"], np.nan,
    )

    # recent pace: 5-game WMA of per-game possessions, strictly pre-game
    df["pace_wma5"] = g["Possessions"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).apply(
            _wma5_rolling, raw=True,
        )
    )

    # ---------- Four Factors: per-game rates → shifted 5-game WMAs ----------
    _ff_cols = [
        "FGM", "FGA", "FGM3", "ORB", "DRB", "TOV", "FTA",
        "opp_DRB", "opp_ORB", "opp_FGM", "opp_FGA", "opp_FGM3", "opp_TOV", "opp_FTA",
    ]
    _has_ff = all(c in df.columns for c in _ff_cols)
    if _has_ff:
        df["game_efg"] = np.where(
            df["FGA"] > 0,
            (df["FGM"] + 0.5 * df["FGM3"]) / df["FGA"],
            np.nan,
        )
        _game_tov_denom = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
        df["game_tov"] = np.where(
            _game_tov_denom > 0,
            df["TOV"] / _game_tov_denom,
            np.nan,
        )
        _game_orb_denom = df["ORB"] + df["opp_DRB"]
        df["game_orb"] = np.where(
            _game_orb_denom > 0,
            df["ORB"] / _game_orb_denom,
            np.nan,
        )
        df["game_ftr"] = np.where(
            df["FGA"] > 0,
            df["FTA"] / df["FGA"],
            np.nan,
        )

        for _rate_col in ["game_efg", "game_tov", "game_orb", "game_ftr"]:
            _wma_col = _rate_col.replace("game_", "") + "_wma5"
            df[_wma_col] = g[_rate_col].transform(
                lambda x: x.shift(1).rolling(5, min_periods=1).apply(
                    _wma5_rolling, raw=True,
                )
            )

    # ---------- home / away split ----------
    for is_home_val, prefix in [(1, "hs_"), (0, "as_")]:
        mask = df["IsHome"] == is_home_val
        sub = df.loc[mask].copy()
        if sub.empty:
            continue
        gs = sub.groupby("Team")

        sub[f"{prefix}cum_pf"]   = gs["PointsFor"].transform(lambda x: x.shift(1).expanding().sum())
        sub[f"{prefix}cum_pa"]   = gs["PointsAgainst"].transform(lambda x: x.shift(1).expanding().sum())
        sub[f"{prefix}cum_poss"] = gs["Possessions"].transform(lambda x: x.shift(1).expanding().sum())

        sub[f"{prefix}off_rtg"] = np.where(
            sub[f"{prefix}cum_poss"] > 0,
            100.0 * sub[f"{prefix}cum_pf"] / sub[f"{prefix}cum_poss"],
            np.nan,
        )
        sub[f"{prefix}def_rtg"] = np.where(
            sub[f"{prefix}cum_poss"] > 0,
            100.0 * sub[f"{prefix}cum_pa"] / sub[f"{prefix}cum_poss"],
            np.nan,
        )

        split_cols = [c for c in sub.columns if c.startswith(prefix)]
        for col in split_cols:
            df.loc[mask, col] = sub[col].values

    return df


# ---------------------------------------------------------------------------
# Build training dataset (multiple seasons, point-in-time)
# ---------------------------------------------------------------------------

def build_training_dataset(
    seasons_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    max_rounds: int = 34,
    default_pace: float = 70.0,
) -> pd.DataFrame:
    """Create a labelled feature matrix from historical game data.

    Parameters
    ----------
    seasons_data : {season: (games_df, team_game_df, elo_game_log_df)}
        ``games_df``           – one row per game (from ``build_games_with_possessions``)
        ``team_game_df``       – one row per team-game (from ``build_team_game_net_ratings``)
        ``elo_game_log_df``    – per-game Elo log (``EloResult.game_elos``)
    max_rounds : normalisation constant for ``round_progress``
    default_pace : fallback pace for early-season NaNs

    Returns
    -------
    DataFrame with ``FEATURE_COLS`` + target columns ``home_win``, ``margin``,
    and metadata columns ``season``, ``gamecode``, ``round``.
    """
    from ..features.context import build_game_context

    all_rows: List[dict] = []

    for season in sorted(seasons_data):
        games_df, team_game_df, elo_game_log = seasons_data[season]
        cum_df = compute_cumulative_features(team_game_df)

        # context features (rest + travel) for this season
        ctx_df = build_game_context(games_df, team_game_df, season)
        ctx_idx = ctx_df.set_index("Gamecode")

        cum_idx = cum_df.set_index(["Team", "Gamecode", "IsHome"])

        for _, game in games_df.iterrows():
            gc  = int(game["Gamecode"])
            ht  = str(game["home_team"])
            at  = str(game["away_team"])
            rnd = int(game["Round"])

            try:
                h_row = cum_idx.loc[(ht, gc, 1)]
            except KeyError:
                continue
            try:
                a_row = cum_idx.loc[(at, gc, 0)]
            except KeyError:
                continue

            # Elo from the game log
            elo_rows = elo_game_log[elo_game_log["Gamecode"] == gc]
            if elo_rows.empty:
                elo_h, elo_a = 1500.0, 1500.0
            else:
                elo_h = float(elo_rows.iloc[0]["elo_home_pre"])
                elo_a = float(elo_rows.iloc[0]["elo_away_pre"])

            _wma_features: dict = {}
            _wma_valid = True
            for _factor in ["efg", "tov", "orb", "ftr"]:
                _wma_col = f"{_factor}_wma5"
                _h_val = h_row.get(_wma_col)
                _a_val = a_row.get(_wma_col)
                if (_h_val is not None and _a_val is not None
                        and np.isfinite(float(_h_val))
                        and np.isfinite(float(_a_val))):
                    _wma_features[f"net_{_factor}_wma5"] = float(_h_val) - float(_a_val)
                else:
                    _wma_valid = False
                    break
            if not _wma_valid:
                _wma_features = {
                    f"net_{f}_wma5": np.nan
                    for f in ["efg", "tov", "orb", "ftr"]
                }

            # context: rest + travel
            if gc in ctx_idx.index:
                ctx = ctx_idx.loc[gc]
                _rest_diff = float(ctx["el_rest_days_diff"])
                _h_travel = float(ctx["home_recent_travel_km"])
                _a_travel = float(ctx["away_recent_travel_km"])
            else:
                _rest_diff = 0.0
                _h_travel = 0.0
                _a_travel = 0.0

            # pace: two candidates, both strictly pre-game
            _h_cum = h_row.get("cum_pace")
            _a_cum = a_row.get("cum_pace")
            _h_cum = float(_h_cum) if _h_cum is not None and np.isfinite(float(_h_cum)) else default_pace
            _a_cum = float(_a_cum) if _a_cum is not None and np.isfinite(float(_a_cum)) else default_pace
            _expected_pace = (_h_cum + _a_cum) / 2.0

            _h_recent = h_row.get("pace_wma5")
            _a_recent = a_row.get("pace_wma5")
            _h_recent = float(_h_recent) if _h_recent is not None and np.isfinite(float(_h_recent)) else default_pace
            _a_recent = float(_a_recent) if _a_recent is not None and np.isfinite(float(_a_recent)) else default_pace
            _expected_pace_recent = (_h_recent + _a_recent) / 2.0

            feat: dict = {
                "elo_diff_scaled": (elo_h - elo_a) / 25.0,
                **_wma_features,
                "round_progress":  rnd / max_rounds,
                "el_rest_days_diff": _rest_diff,
                "home_recent_travel_km": _h_travel,
                "away_recent_travel_km": _a_travel,
                "expected_pace": _expected_pace,
                "expected_pace_recent": _expected_pace_recent,
                # Targets
                "home_win": int(float(game["home_points"]) > float(game["away_points"])),
                "margin":   float(game["margin_home"]),
                # Metadata (not used as features)
                "season":   season,
                "gamecode":  gc,
                "round":    rnd,
            }
            all_rows.append(feat)

    result = pd.DataFrame(all_rows)
    _wma_subset = ["net_efg_wma5", "net_tov_wma5", "net_orb_wma5", "net_ftr_wma5"]
    result = result.dropna(subset=_wma_subset)
    return result


# ---------------------------------------------------------------------------
# Build prediction features (upcoming games)
# ---------------------------------------------------------------------------

def build_prediction_features(
    schedule_df: pd.DataFrame,
    team_ratings_df: pd.DataFrame,
    current_elos: Dict[str, float],
    team_game_df: pd.DataFrame,
    round_number: int,
    elo_base: float = 1500.0,
    max_rounds: int = 34,
    season: int = 2025,
    default_pace: float = 70.0,
) -> pd.DataFrame:
    """Build the feature matrix for upcoming (unplayed) games.

    Uses current Elo values, ``round_progress``, per-team recent-form WMA
    Four Factors, EuroLeague rest/travel context, and expected pace.
    """
    from ..features.context import build_prediction_context

    _ = team_ratings_df  # kept for API compatibility with existing pipeline call sites

    _ff_pred_cols = [
        "FGM", "FGA", "FGM3", "ORB", "DRB", "TOV", "FTA",
        "opp_DRB", "opp_ORB", "opp_FGM", "opp_FGA", "opp_FGM3", "opp_TOV", "opp_FTA",
    ]
    _ff_avail = (
        team_game_df is not None
        and not team_game_df.empty
        and all(c in team_game_df.columns for c in _ff_pred_cols)
    )

    team_stats: Dict[str, Dict[str, float]] = {}
    if team_game_df is not None and not team_game_df.empty and _ff_avail:
        for team, grp in team_game_df.groupby("Team"):
            grp_sorted = grp.sort_values(["Round", "Gamecode"])
            stats: Dict[str, float] = {}

            _g_fga = grp_sorted["FGA"].values.astype(float)
            _g_fgm = grp_sorted["FGM"].values.astype(float)
            _g_fgm3 = grp_sorted["FGM3"].values.astype(float)
            _g_fta = grp_sorted["FTA"].values.astype(float)
            _g_orb = grp_sorted["ORB"].values.astype(float)
            _g_tov = grp_sorted["TOV"].values.astype(float)
            _g_opp_drb = grp_sorted["opp_DRB"].values.astype(float)

            _game_efg = np.where(
                _g_fga > 0,
                (_g_fgm + 0.5 * _g_fgm3) / _g_fga,
                np.nan,
            )
            _td = _g_fga + 0.44 * _g_fta + _g_tov
            _game_tov = np.where(_td > 0, _g_tov / _td, np.nan)
            _od = _g_orb + _g_opp_drb
            _game_orb = np.where(_od > 0, _g_orb / _od, np.nan)
            _game_ftr = np.where(_g_fga > 0, _g_fta / _g_fga, np.nan)

            for _arr, _key in [
                (_game_efg, "efg_wma5"),
                (_game_tov, "tov_wma5"),
                (_game_orb, "orb_wma5"),
                (_game_ftr, "ftr_wma5"),
            ]:
                _tail = _arr[np.isfinite(_arr)][-5:]
                if len(_tail) > 0:
                    stats[_key] = _wma5_rolling(_tail)

            # pace for prediction: both season-to-date and recent WMA5
            _poss = grp_sorted["Possessions"].values.astype(float)
            _finite_poss = _poss[np.isfinite(_poss)]
            if len(_finite_poss) > 0:
                stats["cum_pace"] = float(np.mean(_finite_poss))
                _tail_poss = _finite_poss[-5:]
                stats["pace_wma5"] = _wma5_rolling(_tail_poss)
            team_stats[str(team)] = stats

    # context features (rest + travel) for upcoming games
    ctx_df = build_prediction_context(schedule_df, team_game_df, season)
    ctx_idx = ctx_df.set_index("Gamecode")

    rows: List[dict] = []
    for _, game in schedule_df.iterrows():
        ht = str(game["home_team"])
        at = str(game["away_team"])
        gc = int(game["Gamecode"])

        elo_h = float(current_elos.get(ht, elo_base))
        elo_a = float(current_elos.get(at, elo_base))

        h_s = team_stats.get(ht, {})
        a_s = team_stats.get(at, {})

        _wma_features: dict = {}
        for _factor in ["efg", "tov", "orb", "ftr"]:
            _h_val = h_s.get(f"{_factor}_wma5")
            _a_val = a_s.get(f"{_factor}_wma5")
            if _h_val is not None and _a_val is not None:
                _wma_features[f"net_{_factor}_wma5"] = _h_val - _a_val
            else:
                _wma_features[f"net_{_factor}_wma5"] = np.nan

        # context
        if gc in ctx_idx.index:
            ctx = ctx_idx.loc[gc]
            _rest_diff = float(ctx["el_rest_days_diff"])
            _h_travel = float(ctx["home_recent_travel_km"])
            _a_travel = float(ctx["away_recent_travel_km"])
        else:
            _rest_diff = 0.0
            _h_travel = 0.0
            _a_travel = 0.0

        # pace (two candidates; FEATURE_COLS uses expected_pace by default)
        _h_cum = h_s.get("cum_pace", default_pace)
        _a_cum = a_s.get("cum_pace", default_pace)
        _expected_pace = (_h_cum + _a_cum) / 2.0

        _h_recent = h_s.get("pace_wma5", default_pace)
        _a_recent = a_s.get("pace_wma5", default_pace)
        _expected_pace_recent = (_h_recent + _a_recent) / 2.0

        feat: dict = {
            "elo_diff_scaled": (elo_h - elo_a) / 25.0,
            **_wma_features,
            "round_progress":  round_number / max_rounds,
            "el_rest_days_diff": _rest_diff,
            "home_recent_travel_km": _h_travel,
            "away_recent_travel_km": _a_travel,
            "expected_pace": _expected_pace,
            "expected_pace_recent": _expected_pace_recent,
        }
        rows.append(feat)

    return pd.DataFrame(rows)
