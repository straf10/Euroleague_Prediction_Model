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
    "elo_diff_scaled",    # (elo_home − elo_away) / 25
    "home_off_efg_matchup",  # home off eFG% − away def eFG%
    "home_off_tov_matchup",  # home off TOV% − away def TOV%
    "home_off_orb_matchup",  # home off ORB% − away def DRB%
    "home_off_ftr_matchup",  # home off FT rate − away def FT rate
    "away_off_efg_matchup",  # away off eFG% − home def eFG%
    "away_off_tov_matchup",  # away off TOV% − home def TOV%
    "away_off_orb_matchup",  # away off ORB% − home def DRB%
    "away_off_ftr_matchup",  # away off FT rate − home def FT rate
    "round_progress",     # round / max_rounds
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val, default: float = 0.0) -> float:
    """Return *val* as a float if finite, else *default*."""
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return default
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


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

    # ---------- Four Factors cumulative ----------
    _ff_cols = [
        "FGM", "FGA", "FGM3", "ORB", "DRB", "TOV", "FTA",
        "opp_DRB", "opp_ORB", "opp_FGM", "opp_FGA", "opp_FGM3", "opp_TOV", "opp_FTA",
    ]
    _has_ff = all(c in df.columns for c in _ff_cols)
    if _has_ff:
        for c in _ff_cols:
            df[f"cum_{c}"] = g[c].transform(
                lambda x: x.shift(1).expanding().sum()
            )
        _off_fga = df["cum_FGA"].fillna(0)
        _off_fgm = df["cum_FGM"].fillna(0)
        _off_fgm3 = df["cum_FGM3"].fillna(0)
        _off_fta = df["cum_FTA"].fillna(0)
        _off_tov = df["cum_TOV"].fillna(0)
        _off_orb = df["cum_ORB"].fillna(0)
        _opp_drb = df["cum_opp_DRB"].fillna(0)

        _def_fga = df["cum_opp_FGA"].fillna(0)
        _def_fgm = df["cum_opp_FGM"].fillna(0)
        _def_fgm3 = df["cum_opp_FGM3"].fillna(0)
        _def_fta = df["cum_opp_FTA"].fillna(0)
        _def_tov = df["cum_opp_TOV"].fillna(0)
        _def_drb = df["cum_DRB"].fillna(0)
        _opp_orb = df["cum_opp_ORB"].fillna(0)

        # Rates are derived from cumulative raw counts (not mean of game-level percentages).
        df["cum_off_efg"] = np.where(
            _off_fga > 0,
            (_off_fgm + 0.5 * _off_fgm3) / _off_fga,
            np.nan,
        )
        _off_tov_denom = _off_fga + 0.44 * _off_fta + _off_tov
        df["cum_off_tov_pct"] = np.where(
            _off_tov_denom > 0,
            _off_tov / _off_tov_denom,
            np.nan,
        )
        _off_orb_denom = _off_orb + _opp_drb
        df["cum_off_orb_pct"] = np.where(
            _off_orb_denom > 0,
            _off_orb / _off_orb_denom,
            np.nan,
        )
        df["cum_off_ft_rate"] = np.where(
            _off_fga > 0,
            _off_fta / _off_fga,
            np.nan,
        )

        df["cum_def_efg"] = np.where(
            _def_fga > 0,
            (_def_fgm + 0.5 * _def_fgm3) / _def_fga,
            np.nan,
        )
        _def_tov_denom = _def_fga + 0.44 * _def_fta + _def_tov
        df["cum_def_tov_pct"] = np.where(
            _def_tov_denom > 0,
            _def_tov / _def_tov_denom,
            np.nan,
        )
        _def_drb_denom = _def_drb + _opp_orb
        df["cum_def_drb_pct"] = np.where(
            _def_drb_denom > 0,
            _def_drb / _def_drb_denom,
            np.nan,
        )
        df["cum_def_ft_rate"] = np.where(
            _def_fga > 0,
            _def_fta / _def_fga,
            np.nan,
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
) -> pd.DataFrame:
    """Create a labelled feature matrix from historical game data.

    Parameters
    ----------
    seasons_data : {season: (games_df, team_game_df, elo_game_log_df)}
        ``games_df``           – one row per game (from ``build_games_with_possessions``)
        ``team_game_df``       – one row per team-game (from ``build_team_game_net_ratings``)
        ``elo_game_log_df``    – per-game Elo log (``EloResult.game_elos``)
    max_rounds : normalisation constant for ``round_progress``

    Returns
    -------
    DataFrame with ``FEATURE_COLS`` + target columns ``home_win``, ``margin``,
    and metadata columns ``season``, ``gamecode``, ``round``.
    """
    all_rows: List[dict] = []

    for season in sorted(seasons_data):
        games_df, team_game_df, elo_game_log = seasons_data[season]
        cum_df = compute_cumulative_features(team_game_df)

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

            home_off_efg = _safe(h_row.get("cum_off_efg"), 0.50)
            home_off_tov = _safe(h_row.get("cum_off_tov_pct"), 0.15)
            home_off_orb = _safe(h_row.get("cum_off_orb_pct"), 0.25)
            home_off_ftr = _safe(h_row.get("cum_off_ft_rate"), 0.30)
            home_def_efg = _safe(h_row.get("cum_def_efg"), 0.50)
            home_def_tov = _safe(h_row.get("cum_def_tov_pct"), 0.15)
            home_def_drb = _safe(h_row.get("cum_def_drb_pct"), 0.75)
            home_def_ftr = _safe(h_row.get("cum_def_ft_rate"), 0.30)

            away_off_efg = _safe(a_row.get("cum_off_efg"), 0.50)
            away_off_tov = _safe(a_row.get("cum_off_tov_pct"), 0.15)
            away_off_orb = _safe(a_row.get("cum_off_orb_pct"), 0.25)
            away_off_ftr = _safe(a_row.get("cum_off_ft_rate"), 0.30)
            away_def_efg = _safe(a_row.get("cum_def_efg"), 0.50)
            away_def_tov = _safe(a_row.get("cum_def_tov_pct"), 0.15)
            away_def_drb = _safe(a_row.get("cum_def_drb_pct"), 0.75)
            away_def_ftr = _safe(a_row.get("cum_def_ft_rate"), 0.30)

            feat: dict = {
                "elo_diff_scaled": (elo_h - elo_a) / 25.0,
                "home_off_efg_matchup": home_off_efg - away_def_efg,
                "home_off_tov_matchup": home_off_tov - away_def_tov,
                "home_off_orb_matchup": home_off_orb - away_def_drb,
                "home_off_ftr_matchup": home_off_ftr - away_def_ftr,
                "away_off_efg_matchup": away_off_efg - home_def_efg,
                "away_off_tov_matchup": away_off_tov - home_def_tov,
                "away_off_orb_matchup": away_off_orb - home_def_drb,
                "away_off_ftr_matchup": away_off_ftr - home_def_ftr,
                "round_progress":  rnd / max_rounds,
                # Targets
                "home_win": int(float(game["home_points"]) > float(game["away_points"])),
                "margin":   float(game["margin_home"]),
                # Metadata (not used as features)
                "season":   season,
                "gamecode":  gc,
                "round":    rnd,
            }
            all_rows.append(feat)

    return pd.DataFrame(all_rows)


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
) -> pd.DataFrame:
    """Build the feature matrix for upcoming (unplayed) games.

    Uses current Elo values and full-season Four Factors from ``team_game_df``.
    """
    _ = team_ratings_df  # kept for API compatibility with existing pipeline call sites

    # Per-team aggregate stats from team_game_df
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
    if team_game_df is not None and not team_game_df.empty:
        for team, grp in team_game_df.groupby("Team"):
            grp_sorted = grp.sort_values(["Round", "Gamecode"])
            stats: Dict[str, float] = {}
            # Four Factors from full-season cumulative raw totals
            if _ff_avail:
                t_fga = float(grp_sorted["FGA"].sum())
                t_fgm = float(grp_sorted["FGM"].sum())
                t_fgm3 = float(grp_sorted["FGM3"].sum())
                t_fta = float(grp_sorted["FTA"].sum())
                t_orb = float(grp_sorted["ORB"].sum())
                t_drb = float(grp_sorted["DRB"].sum())
                t_tov = float(grp_sorted["TOV"].sum())
                t_opp_drb = float(grp_sorted["opp_DRB"].sum())
                t_opp_orb = float(grp_sorted["opp_ORB"].sum())
                t_opp_fga = float(grp_sorted["opp_FGA"].sum())
                t_opp_fgm = float(grp_sorted["opp_FGM"].sum())
                t_opp_fgm3 = float(grp_sorted["opp_FGM3"].sum())
                t_opp_tov = float(grp_sorted["opp_TOV"].sum())
                t_opp_fta = float(grp_sorted["opp_FTA"].sum())

                stats["off_efg"] = (
                    (t_fgm + 0.5 * t_fgm3) / t_fga if t_fga > 0 else 0.50
                )
                off_tov_denom = t_fga + 0.44 * t_fta + t_tov
                stats["off_tov_pct"] = t_tov / off_tov_denom if off_tov_denom > 0 else 0.15
                stats["off_orb_pct"] = (
                    t_orb / (t_orb + t_opp_drb)
                    if (t_orb + t_opp_drb) > 0 else 0.25
                )
                stats["off_ft_rate"] = t_fta / t_fga if t_fga > 0 else 0.30

                stats["def_efg"] = (
                    (t_opp_fgm + 0.5 * t_opp_fgm3) / t_opp_fga
                    if t_opp_fga > 0 else 0.50
                )
                def_tov_denom = t_opp_fga + 0.44 * t_opp_fta + t_opp_tov
                stats["def_tov_pct"] = t_opp_tov / def_tov_denom if def_tov_denom > 0 else 0.15
                stats["def_drb_pct"] = (
                    t_drb / (t_drb + t_opp_orb)
                    if (t_drb + t_opp_orb) > 0 else 0.75
                )
                stats["def_ft_rate"] = t_opp_fta / t_opp_fga if t_opp_fga > 0 else 0.30
            else:
                stats.update({
                    "off_efg": 0.50, "off_tov_pct": 0.15, "off_orb_pct": 0.25, "off_ft_rate": 0.30,
                    "def_efg": 0.50, "def_tov_pct": 0.15, "def_drb_pct": 0.75, "def_ft_rate": 0.30,
                })
            team_stats[str(team)] = stats

    _default_stats: Dict[str, float] = {
        "off_efg": 0.50, "off_tov_pct": 0.15, "off_orb_pct": 0.25, "off_ft_rate": 0.30,
        "def_efg": 0.50, "def_tov_pct": 0.15, "def_drb_pct": 0.75, "def_ft_rate": 0.30,
    }

    rows: List[dict] = []
    for _, game in schedule_df.iterrows():
        ht = str(game["home_team"])
        at = str(game["away_team"])

        elo_h = float(current_elos.get(ht, elo_base))
        elo_a = float(current_elos.get(at, elo_base))

        h_s = team_stats.get(ht, _default_stats)
        a_s = team_stats.get(at, _default_stats)

        feat: dict = {
            "elo_diff_scaled": (elo_h - elo_a) / 25.0,
            "home_off_efg_matchup": h_s["off_efg"] - a_s["def_efg"],
            "home_off_tov_matchup": h_s["off_tov_pct"] - a_s["def_tov_pct"],
            "home_off_orb_matchup": h_s["off_orb_pct"] - a_s["def_drb_pct"],
            "home_off_ftr_matchup": h_s["off_ft_rate"] - a_s["def_ft_rate"],
            "away_off_efg_matchup": a_s["off_efg"] - h_s["def_efg"],
            "away_off_tov_matchup": a_s["off_tov_pct"] - h_s["def_tov_pct"],
            "away_off_orb_matchup": a_s["off_orb_pct"] - h_s["def_drb_pct"],
            "away_off_ftr_matchup": a_s["off_ft_rate"] - h_s["def_ft_rate"],
            "round_progress":  round_number / max_rounds,
        }
        rows.append(feat)

    return pd.DataFrame(rows)
