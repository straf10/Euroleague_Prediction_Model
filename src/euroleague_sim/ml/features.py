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
    # Differential features
    "net_rtg_diff",       # home net rating (split) − away net rating (split)
    "elo_diff_scaled",    # (elo_home − elo_away) / 25
    "off_matchup",        # home_off − away_def  (home scoring advantage)
    "def_matchup",        # away_off − home_def  (away scoring advantage)
    "win_pct_diff",       # home win% − away win%
    "form_diff",          # home last-5 NetPer100 − away last-5 NetPer100
    "pace_diff",          # home possessions/game − away possessions/game
    # Absolute – home team
    "home_off_rtg",
    "home_def_rtg",
    "home_win_pct",
    "elo_home",
    # Absolute – away team
    "away_off_rtg",
    "away_def_rtg",
    "away_win_pct",
    "elo_away",
    # Context
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
    _ff_cols = ["FGA", "FGM2", "FGM3", "FTA", "ORB", "TOV", "opp_DRB"]
    _has_ff = all(c in df.columns for c in _ff_cols)
    if _has_ff:
        for c in _ff_cols:
            df[f"cum_{c}"] = g[c].transform(
                lambda x: x.shift(1).expanding().sum()
            )
        _fga = df["cum_FGA"]
        df["cum_efg"] = np.where(
            _fga > 0,
            (df["cum_FGM2"].fillna(0) + 1.5 * df["cum_FGM3"].fillna(0)) / _fga,
            np.nan,
        )
        df["cum_tov_pct"] = np.where(
            df["cum_poss"] > 0,
            df["cum_TOV"].fillna(0) / df["cum_poss"],
            np.nan,
        )
        _orb_total = df["cum_ORB"].fillna(0) + df["cum_opp_DRB"].fillna(0)
        df["cum_orb_pct"] = np.where(
            _orb_total > 0,
            df["cum_ORB"].fillna(0) / _orb_total,
            np.nan,
        )
        df["cum_ft_rate"] = np.where(
            _fga > 0,
            df["cum_FTA"].fillna(0) / _fga,
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

            # Ratings – prefer split, fall back to overall
            home_off = _safe(h_row.get("hs_off_rtg"),
                             _safe(h_row.get("cum_off_rtg"), 100.0))
            home_def = _safe(h_row.get("hs_def_rtg"),
                             _safe(h_row.get("cum_def_rtg"), 100.0))
            away_off = _safe(a_row.get("as_off_rtg"),
                             _safe(a_row.get("cum_off_rtg"), 100.0))
            away_def = _safe(a_row.get("as_def_rtg"),
                             _safe(a_row.get("cum_def_rtg"), 100.0))

            home_net = home_off - home_def
            away_net = away_off - away_def

            home_wpct = _safe(h_row.get("cum_win_pct"), 0.5)
            away_wpct = _safe(a_row.get("cum_win_pct"), 0.5)
            home_form = _safe(h_row.get("form"), 0.0)
            away_form = _safe(a_row.get("form"), 0.0)
            home_pace = _safe(h_row.get("cum_pace"), 72.0)
            away_pace = _safe(a_row.get("cum_pace"), 72.0)

            feat: dict = {
                # Differential
                "net_rtg_diff":    home_net - away_net,
                "elo_diff_scaled": (elo_h - elo_a) / 25.0,
                "off_matchup":     home_off - away_def,
                "def_matchup":     away_off - home_def,
                "win_pct_diff":    home_wpct - away_wpct,
                "form_diff":       home_form - away_form,
                "pace_diff":       home_pace - away_pace,
                # Absolute – home
                "home_off_rtg":    home_off,
                "home_def_rtg":    home_def,
                "home_win_pct":    home_wpct,
                "elo_home":        elo_h,
                # Absolute – away
                "away_off_rtg":    away_off,
                "away_def_rtg":    away_def,
                "away_win_pct":    away_wpct,
                "elo_away":        elo_a,
                # Context
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

    Uses the full-season cumulative ``team_ratings_df`` (with home/away splits)
    and current Elo values – same data available at prediction time.
    """
    tr = (
        team_ratings_df.set_index("Team")
        if "Team" in team_ratings_df.columns
        else team_ratings_df
    )

    # Per-team aggregate stats from team_game_df
    _ff_pred_cols = ["FGA", "FGM2", "FGM3", "FTA", "ORB", "TOV", "opp_DRB"]
    _ff_avail = (
        team_game_df is not None
        and not team_game_df.empty
        and all(c in team_game_df.columns for c in _ff_pred_cols)
    )

    team_stats: Dict[str, Dict[str, float]] = {}
    if team_game_df is not None and not team_game_df.empty:
        for team, grp in team_game_df.groupby("Team"):
            grp_sorted = grp.sort_values(["Round", "Gamecode"])
            n = len(grp_sorted)
            wins = (grp_sorted["PointsFor"] > grp_sorted["PointsAgainst"]).sum()
            wpct = float(wins / n) if n > 0 else 0.5
            form = float(grp_sorted["NetPer100"].iloc[-5:].mean()) if n >= 1 else 0.0
            pace = float(grp_sorted["Possessions"].mean()) if n >= 1 else 72.0
            stats: Dict[str, float] = {
                "win_pct": wpct, "form": form, "pace": pace,
            }
            # Four Factors from full-season cumulative totals
            if _ff_avail:
                t_fga = float(grp_sorted["FGA"].sum())
                t_fgm2 = float(grp_sorted["FGM2"].sum())
                t_fgm3 = float(grp_sorted["FGM3"].sum())
                t_fta = float(grp_sorted["FTA"].sum())
                t_orb = float(grp_sorted["ORB"].sum())
                t_tov = float(grp_sorted["TOV"].sum())
                t_opp_drb = float(grp_sorted["opp_DRB"].sum())
                t_poss = float(grp_sorted["Possessions"].sum())

                stats["efg"] = (
                    (t_fgm2 + 1.5 * t_fgm3) / t_fga if t_fga > 0 else 0.50
                )
                stats["tov_pct"] = t_tov / t_poss if t_poss > 0 else 0.15
                stats["orb_pct"] = (
                    t_orb / (t_orb + t_opp_drb)
                    if (t_orb + t_opp_drb) > 0 else 0.25
                )
                stats["ft_rate"] = t_fta / t_fga if t_fga > 0 else 0.30
            else:
                stats.update({"efg": 0.50, "tov_pct": 0.15,
                              "orb_pct": 0.25, "ft_rate": 0.30})
            team_stats[str(team)] = stats

    _default_stats: Dict[str, float] = {
        "win_pct": 0.5, "form": 0.0, "pace": 72.0,
        "efg": 0.50, "tov_pct": 0.15, "orb_pct": 0.25, "ft_rate": 0.30,
    }

    rows: List[dict] = []
    for _, game in schedule_df.iterrows():
        ht = str(game["home_team"])
        at = str(game["away_team"])

        # Ratings from team_ratings (has home/away splits from aggregate_team_ratings)
        def _get(team: str, col: str, default: float = 100.0) -> float:
            if team not in tr.index or col not in tr.columns:
                return default
            v = tr.at[team, col]
            return _safe(v, default)

        home_off = _get(ht, "Home_OffPer100")
        home_def = _get(ht, "Home_DefPer100")
        away_off = _get(at, "Away_OffPer100")
        away_def = _get(at, "Away_DefPer100")

        elo_h = float(current_elos.get(ht, elo_base))
        elo_a = float(current_elos.get(at, elo_base))

        h_s = team_stats.get(ht, _default_stats)
        a_s = team_stats.get(at, _default_stats)

        home_net = home_off - home_def
        away_net = away_off - away_def

        feat: dict = {
            "net_rtg_diff":    home_net - away_net,
            "elo_diff_scaled": (elo_h - elo_a) / 25.0,
            "off_matchup":     home_off - away_def,
            "def_matchup":     away_off - home_def,
            "win_pct_diff":    h_s["win_pct"] - a_s["win_pct"],
            "form_diff":       h_s["form"] - a_s["form"],
            "pace_diff":       h_s["pace"] - a_s["pace"],
            "home_off_rtg":    home_off,
            "home_def_rtg":    home_def,
            "home_win_pct":    h_s["win_pct"],
            "elo_home":        elo_h,
            "away_off_rtg":    away_off,
            "away_def_rtg":    away_def,
            "away_win_pct":    a_s["win_pct"],
            "elo_away":        elo_a,
            "round_progress":  round_number / max_rounds,
        }
        rows.append(feat)

    return pd.DataFrame(rows)
