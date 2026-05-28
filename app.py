"""Euroleague Prediction Model — Streamlit landing page.

This is the multi-page entry point. The detailed per-round view lives at
``pages/1_Daily_Predictions.py`` so each page has its own script lifecycle
(independent caches, no cross-page recomputation).

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from euroleague_sim.data.team_registry import (
    load_season_meta,
    load_teams_registry,
    team_label,
)


CACHE_DIR = Path(__file__).parent / "data_cache"
PREDICTIONS_CSV = CACHE_DIR / "latest_predictions.csv"


st.set_page_config(
    page_title="Euroleague Prediction Model",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
        h1, h2, h3 { letter-spacing: -0.01em; }
        .footer-meta { color: #6c757d; font-size: 0.78rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, show_spinner=False)
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["game_date"])
    df["p_home_win"] = df["p_home_win"].astype(float).clip(0, 1)
    return df.sort_values(["game_date", "gamecode"], na_position="last").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_registry(season: int):
    return load_teams_registry(CACHE_DIR, season)


@st.cache_resource(show_spinner=False)
def load_meta(season: int):
    return load_season_meta(CACHE_DIR, season)


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except FileNotFoundError:
        return None


def _elo_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    home = df[["home_team", "elo_home"]].rename(columns={"home_team": "team", "elo_home": "elo"})
    away = df[["away_team", "elo_away"]].rename(columns={"away_team": "team", "elo_away": "elo"})
    elos = pd.concat([home, away], ignore_index=True).dropna()
    return elos.groupby("team", as_index=False)["elo"].max().sort_values("elo", ascending=False).reset_index(drop=True)


st.title("🏀 Euroleague Prediction Model")
st.caption(
    "CatBoost + Beta-calibrated probabilities · expanding-window walk-forward · "
    "Monte-Carlo margin simulation"
)

if not PREDICTIONS_CSV.exists():
    st.error(
        f"No predictions found at `{PREDICTIONS_CSV.relative_to(Path(__file__).parent)}`. "
        "Run the predict pipeline locally or wait for the next nightly GitHub Actions update."
    )
    st.stop()

df = load_predictions(PREDICTIONS_CSV)
if df.empty:
    st.warning("Predictions file is empty.")
    st.stop()

season = int(df["season"].iloc[0])
round_n = int(df["round"].iloc[0])
model_name = str(df["model"].iloc[0])
registry = load_registry(season)
meta = load_meta(season)

# ── Top metric strip ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Season", f"{season}-{(season + 1) % 100:02d}")
c2.metric("Current Round", round_n)
c3.metric("Teams Tracked", meta.get("n_teams") if meta else "—")
c4.metric("Total Rounds", meta.get("n_rounds") if meta else "—")

mtime = _mtime(PREDICTIONS_CSV)
if mtime:
    st.caption(f"Last updated: **{mtime:%Y-%m-%d %H:%M}** local · model: `{model_name}`")

st.info(
    "👉 Open **Daily Predictions** from the sidebar for the game-by-game view, "
    "feature differentials, and the round's win-probability bars."
)

st.divider()

# ── Elo leaderboard with full team names ────────────────────────────────────
st.subheader("Current Elo standings")
elos = _elo_leaderboard(df)
if not elos.empty:
    elos["Team"] = elos["team"].map(lambda c: team_label(c, registry, "short"))
    elos["Rank"] = range(1, len(elos) + 1)
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(
            elos[["Rank", "Team", "elo"]].rename(columns={"elo": "Elo"})
                .style.format({"Elo": "{:.0f}"}),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        chart_df = elos.set_index("Team")["elo"]
        st.bar_chart(chart_df, height=400)
else:
    st.write("No Elo data available in the current snapshot.")

# ── Season phases panel ─────────────────────────────────────────────────────
if meta and meta.get("phases"):
    st.divider()
    st.subheader("Season structure")
    phase_labels = {
        "regular_season": "Regular Season",
        "playin": "Play-In",
        "playoff": "Playoffs",
        "final_four": "Final Four",
    }
    phase_rows = []
    for key, (lo, hi) in meta["phases"].items():
        phase_rows.append({
            "Phase": phase_labels.get(key, key),
            "Rounds": f"{lo}" if lo == hi else f"{lo}–{hi}",
        })
    st.dataframe(pd.DataFrame(phase_rows), hide_index=True, use_container_width=False)
    st.caption(
        f"Derived from `season_meta_E{season}.json`. "
        f"Round-progress feature normalised by **{meta.get('regular_season_end_round', 34)}** "
        f"(end of regular season)."
    )

st.markdown(
    f"<div class='footer-meta'>Snapshot: <code>{PREDICTIONS_CSV.name}</code> · "
    f"Refreshed nightly via GitHub Actions.</div>",
    unsafe_allow_html=True,
)
