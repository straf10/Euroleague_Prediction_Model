"""Daily Predictions — per-game cards, feature differentials, win-probability bars.

Renders ``data_cache/latest_predictions.csv`` (refreshed nightly by the
GitHub Actions workflow). All team labels go through ``team_label`` so the
3-letter API codes never leak into the UI.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from euroleague_sim.data.team_registry import load_teams_registry, team_label


CACHE_DIR = Path(__file__).parent.parent / "data_cache"
PREDICTIONS_CSV = CACHE_DIR / "latest_predictions.csv"


st.set_page_config(
    page_title="Daily Predictions · Euroleague",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
        h1, h2, h3 { letter-spacing: -0.01em; }
        .game-card {
            background: linear-gradient(135deg, #15171c 0%, #1c1f27 100%);
            border: 1px solid #2a2e39;
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
        }
        .game-card .teams {
            font-size: 1.15rem; font-weight: 600; color: #f1f3f5;
            display: flex; justify-content: space-between; align-items: baseline;
        }
        .game-card .date { color: #868e96; font-size: 0.82rem; }
        .game-card .pred { margin-top: 0.4rem; color: #adb5bd; font-size: 0.9rem; }
        .winner-pill {
            background: #1f3a2b; color: #51cf66;
            border-radius: 999px; padding: 2px 10px; font-size: 0.78rem;
            font-weight: 600; margin-left: 0.4rem;
        }
        .prob-bar-wrap {
            background: #2a2e39; height: 10px; border-radius: 6px;
            overflow: hidden; margin-top: 0.7rem;
        }
        .prob-bar-fill {
            height: 100%; background: linear-gradient(90deg, #339af0, #51cf66);
        }
        .prob-row {
            display: flex; justify-content: space-between;
            font-size: 0.82rem; color: #adb5bd; margin-top: 4px;
        }
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


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except FileNotFoundError:
        return None


st.title("Daily Predictions")
st.caption("Round-by-round model output with win probability, expected margin, and feature deltas.")

if not PREDICTIONS_CSV.exists():
    st.error(
        f"No predictions found at `{PREDICTIONS_CSV.relative_to(Path(__file__).parent.parent)}`. "
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


def label(code: str) -> str:
    return team_label(code, registry, "short")


# Off-season detection: the pipeline sets ``is_offseason``; older snapshots
# may not have the column, so we fall back to "all game dates in the past".
flagged_offseason = bool(df["is_offseason"].iloc[0]) if "is_offseason" in df.columns else False
dates_all_past = (
    df["game_date"].notna().all()
    and (df["game_date"].dt.tz_localize(None) < pd.Timestamp.now()).all()
) if "game_date" in df.columns else False
is_offseason = flagged_offseason or dates_all_past

if is_offseason:
    st.info(
        f"🏖️ **Off-season mode** — showing the last played round (Round {round_n}) "
        f"of season {season}-{(season + 1) % 100:02d}. New predictions resume "
        f"once the next season schedule goes live."
    )

# ── Top metric strip ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Season", f"{season}-{(season + 1) % 100:02d}")
c2.metric("Round", round_n)
c3.metric("Games", len(df))
c4.metric("Model", model_name)

mtime = _mtime(PREDICTIONS_CSV)
if mtime:
    st.caption(f"Last updated: **{mtime:%Y-%m-%d %H:%M}** local")

st.divider()

# ── Per-game cards ──────────────────────────────────────────────────────────
st.subheader(f"Round {round_n} — game-by-game")

for _, r in df.iterrows():
    home_lbl = label(r["home_team"])
    away_lbl = label(r["away_team"])
    p_home = float(r["p_home_win"])
    winner_lbl = label(r["predicted_winner"])
    conf = max(p_home, 1 - p_home) * 100
    margin = float(r["expected_margin"]) if pd.notna(r["expected_margin"]) else 0.0
    q10 = float(r["margin_q10"]) if pd.notna(r.get("margin_q10")) else float("nan")
    q90 = float(r["margin_q90"]) if pd.notna(r.get("margin_q90")) else float("nan")
    date_str = (
        pd.to_datetime(r["game_date"]).strftime("%a %d %b %Y, %H:%M")
        if pd.notna(r["game_date"]) else "Date TBD"
    )

    st.markdown(
        f"""
        <div class="game-card">
          <div class="teams">
            <span>{home_lbl}  <span style="color:#6c757d">vs</span>  {away_lbl}</span>
            <span class="date">{date_str}</span>
          </div>
          <div class="pred">
            Pick: <b>{winner_lbl}</b>
            <span class="winner-pill">{conf:.0f}% confidence</span>
            &nbsp;·&nbsp; Expected margin (home): <b>{margin:+.1f}</b>
            &nbsp;·&nbsp; MC band: [{q10:+.1f}, {q90:+.1f}]
          </div>
          <div class="prob-bar-wrap">
            <div class="prob-bar-fill" style="width:{p_home * 100:.1f}%"></div>
          </div>
          <div class="prob-row">
            <span>{home_lbl} · {p_home:.1%}</span>
            <span>{away_lbl} · {1 - p_home:.1%}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ── Feature diff table ──────────────────────────────────────────────────────
st.subheader("Why the model picked it")
st.caption(
    "The three differentials the model leans on most — "
    "positive favours home, negative favours away."
)
diff_view = df[[
    "home_team", "away_team", "p_home_win", "expected_margin",
    "elo_diff", "bpm_diff", "fatigue_diff",
]].copy()
diff_view["home_team"] = diff_view["home_team"].map(label)
diff_view["away_team"] = diff_view["away_team"].map(label)
diff_view = diff_view.rename(columns={
    "home_team":       "Home",
    "away_team":       "Away",
    "p_home_win":      "P(Home win)",
    "expected_margin": "E[Margin]",
    "elo_diff":        "Elo diff",
    "bpm_diff":        "BPM diff",
    "fatigue_diff":    "Fatigue diff",
})
st.dataframe(
    diff_view.style.format({
        "P(Home win)":  "{:.1%}",
        "E[Margin]":    "{:+.2f}",
        "Elo diff":     "{:+.1f}",
        "BPM diff":     "{:+.2f}",
        "Fatigue diff": "{:+.1f}",
    }).background_gradient(subset=["Elo diff", "BPM diff", "Fatigue diff"], cmap="RdYlGn"),
    hide_index=True,
    use_container_width=True,
)

st.markdown(
    f"<div class='footer-meta'>Snapshot: <code>{PREDICTIONS_CSV.name}</code> · "
    f"Refreshed nightly via GitHub Actions.</div>",
    unsafe_allow_html=True,
)
