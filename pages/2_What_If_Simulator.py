"""What-If Injury Simulator — toggle players, re-run the model live.

Reads three artefacts produced by the nightly Action / local pipeline:

* ``data_cache/latest_predictions.csv``  → matchup list + baseline display
* ``data_cache/latest_features.csv``     → full 9-column feature row per game
* ``data_cache/rosters_E{season}.json``  → per-team roster (minutes, BPM, anchor)

For the selected matchup we keep every feature constant and overwrite only
``net_bpm_diff`` with the synthetic value derived from the active roster.
``ModelPredictor.predict`` is a pure 1-row forward pass — sub-millisecond,
no caching needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from euroleague_sim.data.rosters import (
    active_team_bpm,
    default_active_ids,
    load_rosters,
)
from euroleague_sim.data.team_registry import load_teams_registry, team_label
from euroleague_sim.ml.features import FEATURE_COLS
from euroleague_sim.ml.predict import ModelPredictor, load_predictor


ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data_cache"
MODEL_DIR = ROOT / "models"
PREDICTIONS_CSV = CACHE_DIR / "latest_predictions.csv"
FEATURES_CSV = CACHE_DIR / "latest_features.csv"


st.set_page_config(
    page_title="What-If Simulator · Euroleague",
    page_icon="🩹",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
        h1, h2, h3 { letter-spacing: -0.01em; }
        .roster-hdr { font-size: 1.05rem; font-weight: 600; color: #f1f3f5; margin-bottom: 0.4rem; }
        .out-pill {
            background: #3a1f1f; color: #ff8787;
            border-radius: 999px; padding: 1px 8px; font-size: 0.7rem;
            margin-left: 0.4rem;
        }
        .footer-meta { color: #6c757d; font-size: 0.78rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model() -> Optional[ModelPredictor]:
    return load_predictor(MODEL_DIR, model_name="baseline")


@st.cache_data(ttl=3600, show_spinner=False)
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["game_date"])
    return df.sort_values(["game_date", "gamecode"], na_position="last").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_features(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(ttl=3600, show_spinner=False)
def load_rosters_cached(season: int) -> Optional[dict]:
    return load_rosters(CACHE_DIR, season)


@st.cache_resource(show_spinner=False)
def load_registry(season: int):
    return load_teams_registry(CACHE_DIR, season)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("What-If Injury Simulator")
st.caption(
    "Toggle players in or out of the active roster. The model re-runs on every "
    "click — see exactly how a missing star moves the win probability."
)

# --- Required artefacts --------------------------------------------------
missing = [p for p in (PREDICTIONS_CSV, FEATURES_CSV) if not p.exists()]
if missing:
    st.error(
        "Missing snapshot files: "
        + ", ".join(f"`{p.relative_to(ROOT)}`" for p in missing)
        + ". Run `euroleague-sim predict` locally or wait for the next nightly "
        "GitHub Actions update."
    )
    st.stop()

predictor = load_model()
if predictor is None:
    st.error(
        f"No trained model found at `{MODEL_DIR.relative_to(ROOT)}/baseline/`. "
        "Run `euroleague-sim train` to produce model artefacts."
    )
    st.stop()

preds = load_predictions(PREDICTIONS_CSV)
feats = load_features(FEATURES_CSV)
if preds.empty or feats.empty:
    st.warning("Snapshot files are empty.")
    st.stop()

season = int(preds["season"].iloc[0])
rosters = load_rosters_cached(season)
if rosters is None or not rosters.get("teams"):
    st.error(
        f"No roster cache for season {season}. Run `euroleague-sim update-data --season {season}` "
        "to regenerate `rosters_E{season}.json`."
    )
    st.stop()

registry = load_registry(season)


def label(code: str) -> str:
    return team_label(code, registry, "short")


# --- Matchup picker ------------------------------------------------------
preds = preds.copy()
preds["label"] = preds.apply(
    lambda r: f"{label(r['home_team'])} vs {label(r['away_team'])}  ·  Round {int(r['round'])}",
    axis=1,
)
options = preds["gamecode"].tolist()
selected_gc = st.selectbox(
    "Matchup",
    options=options,
    format_func=lambda gc: preds.loc[preds["gamecode"] == gc, "label"].iloc[0],
)

pred_row = preds.loc[preds["gamecode"] == selected_gc].iloc[0]
home_code = str(pred_row["home_team"])
away_code = str(pred_row["away_team"])

# --- Baseline feature row -----------------------------------------------
feat_match = feats.loc[feats["Gamecode"] == selected_gc]
if feat_match.empty:
    st.error(
        f"No feature row found for gamecode {selected_gc} in `latest_features.csv`. "
        "The predictions and features snapshots are out of sync — re-run predict."
    )
    st.stop()
baseline_row = feat_match.iloc[[0]].copy()

# Sanity: every model feature must be present in the snapshot.
missing_cols = [c for c in FEATURE_COLS if c not in baseline_row.columns]
if missing_cols:
    st.error(f"Feature snapshot missing columns: {missing_cols}. Re-run predict.")
    st.stop()

baseline_pred = predictor.predict(baseline_row).iloc[0]
baseline_p = float(baseline_pred["pHomeWin_ml"])
baseline_margin = float(baseline_pred["margin_ml"])
baseline_bpm_diff = float(baseline_row["net_bpm_diff"].iloc[0])

# --- Roster lookup -------------------------------------------------------
teams = rosters["teams"]
if home_code not in teams or away_code not in teams:
    missing_codes = [c for c in (home_code, away_code) if c not in teams]
    st.warning(
        f"No roster data for {', '.join(missing_codes)}. Showing baseline only."
    )
    home_entry = teams.get(home_code, {"players": [], "net_anchor": 0.0})
    away_entry = teams.get(away_code, {"players": [], "net_anchor": 0.0})
else:
    home_entry = teams[home_code]
    away_entry = teams[away_code]


# --- Checkbox UI ---------------------------------------------------------
def render_roster(team_code: str, team_entry: dict, side: str) -> set[str]:
    """Render checkboxes for one team and return the set of active player ids."""
    state_key = f"active_{selected_gc}_{team_code}"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_active_ids(team_entry)

    st.markdown(
        f"<div class='roster-hdr'>{side} · {label(team_code)}</div>",
        unsafe_allow_html=True,
    )

    btn_col1, btn_col2 = st.columns(2)
    if btn_col1.button("Reset to last game", key=f"reset_{state_key}"):
        st.session_state[state_key] = default_active_ids(team_entry)
        # Also reset individual checkbox states.
        for p in team_entry["players"]:
            st.session_state[f"chk_{state_key}_{p['player_id']}"] = (
                p["player_id"] in st.session_state[state_key]
            )
        st.rerun()
    if btn_col2.button("Clear all", key=f"clear_{state_key}"):
        st.session_state[state_key] = set()
        for p in team_entry["players"]:
            st.session_state[f"chk_{state_key}_{p['player_id']}"] = False
        st.rerun()

    active: set[str] = set()
    default_ids = st.session_state[state_key]
    for p in team_entry["players"]:
        pid = p["player_id"]
        last_min = float(p["last_game_minutes"])
        avg_min = float(p["avg_minutes"])
        bpm = float(p["season_bpm"])
        out_badge = "  ⚕️" if last_min == 0 else ""
        lbl = (
            f"{p['name']}{out_badge}  ·  "
            f"last {last_min:.0f}m / avg {avg_min:.1f}m  ·  BPM {bpm:+.1f}"
        )
        chk_key = f"chk_{state_key}_{pid}"
        if chk_key not in st.session_state:
            st.session_state[chk_key] = pid in default_ids
        checked = st.checkbox(lbl, key=chk_key)
        if checked:
            active.add(pid)
    return active


col_home, col_away = st.columns(2)
with col_home:
    active_home = render_roster(home_code, home_entry, "Home")
with col_away:
    active_away = render_roster(away_code, away_entry, "Away")

# --- Synthetic feature + live inference ---------------------------------
bpm_home = active_team_bpm(home_entry, active_home)
bpm_away = active_team_bpm(away_entry, active_away)
synth_diff = float(bpm_home - bpm_away)

synth_row = baseline_row.copy()
synth_row["net_bpm_diff"] = synth_diff
synth_pred = predictor.predict(synth_row).iloc[0]
synth_p = float(synth_pred["pHomeWin_ml"])
synth_margin = float(synth_pred["margin_ml"])

# --- Extrapolation guard -------------------------------------------------
all_baseline = feats["net_bpm_diff"].astype(float).abs().max()
if pd.notna(all_baseline) and abs(synth_diff) > all_baseline:
    st.warning(
        f"Synthetic `net_bpm_diff` = {synth_diff:+.2f} sits outside the "
        f"in-sample range (±{all_baseline:.2f}). Treat the prediction as "
        "extrapolation."
    )

# --- Output --------------------------------------------------------------
st.divider()
st.subheader("Cause and effect")

m1, m2, m3 = st.columns(3)
m1.metric(
    label=f"P({label(home_code)} win)",
    value=f"{synth_p:.1%}",
    delta=f"{(synth_p - baseline_p) * 100:+.1f} pp",
)
m2.metric(
    label="Expected margin (home)",
    value=f"{synth_margin:+.2f}",
    delta=f"{synth_margin - baseline_margin:+.2f}",
)
m3.metric(
    label="net_bpm_diff",
    value=f"{synth_diff:+.2f}",
    delta=f"{synth_diff - baseline_bpm_diff:+.2f}",
)

st.caption(
    f"Active BPM — {label(home_code)}: **{bpm_home:+.2f}**  ·  "
    f"{label(away_code)}: **{bpm_away:+.2f}**  ·  "
    f"baseline diff held: {baseline_bpm_diff:+.2f}"
)

with st.expander("Baseline vs What-If — raw numbers"):
    st.dataframe(
        pd.DataFrame(
            {
                "Baseline": [baseline_p, baseline_margin, baseline_bpm_diff],
                "What-If":  [synth_p, synth_margin, synth_diff],
            },
            index=["P(home win)", "E[margin home]", "net_bpm_diff"],
        ).style.format("{:+.4f}"),
        use_container_width=True,
    )

st.markdown(
    "<div class='footer-meta'>Inference: single 1-row forward pass through the "
    "CatBoost predictor. No retraining, no Monte Carlo — only "
    "<code>net_bpm_diff</code> is perturbed.</div>",
    unsafe_allow_html=True,
)
