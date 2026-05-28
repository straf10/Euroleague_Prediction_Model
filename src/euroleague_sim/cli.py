from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from .config import ProjectConfig
from .data.cache import Cache
from .data.fetch import EuroleagueFetcher, FetchParams
from .pipeline import (
    update_season_cache,
    build_features_for_season,
    predict_next_round,
    train_ml_pipeline,
    save_predictions,
)


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="euroleague-sim",
        description="Euroleague next-round predictions (Logistic Regression + Ridge + Monte Carlo)",
    )
    p.add_argument("--cache-dir", default="data_cache", help="Folder for cached data (default: data_cache)")
    p.add_argument("--config", default=None, help="Path to config.json (optional)")
    p.add_argument("--dump-config", default=None, help="Write default config.json to this path and exit")

    sub = p.add_subparsers(dest="cmd", required=True)

    # update-data
    upd = sub.add_parser("update-data", help="Fetch & cache raw data and build features")
    upd.add_argument("--season", type=int, default=2025, help="Season start year (default: 2025 => 2025-26)")
    upd.add_argument("--history", type=int, default=2, help="Past seasons to cache for Elo (default: 2)")
    upd.add_argument("--force", action="store_true", help="Force re-download raw data and rebuild features")

    # train
    trn = sub.add_parser("train", help="Train ML models")
    trn.add_argument("--season", type=int, default=2025, help="Current season start year (default: 2025)")
    trn.add_argument("--model", type=str, default="baseline", help="Model name from registry, or 'all' for leaderboard (default: baseline)")

    # predict
    pred = sub.add_parser("predict", help="Predict a round (default: next unplayed round)")
    pred.add_argument("--season", type=int, default=2025, help="Season start year (default: 2025)")
    pred.add_argument("--round", default="next", help="Round number (int) or 'next'")
    pred.add_argument("--model", type=str, default="baseline", help="Model name from registry (default: baseline)")
    pred.add_argument("--n-sims", type=int, default=None, help="Number of MC simulations (default from config: 20000)")
    pred.add_argument("--seed", type=int, default=42, help="Random seed")
    pred.add_argument("--out", default=None, help="Output CSV path (default: outputs/round_R_predictions.csv)")

    return p.parse_args(argv)


def _save_latest_predictions(
    pred_df: pd.DataFrame,
    season: int,
    round_number: int,
    model_name: str,
    out_path: Path,
) -> None:
    """Write a presentation-ready CSV consumed by the Streamlit dashboard.

    Schema is intentionally narrow and stable so the dashboard never has to
    care about MC internals: one row per game with date, teams, the Beta-
    calibrated win probability (falling back to MC if the ML model wasn't
    available), expected margin and the three feature differentials our
    diagnostics actually use to explain the call.
    """
    df = pred_df.copy()
    p_home = df.get("pHomeWin_ml")
    if p_home is None or p_home.isna().all():
        p_home = df.get("pHomeWin")
    expected_margin = df.get("margin_ml")
    if expected_margin is None or expected_margin.isna().all():
        expected_margin = df.get("q50")

    elo_diff = (
        df["EloCurrent_home"] - df["EloCurrent_away"]
        if {"EloCurrent_home", "EloCurrent_away"}.issubset(df.columns)
        else pd.Series([float("nan")] * len(df))
    )

    is_offseason = bool(df["is_offseason"].iloc[0]) if "is_offseason" in df.columns else False

    out = pd.DataFrame({
        "season":           season,
        "round":            int(round_number),
        "is_offseason":     is_offseason,
        "model":            model_name,
        "gamecode":         df.get("Gamecode"),
        "game_date":        pd.to_datetime(df.get("game_date"), errors="coerce"),
        "home_team":        df.get("home_team"),
        "away_team":        df.get("away_team"),
        "p_home_win":       p_home.astype(float).round(4),
        "expected_margin":  expected_margin.astype(float).round(2),
        "margin_q10":       df.get("q10"),
        "margin_q90":       df.get("q90"),
        "elo_home":         df.get("EloCurrent_home"),
        "elo_away":         df.get("EloCurrent_away"),
        "elo_diff":         elo_diff.round(1),
        "bpm_diff":         df.get("net_bpm_diff"),
        "fatigue_diff":     df.get("domestic_fatigue_diff"),
    })

    # Predicted winner + confidence for one-glance display.
    out["predicted_winner"] = out.apply(
        lambda r: r["home_team"] if r["p_home_win"] >= 0.5 else r["away_team"], axis=1,
    )
    out["confidence"] = (out["p_home_win"].clip(0, 1) - 0.5).abs().mul(2).round(4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def _print_predictions(pred_df: pd.DataFrame, round_number: int) -> None:
    """Pretty-print predictions to terminal."""
    has_ml = "pHomeWin_ml" in pred_df.columns

    print(f"\n{'='*80}")
    print(f"  EUROLEAGUE PREDICTIONS — Round {round_number}")
    if has_ml:
        print(f"  Models: Logistic Regression + Ridge + Monte Carlo")
    else:
        print(f"  Models: Monte Carlo only  (run 'train' to enable ML models)")
    print(f"{'='*80}\n")

    for _, row in pred_df.iterrows():
        home = row.get("home_team", "?")
        away = row.get("away_team", "?")
        p_home_mc  = row.get("pHomeWin", 0)
        margin     = row.get("q50", 0)
        q10        = row.get("q10", 0)
        q90        = row.get("q90", 0)

        if has_ml:
            p_ml = row.get("pHomeWin_ml", 0)
            winner = home if p_ml > 0.5 else away
            conf = max(p_ml, 1 - p_ml) * 100

            print(f"  {home:>5s}  vs  {away:<5s}")
            print(f"    P(Home Win):  ML: {p_ml:.1%}   MC: {p_home_mc:.1%}")
        else:
            winner = home if p_home_mc > 0.5 else away
            conf = max(p_home_mc, 1 - p_home_mc) * 100

            print(f"  {home:>5s}  vs  {away:<5s}")
            print(f"    P(Home Win): {p_home_mc:.1%} (MC)")

        print(f"    Expected margin: {margin:+.1f}  [{q10:+.1f} / {q90:+.1f}]")
        print(f"    -> Prediction: {winner} ({conf:.0f}% confidence)")
        print()

    n_sims = int(pred_df["n_sims"].iloc[0]) if "n_sims" in pred_df.columns else "?"
    print(f"  Simulations: {n_sims}")
    print(f"{'='*80}\n")


def main(argv=None):
    args = _parse_args(argv or sys.argv[1:])

    if args.dump_config:
        cfg = ProjectConfig.default()
        Path(args.dump_config).write_text(
            json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote default config to: {args.dump_config}")
        return 0

    cfg = ProjectConfig.default()
    if args.config:
        cfg = ProjectConfig.load(Path(args.config))

    cache = Cache(Path(args.cache_dir))
    fetcher = EuroleagueFetcher(FetchParams(competition=cfg.season.competition))

    if args.cmd == "update-data":
        seasons = [args.season] + [args.season - i for i in range(1, args.history + 1)]
        for s in seasons:
            print(f"[update-data] season {s} ...")
            update_season_cache(cache, fetcher, s, force=args.force)
            build_features_for_season(cache, s, cfg=cfg, force=args.force)
        print(f"[update-data] Done. Cache at: {Path(args.cache_dir).resolve()}")
        return 0

    if args.cmd == "train":
        season = int(args.season)
        model_name = args.model
        print(f"[train] Training ML models ({model_name}) for season {season} "
              f"(+ {cfg.season.history_seasons} history seasons) …")
        metrics = train_ml_pipeline(cache, cfg, current_season=season, model_name=model_name, verbose=True)
        if model_name != "all":
            print(f"[train] Done. Models at: {Path(cfg.ml.model_dir).resolve()}/{model_name}")
        return 0

    if args.cmd == "predict":
        season = int(args.season)

        round_number = None
        round_arg = args.round
        if isinstance(round_arg, str) and round_arg.lower() != "next":
            try:
                round_number = int(round_arg)
            except ValueError:
                raise SystemExit("--round must be an integer or 'next'")

        pred_df = predict_next_round(
            cache=cache,
            fetcher=fetcher,
            cfg=cfg,
            season=season,
            round_number=round_number,
            model_name=args.model,
            n_sims=args.n_sims,
            seed=int(args.seed),
        )

        # Determine effective round number for output
        eff_round = round_number
        if eff_round is None and "Round" in pred_df.columns:
            eff_round = int(pred_df["Round"].iloc[0])
        eff_round = eff_round or 0

        if bool(pred_df.get("is_offseason", pd.Series([False])).iloc[0]):
            print(
                f"[predict] Off-season detected — falling back to the last played "
                f"round of season {season} (round {eff_round})."
            )

        _print_predictions(pred_df, eff_round)

        # Save CSV
        out_path = args.out
        if out_path:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(p, index=False)
        else:
            p = save_predictions(pred_df, season, eff_round, Path("outputs"))

        print(f"Saved predictions to: {p.resolve()}")

        # Presentation-ready snapshot consumed by the Streamlit dashboard.
        latest_path = Path(args.cache_dir) / "latest_predictions.csv"
        _save_latest_predictions(pred_df, season, eff_round, args.model, latest_path)
        print(f"Wrote dashboard snapshot to: {latest_path.resolve()}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
