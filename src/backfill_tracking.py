"""
backfill_tracking.py
Seeds data/tracking/predictions_log.csv with genuinely out-of-sample
predictions for BACKFILL_SEASONS, so the email's "Track Record" section has
real, graded history from day one instead of sitting empty for weeks while
live predictions slowly accumulate.

HOW THIS STAYS HONEST: coefficients are fit ONLY on seasons before
BACKFILL_SEASONS (same walk-forward, no-leakage features as fit_model.py) -
this is a true holdout simulation of "what would this system have predicted
at the time," not the final production coefficients grading their own
training data. The market-blend weights, however, come from the final
committed fitted_coefficients.json (those were already validated on a
holdout that overlaps BACKFILL_SEASONS) - a small, acknowledged approximation
made so the backfilled "sharp" line matches what the live system actually
uses, rather than a third, throwaway set of blend weights.

Run this once, manually, after fit_model.py has produced
src/models/fitted_coefficients.json. It's not part of any scheduled
workflow - it's a one-time seed, not something to re-run every week (that's
what track_results.py's live snapshot/grade cycle is for).
"""

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import fit_model as fm
import track_results as tr
from historical_features import compute_walkforward_features
from scipy.stats import norm

BACKFILL_SEASONS = [2024, 2025]

def run():
    coeff_path = os.path.join(fm.MODELS_DIR, "fitted_coefficients.json")
    if not os.path.exists(coeff_path):
        print("No fitted_coefficients.json found - run src/fit_model.py first.")
        return
    with open(coeff_path) as f:
        fitted = json.load(f)
    blend = {
        "spread": fitted.get("blend_weight_on_model_spread", 1.0),
        "total": fitted.get("blend_weight_on_model_total", 1.0),
        "winprob": fitted.get("blend_weight_on_model_winprob", 1.0),
    }

    print(f"Fetching play-by-play for {fm.TRAIN_START_SEASON}-present...")
    seasons = list(range(fm.TRAIN_START_SEASON, fm.current_nfl_season() + 1))
    pbp = fm.fetch_historical_pbp(seasons)

    print("Rebuilding walk-forward features...")
    off_game, def_game = fm.load_pbp_offense_defense(pbp)
    off_walk = compute_walkforward_features(off_game, fm.OFF_METRICS)
    def_walk = compute_walkforward_features(def_game, fm.DEF_METRICS)

    print("Building historical dataset...")
    schedules = fm.fetch_historical_schedules()
    dataset = fm.build_dataset(off_walk, def_walk, schedules)
    dataset = dataset[dataset["season"] >= fm.TRAIN_START_SEASON]

    train = dataset[dataset["season"] < min(BACKFILL_SEASONS)]
    target = dataset[dataset["season"].isin(BACKFILL_SEASONS)].copy()
    print(f"  Fitting on {len(train)} games strictly before {min(BACKFILL_SEASONS)} "
          f"(a true holdout for the {BACKFILL_SEASONS} backfill)")

    train_feats = fm.make_features(train)
    coefs, _ = fm.fit_margin_and_total(train_feats, train)

    target_feats = fm.make_features(target)
    valid = target_feats.notna().all(axis=1) & target["actual_margin"].notna() & target["vegas_home_win_prob"].notna()
    target_feats, target = target_feats[valid], target[valid]
    print(f"  {len(target)} backfill games with complete features + Vegas lines + results")

    model_spread, model_total = fm.model_predict(target_feats, coefs)
    model_wp = norm.cdf(model_spread / coefs["margin_std_dev"])

    sharp_spread = blend["spread"] * model_spread + (1 - blend["spread"]) * target["vegas_home_favored_by"]
    sharp_total = blend["total"] * model_total + (1 - blend["total"]) * target["vegas_total"]
    sharp_wp = blend["winprob"] * model_wp + (1 - blend["winprob"]) * target["vegas_home_win_prob"]

    log = pd.DataFrame({
        "season": target["season"].values,
        "week": target["week"].values,
        "home_team": target["home_team"].values,
        "away_team": target["away_team"].values,
        "gameday": target["gameday"].values,
        "model_spread": model_spread.values,
        "model_total": model_total.values,
        "model_home_win_prob": model_wp,
        "sharp_spread": sharp_spread.values,
        "sharp_total": sharp_total.values,
        "sharp_home_win_prob": sharp_wp.values,
        "vegas_home_favored_by": target["vegas_home_favored_by"].values,
        "vegas_total": target["vegas_total"].values,
        "vegas_home_win_prob": target["vegas_home_win_prob"].values,
        "actual_margin": target["actual_margin"].values,
        "actual_total": target["actual_total"].values,
    })

    print("Grading backfilled predictions against actual results (already known - "
          "these seasons are complete)...")
    home_won = (log["actual_margin"] > 0).astype(float)
    for label, spread_col, total_col, wp_col in [
        ("model", "model_spread", "model_total", "model_home_win_prob"),
        ("sharp", "sharp_spread", "sharp_total", "sharp_home_win_prob"),
        ("vegas", "vegas_home_favored_by", "vegas_total", "vegas_home_win_prob"),
    ]:
        picked_home_won = (log[spread_col] > 0) == (log["actual_margin"] > 0)
        log[f"{label}_correct_pick"] = picked_home_won.astype(float)
        log[f"{label}_spread_error"] = (log[spread_col] - log["actual_margin"]).abs()
        log[f"{label}_total_error"] = (log[total_col] - log["actual_total"]).abs()
        log[f"{label}_brier"] = (log[wp_col] - home_won) ** 2

    existing_path = tr.LOG_PATH
    existing = pd.read_csv(existing_path) if os.path.exists(existing_path) else None
    combined = tr.upsert_snapshot(existing, log) if existing is not None else log
    combined.to_csv(existing_path, index=False)

    summary = tr.summarize(combined)
    with open(tr.SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBackfilled {len(log)} graded games ({BACKFILL_SEASONS}) into {existing_path}")
    std = summary.get("season_to_date", {})
    for label in ["model", "sharp", "vegas"]:
        s = std.get(label, {})
        if s:
            print(f"  {label:>6}: {s['pick_accuracy']:.1%} straight-up | "
                  f"spread MAE {s['spread_mae']:.2f} | total MAE {s['total_mae']:.2f} | Brier {s['brier_score']:.4f}")

if __name__ == "__main__":
    run()
