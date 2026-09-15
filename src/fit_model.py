"""
fit_model.py
Replaces every hand-picked constant in game_predictions.py with a coefficient
regressed against real NFL outcomes, and measures how much (if at all) our
model adds on top of the market once blended with it.

WHY THIS EXISTS: game_predictions.py's original constants (home field = 1.5
points, EPA-to-points scaling via a flat 65 plays/game, a fixed 13.5-point
margin std-dev, hand-typed secondary-factor weights) were "reasonable
industry-standard approximations" per its own docstring - never checked
against this project's own data. You cannot get sharp by guessing; you get
sharp by fitting coefficients to outcomes and validating out-of-sample.

DATA: nflverse's games.csv (via nfldata) carries the CLOSING Vegas
spread/total/moneyline for every game back to 1999 - no paid historical odds
subscription needed. That's the market benchmark this script fits and grades
against.

METHOD:
  1. Pull play-by-play for TRAIN_START_SEASON..present, rebuild the same
     per-game team/defense stats build_features.py computes, then re-derive
     them WALK-FORWARD (historical_features.py) so every game's features only
     ever see strictly prior games - no leakage from the outcome being fit.
  2. Split into a TRAIN window and a held-out VALIDATION window (the most
     recent two seasons). Fit margin/total regressions on TRAIN only.
  3. On the untouched VALIDATION window, compare: our model alone, Vegas
     alone, and a grid of model/Vegas blends - by mean absolute error against
     the actual final margin/total, and win-probability calibration (Brier
     score) for straight-up outcomes. Pick the blend weight that wins on
     held-out data, not on the data it was fit to.
  4. Refit final coefficients on the FULL dataset (train+validation) for
     production use, but keep the blend weight and error metrics from the
     honest holdout evaluation - a model shouldn't grade its own homework.
  5. Write everything to src/models/fitted_coefficients.json, versioned in
     git, loaded by game_predictions.py at prediction time.

Run this occasionally (see .github/workflows/refit-model.yml), NOT on every
weekly picks run - it downloads a decade-plus of play-by-play, which is slow
and unnecessary to redo twice a week.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from fetch_data import download_csv_gz, current_nfl_season
from build_features import build_team_game_stats, build_team_defense_game_stats
from historical_features import compute_walkforward_features

HISTORICAL_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical_raw")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(HISTORICAL_RAW_DIR, exist_ok=True)

BASE = "https://github.com/nflverse/nflverse-data/releases/download"

TRAIN_START_SEASON = 2013          # modern EPA-era data, ~13 seasons by default
VALIDATION_SEASONS = 2             # most recent N completed seasons held out
BLEND_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)

OFF_METRICS = ["epa_per_play", "third_down_rate", "redzone_td_rate", "explosive_rate", "sack_rate"]
DEF_METRICS = ["def_epa_per_play_allowed"]

def fetch_historical_pbp(seasons):
    frames = []
    for season in seasons:
        dest = os.path.join(HISTORICAL_RAW_DIR, f"pbp_{season}.csv.gz")
        if not os.path.exists(dest):
            url = f"{BASE}/pbp/play_by_play_{season}.csv.gz"
            download_csv_gz(url, dest)
        df = pd.read_csv(dest, compression="gzip", low_memory=False)
        df["season"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def fetch_historical_schedules():
    url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    return pd.read_csv(url)

def load_pbp_offense_defense(pbp):
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])]
    pbp = pbp[pbp["posteam"].notna()]
    off = build_team_game_stats(pbp)
    defn = build_team_defense_game_stats(pbp)
    return off, defn

def _merge_side(games, stats, side):
    """Rename a per-team stats frame's team/metric columns to home_/away_
    and merge onto games on (season, week, <side>_team)."""
    rename = {"team": f"{side}_team"}
    rename.update({c: f"{side}_{c}" for c in stats.columns if c not in ("season", "week", "team")})
    renamed = stats.rename(columns=rename)
    return games.merge(renamed, on=["season", "week", f"{side}_team"], how="left")

def build_dataset(off_walk, def_walk, schedules):
    """
    One row per completed REG/POST game with actual results, closing Vegas
    lines, and both teams' pre-game (walk-forward) features.
    """
    off_cols = ["season", "week", "team"] + [f"{c}_asof" for c in OFF_METRICS] + [f"{c}_asof_season_avg" for c in OFF_METRICS]
    def_cols = ["season", "week", "team"] + [f"{c}_asof" for c in DEF_METRICS] + [f"{c}_asof_season_avg" for c in DEF_METRICS]
    off = off_walk[off_cols]
    defn = def_walk[def_cols]

    games = schedules[schedules["result"].notna()].copy()
    games = games[games["spread_line"].notna() & games["total_line"].notna()]

    games = _merge_side(games, off, "home")
    games = _merge_side(games, off, "away")
    games = _merge_side(games, defn, "home")
    games = _merge_side(games, defn, "away")

    games["actual_margin"] = games["home_score"] - games["away_score"]
    games["actual_total"] = games["home_score"] + games["away_score"]
    games["vegas_home_favored_by"] = -games["spread_line"]
    games["vegas_total"] = games["total_line"]
    games["vegas_home_ml_prob"] = games["home_moneyline"].apply(_moneyline_to_prob)
    games["vegas_away_ml_prob"] = games["away_moneyline"].apply(_moneyline_to_prob)
    vig_sum = games["vegas_home_ml_prob"] + games["vegas_away_ml_prob"]
    games["vegas_home_win_prob"] = games["vegas_home_ml_prob"] / vig_sum
    games["home_won"] = (games["actual_margin"] > 0).astype(float)

    return games

def _moneyline_to_prob(ml):
    if pd.isna(ml):
        return np.nan
    return -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)

def make_features(games, suffix="_asof"):
    """Build the exact feature set predict_game() uses, for every row."""
    home_off = games[f"home_epa_per_play{suffix}"]
    away_def = games[f"away_def_epa_per_play_allowed{suffix}"]
    away_off = games[f"away_epa_per_play{suffix}"]
    home_def = games[f"home_def_epa_per_play_allowed{suffix}"]

    home_exp_eff = (home_off + away_def) / 2
    away_exp_eff = (away_off + home_def) / 2

    feats = pd.DataFrame({
        "epa_diff": home_exp_eff - away_exp_eff,
        "third_down_diff": games[f"home_third_down_rate{suffix}"] - games[f"away_third_down_rate{suffix}"],
        "redzone_diff": games[f"home_redzone_td_rate{suffix}"] - games[f"away_redzone_td_rate{suffix}"],
        "explosive_diff": games[f"home_explosive_rate{suffix}"] - games[f"away_explosive_rate{suffix}"],
        "sack_diff": games[f"home_sack_rate{suffix}"] - games[f"away_sack_rate{suffix}"],
    })
    feats["combined_exp_eff"] = home_exp_eff + away_exp_eff
    return feats

def fit_ols(X, y):
    """Plain OLS via lstsq (no sklearn dependency): y = X @ beta, X's last
    column is a constant 1s column so beta[-1] is the fitted intercept."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    return beta, residuals

def fit_margin_and_total(feats, games):
    valid = feats.notna().all(axis=1) & games["actual_margin"].notna() & games["actual_total"].notna()
    feats, games = feats[valid], games[valid]

    n = len(feats)
    ones = np.ones(n)

    X_margin = np.column_stack([
        feats["epa_diff"], feats["third_down_diff"], feats["redzone_diff"],
        feats["explosive_diff"], feats["sack_diff"], ones,
    ])
    beta_margin, resid_margin = fit_ols(X_margin, games["actual_margin"].values)

    X_total = np.column_stack([feats["combined_exp_eff"], ones])
    beta_total, resid_total = fit_ols(X_total, games["actual_total"].values)

    return {
        "epa_diff_coef": float(beta_margin[0]),
        "third_down_weight": float(beta_margin[1]),
        "redzone_weight": float(beta_margin[2]),
        "explosive_weight": float(beta_margin[3]),
        "sack_weight": float(beta_margin[4]),
        "home_field_advantage": float(beta_margin[5]),
        "margin_std_dev": float(np.std(resid_margin, ddof=X_margin.shape[1])),
        "total_coef": float(beta_total[0]),
        "total_intercept": float(beta_total[1]),
        "total_std_dev": float(np.std(resid_total, ddof=X_total.shape[1])),
        "n_games": int(n),
    }, valid

def model_predict(feats, coefs):
    spread = (
        feats["epa_diff"] * coefs["epa_diff_coef"]
        + feats["third_down_diff"] * coefs["third_down_weight"]
        + feats["redzone_diff"] * coefs["redzone_weight"]
        + feats["explosive_diff"] * coefs["explosive_weight"]
        + feats["sack_diff"] * coefs["sack_weight"]
        + coefs["home_field_advantage"]
    )
    total = feats["combined_exp_eff"] * coefs["total_coef"] + coefs["total_intercept"]
    return spread, total

def evaluate_blend(model_spread, vegas_spread, actual_margin, model_wp, vegas_wp, home_won, sigma):
    """For every blend weight in BLEND_GRID, compute spread MAE and win-prob
    Brier score. Returns the grid as a DataFrame plus the best weight for each."""
    from scipy.stats import norm
    rows = []
    for w in BLEND_GRID:
        blended_spread = w * model_spread + (1 - w) * vegas_spread
        spread_mae = float(np.mean(np.abs(blended_spread - actual_margin)))

        blended_wp = w * model_wp + (1 - w) * vegas_wp
        brier = float(np.mean((blended_wp - home_won) ** 2))

        rows.append({"weight_on_model": float(w), "spread_mae": spread_mae, "brier_score": brier})

    grid = pd.DataFrame(rows)
    best_spread_w = float(grid.loc[grid["spread_mae"].idxmin(), "weight_on_model"])
    best_wp_w = float(grid.loc[grid["brier_score"].idxmin(), "weight_on_model"])
    return grid, best_spread_w, best_wp_w

def run():
    current_season = current_nfl_season()
    seasons = list(range(TRAIN_START_SEASON, current_season + 1))
    print(f"Fetching play-by-play for {len(seasons)} seasons ({seasons[0]}-{seasons[-1]})...")
    pbp = fetch_historical_pbp(seasons)
    print(f"  {len(pbp):,} rows loaded")

    print("Rebuilding per-game team/defense stats...")
    off_game, def_game = load_pbp_offense_defense(pbp)

    print("Computing walk-forward (no-leakage) features...")
    off_walk = compute_walkforward_features(off_game, OFF_METRICS)
    def_walk = compute_walkforward_features(def_game, DEF_METRICS)

    print("Fetching historical schedules + closing Vegas lines...")
    schedules = fetch_historical_schedules()
    dataset = build_dataset(off_walk, def_walk, schedules)
    dataset = dataset[dataset["season"] >= TRAIN_START_SEASON]
    print(f"  {len(dataset)} completed games with usable Vegas lines")

    holdout_seasons = sorted(dataset["season"].unique())[-VALIDATION_SEASONS:]
    train = dataset[~dataset["season"].isin(holdout_seasons)]
    validation = dataset[dataset["season"].isin(holdout_seasons)]
    print(f"  Train: {len(train)} games ({train['season'].min()}-{train['season'].max()})")
    print(f"  Validation (held out): {len(validation)} games (seasons {holdout_seasons})")

    print("\nFitting margin/total regressions on TRAIN only...")
    train_feats = make_features(train)
    train_coefs, _ = fit_margin_and_total(train_feats, train)
    print(f"  {json.dumps(train_coefs, indent=2)}")

    print("\nEvaluating model vs Vegas vs blends on the HELD-OUT validation seasons...")
    val_feats = make_features(validation)
    valid_mask = val_feats.notna().all(axis=1) & validation["actual_margin"].notna() & validation["vegas_home_win_prob"].notna()
    val_feats, val_games = val_feats[valid_mask], validation[valid_mask]

    from scipy.stats import norm
    model_spread, model_total = model_predict(val_feats, train_coefs)
    model_wp = norm.cdf(model_spread / train_coefs["margin_std_dev"])

    vegas_spread = val_games["vegas_home_favored_by"].values
    vegas_total = val_games["vegas_total"].values
    vegas_wp = val_games["vegas_home_win_prob"].values
    actual_margin = val_games["actual_margin"].values
    actual_total = val_games["actual_total"].values
    home_won = val_games["home_won"].values

    model_only_mae = float(np.mean(np.abs(model_spread - actual_margin)))
    vegas_only_mae = float(np.mean(np.abs(vegas_spread - actual_margin)))
    model_only_brier = float(np.mean((model_wp - home_won) ** 2))
    vegas_only_brier = float(np.mean((vegas_wp - home_won) ** 2))
    model_total_mae = float(np.mean(np.abs(model_total - actual_total)))
    vegas_total_mae = float(np.mean(np.abs(vegas_total - actual_total)))

    print(f"  Spread MAE  - model only: {model_only_mae:.2f} pts | Vegas only: {vegas_only_mae:.2f} pts")
    print(f"  Total MAE   - model only: {model_total_mae:.2f} pts | Vegas only: {vegas_total_mae:.2f} pts")
    print(f"  Win Brier   - model only: {model_only_brier:.4f} | Vegas only: {vegas_only_brier:.4f}")

    grid, best_spread_w, best_wp_w = evaluate_blend(
        model_spread, vegas_spread, actual_margin, model_wp, vegas_wp, home_won, train_coefs["margin_std_dev"]
    )
    best_row_spread = grid.loc[grid["weight_on_model"] == best_spread_w].iloc[0]
    best_row_wp = grid.loc[grid["weight_on_model"] == best_wp_w].iloc[0]
    print(f"\n  Best blend weight on OUR model for spread: {best_spread_w} "
          f"(blended MAE {best_row_spread['spread_mae']:.2f} pts, vs Vegas-only {vegas_only_mae:.2f})")
    print(f"  Best blend weight on OUR model for win-prob: {best_wp_w} "
          f"(blended Brier {best_row_wp['brier_score']:.4f}, vs Vegas-only {vegas_only_brier:.4f})")

    # Also fit a blend weight for totals the same way (grid search on total MAE)
    total_rows = []
    for w in BLEND_GRID:
        blended_total = w * model_total + (1 - w) * vegas_total
        total_rows.append({"weight_on_model": float(w), "total_mae": float(np.mean(np.abs(blended_total - actual_total)))})
    total_grid = pd.DataFrame(total_rows)
    best_total_w = float(total_grid.loc[total_grid["total_mae"].idxmin(), "weight_on_model"])
    print(f"  Best blend weight on OUR model for total: {best_total_w} "
          f"(blended MAE {total_grid['total_mae'].min():.2f} pts, vs Vegas-only {vegas_total_mae:.2f})")

    print("\nRefitting final coefficients on FULL dataset (train + validation) for production...")
    full_feats = make_features(dataset)
    final_coefs, _ = fit_margin_and_total(full_feats, dataset)

    output = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "train_seasons": [int(s) for s in seasons],
        "n_games_used": final_coefs["n_games"],
        "coefficients": final_coefs,
        "blend_weight_on_model_spread": best_spread_w,
        "blend_weight_on_model_winprob": best_wp_w,
        "blend_weight_on_model_total": best_total_w,
        "holdout_validation": {
            "seasons": [int(s) for s in holdout_seasons],
            "n_games": int(len(val_games)),
            "model_only_spread_mae": model_only_mae,
            "vegas_only_spread_mae": vegas_only_mae,
            "blended_spread_mae": float(best_row_spread["spread_mae"]),
            "model_only_total_mae": model_total_mae,
            "vegas_only_total_mae": vegas_total_mae,
            "blended_total_mae": float(total_grid["total_mae"].min()),
            "model_only_brier": model_only_brier,
            "vegas_only_brier": vegas_only_brier,
            "blended_brier": float(best_row_wp["brier_score"]),
        },
    }

    out_path = os.path.join(MODELS_DIR, "fitted_coefficients.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved fitted coefficients + holdout validation report to {out_path}")

if __name__ == "__main__":
    run()
