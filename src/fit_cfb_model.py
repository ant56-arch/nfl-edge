"""
fit_cfb_model.py
College football's version of fit_model.py: same architecture (fit margin/
total regressions on team-efficiency diffs, validate the blend against
Vegas on held-out seasons, refit on the full dataset for production), applied
to CFBD's advanced-stats metrics (PPA, success rate, explosiveness) instead
of nflverse's play-by-play EPA.

Pulls several seasons of history using EACH season's own Power-conference
membership (see fetch_cfb_data.fetch_team_info), rather than today's
alignment, since conference realignment (USC/UCLA to the Big Ten,
Oklahoma/Texas to the SEC, the Pac-12's collapse) means "who's Power
conference" has genuinely changed year to year - filtering every season by
the current season's membership would silently drop teams from their own
historical seasons.

Run this the same way as fit_model.py (see .github/workflows/refit-model.yml),
not on every weekly picks run.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from fetch_cfb_data import fetch_team_info, fetch_games, fetch_lines, fetch_advanced_stats, current_cfb_season, _headers
from historical_features import compute_walkforward_features
from build_features import RECENCY_HALF_LIFE_GAMES
import model_guard as guard

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

TRAIN_SEASONS_BACK = 5   # seasons of history to pull, ending at the current season
VALIDATION_SEASONS = 2
BLEND_GRID = np.round(np.arange(0.0, 1.01, 0.05), 2)

# Recipe search (see model_guard.py): team-form recency half-life (used by
# build_cfb_features.py for the live stats) and how many seasons to fit on,
# scored on the most recent RECENT_HOLDOUT_GAMES lined games (about a season).
RECENT_HOLDOUT_GAMES = 350
DEFAULT_RECIPE = {"team_half_life": RECENCY_HALF_LIFE_GAMES, "seasons": "all"}
RECIPES = [{"team_half_life": h, "seasons": n} for h in (2, 4, 8) for n in ("all", 3)]
COEFFICIENTS_PATH = os.path.join(MODELS_DIR, "fitted_cfb_coefficients.json")

def recipe_name(r):
    return f"half-life {r['team_half_life']} games, {r['seasons']} seasons"

def window(recipe, dataset):
    if recipe["seasons"] == "all" or dataset.empty:
        return dataset
    return dataset[dataset["season"] > dataset["season"].max() - recipe["seasons"]]

METRICS = ["off_ppa_per_play", "off_success_rate", "off_explosiveness",
           "def_ppa_per_play_allowed", "def_success_rate_allowed", "def_explosiveness_allowed"]

def fetch_historical(seasons):
    all_games, all_stats = [], []
    for season in seasons:
        print(f"  Fetching {season}...")
        teams = fetch_team_info(season)
        covered = set(teams["team"]) if not teams.empty else set()

        games = fetch_games(season)
        if not games.empty and covered:
            games = games[games["home_team"].isin(covered) | games["away_team"].isin(covered)]
        lines = fetch_lines(season)
        if not games.empty and not lines.empty:
            games = games.merge(lines, on=["home_team", "away_team"], how="left")
        if not games.empty:
            all_games.append(games)

        stats = fetch_advanced_stats(season)
        if not stats.empty and covered:
            stats = stats[stats["team"].isin(covered)]
        if not stats.empty:
            all_stats.append(stats)

    games = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()
    stats = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    return games, stats

def _merge_side(games, stats, side):
    rename = {"team": f"{side}_team"}
    rename.update({c: f"{side}_{c}" for c in stats.columns if c not in ("season", "week", "team")})
    renamed = stats.rename(columns=rename)
    return games.merge(renamed, on=["season", "week", f"{side}_team"], how="left")

def build_dataset(stats_walk, games):
    cols = ["season", "week", "team"] + [f"{c}_asof" for c in METRICS]
    stats = stats_walk[cols]

    games = games[games["home_score"].notna() & games["away_score"].notna()].copy()
    games = games[games["vegas_home_favored_by"].notna() & games["vegas_total"].notna()]

    games = _merge_side(games, stats, "home")
    games = _merge_side(games, stats, "away")

    games["actual_margin"] = games["home_score"] - games["away_score"]
    games["actual_total"] = games["home_score"] + games["away_score"]
    games["home_won"] = (games["actual_margin"] > 0).astype(float)
    # CFBD closing lines don't reliably carry a moneyline the way nflverse's
    # games.csv does for the NFL side - approximate the market's implied
    # win probability from its own spread via the same normal-distribution
    # conversion the model itself uses (a standard, well-documented approx).
    from scipy.stats import norm
    games["vegas_home_win_prob"] = norm.cdf(games["vegas_home_favored_by"] / 13.5)

    return games

def make_features(games, suffix="_asof"):
    home_off = games[f"home_off_ppa_per_play{suffix}"]
    away_def = games[f"away_def_ppa_per_play_allowed{suffix}"]
    away_off = games[f"away_off_ppa_per_play{suffix}"]
    home_def = games[f"home_def_ppa_per_play_allowed{suffix}"]

    home_exp_eff = (home_off + away_def) / 2
    away_exp_eff = (away_off + home_def) / 2

    feats = pd.DataFrame({
        "ppa_diff": home_exp_eff - away_exp_eff,
        "success_rate_diff": games[f"home_off_success_rate{suffix}"] - games[f"away_off_success_rate{suffix}"],
        "explosiveness_diff": games[f"home_off_explosiveness{suffix}"] - games[f"away_off_explosiveness{suffix}"],
    })
    feats["combined_exp_eff"] = home_exp_eff + away_exp_eff
    return feats

def fit_ols(X, y):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    return beta, residuals

def fit_margin_and_total(feats, games):
    valid = feats.notna().all(axis=1) & games["actual_margin"].notna() & games["actual_total"].notna()
    feats, games = feats[valid], games[valid]

    n = len(feats)
    ones = np.ones(n)

    X_margin = np.column_stack([feats["ppa_diff"], feats["success_rate_diff"], feats["explosiveness_diff"], ones])
    beta_margin, resid_margin = fit_ols(X_margin, games["actual_margin"].values)

    X_total = np.column_stack([feats["combined_exp_eff"], ones])
    beta_total, resid_total = fit_ols(X_total, games["actual_total"].values)

    return {
        "ppa_diff_coef": float(beta_margin[0]),
        "success_rate_weight": float(beta_margin[1]),
        "explosiveness_weight": float(beta_margin[2]),
        "home_field_advantage": float(beta_margin[3]),
        "margin_std_dev": float(np.std(resid_margin, ddof=X_margin.shape[1])),
        "total_coef": float(beta_total[0]),
        "total_intercept": float(beta_total[1]),
        "total_std_dev": float(np.std(resid_total, ddof=X_total.shape[1])),
        "n_games": int(n),
    }

def model_predict(feats, coefs):
    spread = (
        feats["ppa_diff"] * coefs["ppa_diff_coef"]
        + feats["success_rate_diff"] * coefs["success_rate_weight"]
        + feats["explosiveness_diff"] * coefs["explosiveness_weight"]
        + coefs["home_field_advantage"]
    )
    total = feats["combined_exp_eff"] * coefs["total_coef"] + coefs["total_intercept"]
    return spread, total

def evaluate_blend(model_spread, vegas_spread, actual_margin, model_wp, vegas_wp, home_won):
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

def validate_and_fit(dataset):
    """Fit on all but the last VALIDATION_SEASONS seasons, pick blend weights
    on those held-out seasons, then refit on everything."""
    holdout_seasons = sorted(dataset["season"].unique())[-VALIDATION_SEASONS:]
    train = dataset[~dataset["season"].isin(holdout_seasons)]
    validation = dataset[dataset["season"].isin(holdout_seasons)]
    print(f"  Train: {len(train)} games, Validation: {len(validation)} games (seasons {holdout_seasons})")

    print("\nFitting margin/total regressions on TRAIN only...")
    train_feats = make_features(train)
    train_coefs = fit_margin_and_total(train_feats, train)

    print("\nEvaluating model vs Vegas vs blends on the HELD-OUT validation seasons...")
    val_feats = make_features(validation)
    valid_mask = val_feats.notna().all(axis=1) & validation["actual_margin"].notna() & validation["vegas_home_win_prob"].notna()
    val_feats, val_games = val_feats[valid_mask], validation[valid_mask]

    if val_feats.empty:
        print("  No valid held-out games to evaluate against - aborting.")
        return None

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

    print(f"  Spread MAE - model: {model_only_mae:.2f} | Vegas: {vegas_only_mae:.2f}")
    print(f"  Total MAE  - model: {model_total_mae:.2f} | Vegas: {vegas_total_mae:.2f}")
    print(f"  Win Brier  - model: {model_only_brier:.4f} | Vegas: {vegas_only_brier:.4f}")

    grid, best_spread_w, best_wp_w = evaluate_blend(model_spread, vegas_spread, actual_margin, model_wp, vegas_wp, home_won)
    best_row_spread = grid.loc[grid["weight_on_model"] == best_spread_w].iloc[0]
    best_row_wp = grid.loc[grid["weight_on_model"] == best_wp_w].iloc[0]

    total_rows = []
    for w in BLEND_GRID:
        blended_total = w * model_total + (1 - w) * vegas_total
        total_rows.append({"weight_on_model": float(w), "total_mae": float(np.mean(np.abs(blended_total - actual_total)))})
    total_grid = pd.DataFrame(total_rows)
    best_total_w = float(total_grid.loc[total_grid["total_mae"].idxmin(), "weight_on_model"])

    print(f"\n  Best blend weight on OUR model - spread: {best_spread_w}, total: {best_total_w}, win-prob: {best_wp_w}")

    print("\nRefitting final coefficients on FULL dataset for production...")
    full_feats = make_features(dataset)
    final_coefs = fit_margin_and_total(full_feats, dataset)

    output = {
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "train_seasons": [int(s) for s in sorted(dataset["season"].unique())],
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

    return output

def run():
    if _headers() is None:
        print("CFBD_API_KEY not set - skipping college football model fit. "
              "Get a free key at https://collegefootballdata.com/key and add it as a repo secret.")
        return

    current_season = current_cfb_season()
    seasons = list(range(current_season - TRAIN_SEASONS_BACK, current_season + 1))
    print(f"Fetching {len(seasons)} seasons of CFB data ({seasons[0]}-{seasons[-1]})...")
    games, stats = fetch_historical(seasons)
    if games.empty or stats.empty:
        print("Not enough CFB data returned to fit a model - aborting.")
        return
    print(f"  {len(games)} games, {len(stats)} team-game stat rows")

    datasets = {}

    def dataset_for(recipe):
        """Completed lined games with walk-forward (no-leakage) features at
        this recipe's half-life."""
        hl = recipe["team_half_life"]
        if hl not in datasets:
            datasets[hl] = build_dataset(compute_walkforward_features(stats, METRICS, half_life=hl), games)
        return datasets[hl]

    live = guard.load_json(COEFFICIENTS_PATH)
    live_recipe = live.get("recipe", DEFAULT_RECIPE)
    base = dataset_for(live_recipe)
    print(f"  {len(base)} completed games with usable lines")
    if len(base) < 100:
        print("  Not enough graded games with lines to fit a reliable model yet - aborting.")
        return
    if guard.nothing_new(live, base):
        print(f"No games since the live model's last fit ({live['trained_through']}) - keeping it.")
        return

    print(f"\nScoring recipes on the most recent {RECENT_HOLDOUT_GAMES} games (each fit only on earlier games)...")
    chosen, report = guard.search(RECIPES, live_recipe, dataset_for, window, fit_margin_and_total, make_features,
                                  model_predict, RECENT_HOLDOUT_GAMES, recipe_name)
    dataset = window(chosen, dataset_for(chosen))
    print(f"\nRecipe: {recipe_name(chosen)}{' (switched)' if report['switched'] else ''}")

    output = validate_and_fit(dataset)
    if output is None:
        return
    trained_through = guard.latest_gameday(dataset)
    output.update({"recipe": chosen, "team_half_life_games": chosen["team_half_life"],
                   "trained_through": trained_through})
    ok, why, new, live_score = guard.deploy_ok(output["coefficients"], live.get("coefficients"), live_recipe, chosen,
                                               dataset_for, make_features, model_predict, report["keys"])
    if ok:
        with open(COEFFICIENTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved fitted CFB coefficients + holdout validation report to {COEFFICIENTS_PATH}")
    reason = why or ("new recipe beat the live one on recent games" if report["switched"]
                     else "kept the recipe, refit with the newest games")
    guard.log_run("cfb", live_recipe, report, ok, reason, new, live_score, trained_through,
                  {"holdout_validation": output["holdout_validation"],
                   "weights_before": guard.weights_snapshot(live),
                   "weights_after": guard.weights_snapshot(output if ok else live)})
    guard.summary([f"### CFB game model refit ({trained_through})",
                   f"- Recipe: {'switched to ' if report['switched'] else 'kept '}{recipe_name(chosen)}",
                   f"- Deployed: {'yes' if ok else 'no - ' + why}"])

if __name__ == "__main__":
    run()
