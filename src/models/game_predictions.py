"""
game_predictions.py
Predicts game winners, spreads, and totals for upcoming NFL games.

APPROACH:
Anchored on net EPA/play (expected points added), the strongest single
team-strength metric in modern NFL analytics. Secondary factors (third-down
rate, red zone efficiency, explosiveness, sack rate) act as smaller
adjustments layered on top.

MATCHUP LOGIC (not just raw team strength):
Each team's expected efficiency in THIS game = average of (their own
offensive efficiency) and (their opponent's defensive efficiency allowed).

MANUAL OVERRIDES: before computing anything, team_stats gets adjusted by
team_overrides.py and auto_defense_adjustments.py for personnel changes
this season's data can't fully see yet.

CALIBRATION: every point/probability conversion below (EPA-to-points scaling,
spread-to-win% curve, home field edge, secondary-factor weights) is a
coefficient regressed against this project's own historical outcomes by
fit_model.py, not a hand-picked guess - see src/models/fitted_coefficients.json
and that script's holdout-validation report for how well it actually performs.
If that file is missing (e.g. first run before fit_model.py has ever been
run), we fall back to the original hand-picked constants so the pipeline
still produces something, but sharpness depends on running the refit.

MARKET BLENDING: the market (Vegas closing lines) is, empirically, very hard
to beat outright. Rather than pretend our standalone model is the sharpest
number in the building, we blend it with the current market line using
weights fit_model.py validated on held-out seasons to minimize error. That
blended "sharp" line is the headline prediction; the pure, unblended model
output is kept alongside it specifically so compare_to_vegas.py can still
flag real model/market disagreements (a blended number can't disagree with
itself).
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from team_overrides import apply_overrides
from auto_defense_adjustments import apply_auto_adjustments

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
COEFFICIENTS_PATH = os.path.join(os.path.dirname(__file__), "fitted_coefficients.json")

# Fallback constants, used ONLY if fitted_coefficients.json doesn't exist yet
# (run src/fit_model.py to generate it). These are the original hand-picked
# starting points - kept purely so the pipeline degrades gracefully instead
# of crashing, not because they're a good source of sharpness.
DEFAULT_COEFFICIENTS = {
    "epa_diff_coef": 65.0,
    "third_down_weight": 8.0,
    "redzone_weight": 6.0,
    "explosive_weight": 10.0,
    "sack_weight": -6.0,
    "home_field_advantage": 1.5,
    "margin_std_dev": 13.5,
    "total_coef": 65.0,
    "total_intercept": 45.0,
    "total_std_dev": 10.0,
}
DEFAULT_BLEND_WEIGHTS = {
    "blend_weight_on_model_spread": 1.0,
    "blend_weight_on_model_winprob": 1.0,
    "blend_weight_on_model_total": 1.0,
}

def load_coefficients():
    if not os.path.exists(COEFFICIENTS_PATH):
        print("  WARNING: fitted_coefficients.json not found - using un-backtested "
              "fallback constants. Run src/fit_model.py to fit real coefficients.")
        return DEFAULT_COEFFICIENTS, DEFAULT_BLEND_WEIGHTS, None
    with open(COEFFICIENTS_PATH) as f:
        fitted = json.load(f)
    coefs = fitted["coefficients"]
    blend = {
        "blend_weight_on_model_spread": fitted.get("blend_weight_on_model_spread", 1.0),
        "blend_weight_on_model_winprob": fitted.get("blend_weight_on_model_winprob", 1.0),
        "blend_weight_on_model_total": fitted.get("blend_weight_on_model_total", 1.0),
    }
    return coefs, blend, fitted.get("holdout_validation")

def load_team_stats():
    team_stats = pd.read_csv(os.path.join(PROCESSED_DIR, "team_stats.csv")).set_index("team")
    team_stats = apply_overrides(team_stats)
    team_stats = apply_auto_adjustments(team_stats)
    return team_stats

def load_upcoming_games():
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    upcoming = schedules[schedules["result"].isna()]
    return upcoming

def load_vegas_odds():
    odds_path = os.path.join(RAW_DIR, "odds.csv")
    if not os.path.exists(odds_path):
        return None
    odds = pd.read_csv(odds_path)
    if odds.empty:
        return None
    return odds.set_index(["home_team", "away_team"])

def matchup_expected_efficiency(offense_row, defense_row, use_current_form=True):
    suffix = "" if use_current_form else "_season_avg"
    own_offense = offense_row[f"epa_per_play{suffix}"]
    opp_defense_allowed = defense_row[f"def_epa_per_play_allowed{suffix}"]
    return (own_offense + opp_defense_allowed) / 2

def secondary_adjustment(home_row, away_row, coefs, use_current_form=True):
    suffix = "" if use_current_form else "_season_avg"
    weights = {
        "third_down_rate": coefs["third_down_weight"],
        "redzone_td_rate": coefs["redzone_weight"],
        "explosive_rate": coefs["explosive_weight"],
        "sack_rate": coefs["sack_weight"],
    }
    adjustment = 0.0
    for factor, weight in weights.items():
        home_val = home_row.get(f"{factor}{suffix}", np.nan)
        away_val = away_row.get(f"{factor}{suffix}", np.nan)
        if pd.notna(home_val) and pd.notna(away_val):
            adjustment += (home_val - away_val) * weight
    return adjustment

def get_vegas_line(vegas_odds, home_team, away_team):
    if vegas_odds is None or (home_team, away_team) not in vegas_odds.index:
        return None
    row = vegas_odds.loc[(home_team, away_team)]
    return {
        "home_favored_by": row.get("vegas_home_favored_by"),
        "total": row.get("total_line"),
        "home_win_prob": row.get("vegas_home_win_prob"),
    }

def predict_game(home_team, away_team, team_stats, coefs, blend_weights, vegas_odds=None, use_current_form=True):
    if home_team not in team_stats.index or away_team not in team_stats.index:
        return None

    home_row = team_stats.loc[home_team]
    away_row = team_stats.loc[away_team]

    home_expected_eff = matchup_expected_efficiency(home_row, away_row, use_current_form)
    away_expected_eff = matchup_expected_efficiency(away_row, home_row, use_current_form)

    epa_diff = home_expected_eff - away_expected_eff
    base_spread = epa_diff * coefs["epa_diff_coef"]
    adjustment = secondary_adjustment(home_row, away_row, coefs, use_current_form)

    model_spread = base_spread + adjustment + coefs["home_field_advantage"]
    model_home_win_prob = norm.cdf(model_spread / coefs["margin_std_dev"])

    combined_expected_eff = home_expected_eff + away_expected_eff
    model_total = coefs["total_intercept"] + (combined_expected_eff * coefs["total_coef"])

    vegas = get_vegas_line(vegas_odds, home_team, away_team)
    has_market = vegas is not None and pd.notna(vegas.get("home_favored_by"))

    if has_market:
        w_spread = blend_weights["blend_weight_on_model_spread"]
        sharp_spread = w_spread * model_spread + (1 - w_spread) * vegas["home_favored_by"]

        w_total = blend_weights["blend_weight_on_model_total"]
        sharp_total = (w_total * model_total + (1 - w_total) * vegas["total"]
                       if pd.notna(vegas.get("total")) else model_total)

        w_wp = blend_weights["blend_weight_on_model_winprob"]
        sharp_home_win_prob = (w_wp * model_home_win_prob + (1 - w_wp) * vegas["home_win_prob"]
                                if pd.notna(vegas.get("home_win_prob")) else model_home_win_prob)
    else:
        sharp_spread, sharp_total, sharp_home_win_prob = model_spread, model_total, model_home_win_prob

    return {
        "home_team": home_team,
        "away_team": away_team,
        "has_market_line": has_market,
        "model_home_win_prob": round(model_home_win_prob, 3),
        "model_spread": round(model_spread, 1),
        "model_total": round(model_total, 1),
        "home_win_prob": round(sharp_home_win_prob, 3),
        "away_win_prob": round(1 - sharp_home_win_prob, 3),
        "projected_spread": round(sharp_spread, 1),
        "projected_total": round(sharp_total, 1),
        "favored_team": home_team if sharp_spread > 0 else away_team,
        "favored_by": round(abs(sharp_spread), 1),
    }

def predict_all_upcoming(use_current_form=True):
    coefs, blend_weights, holdout = load_coefficients()
    team_stats = load_team_stats()
    upcoming = load_upcoming_games()
    vegas_odds = load_vegas_odds()

    if vegas_odds is None:
        print("  No Vegas odds available yet - predictions will be model-only "
              "(run fetch_odds.py before this step to enable market blending).")

    predictions = []
    for _, game in upcoming.iterrows():
        pred = predict_game(game["home_team"], game["away_team"], team_stats, coefs, blend_weights, vegas_odds, use_current_form)
        if pred:
            pred["season"] = game["season"]
            pred["week"] = game["week"]
            pred["gameday"] = game["gameday"]
            pred["gametime"] = game.get("gametime")
            pred["weekday"] = game.get("weekday")
            pred["game_type"] = game.get("game_type")
            predictions.append(pred)

    return pd.DataFrame(predictions), holdout

def main():
    print("Loading team stats, upcoming schedule, and fitted coefficients...")
    predictions, holdout = predict_all_upcoming()

    if predictions.empty:
        print("No upcoming games found.")
        return

    predictions = predictions.sort_values("favored_by", ascending=False)
    out_path = os.path.join(PROCESSED_DIR, "game_predictions.csv")
    predictions.to_csv(out_path, index=False)

    if holdout:
        print(f"\nModel calibration (holdout seasons {holdout['seasons']}, {holdout['n_games']} games):")
        print(f"  Spread MAE  - model: {holdout['model_only_spread_mae']:.2f} pts | "
              f"blended: {holdout['blended_spread_mae']:.2f} pts | Vegas: {holdout['vegas_only_spread_mae']:.2f} pts")
        print(f"  Win Brier   - model: {holdout['model_only_brier']:.4f} | "
              f"blended: {holdout['blended_brier']:.4f} | Vegas: {holdout['vegas_only_brier']:.4f}")

    print(f"\nGenerated {len(predictions)} game predictions "
          f"({predictions['has_market_line'].sum()} with a live market line to blend against).")
    print("\nTop 5 most confident predictions:")
    print(predictions[["week", "home_team", "away_team", "favored_team", "favored_by", "home_win_prob", "projected_total"]].head(5).to_string(index=False))
    print(f"\nSaved full results to {out_path}")

if __name__ == "__main__":
    main()
