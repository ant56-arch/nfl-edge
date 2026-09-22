"""
cfb_game_predictions.py
College football's version of game_predictions.py - same architecture (own
model anchored on team efficiency, blended with the market using weights
fit_cfb_model.py validated on held-out seasons), applied to CFBD's PPA/
success-rate/explosiveness metrics instead of nflverse's EPA-based ones, and
scoped to Power-conference + independent FBS teams only (see fetch_cfb_data.py).

No team_overrides/auto_defense_adjustments layer here - those exist on the
NFL side for known personnel changes the current season's data can't fully
see yet; college football's much larger team count and higher roster
turnover make an equivalent by-hand override list impractical for a first
pass.

Degrades gracefully at every step: if CFBD_API_KEY hasn't been added yet (so
none of the upstream cfb_*.csv files exist), this produces no predictions
rather than crashing.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import json
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
COEFFICIENTS_PATH = os.path.join(os.path.dirname(__file__), "fitted_cfb_coefficients.json")

DEFAULT_COEFFICIENTS = {
    "ppa_diff_coef": 30.0, "success_rate_weight": 10.0, "explosiveness_weight": 5.0,
    "home_field_advantage": 2.0, "margin_std_dev": 16.0,
    "total_coef": 30.0, "total_intercept": 50.0, "total_std_dev": 13.0,
}
DEFAULT_BLEND_WEIGHTS = {
    "blend_weight_on_model_spread": 1.0, "blend_weight_on_model_winprob": 1.0, "blend_weight_on_model_total": 1.0,
}

def load_coefficients():
    if not os.path.exists(COEFFICIENTS_PATH):
        print("  WARNING: fitted_cfb_coefficients.json not found - using un-backtested "
              "fallback constants. Run src/fit_cfb_model.py to fit real coefficients.")
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
    path = os.path.join(PROCESSED_DIR, "cfb_team_stats.csv")
    if not os.path.exists(path):
        return None
    stats = pd.read_csv(path)
    return stats.set_index("team") if not stats.empty else None

def load_upcoming_games():
    path = os.path.join(RAW_DIR, "cfb_games.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    games = pd.read_csv(path)
    return games[games["home_score"].isna()]

def load_vegas_odds():
    odds_path = os.path.join(RAW_DIR, "cfb_odds.csv")
    if not os.path.exists(odds_path):
        return None
    odds = pd.read_csv(odds_path)
    if odds.empty:
        return None
    return odds.set_index(["home_team", "away_team"])

def matchup_expected_efficiency(offense_row, defense_row, use_current_form=True):
    suffix = "" if use_current_form else "_season_avg"
    return (offense_row[f"off_ppa_per_play{suffix}"] + defense_row[f"def_ppa_per_play_allowed{suffix}"]) / 2

def secondary_adjustment(home_row, away_row, coefs, use_current_form=True):
    suffix = "" if use_current_form else "_season_avg"
    weights = {"off_success_rate": coefs["success_rate_weight"], "off_explosiveness": coefs["explosiveness_weight"]}
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

    ppa_diff = home_expected_eff - away_expected_eff
    base_spread = ppa_diff * coefs["ppa_diff_coef"]
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
        "home_team": home_team, "away_team": away_team, "has_market_line": has_market,
        "model_home_win_prob": round(model_home_win_prob, 3),
        "model_spread": round(model_spread, 1), "model_total": round(model_total, 1),
        "home_win_prob": round(sharp_home_win_prob, 3), "away_win_prob": round(1 - sharp_home_win_prob, 3),
        "projected_spread": round(sharp_spread, 1), "projected_total": round(sharp_total, 1),
        "favored_team": home_team if sharp_spread > 0 else away_team,
        "favored_by": round(abs(sharp_spread), 1),
    }

def predict_all_upcoming(use_current_form=True):
    coefs, blend_weights, holdout = load_coefficients()
    team_stats = load_team_stats()
    upcoming = load_upcoming_games()
    vegas_odds = load_vegas_odds()

    if team_stats is None or upcoming.empty:
        return pd.DataFrame(), holdout

    predictions = []
    for _, game in upcoming.iterrows():
        pred = predict_game(game["home_team"], game["away_team"], team_stats, coefs, blend_weights, vegas_odds, use_current_form)
        if pred:
            pred["season"] = game["season"]
            pred["week"] = game["week"]
            pred["gameday"] = game.get("gameday")
            pred["gametime"] = game.get("gametime")
            pred["weekday"] = game.get("weekday")
            pred["game_type"] = game.get("game_type")
            predictions.append(pred)

    return pd.DataFrame(predictions), holdout

def main():
    print("Loading CFB team stats, upcoming schedule, and fitted coefficients...")
    predictions, holdout = predict_all_upcoming()

    if predictions.empty:
        print("No upcoming CFB games found (or CFBD_API_KEY not set yet / no team stats).")
        return

    predictions = predictions.sort_values("favored_by", ascending=False)
    out_path = os.path.join(PROCESSED_DIR, "cfb_game_predictions.csv")
    predictions.to_csv(out_path, index=False)

    if holdout:
        print(f"\nModel calibration (holdout seasons {holdout['seasons']}, {holdout['n_games']} games):")
        print(f"  Spread MAE - model: {holdout['model_only_spread_mae']:.2f} | "
              f"blended: {holdout['blended_spread_mae']:.2f} | Vegas: {holdout['vegas_only_spread_mae']:.2f}")

    print(f"\nGenerated {len(predictions)} CFB game predictions ({predictions['has_market_line'].sum()} with a live market line).")
    print(f"Saved full results to {out_path}")

if __name__ == "__main__":
    main()
