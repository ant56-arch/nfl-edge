"""
game_predictions.py
Predicts game winners, spreads, and totals for upcoming NFL games.

APPROACH:
Anchored on net EPA/play (expected points added), the strongest single
team-strength metric in modern NFL analytics — it captures scoring efficiency
better than yards, and correlates with wins far more reliably than counting
stats. Secondary factors (third-down rate, red zone efficiency, explosiveness,
pressure rate) act as smaller adjustments layered on top.

MATCHUP LOGIC (not just raw team strength):
Each team's expected efficiency in THIS game = average of (their own
offensive efficiency) and (their opponent's defensive efficiency allowed).
An elite offense facing a weak defense should be expected to outperform its
season-long average, and vice versa.

MANUAL OVERRIDES: before computing anything, team_stats gets adjusted by
team_overrides.py for known offseason changes (trades, major signings) that
2024-2025 data can't see. See that file to add/edit adjustments.

CALIBRATION NOTE (read this):
The point/probability conversions below (EPA-to-points scaling, spread-to-win%
curve, home field edge) are reasonable industry-standard approximations, NOT
yet backtested against our own data.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from team_overrides import apply_overrides
from auto_defense_adjustments import apply_auto_adjustments

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

HOME_FIELD_ADVANTAGE_POINTS = 1.5
AVG_PLAYS_PER_TEAM_PER_GAME = 65
NFL_MARGIN_STD_DEV = 13.5
LEAGUE_AVG_TOTAL_POINTS = 45.0

SECONDARY_WEIGHTS = {
    "third_down_rate": 8.0,
    "redzone_td_rate": 6.0,
    "explosive_rate": 10.0,
    "sack_rate": -6.0,
}

def load_team_stats():
    team_stats = pd.read_csv(os.path.join(PROCESSED_DIR, "team_stats.csv")).set_index("team")
    team_stats = apply_overrides(team_stats)
    team_stats = apply_auto_adjustments(team_stats)
    return team_stats

def load_upcoming_games():
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    upcoming = schedules[schedules["result"].isna()]
    return upcoming

def matchup_expected_efficiency(offense_row, defense_row, use_current_form=True):
    suffix = "" if use_current_form else "_season_avg"
    own_offense = offense_row[f"epa_per_play{suffix}"]
    opp_defense_allowed = defense_row[f"def_epa_per_play_allowed{suffix}"]
    return (own_offense + opp_defense_allowed) / 2

def secondary_adjustment(home_row, away_row, use_current_form=True):
    suffix = "" if use_current_form else "_season_avg"
    adjustment = 0.0
    for factor, weight in SECONDARY_WEIGHTS.items():
        home_val = home_row.get(f"{factor}{suffix}", np.nan)
        away_val = away_row.get(f"{factor}{suffix}", np.nan)
        if pd.notna(home_val) and pd.notna(away_val):
            adjustment += (home_val - away_val) * weight
    return adjustment

def predict_game(home_team, away_team, team_stats, use_current_form=True):
    if home_team not in team_stats.index or away_team not in team_stats.index:
        return None

    home_row = team_stats.loc[home_team]
    away_row = team_stats.loc[away_team]

    home_expected_eff = matchup_expected_efficiency(home_row, away_row, use_current_form)
    away_expected_eff = matchup_expected_efficiency(away_row, home_row, use_current_form)

    epa_diff = home_expected_eff - away_expected_eff
    base_spread = epa_diff * AVG_PLAYS_PER_TEAM_PER_GAME

    adjustment = secondary_adjustment(home_row, away_row, use_current_form)

    projected_spread = base_spread + adjustment + HOME_FIELD_ADVANTAGE_POINTS
    home_win_prob = norm.cdf(projected_spread / NFL_MARGIN_STD_DEV)

    combined_expected_eff = home_expected_eff + away_expected_eff
    projected_total = LEAGUE_AVG_TOTAL_POINTS + (combined_expected_eff * AVG_PLAYS_PER_TEAM_PER_GAME)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": round(home_win_prob, 3),
        "away_win_prob": round(1 - home_win_prob, 3),
        "projected_spread": round(projected_spread, 1),
        "projected_total": round(projected_total, 1),
        "favored_team": home_team if projected_spread > 0 else away_team,
        "favored_by": round(abs(projected_spread), 1),
    }

def predict_all_upcoming(use_current_form=True):
    team_stats = load_team_stats()
    upcoming = load_upcoming_games()

    predictions = []
    for _, game in upcoming.iterrows():
        pred = predict_game(game["home_team"], game["away_team"], team_stats, use_current_form)
        if pred:
            pred["season"] = game["season"]
            pred["week"] = game["week"]
            pred["gameday"] = game["gameday"]
            predictions.append(pred)

    return pd.DataFrame(predictions)

def main():
    print("Loading team stats and upcoming schedule...")
    predictions = predict_all_upcoming()

    if predictions.empty:
        print("No upcoming games found.")
        return

    predictions = predictions.sort_values("favored_by", ascending=False)
    out_path = os.path.join(PROCESSED_DIR, "game_predictions.csv")
    predictions.to_csv(out_path, index=False)

    print(f"\nGenerated {len(predictions)} game predictions.")
    print("\nTop 5 most confident predictions:")
    print(predictions[["week", "home_team", "away_team", "favored_team", "favored_by", "home_win_prob", "projected_total"]].head(5).to_string(index=False))
    print(f"\nSaved full results to {out_path}")

if __name__ == "__main__":
    main()
