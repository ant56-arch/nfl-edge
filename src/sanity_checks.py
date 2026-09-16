"""
sanity_checks.py
Runs basic plausibility checks on our own output before it goes out in an
email. Catches the "something upstream broke and now every spread is 400
points" class of bug, which a crashed script wouldn't catch (the script ran
fine, it just produced nonsense).

Exits with a non-zero status if something looks broken, which fails the
GitHub Actions step and triggers the alert email instead of sending a
garbage picks email.
"""

import pandas as pd
import os
import sys

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Generous bounds - wide enough to never flag a real NFL game, tight enough
# to catch a genuinely broken calculation.
CHECKS = {
    "projected_spread": (-40, 40),
    "projected_total": (10, 90),
    "home_win_prob": (0.01, 0.99),
    "model_spread": (-40, 40),
    "model_total": (10, 90),
}

PROPS_CHECKS = {
    "proj_pass_yards": (0, 600),
    "proj_rush_yards": (0, 350),
    "proj_rec_yards": (0, 300),
}

def check_game_predictions():
    path = os.path.join(PROCESSED_DIR, "game_predictions.csv")
    if not os.path.exists(path):
        return [f"game_predictions.csv is missing entirely"]

    df = pd.read_csv(path)
    problems = []

    if df.empty:
        problems.append("game_predictions.csv has zero rows - no games to predict")
        return problems

    for col, (lo, hi) in CHECKS.items():
        if col not in df.columns:
            continue
        out_of_bounds = df[(df[col] < lo) | (df[col] > hi)]
        if not out_of_bounds.empty:
            examples = out_of_bounds[["home_team", "away_team", col]].head(3).to_dict("records")
            problems.append(f"{col} out of plausible range ({lo} to {hi}) for {len(out_of_bounds)} games. Examples: {examples}")

    return problems

def check_player_props():
    path = os.path.join(PROCESSED_DIR, "player_props.csv")
    if not os.path.exists(path):
        return [f"player_props.csv is missing entirely"]

    df = pd.read_csv(path)
    problems = []

    if df.empty:
        problems.append("player_props.csv has zero rows - no player projections generated")
        return problems

    for col, (lo, hi) in PROPS_CHECKS.items():
        if col not in df.columns:
            continue
        out_of_bounds = df[(df[col] < lo) | (df[col] > hi)]
        if not out_of_bounds.empty:
            examples = out_of_bounds[["player_name", col]].head(3).to_dict("records")
            problems.append(f"{col} out of plausible range ({lo} to {hi}) for {len(out_of_bounds)} players. Examples: {examples}")

    return problems

def main():
    print("Running sanity checks on generated predictions...")
    problems = check_game_predictions() + check_player_props()

    if problems:
        print("\nSANITY CHECK FAILURES:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)  # non-zero exit fails the GitHub Actions step

    print("All sanity checks passed.")

if __name__ == "__main__":
    main()
