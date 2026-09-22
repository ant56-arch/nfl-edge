"""
build_cfb_features.py
Turns CFBD's per-game advanced stats (data/raw/cfb_advanced_stats.csv) into
recency-weighted "current form" team stats - the college football analog of
build_features.py, reusing its exact recency-weighting logic
(apply_recency_weighting) since the math is identical, just applied to a
different metric set.

CFBD's advanced-stats endpoint already gives both sides (offense AND
defense) in one row per team per game, unlike nflverse's raw play-by-play
which needs separate offense/defense aggregation passes first - so there's
no equivalent of build_team_game_stats/build_team_defense_game_stats here,
just the recency-weighting step directly on top of what CFBD already
computed.

Input:  data/raw/cfb_advanced_stats.csv
Output: data/processed/cfb_team_stats.csv
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from build_features import apply_recency_weighting, RECENCY_HALF_LIFE_GAMES

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

METRICS = ["off_ppa_per_play", "off_success_rate", "off_explosiveness",
           "def_ppa_per_play_allowed", "def_success_rate_allowed", "def_explosiveness_allowed"]

def main():
    path = os.path.join(RAW_DIR, "cfb_advanced_stats.csv")
    if not os.path.exists(path):
        print("No cfb_advanced_stats.csv found (CFBD_API_KEY likely not set yet) - skipping.")
        return

    game_stats = pd.read_csv(path)
    if game_stats.empty:
        print("cfb_advanced_stats.csv is empty - skipping.")
        return

    print("Applying recency weighting (current form vs season-long)...")
    team_stats = apply_recency_weighting(game_stats, "team", METRICS, half_life=RECENCY_HALF_LIFE_GAMES)
    team_stats.to_csv(os.path.join(PROCESSED_DIR, "cfb_team_stats.csv"), index=False)
    print(f"  Saved cfb_team_stats.csv ({len(team_stats)} teams)")

if __name__ == "__main__":
    main()
