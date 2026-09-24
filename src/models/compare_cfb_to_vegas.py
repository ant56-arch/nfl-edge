"""
compare_cfb_to_vegas.py
College football's version of compare_to_vegas.py - same edge-detection
logic (pure model vs. the market), pointed at the CFB prediction/odds files.
See compare_to_vegas.py's own docstring for the full rationale.

Degrades gracefully: if either input file is missing (CFBD_API_KEY not set
yet, or no odds this run), this skips rather than crashing.
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import moneyline

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

SPREAD_EDGE_THRESHOLD = 3.0
TOTAL_EDGE_THRESHOLD = 3.0
WIN_PROB_EDGE_THRESHOLD = 0.08

def next_week_only(predictions):
    if predictions.empty:
        return predictions
    next_week = predictions.sort_values(["season", "week"]).iloc[0][["season", "week"]]
    return predictions[(predictions["season"] == next_week["season"]) & (predictions["week"] == next_week["week"])]

def compare(predictions, odds):
    predictions = next_week_only(predictions)
    merged = predictions.merge(odds, on=["home_team", "away_team"], how="inner", suffixes=("", "_vegas"))
    if merged.empty:
        return merged

    merged["spread_edge"] = merged["model_spread"] - merged["vegas_home_favored_by"]
    merged["total_edge"] = merged["model_total"] - merged["total_line"]
    merged["win_prob_edge"] = merged["model_home_win_prob"] - merged["vegas_home_win_prob"]

    merged["notable_spread_edge"] = merged["spread_edge"].abs() >= SPREAD_EDGE_THRESHOLD
    merged["notable_total_edge"] = merged["total_edge"].abs() >= TOTAL_EDGE_THRESHOLD
    merged["notable_win_prob_edge"] = merged["win_prob_edge"].abs() >= WIN_PROB_EDGE_THRESHOLD
    merged["has_notable_edge"] = merged["notable_spread_edge"] | merged["notable_total_edge"] | merged["notable_win_prob_edge"]

    merged["vegas_favored_team"] = merged.apply(lambda r: r["home_team"] if r["vegas_home_favored_by"] > 0 else r["away_team"], axis=1)
    merged["model_favored_team"] = merged.apply(lambda r: r["home_team"] if r["model_spread"] > 0 else r["away_team"], axis=1)
    merged["model_favored_by"] = merged["model_spread"].abs()
    merged["picks_flip"] = merged["model_favored_team"] != merged["vegas_favored_team"]

    # Moneyline pick: the side where our pure model's win probability beats
    # the no-vig book probability by more (see src/moneyline.py). Uses the
    # h2h prices fetch_odds already pulls - no extra odds-API calls.
    merged = moneyline.add_pick_columns(merged)

    return merged

def main():
    preds_path = os.path.join(PROCESSED_DIR, "cfb_game_predictions.csv")
    odds_path = os.path.join(RAW_DIR, "cfb_odds.csv")
    if not os.path.exists(preds_path) or not os.path.exists(odds_path):
        print("CFB predictions or odds not available yet - skipping comparison.")
        return

    predictions = pd.read_csv(preds_path)
    odds = pd.read_csv(odds_path)
    if predictions.empty or odds.empty:
        print("No CFB predictions/odds to compare - skipping.")
        return

    comparison = compare(predictions, odds)
    if comparison.empty:
        print("No matching CFB games found between predictions and odds.")
        return

    out_path = os.path.join(PROCESSED_DIR, "cfb_vegas_comparison.csv")
    comparison.to_csv(out_path, index=False)
    notable = comparison[comparison["has_notable_edge"]]
    print(f"{len(comparison)} CFB games compared, {len(notable)} with a notable edge vs Vegas.")
    ml = comparison[comparison["ml_pick_side"].notna()]
    print(f"Moneyline picks: {len(ml)} games, {int(ml['ml_value'].astype(bool).sum())} flagged Value (edge >= 3 pts).")
    print(f"Saved full comparison to {out_path}")

if __name__ == "__main__":
    main()
