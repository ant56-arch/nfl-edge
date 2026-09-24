"""
compare_to_vegas.py
Joins our own model's projections against real sportsbook lines and flags
games/props where we meaningfully disagree with the market - these are the
"value" spots worth a second look.

IMPORTANT: edges are computed from the PURE, unblended model output
(model_spread / model_total / model_home_win_prob), not the headline
"projected_spread" shown elsewhere - that number is already blended with
the market (see game_predictions.py), so comparing it back against the
market would trivially show almost no disagreement. The pure model is what
can actually disagree with Vegas; the blended number is what's most accurate.

EDGE DEFINITIONS:
  spread_edge = our_model_home_favor - vegas_home_favor
    Positive = we like the home team MORE than Vegas does.
  total_edge = our_model_total - vegas_total
    Positive = we expect a higher-scoring game than the market does.
  win_prob_edge = our_model_home_win_prob - vegas_home_win_prob (de-vigged)
    Positive = we're more confident in the home team than the market is.

THRESHOLDS (starting points, adjust as you get a feel for the model):
  A 3+ point spread edge or 3+ point total edge is generally considered
  notable in NFL betting circles. An 8+ percentage point win-prob edge is
  a meaningful model/market disagreement.
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

def load_data():
    predictions = pd.read_csv(os.path.join(PROCESSED_DIR, "game_predictions.csv"))
    odds = pd.read_csv(os.path.join(RAW_DIR, "odds.csv"))
    return predictions, odds

def next_week_only(predictions):
    """Match build_site.py's logic: only compare the soonest upcoming week,
    not the entire season's worth of games."""
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

    merged["has_notable_edge"] = (
        merged["notable_spread_edge"] | merged["notable_total_edge"] | merged["notable_win_prob_edge"]
    )

    # Readable pick strings, and whether we disagree on who's even favored
    # (a bigger deal than just disagreeing on the margin). Uses the PURE
    # model pick, not the blended "sharp" one - the blended line is designed
    # to hug the market, so it's rarely the side that actually disagrees.
    merged["vegas_favored_team"] = merged.apply(
        lambda r: r["home_team"] if r["vegas_home_favored_by"] > 0 else r["away_team"], axis=1
    )
    merged["model_favored_team"] = merged.apply(
        lambda r: r["home_team"] if r["model_spread"] > 0 else r["away_team"], axis=1
    )
    merged["model_favored_by"] = merged["model_spread"].abs()
    merged["picks_flip"] = merged["model_favored_team"] != merged["vegas_favored_team"]

    # Moneyline pick: the side where our pure model's win probability beats
    # the no-vig book probability by more (see src/moneyline.py). Uses the
    # h2h prices fetch_odds already pulls - no extra odds-API calls.
    merged = moneyline.add_pick_columns(merged)

    return merged

def main():
    print("Loading predictions and odds...")
    predictions, odds = load_data()

    if odds.empty:
        print("No odds data available - skipping comparison.")
        return

    print("Comparing model vs Vegas...")
    comparison = compare(predictions, odds)

    if comparison.empty:
        print("No matching games found between predictions and odds (check team name mapping or schedules).")
        return

    out_path = os.path.join(PROCESSED_DIR, "vegas_comparison.csv")
    comparison.to_csv(out_path, index=False)

    notable = comparison[comparison["has_notable_edge"]].sort_values("spread_edge", key=abs, ascending=False)
    print(f"\n{len(comparison)} games compared, {len(notable)} with a notable edge vs Vegas.")

    if not notable.empty:
        print("\nNotable disagreements with the market:")
        cols = ["home_team", "away_team", "model_spread", "vegas_home_favored_by", "spread_edge",
                "model_total", "total_line", "total_edge", "model_home_win_prob", "vegas_home_win_prob", "win_prob_edge"]
        print(notable[cols].round(3).to_string(index=False))

    ml = comparison[comparison["ml_pick_side"].notna()]
    print(f"\nMoneyline picks: {len(ml)} games, {int(ml['ml_value'].astype(bool).sum())} flagged Value (edge >= 6 pts).")
    print(f"\nSaved full comparison to {out_path}")

if __name__ == "__main__":
    main()
