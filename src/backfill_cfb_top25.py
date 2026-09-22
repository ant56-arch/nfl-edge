"""
backfill_cfb_top25.py
Seeds data/tracking/cfb_top25_summary.json with Vegas's own closing-line
record (straight-up and against the spread) for the CURRENT AP Top 25
teams' games since 2024 - the CFB equivalent of what backfill_tracking.py
gave the NFL side, adapted to what's actually meaningful here: the CFB
tracker's own live history (see track_cfb_results.py) only goes back to
whenever the CFBD_API_KEY was added this season, so there's no multi-year
in-house history to show yet. This backfill instead gives the CFB Track
Record card a real historical number to show alongside that live tracking,
using the current Top 25 roster's full 2024-onward schedules.

THIS IS VEGAS'S RECORD, NOT OURS: no model is fit or evaluated here at all -
just how often the market's own closing-line favorite won straight-up and
covered the spread, for the 25 highest-ranked teams right now, going back
to the start of the 2024 season. Disclosed as such on the site, exactly
like the NFL Track Record card's own Vegas-only framing.

Run this manually and occasionally (see .github/workflows/backfill-cfb-top25.yml's
workflow_dispatch) - it re-derives its own Top 25 roster from the most
recent available poll each time, so re-running it later naturally refreshes
both the roster and the games covered. Not part of any scheduled workflow,
same as backfill_tracking.py.
"""

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fetch_cfb_data import fetch_games, fetch_lines, fetch_rankings, current_cfb_season, _headers

TRACKING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tracking")
os.makedirs(TRACKING_DIR, exist_ok=True)
SUMMARY_PATH = os.path.join(TRACKING_DIR, "cfb_top25_summary.json")

BACKFILL_START_SEASON = 2024

def current_top25():
    """The most recent AP Top 25 available - tries this season's rankings
    first (most recent week with a poll), then falls back to last season's
    final poll if the current season doesn't have one out yet (e.g. before
    week 1's poll each fall)."""
    season = current_cfb_season()
    for yr in (season, season - 1):
        weeks = fetch_rankings(yr)
        if not weeks:
            continue
        for week_entry in sorted(weeks, key=lambda w: w.get("week", 0), reverse=True):
            for poll in week_entry.get("polls", []):
                if poll.get("poll") == "AP Top 25" and poll.get("ranks"):
                    ranks = sorted(poll["ranks"], key=lambda r: r.get("rank", 99))[:25]
                    return [r["school"] for r in ranks], yr, week_entry.get("week")
    return [], None, None

def run():
    if _headers() is None:
        print("CFBD_API_KEY not set - skipping Top 25 backfill.")
        return

    top25, poll_season, poll_week = current_top25()
    if not top25:
        print("Could not find a current AP Top 25 poll - aborting.")
        return
    print(f"Top 25 as of {poll_season} week {poll_week}: {', '.join(top25)}")

    current_season = current_cfb_season()
    seasons = list(range(BACKFILL_START_SEASON, current_season + 1))
    print(f"Fetching games + lines for {seasons}...")

    frames = []
    for season in seasons:
        games = fetch_games(season)
        if games.empty:
            continue
        lines = fetch_lines(season)
        if not lines.empty:
            games = games.merge(lines, on=["home_team", "away_team"], how="left")
        frames.append(games)
    all_games = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if all_games.empty:
        print("No games returned - aborting.")
        return

    involved = all_games[all_games["home_team"].isin(top25) | all_games["away_team"].isin(top25)].copy()
    involved = involved[involved["home_score"].notna() & involved["away_score"].notna()]
    involved = involved[involved["vegas_home_favored_by"].notna()]
    print(f"  {len(involved)} completed, lined games involving a current Top 25 team since {BACKFILL_START_SEASON}")

    if involved.empty:
        print("No graded games with lines found - aborting.")
        return

    actual_margin = involved["home_score"] - involved["away_score"]
    spread = involved["vegas_home_favored_by"]

    # Straight-up: did the team Vegas favored win outright?
    picked_home_won = (spread > 0) == (actual_margin > 0)
    wins = int(picked_home_won.sum())
    losses = int(len(involved)) - wins

    # Against the spread: did the favored side win by MORE than the spread?
    cover_margin = actual_margin - spread
    push = cover_margin == 0
    home_covered = cover_margin > 0
    picked_covered = pd.Series(np.where(spread >= 0, home_covered, ~home_covered), index=involved.index)
    decided = ~push
    ats_wins = int(picked_covered[decided].sum())
    ats_losses = int(decided.sum()) - ats_wins
    ats_pushes = int(push.sum())

    spread_mae = float((spread - actual_margin).abs().mean())

    summary = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "since_year": BACKFILL_START_SEASON,
        "poll_season": poll_season,
        "poll_week": poll_week,
        "top25": top25,
        "n_games": int(len(involved)),
        "wins": wins, "losses": losses, "record": f"{wins}-{losses}",
        "pick_accuracy": round(wins / len(involved), 3),
        "ats_wins": ats_wins, "ats_losses": ats_losses, "ats_pushes": ats_pushes,
        "ats_record": f"{ats_wins}-{ats_losses}" + (f"-{ats_pushes}" if ats_pushes else ""),
        "ats_accuracy": round(ats_wins / (ats_wins + ats_losses), 3) if (ats_wins + ats_losses) > 0 else None,
        "spread_mae": round(spread_mae, 2),
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nVegas record for the current AP Top 25, {BACKFILL_START_SEASON}-now ({summary['n_games']} games):")
    print(f"  Straight-up: {summary['record']} ({summary['pick_accuracy']:.1%})")
    print(f"  ATS: {summary['ats_record']} ({summary['ats_accuracy']:.1%})")
    print(f"Saved to {SUMMARY_PATH}")

if __name__ == "__main__":
    run()
