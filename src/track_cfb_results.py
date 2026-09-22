"""
track_cfb_results.py
College football's version of track_results.py - same snapshot-then-grade
architecture, same ATS logic, pointed at the CFB-specific data files instead
of the NFL ones. See track_results.py's own docstring for the full
rationale; this file intentionally mirrors its logic closely rather than
trying to share code across sports, since a shared abstraction over already
dtype-sensitive pandas code (see upsert_snapshot's LossySetitemError history)
is a good way to reintroduce that exact class of bug for BOTH sports at once.

Degrades gracefully: if cfb_game_predictions.csv doesn't exist yet (CFBD_API_KEY
not configured, or no games predicted this run), this exits without writing
anything rather than crashing the pipeline.
"""

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fetch_cfb_data import current_cfb_season

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TRACKING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tracking")
os.makedirs(TRACKING_DIR, exist_ok=True)

LOG_PATH = os.path.join(TRACKING_DIR, "cfb_predictions_log.csv")
SUMMARY_PATH = os.path.join(TRACKING_DIR, "cfb_accuracy_summary.json")

KEY_COLS = ["season", "week", "home_team", "away_team"]
LAST_N_WEEKS = 4

def load_predictions_snapshot():
    preds_path = os.path.join(PROCESSED_DIR, "cfb_game_predictions.csv")
    if not os.path.exists(preds_path):
        return None
    games = pd.read_csv(preds_path)
    if games.empty:
        return None

    vegas_cols = ["home_team", "away_team", "vegas_home_favored_by", "total_line", "vegas_home_win_prob"]
    comparison_path = os.path.join(PROCESSED_DIR, "cfb_vegas_comparison.csv")
    if os.path.exists(comparison_path):
        comparison = pd.read_csv(comparison_path)
        comparison = comparison[[c for c in vegas_cols if c in comparison.columns]].drop_duplicates(["home_team", "away_team"])
        games = games.merge(comparison, on=["home_team", "away_team"], how="left")
    for c in vegas_cols[2:]:
        if c not in games.columns:
            games[c] = np.nan

    if "game_type" not in games.columns:
        games["game_type"] = np.nan

    snapshot = games[KEY_COLS + [
        "gameday", "game_type", "model_spread", "model_total", "model_home_win_prob",
        "projected_spread", "projected_total", "home_win_prob",
        "vegas_home_favored_by", "total_line", "vegas_home_win_prob",
    ]].copy()
    return snapshot.rename(columns={
        "projected_spread": "sharp_spread", "projected_total": "sharp_total", "home_win_prob": "sharp_home_win_prob",
        "total_line": "vegas_total",
    })

def upsert_snapshot(log, snapshot):
    if log is None or log.empty:
        return snapshot

    log = log.set_index(KEY_COLS)
    snapshot_indexed = snapshot.set_index(KEY_COLS)
    log = log.reindex(log.index.union(snapshot_indexed.index))
    for col in snapshot_indexed.columns:
        if col not in log.columns:
            log[col] = pd.Series(index=log.index, dtype=snapshot_indexed[col].dtype)
        log.loc[snapshot_indexed.index, col] = snapshot_indexed[col]
    return log.reset_index()

def grade_completed_games(log):
    games_path = os.path.join(RAW_DIR, "cfb_games.csv")
    if not os.path.exists(games_path):
        return log
    schedules = pd.read_csv(games_path)
    results = schedules[schedules["home_score"].notna() & schedules["away_score"].notna()][KEY_COLS + ["home_score", "away_score"]]

    log = log.drop(columns=[c for c in ["home_score", "away_score"] if c in log.columns])
    log = log.merge(results, on=KEY_COLS, how="left")

    graded = log["home_score"].notna()
    if not graded.any():
        # Nothing graded yet (true for every run until this week's games
        # actually finish, since tracking only just started). Bail out here
        # rather than let the loop below run its .loc[graded, ...] assignments
        # against an all-False mask - pandas' empty-selection assignment into
        # a column reloaded from CSV with a different dtype than the fresh
        # in-memory value can raise (TypeError: Invalid value '[]' for dtype
        # ...) even though logically nothing needs to change, since it goes
        # through the same dtype-compatibility check as a real assignment.
        return log
    log.loc[graded, "actual_margin"] = log.loc[graded, "home_score"] - log.loc[graded, "away_score"]
    log.loc[graded, "actual_total"] = log.loc[graded, "home_score"] + log.loc[graded, "away_score"]
    home_won = (log["actual_margin"] > 0).astype(float)

    for label, spread_col, total_col, wp_col in [
        ("model", "model_spread", "model_total", "model_home_win_prob"),
        ("sharp", "sharp_spread", "sharp_total", "sharp_home_win_prob"),
        ("vegas", "vegas_home_favored_by", "vegas_total", "vegas_home_win_prob"),
    ]:
        picked_home_won = (log[spread_col] > 0) == (log["actual_margin"] > 0)
        log.loc[graded, f"{label}_correct_pick"] = picked_home_won[graded].astype(float)
        log.loc[graded, f"{label}_spread_error"] = (log[spread_col] - log["actual_margin"]).abs()[graded]
        log.loc[graded, f"{label}_total_error"] = (log[total_col] - log["actual_total"]).abs()[graded]
        log.loc[graded, f"{label}_brier"] = ((log[wp_col] - home_won) ** 2)[graded]

        cover_margin = log["actual_margin"] - log[spread_col]
        push = cover_margin == 0
        home_covered = cover_margin > 0
        picked_covered = pd.Series(np.where(log[spread_col] >= 0, home_covered, ~home_covered), index=log.index)
        log.loc[graded, f"{label}_ats_push"] = push[graded]
        decided = graded & ~push
        if decided.any():
            log.loc[decided, f"{label}_ats_correct"] = picked_covered[decided].astype(float)

    return log

def summarize(log):
    graded = log[log["actual_margin"].notna()].copy()
    if graded.empty:
        return {"n_graded_games": 0}

    graded = graded.sort_values(["season", "week"])
    current_season = current_cfb_season()
    this_season = graded[graded["season"] == current_season]

    distinct_weeks = this_season[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    recent_weeks = pd.MultiIndex.from_frame(distinct_weeks.tail(LAST_N_WEEKS))
    recent_mask = pd.MultiIndex.from_frame(this_season[["season", "week"]]).isin(recent_weeks)

    summary = {
        "n_graded_games": int(len(graded)),
        "current_season_year": current_season,
        "last_updated": pd.Timestamp.now("UTC").isoformat(),
    }
    windows = {
        "all_time": graded,
        "current_season": this_season,
        f"last_{LAST_N_WEEKS}_weeks": this_season[recent_mask],
    }

    for window_name, window_df in windows.items():
        if window_df.empty:
            continue
        window_summary = {}
        for label in ["model", "sharp", "vegas"]:
            wins = int(window_df[f"{label}_correct_pick"].sum())
            losses = int(len(window_df)) - wins

            ats_col = f"{label}_ats_correct"
            push_col = f"{label}_ats_push"
            ats_decided = window_df[ats_col].notna() if ats_col in window_df.columns else pd.Series(False, index=window_df.index)
            ats_wins = int(window_df.loc[ats_decided, ats_col].sum())
            ats_losses = int(ats_decided.sum()) - ats_wins
            ats_pushes = int(window_df[push_col].sum()) if push_col in window_df.columns else 0
            ats_record = f"{ats_wins}-{ats_losses}" + (f"-{ats_pushes}" if ats_pushes else "")

            window_summary[label] = {
                "wins": wins, "losses": losses, "record": f"{wins}-{losses}",
                "pick_accuracy": round(float(window_df[f"{label}_correct_pick"].mean()), 3),
                "spread_mae": round(float(window_df[f"{label}_spread_error"].mean()), 2),
                "total_mae": round(float(window_df[f"{label}_total_error"].mean()), 2),
                "brier_score": round(float(window_df[f"{label}_brier"].mean()), 4),
                "n_games": int(len(window_df)),
                "ats_wins": ats_wins, "ats_losses": ats_losses, "ats_pushes": ats_pushes,
                "ats_record": ats_record,
                "ats_accuracy": round(ats_wins / (ats_wins + ats_losses), 3) if (ats_wins + ats_losses) > 0 else None,
            }
        summary[window_name] = window_summary
    return summary

def main():
    print("Loading this run's CFB predictions to snapshot...")
    snapshot = load_predictions_snapshot()
    if snapshot is None:
        print("  No CFB predictions available yet (CFBD_API_KEY not set, or no games this run) - skipping.")
        return

    log = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else None
    log = upsert_snapshot(log, snapshot)
    print(f"  CFB tracking log now has {len(log)} predicted games total")

    print("Grading any CFB games whose results are now in...")
    log = grade_completed_games(log)
    print(f"  {int(log['actual_margin'].notna().sum())} games have a graded actual result")

    log.to_csv(LOG_PATH, index=False)

    summary = summarize(log)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved CFB tracking log to {LOG_PATH} and summary to {SUMMARY_PATH}")

if __name__ == "__main__":
    main()
