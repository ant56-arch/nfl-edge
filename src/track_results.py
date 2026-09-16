"""
track_results.py
Keeps a running, git-committed history of every prediction we've ever sent,
and grades it the moment the actual result shows up in schedules.csv. This is
the only way to know whether changes to the model make it sharper or not -
projecting confidently and never checking the scoreboard is how the original
version of this project ended up with un-backtested constants for years.

Two responsibilities each run:
  1. SNAPSHOT - record this run's upcoming-game predictions (model-only,
     market-blended "sharp", and the Vegas line at prediction time) into
     data/tracking/predictions_log.csv, keyed by (season, week, home_team,
     away_team). Re-running later in the same week (e.g. Tue -> Fri) updates
     that game's snapshot in place, so we always keep the LAST snapshot
     before kickoff - the fairest comparison against Vegas' closing line.
  2. GRADE - for any previously-logged game whose actual result has since
     appeared in schedules.csv, fill in the outcome and compute per-game
     error metrics for the model, the blended "sharp" line, and Vegas itself,
     so accuracy_summary.json can show head-to-head whether we're closing
     the gap on the market or not.

data/tracking/ is committed to git (unlike data/raw/ and data/processed/,
which are regenerated fresh every run) specifically so this history survives
the ephemeral GitHub Actions runner - see the workflow's "commit tracking
data" step.
"""

import pandas as pd
import numpy as np
import json
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
TRACKING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tracking")
os.makedirs(TRACKING_DIR, exist_ok=True)

LOG_PATH = os.path.join(TRACKING_DIR, "predictions_log.csv")
SUMMARY_PATH = os.path.join(TRACKING_DIR, "accuracy_summary.json")

KEY_COLS = ["season", "week", "home_team", "away_team"]
LAST_N_WEEKS = 4

def load_predictions_snapshot():
    games = pd.read_csv(os.path.join(PROCESSED_DIR, "game_predictions.csv"))

    vegas_cols = ["home_team", "away_team", "vegas_home_favored_by", "total_line", "vegas_home_win_prob"]
    comparison_path = os.path.join(PROCESSED_DIR, "vegas_comparison.csv")
    if os.path.exists(comparison_path):
        comparison = pd.read_csv(comparison_path)
        comparison = comparison[[c for c in vegas_cols if c in comparison.columns]].drop_duplicates(["home_team", "away_team"])
        games = games.merge(comparison, on=["home_team", "away_team"], how="left")
    for c in vegas_cols[2:]:
        if c not in games.columns:
            games[c] = np.nan

    snapshot = games[KEY_COLS + [
        "gameday", "model_spread", "model_total", "model_home_win_prob",
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
            log[col] = np.nan
        log.loc[snapshot_indexed.index, col] = snapshot_indexed[col]
    return log.reset_index()

def grade_completed_games(log):
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    results = schedules[schedules["result"].notna()][KEY_COLS + ["home_score", "away_score"]]

    log = log.drop(columns=[c for c in ["home_score", "away_score"] if c in log.columns])
    log = log.merge(results, on=KEY_COLS, how="left")

    graded = log["home_score"].notna()
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

    return log

def summarize(log):
    graded = log[log["actual_margin"].notna()].copy()
    if graded.empty:
        return {"n_graded_games": 0}

    graded = graded.sort_values(["season", "week"])
    distinct_weeks = graded[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    recent_weeks = pd.MultiIndex.from_frame(distinct_weeks.tail(LAST_N_WEEKS))
    recent_mask = pd.MultiIndex.from_frame(graded[["season", "week"]]).isin(recent_weeks)

    summary = {"n_graded_games": int(len(graded)), "last_updated": pd.Timestamp.now("UTC").isoformat()}
    windows = {"season_to_date": graded, f"last_{LAST_N_WEEKS}_weeks": graded[recent_mask]}

    for window_name, window_df in windows.items():
        if window_df.empty:
            continue
        window_summary = {}
        for label in ["model", "sharp", "vegas"]:
            wins = int(window_df[f"{label}_correct_pick"].sum())
            losses = int(len(window_df)) - wins
            window_summary[label] = {
                "wins": wins,
                "losses": losses,
                "record": f"{wins}-{losses}",
                "pick_accuracy": round(float(window_df[f"{label}_correct_pick"].mean()), 3),
                "spread_mae": round(float(window_df[f"{label}_spread_error"].mean()), 2),
                "total_mae": round(float(window_df[f"{label}_total_error"].mean()), 2),
                "brier_score": round(float(window_df[f"{label}_brier"].mean()), 4),
                "n_games": int(len(window_df)),
            }
        summary[window_name] = window_summary
    return summary

def main():
    print("Loading this run's predictions to snapshot...")
    snapshot = load_predictions_snapshot()

    log = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else None
    log = upsert_snapshot(log, snapshot)
    print(f"  Tracking log now has {len(log)} predicted games total")

    print("Grading any games whose results are now in...")
    log = grade_completed_games(log)
    print(f"  {int(log['actual_margin'].notna().sum())} games have a graded actual result")

    log.to_csv(LOG_PATH, index=False)

    summary = summarize(log)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    if summary.get("n_graded_games", 0) > 0:
        std = summary.get("season_to_date", {})
        print("\nSeason-to-date accuracy (us vs the market):")
        for label in ["model", "sharp", "vegas"]:
            s = std.get(label, {})
            if s:
                print(f"  {label:>6}: {s['record']} ({s['pick_accuracy']:.1%}) straight-up | "
                      f"spread MAE {s['spread_mae']:.2f} | total MAE {s['total_mae']:.2f} | Brier {s['brier_score']:.4f}")

    print(f"\nSaved tracking log to {LOG_PATH} and summary to {SUMMARY_PATH}")

if __name__ == "__main__":
    main()
