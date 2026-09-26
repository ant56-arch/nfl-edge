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
import games as espn
import moneyline

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

def load_moneyline_snapshot():
    """This run's moneyline picks (see src/moneyline.py), keyed like the log.
    Written into the log by moneyline.lock_and_merge, which skips any game
    whose kickoff has already passed - the pick and price lock at kickoff."""
    path = os.path.join(PROCESSED_DIR, "cfb_vegas_comparison.csv")
    if not os.path.exists(path):
        return None
    comparison = pd.read_csv(path)
    if comparison.empty or "ml_pick_side" not in comparison.columns:
        return None
    return comparison[[c for c in KEY_COLS + moneyline.ML_COLS if c in comparison.columns]].drop_duplicates(KEY_COLS)

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

def espn_final_scores(log, scored):
    """Final scores from ESPN's scoreboard for logged games CFBD hasn't scored
    yet. CFBD can take hours (sometimes until the next day) to post a result,
    while ESPN has it the moment a game ends, so on a college football
    Saturday this is what lets results show up during the day. Keyed like
    the log; returns an empty frame when ESPN is unreachable."""
    cols = KEY_COLS + ["home_score", "away_score"]
    if log is None or log.empty or "gameday" not in log.columns:
        return pd.DataFrame(columns=cols)
    today = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    pending = log[log["gameday"].notna() & (log["gameday"].astype(str) <= today)]
    def key(r):
        return (int(r["season"]), int(r["week"]), r["home_team"], r["away_team"])
    have = {key(r) for _, r in scored.iterrows()}
    pending = pending[[key(r) not in have for _, r in pending.iterrows()]]
    if pending.empty:
        return pd.DataFrame(columns=cols)

    known = sorted(set(log["home_team"]) | set(log["away_team"]), key=len, reverse=True)

    def school(team):
        if team.get("location") in known:
            return team["location"]
        return next((k for k in known if team.get("name", "").startswith(k)), None)

    finals = {}
    for day in sorted(pending["gameday"].astype(str).unique()):
        params = dict(espn.SPORTS["cfb"]["params"], dates=day.replace("-", ""))
        data = espn._get(espn.ESPN + espn.SPORTS["cfb"]["path"] + "/scoreboard", params)
        for event in (data or {}).get("events", []):
            g = espn.parse(event)
            if not g or g["state"] != "post" or g["home"]["score"] is None or g["away"]["score"] is None:
                continue
            home, away = school(g["home"]), school(g["away"])
            if home and away:
                finals[(home, away)] = (g["home"]["score"], g["away"]["score"])

    rows = []
    for _, r in pending.iterrows():
        key = (r["home_team"], r["away_team"])
        if key in finals:
            h, a = finals[key]
        elif key[::-1] in finals:  # neutral site listed the other way round
            a, h = finals[key[::-1]]
        else:
            continue
        rows.append({**{c: r[c] for c in KEY_COLS}, "home_score": h, "away_score": a})
    print(f"  ESPN final scores filled in for {len(rows)} game(s) CFBD hasn't posted yet")
    return pd.DataFrame(rows, columns=cols)

def grade_completed_games(log):
    games_path = os.path.join(RAW_DIR, "cfb_games.csv")
    if not os.path.exists(games_path):
        return log
    schedules = pd.read_csv(games_path)
    results = schedules[schedules["home_score"].notna() & schedules["away_score"].notna()][KEY_COLS + ["home_score", "away_score"]]
    # CFBD wins when it has a score; ESPN fills in the games it hasn't posted yet.
    espn_results = espn_final_scores(log, results)
    if not espn_results.empty:
        espn_results = espn_results.astype({"season": results["season"].dtype, "week": results["week"].dtype}, errors="ignore")
        results = pd.concat([results, espn_results], ignore_index=True).drop_duplicates(KEY_COLS, keep="first")

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
        # As floats: on the first-ever grade this column comes back from the CSV as
        # all-NaN float64, and pandas refuses to write bools into it.
        log.loc[graded, f"{label}_ats_push"] = push[graded].astype(float)
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
    ml_snapshot = load_moneyline_snapshot()
    if ml_snapshot is not None:
        log = moneyline.lock_and_merge(log, ml_snapshot, KEY_COLS)
    print(f"  CFB tracking log now has {len(log)} predicted games total")

    print("Grading any CFB games whose results are now in...")
    log = grade_completed_games(log)
    log = moneyline.grade(log)
    print(f"  {int(log['actual_margin'].notna().sum())} games have a graded actual result")

    log.to_csv(LOG_PATH, index=False)

    summary = summarize(log)
    # This season's live moneyline record only (never backfill/past seasons).
    summary["moneyline"] = moneyline.summarize(log, current_cfb_season())
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved CFB tracking log to {LOG_PATH} and summary to {SUMMARY_PATH}")

if __name__ == "__main__":
    main()
