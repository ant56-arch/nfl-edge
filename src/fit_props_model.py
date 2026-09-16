"""
fit_props_model.py
Same treatment fit_model.py gave game_predictions.py, applied to
props_predictions.py: every hand-picked constant (the 2-season blend
weights, the matchup scaling factor, the injury-status discount multipliers)
gets replaced with a value fit against real historical player performance
and validated on a held-out season.

METHOD:
  1. Rebuild player-game actual stat lines from play-by-play (per season,
     week, player - build_features.build_player_game_stats), for
     TRAIN_START_SEASON..present.
  2. Recency decay: the old props model blended "this season" (weight 1.0)
     and "last season" (weight 0.5) as two discrete chunks - a coarse
     approximation copy-pasted as a starting point, never checked against
     data. Instead, grid-search the same continuous exponential half-life
     decay team stats already use (historical_features.compute_walkforward_features,
     generalized to group by player_id), picking whichever half-life
     minimizes held-out projection error.
  3. Matchup scaling: fit how much an opponent's walk-forward defensive
     EPA-allowed should move a player's projection, separately for
     receiving/rushing/passing (the original 2.5 was a single guess shared
     across all three), via OLS: actual_stat = a*baseline + b*(baseline*matchup_diff).
  4. Injury multipliers: join historical injury reports (report_status per
     player per week) against each player's own walk-forward baseline usage,
     to see what fraction of expected volume "Questionable"/"Doubtful"
     players actually saw - replacing the hand-picked 0.80/0.35.
  5. Validate everything on the most recent completed season, held out from
     every fit above, then refit final coefficients on the full dataset for
     production. Write src/models/fitted_props_coefficients.json.

Like fit_model.py, this is NOT part of the twice-weekly pipeline - see
refit-model.yml, which runs this alongside the game model monthly.
"""

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import fit_model as fm
from build_features import build_player_game_stats, build_team_defense_game_stats
from historical_features import compute_walkforward_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

HALF_LIFE_GRID = [3, 4, 6, 8, 10, 13, 16, 20, 26, 34]
VALIDATION_SEASONS = 2  # most recent seasons held out (matches fit_model.py - the
# last "season" in range() is always the current, in-progress one, so this
# needs to be 2 to actually capture one FULL completed season (e.g. 2025) in
# the holdout rather than just a handful of played-so-far 2026 games.

STAT_TYPES = {
    "receiving": {"usage": "targets_per_game", "efficiency": "yards_per_target",
                  "actual": "rec_yards", "matchup_metric": "def_pass_epa_allowed"},
    "rushing":   {"usage": "carries_per_game", "efficiency": "yards_per_carry",
                  "actual": "rush_yards", "matchup_metric": "def_rush_epa_allowed"},
    "passing":   {"usage": "pass_attempts_per_game", "efficiency": "yards_per_pass_attempt",
                  "actual": "pass_yards", "matchup_metric": "def_pass_epa_allowed"},
}
MIN_USAGE_ASOF = {"receiving": 2.0, "rushing": 2.0, "passing": 8.0}  # ignore token/garbage-time usage

def fetch_historical_injuries(seasons):
    frames = []
    for season in seasons:
        url = f"{fm.BASE}/injuries/injuries_{season}.csv"
        try:
            df = pd.read_csv(url, low_memory=False)
            df["season"] = season
            frames.append(df)
        except Exception as e:
            print(f"  Skipping injuries {season}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

PLAYER_WALKFORWARD_METRICS = ["targets_per_game", "carries_per_game", "pass_attempts_per_game",
                               "yards_per_target", "yards_per_carry", "yards_per_pass_attempt",
                               "rec_yards", "rush_yards", "pass_yards"]

def build_player_dataset(player_game, half_life, opponent_map, off_walk_def):
    """One row per (season, week, player) with pre-game walk-forward usage/
    efficiency, the opponent's walk-forward defensive strength in that phase,
    and the actual result - for every stat type at once. `player_game` is
    the (expensive, half-life-independent) per-game aggregation, built once
    by the caller and re-decayed here for each half-life candidate."""
    # rec_yards/rush_yards/pass_yards ride along as raw (non-decayed) columns
    # purely so we can pull the ACTUAL value per row after walk-forward runs.
    walk = compute_walkforward_features(player_game, PLAYER_WALKFORWARD_METRICS, half_life=half_life, group_col="player_id")

    walk["opponent"] = walk.apply(lambda r: opponent_map.get((r["season"], r["week"], r["team"])), axis=1)
    walk = walk.merge(
        off_walk_def.rename(columns={"team": "opponent"}),
        on=["season", "week", "opponent"], how="left"
    )
    return walk

def league_avg_by_week(def_walk):
    """League-average defensive strength AS OF each (season, week) - the
    same no-leakage walk-forward values, averaged across whichever teams
    have a valid figure at that point."""
    return def_walk.groupby(["season", "week"])[
        ["def_pass_epa_allowed_asof", "def_rush_epa_allowed_asof"]
    ].mean().rename(columns={
        "def_pass_epa_allowed_asof": "league_avg_pass", "def_rush_epa_allowed_asof": "league_avg_rush"
    }).reset_index()

def fit_matchup_scaling(df, stat_type, min_usage):
    cfg = STAT_TYPES[stat_type]
    usage_col, eff_col, actual_col = f"{cfg['usage']}_asof", f"{cfg['efficiency']}_asof", cfg["actual"]
    matchup_col = f"{cfg['matchup_metric']}_asof"
    league_col = "league_avg_pass" if "pass" in cfg["matchup_metric"] else "league_avg_rush"

    d = df[(df[usage_col] >= min_usage) & df[usage_col].notna() & df[eff_col].notna()
           & df[matchup_col].notna() & df[actual_col].notna()].copy()
    baseline = d[usage_col] * d[eff_col]
    matchup_diff = d[matchup_col] - d[league_col]

    X = np.column_stack([baseline, baseline * matchup_diff])
    y = d[actual_col].values
    beta, resid = fm.fit_ols(X, y)
    mae_baseline_only = float(np.mean(np.abs(baseline - y)))
    mae_fitted = float(np.mean(np.abs(X @ beta - y)))
    return {"baseline_scale": float(beta[0]), "matchup_scale": float(beta[1]),
            "n_samples": int(len(d)), "mae_baseline_only": mae_baseline_only, "mae_fitted": mae_fitted}, d.index

def evaluate_holdout(df, coefs, stat_type, min_usage):
    cfg = STAT_TYPES[stat_type]
    usage_col, eff_col, actual_col = f"{cfg['usage']}_asof", f"{cfg['efficiency']}_asof", cfg["actual"]
    matchup_col = f"{cfg['matchup_metric']}_asof"
    league_col = "league_avg_pass" if "pass" in cfg["matchup_metric"] else "league_avg_rush"

    d = df[(df[usage_col] >= min_usage) & df[usage_col].notna() & df[eff_col].notna()
           & df[matchup_col].notna() & df[actual_col].notna()].copy()
    baseline = d[usage_col] * d[eff_col]
    matchup_diff = d[matchup_col] - d[league_col]

    fitted_proj = coefs["baseline_scale"] * baseline + coefs["matchup_scale"] * baseline * matchup_diff
    naive_proj = baseline * (1 + matchup_diff * 2.5)  # original hand-picked constant, for comparison

    return {
        "n_games": int(len(d)),
        "mae_baseline_only": float(np.mean(np.abs(baseline - d[actual_col]))),
        "mae_original_handpicked": float(np.mean(np.abs(naive_proj - d[actual_col]))),
        "mae_fitted": float(np.mean(np.abs(fitted_proj - d[actual_col]))),
    }

MIN_INJURY_SAMPLES = 30  # below this, a fitted multiplier is noise, not signal - fall back to the hand-picked default

def fit_injury_multipliers(pbp_seasons, player_game_asof):
    """
    For every player-week tagged Questionable/Doubtful on that week's own
    injury report, compare actual volume (targets or carries, whichever
    applies) to their walk-forward baseline USAGE going into that game - the
    ratio is a data-driven discount, replacing the hand-picked 0.80/0.35.

    "Doubtful" is a genuinely rare tag among players who actually suit up
    (most truly doubtful players simply don't play, so there's little data on
    what they do WHEN they play) - if a status doesn't clear MIN_INJURY_SAMPLES,
    it's dropped here rather than reported with false confidence; the caller
    (props_predictions.py) falls back to the hand-picked default for it.
    """
    print("  Fetching historical injury reports...")
    injuries = fetch_historical_injuries(pbp_seasons)
    if injuries.empty or "gsis_id" not in injuries.columns or "report_status" not in injuries.columns:
        print("  No usable historical injury data - keeping hand-picked injury multipliers.")
        return None

    injuries = injuries[injuries["report_status"].isin(["Questionable", "Doubtful"])]
    injuries = injuries[["season", "week", "gsis_id", "report_status"]].rename(columns={"gsis_id": "player_id"})

    merged = player_game_asof.merge(injuries, on=["season", "week", "player_id"], how="inner")

    results = {}
    for status in ["Questionable", "Doubtful"]:
        sub = merged[merged["report_status"] == status]
        ratios = []
        for usage_col, actual_col in [("targets_per_game_asof", "targets_per_game"), ("carries_per_game_asof", "carries_per_game")]:
            valid = sub[(sub[usage_col] >= 2.0) & sub[usage_col].notna() & sub[actual_col].notna()]
            if len(valid):
                ratios.extend((valid[actual_col] / valid[usage_col]).clip(0, 3).tolist())
        if len(ratios) >= MIN_INJURY_SAMPLES:
            results[status] = {"multiplier": round(float(np.median(ratios)), 2), "n_samples": len(ratios)}
        elif ratios:
            print(f"  {status}: only {len(ratios)} samples (< {MIN_INJURY_SAMPLES}) - "
                  f"too few to trust, keeping the hand-picked default for this status.")

    return results if results else None

def run():
    current_season = fm.current_nfl_season()
    seasons = list(range(fm.TRAIN_START_SEASON, current_season + 1))
    print(f"Fetching play-by-play for {len(seasons)} seasons ({seasons[0]}-{seasons[-1]})...")
    pbp = fm.fetch_historical_pbp(seasons)

    print("Rebuilding team defensive walk-forward features (for matchup context)...")
    # Only defense is needed here (matchup context is "how good is the
    # opponent's defense") - calling build_team_defense_game_stats directly
    # instead of fm.load_pbp_offense_defense avoids computing the (equally
    # expensive) offensive side of the same pipeline for nothing.
    pbp_filtered = pbp[pbp["play_type"].isin(["pass", "run"])]
    pbp_filtered = pbp_filtered[pbp_filtered["posteam"].notna()]
    def_game = build_team_defense_game_stats(pbp_filtered)
    def_walk = compute_walkforward_features(def_game, ["def_pass_epa_allowed", "def_rush_epa_allowed"], group_col="team")
    league_avg = league_avg_by_week(def_walk)
    def_walk_for_join = def_walk.merge(league_avg, on=["season", "week"], how="left")

    print("Building each team's opponent-by-week map (for matchup lookups)...")
    schedules = fm.fetch_historical_schedules()
    opp_rows = pd.concat([
        schedules.rename(columns={"home_team": "team", "away_team": "opponent"})[["season", "week", "team", "opponent"]],
        schedules.rename(columns={"away_team": "team", "home_team": "opponent"})[["season", "week", "team", "opponent"]],
    ])
    opponent_map = {(r.season, r.week, r.team): r.opponent for r in opp_rows.itertuples()}

    print("Building per-game player stats (once - reused across every half-life candidate)...")
    player_game = build_player_game_stats(pbp)

    print(f"\nGrid-searching player recency half-life over {HALF_LIFE_GRID}...")
    holdout_seasons = seasons[-VALIDATION_SEASONS:]
    train_seasons_only = seasons[:-VALIDATION_SEASONS]

    best_half_life, best_mae = None, np.inf
    grid_results = []
    for hl in HALF_LIFE_GRID:
        df = build_player_dataset(player_game, hl, opponent_map, def_walk_for_join)
        val = df[df["season"].isin(holdout_seasons)]
        cfg = STAT_TYPES["receiving"]
        usage_col, eff_col, actual_col = f"{cfg['usage']}_asof", f"{cfg['efficiency']}_asof", cfg["actual"]
        d = val[(val[usage_col] >= MIN_USAGE_ASOF["receiving"]) & val[usage_col].notna() & val[eff_col].notna() & val[actual_col].notna()]
        mae = float(np.mean(np.abs(d[usage_col] * d[eff_col] - d[actual_col]))) if len(d) else np.inf
        grid_results.append({"half_life": hl, "receiving_baseline_mae": round(mae, 3), "n": int(len(d))})
        print(f"  half_life={hl:>2} games -> holdout receiving-yards baseline MAE {mae:.2f} ({len(d)} samples)")
        if mae < best_mae:
            best_mae, best_half_life = mae, hl

    print(f"\nBest half-life: {best_half_life} games")
    full_df = build_player_dataset(player_game, best_half_life, opponent_map, def_walk_for_join)
    train_df = full_df[full_df["season"].isin(train_seasons_only)]
    holdout_df = full_df[full_df["season"].isin(holdout_seasons)]

    print("\nFitting matchup scaling per stat type on TRAIN, validating on HOLDOUT...")
    matchup_coefs, holdout_report = {}, {}
    for stat_type in STAT_TYPES:
        coefs, _ = fit_matchup_scaling(train_df, stat_type, MIN_USAGE_ASOF[stat_type])
        matchup_coefs[stat_type] = coefs
        holdout_report[stat_type] = evaluate_holdout(holdout_df, coefs, stat_type, MIN_USAGE_ASOF[stat_type])
        h = holdout_report[stat_type]
        print(f"  {stat_type:>9}: baseline_scale={coefs['baseline_scale']:.3f} matchup_scale={coefs['matchup_scale']:.3f} | "
              f"holdout MAE - baseline only: {h['mae_baseline_only']:.2f}, original (2.5): {h['mae_original_handpicked']:.2f}, "
              f"fitted: {h['mae_fitted']:.2f} ({h['n_games']} samples)")

    print("\nFitting injury-status discount multipliers from historical injury reports...")
    injury_seasons = [s for s in seasons if s >= max(fm.TRAIN_START_SEASON, current_season - 9)]
    injury_multipliers = fit_injury_multipliers(injury_seasons, full_df)
    if injury_multipliers:
        for status, r in injury_multipliers.items():
            print(f"  {status}: {r['multiplier']}x expected volume ({r['n_samples']} player-weeks)")

    print("\nRefitting final matchup coefficients on FULL dataset (train + holdout) for production...")
    final_matchup_coefs = {}
    for stat_type in STAT_TYPES:
        coefs, _ = fit_matchup_scaling(full_df, stat_type, MIN_USAGE_ASOF[stat_type])
        final_matchup_coefs[stat_type] = coefs

    output = {
        "fitted_at": pd.Timestamp.now("UTC").isoformat(),
        "train_seasons": seasons,
        "half_life_games": best_half_life,
        "half_life_grid_search": grid_results,
        "matchup_scaling": final_matchup_coefs,
        "injury_multipliers": injury_multipliers,
        "holdout_validation": {"seasons": holdout_seasons, "by_stat_type": holdout_report},
    }
    out_path = os.path.join(MODELS_DIR, "fitted_props_coefficients.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved fitted props coefficients + holdout validation to {out_path}")

if __name__ == "__main__":
    run()
