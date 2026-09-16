"""
props_predictions.py
Projects player stat lines (passing/rushing/receiving) for each player's next
upcoming game.

APPROACH:
For each player, take their recency-weighted "current form" usage/efficiency
(player_current_form.csv - the same continuous exponential-decay design as
team stats, half-life fit by fit_props_model.py instead of a hand-picked
2-season blend), then adjust for the upcoming opponent's matchup strength
using coefficients fit_props_model.py regressed against actual historical
outcomes.

Projected stat = (usage rate) x (efficiency) x (matchup multiplier)

INJURY HANDLING: players listed as "Out" are excluded from projections
entirely. "Doubtful"/"Questionable" players are kept but flagged, with a
discount applied to their projected volume - the multipliers come from
fit_props_model.py's join of historical injury reports against actual
performance (see fitted_props_coefficients.json), falling back to
reasonable hand-picked defaults if that file hasn't been generated yet.

Uses current_roster.csv (not historical team from play-by-play) to determine
who a player plays for RIGHT NOW, so offseason signings/trades are reflected
even though the underlying efficiency stats come from their prior team(s).
"""

import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from team_overrides import apply_overrides
from auto_defense_adjustments import apply_auto_adjustments

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
COEFFICIENTS_PATH = os.path.join(os.path.dirname(__file__), "fitted_props_coefficients.json")

MIN_GAMES_PLAYED = 3  # ignore small-sample noise (injury replacements, garbage time, etc.)

# Minimum CAREER volume (within the fetched data window) before we trust a
# rate stat (yards per target/carry/attempt) enough to project from it.
# Below this, a single fluke play can dominate the average.
MIN_CAREER_VOLUME = {
    "targets": 8,
    "carries": 8,
    "pass_attempts": 10,
}

# Fallback constants, used ONLY if fitted_props_coefficients.json doesn't
# exist yet (run src/fit_props_model.py to generate it).
DEFAULT_MATCHUP_SCALING = {
    "receiving": {"baseline_scale": 1.0, "matchup_scale": 2.5},
    "rushing": {"baseline_scale": 1.0, "matchup_scale": 2.5},
    "passing": {"baseline_scale": 1.0, "matchup_scale": 2.5},
}
DEFAULT_INJURY_MULTIPLIERS = {"Doubtful": 0.35, "Questionable": 0.80}

def load_props_coefficients():
    if not os.path.exists(COEFFICIENTS_PATH):
        print("  WARNING: fitted_props_coefficients.json not found - using un-backtested "
              "fallback matchup/injury constants. Run src/fit_props_model.py to fit real coefficients.")
        return DEFAULT_MATCHUP_SCALING, DEFAULT_INJURY_MULTIPLIERS, None
    with open(COEFFICIENTS_PATH) as f:
        fitted = json.load(f)
    matchup = {k: v for k, v in fitted["matchup_scaling"].items()}

    # Merge (not replace): fit_props_model.py omits any status without enough
    # samples to trust (e.g. "Doubtful" - rare among players who actually
    # play), so that status must fall back to its hand-picked default rather
    # than silently becoming a 1.0 (no discount at all).
    injury = dict(DEFAULT_INJURY_MULTIPLIERS)
    for status, info in (fitted.get("injury_multipliers") or {}).items():
        injury[status] = info["multiplier"]

    return matchup, injury, fitted.get("holdout_validation")

def load_data():
    player_form = pd.read_csv(os.path.join(PROCESSED_DIR, "player_current_form.csv"))
    team_stats = pd.read_csv(os.path.join(PROCESSED_DIR, "team_stats.csv")).set_index("team")
    team_stats = apply_overrides(team_stats)
    team_stats = apply_auto_adjustments(team_stats)
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    current_roster = pd.read_csv(os.path.join(RAW_DIR, "current_roster.csv"), low_memory=False)

    injuries_path = os.path.join(RAW_DIR, "injuries.csv")
    if os.path.exists(injuries_path):
        injuries = pd.read_csv(injuries_path)
    else:
        injuries = pd.DataFrame(columns=["gsis_id", "report_status"])

    return player_form, team_stats, schedules, current_roster, injuries

def league_averages(team_stats):
    return {
        "def_pass_epa_allowed": team_stats["def_pass_epa_allowed"].mean(),
        "def_rush_epa_allowed": team_stats["def_rush_epa_allowed"].mean(),
    }

def matchup_multiplier(opponent_def_epa_allowed, league_avg_epa_allowed, matchup_scale):
    """
    > 1.0 means the opponent's defense is worse than average in this phase
    (good matchup for our player); < 1.0 means tougher than average.
    """
    if pd.isna(opponent_def_epa_allowed):
        return 1.0
    diff = opponent_def_epa_allowed - league_avg_epa_allowed
    return 1 + (diff * matchup_scale)

def get_next_game(team, schedules):
    """Find a team's next unplayed game and return (opponent, is_home)."""
    team_games = schedules[
        ((schedules["home_team"] == team) | (schedules["away_team"] == team))
        & (schedules["result"].isna())
    ].sort_values(["season", "week"])
    if team_games.empty:
        return None, None, None, None
    game = team_games.iloc[0]
    is_home = game["home_team"] == team
    opponent = game["away_team"] if is_home else game["home_team"]
    return opponent, is_home, game["week"], game["season"]

def get_injury_status(player_id, injury_map):
    """Returns report_status string, or None if not on the injury report (presumed healthy)."""
    return injury_map.get(player_id)

def project_player(row, team, opponent, team_stats, league_avgs, matchup_coefs, injury_multipliers, injury_status=None):
    """Build a single player's projected stat line for their next game."""
    opp_stats = team_stats.loc[opponent] if opponent in team_stats.index else None

    pass_mult = matchup_multiplier(
        opp_stats["def_pass_epa_allowed"] if opp_stats is not None else np.nan,
        league_avgs["def_pass_epa_allowed"], matchup_coefs["receiving"]["matchup_scale"]
    )
    rush_mult = matchup_multiplier(
        opp_stats["def_rush_epa_allowed"] if opp_stats is not None else np.nan,
        league_avgs["def_rush_epa_allowed"], matchup_coefs["rushing"]["matchup_scale"]
    )
    pass_yds_mult = matchup_multiplier(
        opp_stats["def_pass_epa_allowed"] if opp_stats is not None else np.nan,
        league_avgs["def_pass_epa_allowed"], matchup_coefs["passing"]["matchup_scale"]
    )

    # Injury discount applies on top of the matchup multiplier - a banged-up
    # player facing a bad defense might still project lower than a healthy
    # player facing a good one.
    injury_disc = injury_multipliers.get(injury_status, 1.0)
    pass_mult *= injury_disc
    rush_mult *= injury_disc
    pass_yds_mult *= injury_disc

    proj = {
        "player_id": row["player_id"], "player_name": row["player_name"],
        "team": team, "opponent": opponent,
        "injury_status": injury_status if injury_status else "",
        "games_played": row.get("games_played", 0),
        # Matchup multiplier BEFORE the injury discount, so the email can show
        # "tough/soft matchup" as a pure matchup signal, separate from injury risk.
        "matchup_mult_rec": round(pass_mult / injury_disc, 3) if injury_disc else None,
        "matchup_mult_rush": round(rush_mult / injury_disc, 3) if injury_disc else None,
        "matchup_mult_pass": round(pass_yds_mult / injury_disc, 3) if injury_disc else None,
    }

    rec_scale = matchup_coefs["receiving"]["baseline_scale"]
    rush_scale = matchup_coefs["rushing"]["baseline_scale"]
    pass_scale = matchup_coefs["passing"]["baseline_scale"]

    # Receiving - only project if we have a real career sample of targets,
    # not a single fluke play
    if pd.notna(row.get("targets_per_game")) and row.get("targets_career_total", 0) >= MIN_CAREER_VOLUME["targets"]:
        proj["proj_targets"] = round(row["targets_per_game"] * injury_disc, 1)
        proj["proj_receptions"] = round(proj["proj_targets"] * row.get("catch_rate", np.nan), 1)
        proj["proj_rec_yards"] = round(max(proj["proj_targets"] * row.get("yards_per_target", 0) * pass_mult * rec_scale, 0), 1)
        proj["proj_rec_tds"] = round(max(proj["proj_targets"] * row.get("rec_td_rate", 0) * pass_mult, 0), 2)

    # Rushing - same volume floor
    if pd.notna(row.get("carries_per_game")) and row.get("carries_career_total", 0) >= MIN_CAREER_VOLUME["carries"]:
        proj["proj_carries"] = round(row["carries_per_game"] * injury_disc, 1)
        proj["proj_rush_yards"] = round(max(proj["proj_carries"] * row.get("yards_per_carry", 0) * rush_mult * rush_scale, 0), 1)
        proj["proj_rush_tds"] = round(max(proj["proj_carries"] * row.get("rush_td_rate", 0) * rush_mult, 0), 2)

    # Passing - same volume floor
    if pd.notna(row.get("pass_attempts_per_game")) and row.get("pass_attempts_career_total", 0) >= MIN_CAREER_VOLUME["pass_attempts"]:
        proj["proj_pass_attempts"] = round(row["pass_attempts_per_game"] * injury_disc, 1)
        proj["proj_completions"] = round(proj["proj_pass_attempts"] * row.get("completion_rate", np.nan), 1)
        proj["proj_pass_yards"] = round(max(proj["proj_pass_attempts"] * row.get("yards_per_pass_attempt", 0) * pass_yds_mult * pass_scale, 0), 1)
        proj["proj_pass_tds"] = round(max(proj["proj_pass_attempts"] * row.get("pass_td_rate", 0) * pass_yds_mult, 0), 2)

    return proj

def main():
    print("Loading data...")
    player_form, team_stats, schedules, current_roster, injuries = load_data()
    league_avgs = league_averages(team_stats)
    matchup_coefs, injury_multipliers, holdout = load_props_coefficients()

    if holdout:
        print("\nProps model calibration (holdout validation from fit_props_model.py):")
        for stat_type, h in holdout["by_stat_type"].items():
            print(f"  {stat_type:>9}: baseline-only MAE {h['mae_baseline_only']:.2f} | "
                  f"original hand-picked MAE {h['mae_original_handpicked']:.2f} | fitted MAE {h['mae_fitted']:.2f}")

    # Build a quick lookup: player_id -> report_status, keeping only the
    # most severe/relevant status if a player somehow has multiple rows
    injury_map = {}
    excluded_out = set()
    if not injuries.empty and "gsis_id" in injuries.columns:
        for _, r in injuries.iterrows():
            status = r.get("report_status")
            pid = r.get("gsis_id")
            if status == "Out":
                excluded_out.add(pid)
            elif pd.notna(status):
                injury_map[pid] = status
        print(f"  Injury report: {len(excluded_out)} Out (excluded), {len(injury_map)} Questionable/Doubtful (discounted)")

    eligible = player_form[player_form["games_played"] >= MIN_GAMES_PLAYED]
    eligible = eligible[~eligible["player_id"].isin(excluded_out)]
    print(f"  {len(eligible)} players with enough games to project (after removing 'Out' players)")

    # Map each player to their CURRENT team via current_roster (handles offseason moves)
    roster_map = current_roster.set_index("gsis_id")["team"].to_dict() if "gsis_id" in current_roster.columns else {}
    if not roster_map:
        print("  Warning: could not map player_id -> current team from current_roster.csv (check column name)")

    print("Projecting next-game stat lines...")
    projections = []
    team_next_game_cache = {}

    for _, row in eligible.iterrows():
        current_team = roster_map.get(row["player_id"])
        if current_team is None:
            continue

        if current_team not in team_next_game_cache:
            team_next_game_cache[current_team] = get_next_game(current_team, schedules)
        opponent, is_home, week, season = team_next_game_cache[current_team]

        if opponent is None:
            continue

        injury_status = get_injury_status(row["player_id"], injury_map)
        proj = project_player(row, current_team, opponent, team_stats, league_avgs, matchup_coefs, injury_multipliers, injury_status)
        proj["week"] = week
        proj["season"] = season
        proj["is_home"] = is_home
        projections.append(proj)

    result = pd.DataFrame(projections)
    out_path = os.path.join(PROCESSED_DIR, "player_props.csv")
    result.to_csv(out_path, index=False)

    print(f"\nGenerated {len(result)} player projections.")
    if not result.empty and "proj_rec_yards" in result.columns:
        print("\nTop 5 projected receiving performances:")
        top_rec = result.dropna(subset=["proj_rec_yards"]).sort_values("proj_rec_yards", ascending=False).head(5)
        print(top_rec[["player_name", "team", "opponent", "injury_status", "proj_targets", "proj_receptions", "proj_rec_yards", "proj_rec_tds"]].to_string(index=False))

    print(f"\nSaved full results to {out_path}")

if __name__ == "__main__":
    main()
