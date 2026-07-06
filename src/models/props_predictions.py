"""
props_predictions.py
Projects player stat lines (passing/rushing/receiving) for each player's next
upcoming game.

APPROACH:
For each player, blend their per-game usage rates and efficiency across the
last 2 seasons (recent season weighted more heavily), then adjust that
efficiency based on how their specific upcoming opponent's defense performs
in that phase of the game (e.g., a WR facing a defense that's bad against the
pass should be expected to outperform their season averages).

Projected stat = (usage rate) x (efficiency) x (matchup multiplier)

CALIBRATION NOTE: same caveat as game_predictions.py - the matchup multiplier
scaling factor below is a reasonable starting point, not yet backtested
against actual results. Revisit once real games start rolling in.

Uses current_roster.csv (not historical team from play-by-play) to determine
who a player plays for RIGHT NOW, so offseason signings/trades are reflected
even though the underlying efficiency stats come from their prior team(s).
"""

import pandas as pd
import numpy as np
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

SEASON_RECENCY_WEIGHT = {0: 1.0, 1: 0.5}  # 0 = most recent season, 1 = season before that
MATCHUP_SCALING_FACTOR = 2.5  # converts opponent EPA-allowed differential into a % multiplier
MIN_GAMES_PLAYED = 3  # ignore small-sample noise (injury replacements, garbage time, etc.)

def load_data():
    player_stats = pd.read_csv(os.path.join(PROCESSED_DIR, "player_stats.csv"))
    team_stats = pd.read_csv(os.path.join(PROCESSED_DIR, "team_stats.csv")).set_index("team")
    schedules = pd.read_csv(os.path.join(RAW_DIR, "schedules.csv"))
    current_roster = pd.read_csv(os.path.join(RAW_DIR, "current_roster.csv"), low_memory=False)
    return player_stats, team_stats, schedules, current_roster

def blend_player_seasons(player_stats):
    """
    Collapse multiple season rows per player into one blended profile,
    weighting the more recent season more heavily. Rate/efficiency columns
    get a weighted average; count columns (games_played) get summed.
    """
    rate_cols = [c for c in player_stats.columns if any(
        k in c for k in ["_per_game", "_per_target", "_per_carry", "_per_pass_attempt", "_rate"]
    )]

    blended_rows = []
    for player_id, g in player_stats.groupby("player_id"):
        g = g.sort_values("season", ascending=False).reset_index(drop=True)
        g = g[g["games_played"].fillna(0) >= 1]  # need at least some snaps to be useful
        if g.empty:
            continue

        weights = g.index.map(lambda i: SEASON_RECENCY_WEIGHT.get(i, 0.25))
        row = {"player_id": player_id, "player_name": g.iloc[0]["player_name"]}
        row["total_games_played"] = g["games_played"].sum()

        for col in rate_cols:
            valid = g[col].notna()
            if valid.sum() == 0:
                row[col] = np.nan
            else:
                row[col] = np.average(g.loc[valid, col], weights=weights[valid])

        blended_rows.append(row)

    return pd.DataFrame(blended_rows)

def league_averages(team_stats):
    return {
        "def_pass_epa_allowed": team_stats["def_pass_epa_allowed"].mean(),
        "def_rush_epa_allowed": team_stats["def_rush_epa_allowed"].mean(),
    }

def matchup_multiplier(opponent_def_epa_allowed, league_avg_epa_allowed):
    """
    > 1.0 means the opponent's defense is worse than average in this phase
    (good matchup for our player); < 1.0 means tougher than average.
    """
    if pd.isna(opponent_def_epa_allowed):
        return 1.0
    diff = opponent_def_epa_allowed - league_avg_epa_allowed
    return 1 + (diff * MATCHUP_SCALING_FACTOR)

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

def project_player(row, team, opponent, team_stats, league_avgs):
    """Build a single player's projected stat line for their next game."""
    opp_stats = team_stats.loc[opponent] if opponent in team_stats.index else None

    pass_mult = matchup_multiplier(
        opp_stats["def_pass_epa_allowed"] if opp_stats is not None else np.nan,
        league_avgs["def_pass_epa_allowed"]
    )
    rush_mult = matchup_multiplier(
        opp_stats["def_rush_epa_allowed"] if opp_stats is not None else np.nan,
        league_avgs["def_rush_epa_allowed"]
    )

    proj = {"player_id": row["player_id"], "player_name": row["player_name"], "team": team, "opponent": opponent}

    # Receiving
    if pd.notna(row.get("targets_per_game")):
        proj["proj_targets"] = round(row["targets_per_game"], 1)
        proj["proj_receptions"] = round(row["targets_per_game"] * row.get("catch_rate", np.nan), 1)
        proj["proj_rec_yards"] = round(row["targets_per_game"] * row.get("yards_per_target", 0) * pass_mult, 1)
        proj["proj_rec_tds"] = round(row["targets_per_game"] * row.get("rec_td_rate", 0) * pass_mult, 2)

    # Rushing
    if pd.notna(row.get("carries_per_game")):
        proj["proj_carries"] = round(row["carries_per_game"], 1)
        proj["proj_rush_yards"] = round(row["carries_per_game"] * row.get("yards_per_carry", 0) * rush_mult, 1)
        proj["proj_rush_tds"] = round(row["carries_per_game"] * row.get("rush_td_rate", 0) * rush_mult, 2)

    # Passing
    if pd.notna(row.get("pass_attempts_per_game")) and row["pass_attempts_per_game"] > 5:
        proj["proj_pass_attempts"] = round(row["pass_attempts_per_game"], 1)
        proj["proj_completions"] = round(row["pass_attempts_per_game"] * row.get("completion_rate", np.nan), 1)
        proj["proj_pass_yards"] = round(row["pass_attempts_per_game"] * row.get("yards_per_pass_attempt", 0) * pass_mult, 1)
        proj["proj_pass_tds"] = round(row["pass_attempts_per_game"] * row.get("pass_td_rate", 0) * pass_mult, 2)

    return proj

def main():
    print("Loading data...")
    player_stats, team_stats, schedules, current_roster = load_data()
    league_avgs = league_averages(team_stats)

    print("Blending player seasons (recency-weighted)...")
    blended = blend_player_seasons(player_stats)
    blended = blended[blended["total_games_played"] >= MIN_GAMES_PLAYED]
    print(f"  {len(blended)} players with enough games to project")

    # Map each player to their CURRENT team via current_roster (handles offseason moves)
    roster_map = current_roster.set_index("gsis_id")["team"].to_dict() if "gsis_id" in current_roster.columns else {}
    if not roster_map:
        print("  Warning: could not map player_id -> current team from current_roster.csv (check column name)")

    print("Projecting next-game stat lines...")
    projections = []
    team_next_game_cache = {}

    for _, row in blended.iterrows():
        current_team = roster_map.get(row["player_id"])
        if current_team is None:
            continue

        if current_team not in team_next_game_cache:
            team_next_game_cache[current_team] = get_next_game(current_team, schedules)
        opponent, is_home, week, season = team_next_game_cache[current_team]

        if opponent is None:
            continue

        proj = project_player(row, current_team, opponent, team_stats, league_avgs)
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
        print(top_rec[["player_name", "team", "opponent", "proj_targets", "proj_receptions", "proj_rec_yards", "proj_rec_tds"]].to_string(index=False))

    print(f"\nSaved full results to {out_path}")

if __name__ == "__main__":
    main()
