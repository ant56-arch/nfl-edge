"""
build_features.py
Turns raw nflverse play-by-play data into the advanced stats our models use.

Input:  data/raw/pbp_combined.parquet, data/raw/rosters.parquet, data/raw/schedules.csv
Output: data/processed/team_stats.csv, data/processed/player_stats.csv

Team stats are computed two ways for every team-season:
  - Season-long averages (the full sample)
  - Recency-weighted "current form" (recent games count more, via exponential
    decay). This matters a lot in-season: a team that's turned it around in
    the last month should look different from their week-1 numbers.

Player stats are usage + efficiency metrics for skill positions, since props
predictions depend on "how often does this guy touch the ball" as much as
"how good is he when he does."
"""

import pandas as pd
import numpy as np
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

RECENCY_HALF_LIFE_GAMES = 4  # a game 4 weeks ago counts half as much as this week

def recency_weight(games_ago, half_life=RECENCY_HALF_LIFE_GAMES):
    """Exponential decay weight: more recent games matter more."""
    return 0.5 ** (games_ago / half_life)

def load_pbp():
    path = os.path.join(RAW_DIR, "pbp_combined.parquet")
    df = pd.read_parquet(path)
    # Keep only meaningful offensive plays - drop no-plays (penalties w/o snap),
    # special teams, and rows missing a possession team.
    df = df[df["play_type"].isin(["pass", "run"])]
    df = df[df["posteam"].notna()]
    return df

def build_team_game_stats(pbp):
    """Aggregate play-by-play up to one row per team per game (offense side)."""
    grouped = pbp.groupby(["season", "week", "posteam"])

    rows = []
    for (season, week, team), g in grouped:
        plays = len(g)
        pass_plays = g[g["pass"] == 1]
        rush_plays = g[g["rush"] == 1]
        third_downs = g[g["down"] == 3]
        third_conv = third_downs[third_downs["yards_gained"] >= third_downs["ydstogo"]]
        redzone = g[g["yardline_100"] <= 20]
        redzone_td = redzone[redzone["touchdown"] == 1]

        rows.append({
            "season": season,
            "week": week,
            "team": team,
            "plays": plays,
            "epa_per_play": g["epa"].mean(),
            "success_rate": g["success"].mean(),
            "pass_epa": pass_plays["epa"].mean() if len(pass_plays) else np.nan,
            "rush_epa": rush_plays["epa"].mean() if len(rush_plays) else np.nan,
            "explosive_rate": (g["yards_gained"] >= 15).mean(),
            "third_down_rate": len(third_conv) / len(third_downs) if len(third_downs) else np.nan,
            "redzone_td_rate": len(redzone_td) / len(redzone) if len(redzone) else np.nan,
            "sack_rate": pass_plays["sack"].mean() if len(pass_plays) else np.nan,
            "cpoe": pass_plays["cpoe"].mean() if "cpoe" in pass_plays.columns and len(pass_plays) else np.nan,
        })

    return pd.DataFrame(rows)

def build_team_defense_game_stats(pbp):
    """Same metrics, but from the defense's perspective (what they allowed)."""
    grouped = pbp.groupby(["season", "week", "defteam"])

    rows = []
    for (season, week, team), g in grouped:
        pass_plays = g[g["pass"] == 1]
        rush_plays = g[g["rush"] == 1]
        third_downs = g[g["down"] == 3]
        third_conv = third_downs[third_downs["yards_gained"] >= third_downs["ydstogo"]]
        redzone = g[g["yardline_100"] <= 20]
        redzone_td = redzone[redzone["touchdown"] == 1]

        rows.append({
            "season": season,
            "week": week,
            "team": team,
            "def_epa_per_play_allowed": g["epa"].mean(),
            "def_success_rate_allowed": g["success"].mean(),
            "def_pass_epa_allowed": pass_plays["epa"].mean() if len(pass_plays) else np.nan,
            "def_rush_epa_allowed": rush_plays["epa"].mean() if len(rush_plays) else np.nan,
            "def_explosive_rate_allowed": (g["yards_gained"] >= 15).mean(),
            "def_third_down_rate_allowed": len(third_conv) / len(third_downs) if len(third_downs) else np.nan,
            "def_redzone_td_rate_allowed": len(redzone_td) / len(redzone) if len(redzone) else np.nan,
            "pressure_rate": pass_plays["sack"].mean() if len(pass_plays) else np.nan,
        })

    return pd.DataFrame(rows)

def apply_recency_weighting(game_stats, group_cols, metric_cols):
    """
    For each team, weight each game's stats by how recent it is (within that
    team's own game sequence), and produce one weighted-average row per
    team-season representing "current form."
    """
    results = []
    for team, g in game_stats.groupby(group_cols):
        g = g.sort_values(["season", "week"]).reset_index(drop=True)
        games_ago = (len(g) - 1) - g.index  # 0 = most recent game
        weights = games_ago.map(recency_weight)

        weighted = {}
        for col in metric_cols:
            valid = g[col].notna()
            if valid.sum() == 0:
                weighted[col] = np.nan
            else:
                weighted[col] = np.average(g.loc[valid, col], weights=weights[valid])

        season_avg = {f"{col}_season_avg": g[col].mean() for col in metric_cols}

        row = {"team": team if isinstance(team, str) else team[0], "games_played": len(g)}
        row.update(weighted)
        row.update(season_avg)
        results.append(row)

    return pd.DataFrame(results)

def build_player_stats(pbp):
    """Usage + efficiency stats for skill-position players (for prop predictions)."""
    # Receiving stats
    targets = pbp[pbp["pass"] == 1].copy()
    team_pass_attempts = targets.groupby(["season", "posteam"])["play_id"].count().rename("team_targets")

    receiving = targets.groupby(["season", "posteam", "receiver_player_id", "receiver_player_name"]).agg(
        targets=("play_id", "count"),
        receptions=("complete_pass", "sum"),
        rec_yards=("yards_gained", "sum"),
        air_yards=("air_yards", "sum"),
        rec_tds=("touchdown", "sum"),
    ).reset_index()
    receiving = receiving.merge(team_pass_attempts, left_on=["season", "posteam"], right_index=True)
    receiving["target_share"] = receiving["targets"] / receiving["team_targets"]
    receiving = receiving.rename(columns={"receiver_player_id": "player_id", "receiver_player_name": "player_name", "posteam": "team"})
    receiving = receiving[receiving["player_id"].notna()]

    # Rushing stats
    rushes = pbp[pbp["rush"] == 1].copy()
    team_rush_attempts = rushes.groupby(["season", "posteam"])["play_id"].count().rename("team_carries")

    rushing = rushes.groupby(["season", "posteam", "rusher_player_id", "rusher_player_name"]).agg(
        carries=("play_id", "count"),
        rush_yards=("yards_gained", "sum"),
        rush_tds=("touchdown", "sum"),
    ).reset_index()
    rushing = rushing.merge(team_rush_attempts, left_on=["season", "posteam"], right_index=True)
    rushing["carry_share"] = rushing["carries"] / rushing["team_carries"]
    rushing = rushing.rename(columns={"rusher_player_id": "player_id", "rusher_player_name": "player_name", "posteam": "team"})
    rushing = rushing[rushing["player_id"].notna()]

    player_stats = pd.merge(
        receiving, rushing,
        on=["season", "team", "player_id", "player_name"],
        how="outer", suffixes=("_rec", "_rush")
    )
    return player_stats

def main():
    print("Loading play-by-play data...")
    pbp = load_pbp()
    print(f"  {len(pbp):,} offensive plays loaded")

    print("\nBuilding team offense game-by-game stats...")
    off_game_stats = build_team_game_stats(pbp)

    print("Building team defense game-by-game stats...")
    def_game_stats = build_team_defense_game_stats(pbp)

    print("Applying recency weighting (current form vs season-long)...")
    off_metrics = ["epa_per_play", "success_rate", "pass_epa", "rush_epa",
                   "explosive_rate", "third_down_rate", "redzone_td_rate", "sack_rate", "cpoe"]
    def_metrics = ["def_epa_per_play_allowed", "def_success_rate_allowed", "def_pass_epa_allowed",
                   "def_rush_epa_allowed", "def_explosive_rate_allowed", "def_third_down_rate_allowed",
                   "def_redzone_td_rate_allowed", "pressure_rate"]

    off_weighted = apply_recency_weighting(off_game_stats, "team", off_metrics)
    def_weighted = apply_recency_weighting(def_game_stats, "team", def_metrics)

    team_stats = off_weighted.merge(def_weighted, on="team", suffixes=("", "_def"))
    team_stats.to_csv(os.path.join(PROCESSED_DIR, "team_stats.csv"), index=False)
    print(f"  Saved team_stats.csv ({len(team_stats)} teams)")

    print("\nBuilding player usage/efficiency stats...")
    player_stats = build_player_stats(pbp)
    player_stats.to_csv(os.path.join(PROCESSED_DIR, "player_stats.csv"), index=False)
    print(f"  Saved player_stats.csv ({len(player_stats)} players)")

    print("\nDone. Processed stats saved to data/processed/")

if __name__ == "__main__":
    main()
