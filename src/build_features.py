"""
build_features.py
Turns raw nflverse play-by-play data into the advanced stats our models use.

Input:  data/raw/pbp_combined.parquet, data/raw/rosters.parquet, data/raw/schedules.csv
Output: data/processed/team_stats.csv, data/processed/player_current_form.csv

Team AND player stats are both computed the same two ways:
  - Season-long averages (the full sample)
  - Recency-weighted "current form" (recent games count more, via exponential
    decay - the half-life is fit against real outcomes by fit_model.py for
    teams and fit_props_model.py for players, not hand-picked). This matters
    a lot in-season: a team or player who's turned it around in the last
    month should look different from their week-1 numbers.

Player current-form metrics are usage + efficiency for skill positions,
since props predictions depend on "how often does this guy touch the ball"
as much as "how good is he when he does."
"""

import pandas as pd
import numpy as np
import json
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

PLAYER_HALF_LIFE_PATH = os.path.join(os.path.dirname(__file__), "models", "fitted_props_coefficients.json")
DEFAULT_PLAYER_HALF_LIFE = 6  # used only until fit_props_model.py has produced a fitted value

RECENCY_HALF_LIFE_GAMES = 4  # a game 4 weeks ago counts half as much as this week

def recency_weight(games_ago, half_life=RECENCY_HALF_LIFE_GAMES):
    """Exponential decay weight: more recent games matter more."""
    return 0.5 ** (games_ago / half_life)

def load_player_half_life():
    if not os.path.exists(PLAYER_HALF_LIFE_PATH):
        return DEFAULT_PLAYER_HALF_LIFE
    with open(PLAYER_HALF_LIFE_PATH) as f:
        return json.load(f).get("half_life_games", DEFAULT_PLAYER_HALF_LIFE)

def load_pbp():
    path = os.path.join(RAW_DIR, "pbp_combined.parquet")
    df = pd.read_parquet(path)
    # Keep only meaningful offensive plays - drop no-plays (penalties w/o snap),
    # special teams, and rows missing a possession team.
    df = df[df["play_type"].isin(["pass", "run"])]
    df = df[df["posteam"].notna()]
    return df

def redzone_trip_td_rate(g):
    """
    Proper red-zone TD rate: % of DRIVES that reach the red zone and end in
    a touchdown (the standard definition), not % of red-zone plays that are
    touchdowns (which understates it, since most plays in a trip aren't the
    scoring play).

    Uses nflverse's 'drive' and 'fixed_drive_result' columns when available;
    falls back to the cruder per-play method if a dataset doesn't have them.
    """
    if "drive" not in g.columns or "fixed_drive_result" not in g.columns:
        redzone = g[g["yardline_100"] <= 20]
        redzone_td = redzone[redzone["touchdown"] == 1]
        return len(redzone_td) / len(redzone) if len(redzone) else np.nan

    trips = g.groupby("drive").agg(
        reached_rz=("yardline_100", lambda x: (x <= 20).any()),
        result=("fixed_drive_result", "first"),
    )
    rz_trips = trips[trips["reached_rz"]]
    if len(rz_trips) == 0:
        return np.nan
    td_trips = rz_trips[rz_trips["result"].astype(str).str.contains("Touchdown", case=False, na=False)]
    return len(td_trips) / len(rz_trips)

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
            "redzone_td_rate": redzone_trip_td_rate(g),
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
            "def_redzone_td_rate_allowed": redzone_trip_td_rate(g),
            "pressure_rate": pass_plays["sack"].mean() if len(pass_plays) else np.nan,
        })

    return pd.DataFrame(rows)

def apply_recency_weighting(game_stats, group_cols, metric_cols, half_life=RECENCY_HALF_LIFE_GAMES,
                             key_col="team", volume_cols=None):
    """
    For each entity (team or player), weight each game's stats by how recent
    it is (within that entity's own game sequence), and produce one
    weighted-average row representing "current form" as of right now (i.e.
    using every game on record - this is what a predictor uses for an
    upcoming game, as opposed to historical_features.compute_walkforward_features,
    which evaluates this same math at every PAST cutoff for backtesting).

    volume_cols (optional): raw counting columns to also sum (unweighted)
    across every game - used for minimum-sample-size gating downstream,
    where a rate stat's OWN precision matters more than its recency.
    """
    results = []
    for key, g in game_stats.groupby(group_cols):
        g = g.sort_values(["season", "week"]).reset_index(drop=True)
        games_ago = (len(g) - 1) - g.index  # 0 = most recent game
        weights = games_ago.map(lambda x: recency_weight(x, half_life))

        weighted = {}
        for col in metric_cols:
            valid = g[col].notna()
            if valid.sum() == 0:
                weighted[col] = np.nan
            else:
                weighted[col] = np.average(g.loc[valid, col], weights=weights[valid])

        season_avg = {f"{col}_season_avg": g[col].mean() for col in metric_cols}

        row = {key_col: key if isinstance(key, str) else key[0], "games_played": len(g)}
        row.update(weighted)
        row.update(season_avg)
        if volume_cols:
            for col in volume_cols:
                row[f"{col}_career_total"] = g[col].fillna(0).sum()
        results.append(row)

    return pd.DataFrame(results)

def build_player_game_stats(pbp):
    """
    Usage + efficiency stats for skill-position players, ONE ROW PER GAME
    (season, week, team, player) - not aggregated to a season total. This is
    what lets props be recency-weighted the same continuous way team stats
    are (apply_recency_weighting / historical_features.compute_walkforward_features),
    instead of the coarser "blend last 2 season totals" approach the props
    model used before fit_props_model.py's backtest replaced it.
    """
    # Receiving stats
    targets = pbp[pbp["pass"] == 1].copy()
    team_pass_attempts = targets.groupby(["season", "week", "posteam"])["play_id"].count().rename("team_targets")

    receiving = targets.groupby(["season", "week", "posteam", "receiver_player_id", "receiver_player_name"]).agg(
        targets=("play_id", "count"),
        receptions=("complete_pass", "sum"),
        rec_yards=("yards_gained", "sum"),
        rec_tds=("touchdown", "sum"),
    ).reset_index()
    receiving = receiving.merge(team_pass_attempts, left_on=["season", "week", "posteam"], right_index=True)
    receiving["target_share"] = receiving["targets"] / receiving["team_targets"]
    receiving = receiving.rename(columns={"receiver_player_id": "player_id", "receiver_player_name": "player_name", "posteam": "team"})
    receiving = receiving[receiving["player_id"].notna()]

    # Rushing stats
    rushes = pbp[pbp["rush"] == 1].copy()
    team_rush_attempts = rushes.groupby(["season", "week", "posteam"])["play_id"].count().rename("team_carries")

    rushing = rushes.groupby(["season", "week", "posteam", "rusher_player_id", "rusher_player_name"]).agg(
        carries=("play_id", "count"),
        rush_yards=("yards_gained", "sum"),
        rush_tds=("touchdown", "sum"),
    ).reset_index()
    rushing = rushing.merge(team_rush_attempts, left_on=["season", "week", "posteam"], right_index=True)
    rushing["carry_share"] = rushing["carries"] / rushing["team_carries"]
    rushing = rushing.rename(columns={"rusher_player_id": "player_id", "rusher_player_name": "player_name", "posteam": "team"})
    rushing = rushing[rushing["player_id"].notna()]

    # Passing stats (QBs)
    passes = pbp[pbp["pass"] == 1].copy()
    passing = passes.groupby(["season", "week", "posteam", "passer_player_id", "passer_player_name"]).agg(
        pass_attempts=("play_id", "count"),
        completions=("complete_pass", "sum"),
        pass_yards=("yards_gained", "sum"),
        pass_tds=("touchdown", "sum"),
    ).reset_index()
    passing = passing.rename(columns={"passer_player_id": "player_id", "passer_player_name": "player_name", "posteam": "team"})
    passing = passing[passing["player_id"].notna()]

    player_game = pd.merge(
        receiving, rushing,
        on=["season", "week", "team", "player_id", "player_name"],
        how="outer", suffixes=("_rec", "_rush")
    )
    player_game = pd.merge(
        player_game, passing,
        on=["season", "week", "team", "player_id", "player_name"],
        how="outer"
    )

    # Per-game rate columns - one game's worth, used as the raw metric that
    # apply_recency_weighting / compute_walkforward_features then blends
    # across a player's game history.
    player_game["targets_per_game"] = player_game["targets"]
    player_game["carries_per_game"] = player_game["carries"]
    player_game["pass_attempts_per_game"] = player_game["pass_attempts"]
    player_game["yards_per_target"] = player_game["rec_yards"] / player_game["targets"]
    player_game["yards_per_carry"] = player_game["rush_yards"] / player_game["carries"]
    player_game["yards_per_pass_attempt"] = player_game["pass_yards"] / player_game["pass_attempts"]
    player_game["catch_rate"] = player_game["receptions"] / player_game["targets"]
    player_game["completion_rate"] = player_game["completions"] / player_game["pass_attempts"]
    player_game["rec_td_rate"] = player_game["rec_tds"] / player_game["targets"]
    player_game["rush_td_rate"] = player_game["rush_tds"] / player_game["carries"]
    player_game["pass_td_rate"] = player_game["pass_tds"] / player_game["pass_attempts"]

    return player_game

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

    print("\nBuilding player current form (recency-weighted usage/efficiency)...")
    player_half_life = load_player_half_life()
    print(f"  Using player half-life = {player_half_life} games "
          f"({'fitted' if os.path.exists(PLAYER_HALF_LIFE_PATH) else 'default - run src/fit_props_model.py'})")
    player_game_stats = build_player_game_stats(pbp)
    player_metrics = ["targets_per_game", "carries_per_game", "pass_attempts_per_game",
                       "yards_per_target", "yards_per_carry", "yards_per_pass_attempt",
                       "catch_rate", "completion_rate", "rec_td_rate", "rush_td_rate", "pass_td_rate"]
    player_current_form = apply_recency_weighting(
        player_game_stats, "player_id", player_metrics, half_life=player_half_life,
        key_col="player_id", volume_cols=["targets", "carries", "pass_attempts"],
    )
    # player_name/team drift as players change teams - always take the most
    # recent game's values for display/roster-mapping purposes.
    latest = player_game_stats.sort_values(["season", "week"]).groupby("player_id").last().reset_index()
    player_current_form = player_current_form.merge(
        latest[["player_id", "player_name", "team"]], on="player_id", how="left"
    )
    player_current_form.to_csv(os.path.join(PROCESSED_DIR, "player_current_form.csv"), index=False)
    print(f"  Saved player_current_form.csv ({len(player_current_form)} players)")

    print("\nDone. Processed stats saved to data/processed/")

if __name__ == "__main__":
    main()
