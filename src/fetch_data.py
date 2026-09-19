"""
fetch_data.py
Pulls raw NFL data from nflverse's public data releases.
Source: https://github.com/nflverse/nflverse-data

nflverse publishes free, clean play-by-play and roster data used widely
in the NFL analytics community (this is the same underlying data source
that powers most public EPA/advanced-stats sites).

This script downloads:
  - Play-by-play data for the last 2 completed seasons + current season
  - Weekly schedules (for upcoming matchups)
  - Weekly rosters (for player props - active players, positions)

Output: raw CSVs saved to data/raw/
"""

import requests
import pandas as pd
import os
from datetime import datetime

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

BASE = "https://github.com/nflverse/nflverse-data/releases/download"

def current_nfl_season():
    """
    NFL season is labeled by the year it starts in.
    If we're before March, we're still in last year's season (playoffs/offseason).
    """
    now = datetime.utcnow()
    return now.year if now.month >= 3 else now.year - 1

def get_seasons_to_pull(lookback_years=2):
    """Returns the list of seasons to pull: last N completed + current."""
    current = current_nfl_season()
    return list(range(current - lookback_years, current + 1))

def download_csv_gz(url, dest_path):
    """Download a gzipped CSV from a nflverse release URL."""
    print(f"  Fetching: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)
    print(f"  Saved: {dest_path} ({len(r.content) / 1_000_000:.1f} MB)")

def fetch_play_by_play(seasons):
    """Pull play-by-play data for each season (this is the core advanced-stats source)."""
    frames = []
    for season in seasons:
        dest = os.path.join(RAW_DIR, f"pbp_{season}.csv.gz")
        url = f"{BASE}/pbp/play_by_play_{season}.csv.gz"
        try:
            download_csv_gz(url, dest)
            df = pd.read_csv(dest, compression="gzip", low_memory=False)
            df["season"] = season
            frames.append(df)
        except requests.exceptions.HTTPError as e:
            print(f"  Skipping {season} (not available yet): {e}")
    if not frames:
        raise RuntimeError("No play-by-play data could be fetched.")
    return pd.concat(frames, ignore_index=True)

def fetch_schedules(seasons):
    """Pull game schedules (needed to know who's playing whom, when)."""
    url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    df = pd.read_csv(url)
    return df[df["season"].isin(seasons)]

def fetch_team_info():
    """Pull official team colors and logo URLs - the same public dataset
    nflreadr::load_teams() is built on, redistributed by nflverse for exactly
    this kind of use (team branding on public analytics sites)."""
    url = "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/teams_colors_logos.csv"
    df = pd.read_csv(url)
    return df[["team_abbr", "team_color", "team_color2", "team_logo_espn"]]

def fetch_rosters(seasons):
    """Pull weekly rosters for HISTORICAL seasons (used for feature engineering)."""
    frames = []
    for season in seasons:
        url = f"{BASE}/weekly_rosters/roster_weekly_{season}.csv"
        try:
            df = pd.read_csv(url, low_memory=False)
            frames.append(df)
        except requests.exceptions.HTTPError as e:
            print(f"  Skipping roster {season}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def fetch_current_roster(season):
    """
    Pull the CURRENT full-season roster - this is nflverse's continuously
    updated roster file that reflects free agency signings, trades, and
    releases year-round (unlike weekly_rosters, which only exists for weeks
    that have actually been played). This is our source of truth for
    "who is on what team right now," which matters most in the offseason
    when the previous season's play-by-play data is stale on personnel.
    """
    url = f"{BASE}/rosters/roster_{season}.csv"
    try:
        df = pd.read_csv(url, low_memory=False)
        return df
    except requests.exceptions.HTTPError as e:
        print(f"  Current roster not available yet for {season}: {e}")
        return pd.DataFrame()

def main():
    seasons = get_seasons_to_pull(lookback_years=2)
    print(f"Pulling data for seasons: {seasons}")

    print("\n[1/4] Play-by-play data...")
    pbp = fetch_play_by_play(seasons)
    pbp.to_parquet(os.path.join(RAW_DIR, "pbp_combined.parquet"))
    print(f"  Total plays: {len(pbp):,}")

    print("\n[2/4] Schedules...")
    schedules = fetch_schedules(seasons)
    schedules.to_csv(os.path.join(RAW_DIR, "schedules.csv"), index=False)
    print(f"  Total games: {len(schedules):,}")

    print("\n[3/5] Team colors and logos...")
    teams = fetch_team_info()
    teams.to_csv(os.path.join(RAW_DIR, "teams.csv"), index=False)
    print(f"  Total teams: {len(teams):,}")

    print("\n[4/5] Historical weekly rosters (for feature engineering)...")
    rosters = fetch_rosters(seasons)
    if not rosters.empty:
        rosters.to_parquet(os.path.join(RAW_DIR, "rosters.parquet"))
        print(f"  Total roster entries: {len(rosters):,}")

    print("\n[5/5] Current live roster (who's on what team right now)...")
    current_season = current_nfl_season()
    current_roster = fetch_current_roster(current_season)
    if not current_roster.empty:
        current_roster.to_csv(os.path.join(RAW_DIR, "current_roster.csv"), index=False)
        print(f"  Total current roster entries: {len(current_roster):,}")
        # Sanity check: show a few recently-signed/traded players if the column exists
        if "full_name" in current_roster.columns and "team" in current_roster.columns:
            print(f"  Sample: {current_roster[['full_name', 'team', 'position']].sample(min(3, len(current_roster))).to_dict('records')}")

    print("\nDone. Raw data saved to data/raw/")

if __name__ == "__main__":
    main()
