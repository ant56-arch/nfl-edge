"""
fetch_cfb_data.py
Pulls college football data from CollegeFootballData.com's API (CFBD) - the
de facto free public API for CFB, analogous to what nflverse is for the NFL
side of this project. Unlike nflverse, CFBD requires a free API key (sign up
at collegefootballdata.com/key) passed as a Bearer token - set as the
CFBD_API_KEY repo secret.

Scoped to Power-conference (SEC, Big Ten, Big 12, ACC) and FBS independent
teams (~70 programs, see POWER_CONFERENCES) - Group of Five programs have
much thinner data and far less public betting/market interest, so including
them would mostly add noise to a model this size rather than value.

Degrades gracefully: if CFBD_API_KEY isn't set (e.g. the secret hasn't been
added to the repo yet), this exits without writing anything rather than
crashing the pipeline - every downstream CFB script checks for missing input
files the same way and skips instead of erroring, so the existing NFL
pipeline is unaffected either way.

Output: raw CSVs saved to data/raw/ (cfb_games.csv, cfb_advanced_stats.csv,
cfb_teams.csv), the same directory the NFL raw data lives in.
"""

import requests
import pandas as pd
import os
from datetime import datetime

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

BASE = "https://api.collegefootballdata.com"

POWER_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC", "FBS Independents"}

def current_cfb_season():
    """Same convention as the NFL side: a season runs Aug-Jan, labeled by the
    year it starts in. The CFP championship is mid-January, so before March
    we're still finishing last year's season."""
    now = datetime.utcnow()
    return now.year if now.month >= 3 else now.year - 1

def _headers():
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        return None
    return {"Authorization": f"Bearer {key}"}

def _get(path, params):
    r = requests.get(f"{BASE}{path}", params=params, headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()

def fetch_team_info(season):
    data = _get("/teams/fbs", {"year": season})
    rows = []
    for t in data:
        conference = t.get("conference")
        if conference not in POWER_CONFERENCES:
            continue
        logos = t.get("logos") or []
        rows.append({
            "team": t.get("school"),
            "conference": conference,
            "color": t.get("color") or "#94a3b8",
            "alt_color": t.get("alt_color") or t.get("color") or "#94a3b8",
            "logo": logos[0] if logos else "",
        })
    return pd.DataFrame(rows)

def _add_kickoff_columns(games):
    if "gameday" not in games.columns:
        return games
    dt_utc = pd.to_datetime(games["gameday"], errors="coerce", utc=True)
    dt_et = dt_utc.dt.tz_convert("US/Eastern")
    games["weekday"] = dt_et.dt.day_name()
    games["gametime"] = dt_et.dt.strftime("%H:%M")
    games["gameday"] = dt_et.dt.strftime("%Y-%m-%d")
    return games

def fetch_games(season):
    frames = []
    for season_type in ("regular", "postseason"):
        try:
            data = _get("/games", {"year": season, "seasonType": season_type})
            frames.append(pd.DataFrame(data))
        except requests.exceptions.HTTPError as e:
            print(f"  Skipping {season_type} games for {season}: {e}")
    if not frames or all(f.empty for f in frames):
        return pd.DataFrame()
    games = pd.concat(frames, ignore_index=True)
    games = games.rename(columns={
        "homeTeam": "home_team", "awayTeam": "away_team",
        "homePoints": "home_score", "awayPoints": "away_score",
        "homeConference": "home_conference", "awayConference": "away_conference",
        "seasonType": "game_type", "startDate": "gameday",
    })
    games["game_type"] = games["game_type"].map({"regular": "REG", "postseason": "POST"})
    games = _add_kickoff_columns(games)
    return games

def fetch_lines(season):
    frames = []
    for season_type in ("regular", "postseason"):
        try:
            data = _get("/lines", {"year": season, "seasonType": season_type})
            frames.append(pd.DataFrame(data))
        except requests.exceptions.HTTPError as e:
            print(f"  Skipping {season_type} lines for {season}: {e}")
    if not frames or all(f.empty for f in frames):
        return pd.DataFrame()
    lines = pd.concat(frames, ignore_index=True)
    if lines.empty:
        return lines

    def pick_line(rows):
        """One consistent line source per game, preferring CFBD's own
        consensus line, then DraftKings, then whatever's first - the same
        idea as fetch_odds.py preferring one bookmaker over averaging."""
        if not rows:
            return {}
        for preferred in ("consensus", "DraftKings"):
            match = next((r for r in rows if r.get("provider") == preferred), None)
            if match:
                return match
        return rows[0]

    rows = []
    for _, g in lines.iterrows():
        chosen = pick_line(g.get("lines") or [])
        rows.append({
            "home_team": g.get("homeTeam"), "away_team": g.get("awayTeam"),
            "vegas_home_favored_by": -chosen["spread"] if chosen.get("spread") is not None else None,
            "vegas_total": chosen.get("overUnder"),
        })
    return pd.DataFrame(rows)

def fetch_advanced_stats(season):
    frames = []
    for season_type in ("regular", "postseason"):
        try:
            data = _get("/stats/game/advanced", {"year": season, "seasonType": season_type})
            frames.append(pd.DataFrame(data))
        except requests.exceptions.HTTPError as e:
            print(f"  Skipping {season_type} advanced stats for {season}: {e}")
    if not frames or all(f.empty for f in frames):
        return pd.DataFrame()
    stats = pd.concat(frames, ignore_index=True)
    if stats.empty:
        return stats

    rows = []
    for _, r in stats.iterrows():
        offense = r.get("offense") or {}
        defense = r.get("defense") or {}
        rows.append({
            "season": r.get("season"), "week": r.get("week"), "team": r.get("team"), "opponent": r.get("opponent"),
            "off_ppa_per_play": offense.get("ppa"),
            "off_success_rate": offense.get("successRate"),
            "off_explosiveness": offense.get("explosiveness"),
            "def_ppa_per_play_allowed": defense.get("ppa"),
            "def_success_rate_allowed": defense.get("successRate"),
            "def_explosiveness_allowed": defense.get("explosiveness"),
        })
    return pd.DataFrame(rows)

def main():
    if _headers() is None:
        print("CFBD_API_KEY not set - skipping college football data fetch. "
              "Get a free key at https://collegefootballdata.com/key and add it "
              "as a repo secret to enable college football predictions.")
        return

    season = current_cfb_season()
    print(f"Pulling college football data for season {season}...")

    print("\n[1/3] Team info (Power conferences + independents)...")
    teams = fetch_team_info(season)
    teams.to_csv(os.path.join(RAW_DIR, "cfb_teams.csv"), index=False)
    print(f"  Total teams: {len(teams):,}")
    covered = set(teams["team"]) if not teams.empty else set()

    print("\n[2/3] Games + closing lines...")
    games = fetch_games(season)
    if not games.empty and covered:
        games = games[games["home_team"].isin(covered) | games["away_team"].isin(covered)]
    lines = fetch_lines(season)
    if not games.empty and not lines.empty:
        games = games.merge(lines, on=["home_team", "away_team"], how="left")
    games.to_csv(os.path.join(RAW_DIR, "cfb_games.csv"), index=False)
    print(f"  Total games: {len(games):,}")

    print("\n[3/3] Advanced per-game team stats (PPA, success rate, explosiveness)...")
    stats = fetch_advanced_stats(season)
    if not stats.empty and covered:
        stats = stats[stats["team"].isin(covered)]
    stats.to_csv(os.path.join(RAW_DIR, "cfb_advanced_stats.csv"), index=False)
    print(f"  Total team-game rows: {len(stats):,}")

    print("\nDone. Raw CFB data saved to data/raw/")

if __name__ == "__main__":
    main()
