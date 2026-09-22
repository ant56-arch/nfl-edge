"""
fetch_cfb_odds.py
Pulls current college football betting lines from the-odds-api.com - the
same provider and free tier fetch_odds.py already uses for the NFL, just a
different sport key (americanfootball_ncaaf). Kept as a separate script
(rather than folding into fetch_odds.py) because team-name matching is
genuinely different: the-odds-api returns full "School Mascot" names (e.g.
"Georgia Bulldogs") while our own data (from CollegeFootballData.com) keys
teams by school name alone (e.g. "Georgia") - matched here by longest-prefix
match against our known team list rather than a hand-built ~70-team lookup
table, since a hardcoded map would silently go stale as team names change.

Requires GitHub Secret: ODDS_API_KEY (the same key fetch_odds.py uses - this
just spends a few more of the same free-tier credits on a second sport).
"""

import requests
import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

PREFERRED_BOOKMAKER = "draftkings"

def moneyline_to_implied_prob(ml):
    if pd.isna(ml):
        return None
    if ml < 0:
        return -ml / (-ml + 100)
    return 100 / (ml + 100)

def load_known_teams():
    path = os.path.join(RAW_DIR, "cfb_teams.csv")
    if not os.path.exists(path):
        return []
    teams = pd.read_csv(path)["team"].dropna().tolist()
    return sorted(teams, key=len, reverse=True)

def match_team(odds_api_name, known_teams):
    """Longest-prefix match: the-odds-api names are '{school} {mascot}', so
    the longest known school name that the odds-api name starts with is the
    right match (checking longest-first avoids a short prefix like 'Miami'
    matching before a more specific 'Miami (OH)')."""
    for team in known_teams:
        if odds_api_name.startswith(team):
            return team
    return None

def extract_game_odds(game, known_teams):
    bookmakers = game.get("bookmakers", [])
    if not bookmakers:
        return None
    book = next((b for b in bookmakers if b["key"] == PREFERRED_BOOKMAKER), bookmakers[0])

    home_team = match_team(game["home_team"], known_teams)
    away_team = match_team(game["away_team"], known_teams)
    if not home_team or not away_team:
        return None  # not one of our covered Power/independent teams

    result = {"home_team": home_team, "away_team": away_team, "commence_time": game["commence_time"], "bookmaker": book["key"]}

    for market in book.get("markets", []):
        if market["key"] == "h2h":
            for outcome in market["outcomes"]:
                if outcome["name"] == game["home_team"]:
                    result["moneyline_home"] = outcome["price"]
                elif outcome["name"] == game["away_team"]:
                    result["moneyline_away"] = outcome["price"]
        elif market["key"] == "spreads":
            for outcome in market["outcomes"]:
                if outcome["name"] == game["home_team"]:
                    result["spread_home_point"] = outcome["point"]
        elif market["key"] == "totals":
            for outcome in market["outcomes"]:
                if outcome["name"] == "Over":
                    result["total_line"] = outcome["point"]

    result["vegas_home_favored_by"] = -result["spread_home_point"] if "spread_home_point" in result else None
    home_prob = moneyline_to_implied_prob(result.get("moneyline_home"))
    away_prob = moneyline_to_implied_prob(result.get("moneyline_away"))
    result["vegas_home_win_prob"] = home_prob / (home_prob + away_prob) if home_prob and away_prob else None

    return result

def fetch_cfb_odds():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not set - skipping CFB odds fetch.")
        return pd.DataFrame()

    known_teams = load_known_teams()
    if not known_teams:
        print("No cfb_teams.csv found yet (run fetch_cfb_data.py first) - skipping CFB odds fetch.")
        return pd.DataFrame()

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds/"
    params = {"apiKey": api_key, "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american", "dateFormat": "iso"}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    remaining = response.headers.get("x-requests-remaining")
    print(f"  API credits remaining this month: {remaining}")

    games = response.json()
    rows = [extract_game_odds(g, known_teams) for g in games]
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows)

def main():
    print("Fetching CFB odds from the-odds-api.com...")
    odds = fetch_cfb_odds()
    if odds.empty:
        print("  No CFB odds returned (may be no covered-team games this week, or key/data missing).")
    else:
        print(f"  Fetched odds for {len(odds)} games")
    out_path = os.path.join(RAW_DIR, "cfb_odds.csv")
    odds.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

if __name__ == "__main__":
    main()
