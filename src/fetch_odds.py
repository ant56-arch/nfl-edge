"""
fetch_odds.py
Pulls current NFL betting lines (moneyline, spread, total) from the-odds-api.com's
free tier, for comparison against our own model's projections.

Requires GitHub Secret: ODDS_API_KEY

Free tier note: 500 credits/month. Each call here costs regions x markets
credits (1 region x 3 markets = 3 credits). At our Tue/Fri cadence, that's
roughly 24 credits/month - comfortably within the free allowance.
"""

import requests
import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# the-odds-api.com returns full team names; our data uses nflverse abbreviations.
TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

PREFERRED_BOOKMAKER = "draftkings"  # fall back to first available if not present

def moneyline_to_implied_prob(ml):
    if pd.isna(ml):
        return None
    if ml < 0:
        return -ml / (-ml + 100)
    return 100 / (ml + 100)

def extract_game_odds(game):
    """Pull the odds we care about from one game's bookmaker list, preferring
    one consistent book so lines aren't a weird average of different books."""
    bookmakers = game.get("bookmakers", [])
    if not bookmakers:
        return None

    book = next((b for b in bookmakers if b["key"] == PREFERRED_BOOKMAKER), bookmakers[0])

    home_team = game["home_team"]
    away_team = game["away_team"]

    result = {
        "home_team": TEAM_NAME_TO_ABBR.get(home_team, home_team),
        "away_team": TEAM_NAME_TO_ABBR.get(away_team, away_team),
        "commence_time": game["commence_time"],
        "bookmaker": book["key"],
    }

    for market in book.get("markets", []):
        if market["key"] == "h2h":
            for outcome in market["outcomes"]:
                if outcome["name"] == home_team:
                    result["moneyline_home"] = outcome["price"]
                elif outcome["name"] == away_team:
                    result["moneyline_away"] = outcome["price"]

        elif market["key"] == "spreads":
            for outcome in market["outcomes"]:
                if outcome["name"] == home_team:
                    result["spread_home_point"] = outcome["point"]
                    result["spread_home_price"] = outcome["price"]
                elif outcome["name"] == away_team:
                    result["spread_away_point"] = outcome["point"]

        elif market["key"] == "totals":
            for outcome in market["outcomes"]:
                if outcome["name"] == "Over":
                    result["total_line"] = outcome["point"]
                    result["over_price"] = outcome["price"]
                elif outcome["name"] == "Under":
                    result["under_price"] = outcome["price"]

    # Convenience fields for downstream comparison
    result["vegas_home_favored_by"] = -result.get("spread_home_point", 0) if "spread_home_point" in result else None
    result["vegas_home_win_prob_raw"] = moneyline_to_implied_prob(result.get("moneyline_home"))
    result["vegas_away_win_prob_raw"] = moneyline_to_implied_prob(result.get("moneyline_away"))

    # De-vig: raw implied probabilities always sum to >100% (the bookmaker's
    # margin). Normalize so they sum to 100%, giving a fairer "true" probability
    # to compare against our own model's win probability.
    if result["vegas_home_win_prob_raw"] and result["vegas_away_win_prob_raw"]:
        total_prob = result["vegas_home_win_prob_raw"] + result["vegas_away_win_prob_raw"]
        result["vegas_home_win_prob"] = result["vegas_home_win_prob_raw"] / total_prob
    else:
        result["vegas_home_win_prob"] = None

    return result

def fetch_nfl_odds():
    api_key = os.environ["ODDS_API_KEY"]
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    remaining = response.headers.get("x-requests-remaining")
    used = response.headers.get("x-requests-used")
    print(f"  API credits used this call, remaining this month: {remaining} (used so far: {used})")

    games = response.json()
    rows = [extract_game_odds(g) for g in games]
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows)

def main():
    print("Fetching NFL odds from the-odds-api.com...")
    odds = fetch_nfl_odds()

    if odds.empty:
        print("  No odds returned (may be no games currently listed, or offseason).")
    else:
        print(f"  Fetched odds for {len(odds)} games")

    out_path = os.path.join(RAW_DIR, "odds.csv")
    odds.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

if __name__ == "__main__":
    main()
