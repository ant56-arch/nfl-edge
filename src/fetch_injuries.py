"""
fetch_injuries.py
Pulls the current week's official injury report from nflverse. Used to avoid
projecting big stat lines for players who are Out, Doubtful, or dealing with
something that should discount their projection (Questionable).

report_status values (per nflverse's standard schema): "Out", "Doubtful",
"Questionable". A player with no row in this data is presumed healthy.
"""

import pandas as pd
import os
from fetch_data import current_nfl_season

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
BASE = "https://github.com/nflverse/nflverse-data/releases/download"

def fetch_injuries(season):
    url = f"{BASE}/injuries/injuries_{season}.csv"
    try:
        df = pd.read_csv(url, low_memory=False)
        return df
    except Exception as e:
        print(f"  Could not fetch injuries for {season}: {e}")
        return pd.DataFrame()

def main():
    season = current_nfl_season()
    print(f"Fetching injury report for {season}...")
    injuries = fetch_injuries(season)

    if injuries.empty:
        print("  No injury data available.")
    else:
        # Keep only the most recent report per player (injury reports get
        # updated multiple times during the week - Wed/Thu/Fri practice
        # reports, then a final designation before kickoff)
        if "week" in injuries.columns:
            latest_week = injuries["week"].max()
            injuries = injuries[injuries["week"] == latest_week]
        print(f"  {len(injuries)} injury report entries (most recent week)")
        if "report_status" in injuries.columns:
            print(f"  Status breakdown: {injuries['report_status'].value_counts().to_dict()}")

    out_path = os.path.join(RAW_DIR, "injuries.csv")
    injuries.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

if __name__ == "__main__":
    main()
