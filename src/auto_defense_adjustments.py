"""
auto_defense_adjustments.py
Automatically computes each team's defensive rating adjustment based on
personnel changes - no manual research required. This is the automated
replacement for hand-typing things like the Myles Garrett trade into
team_overrides.py.

HOW IT WORKS:
1. For each team, sum the current roster's defensive players' portable
   career defensive value (their own track record, wherever they earned it).
2. Compare that to the team's historical defensive value (how much value the
   PREVIOUS personnel who generated last year's team_stats number actually
   produced).
3. The ratio between these tells us whether this year's defensive personnel
   is stronger or weaker than what our historical stats are measuring, and
   by how much - which becomes an automatic adjustment to def_epa_per_play_allowed.

This directly replaces the need to notice and hand-code trades like
Garrett-to-the-Rams: the moment current_roster.csv reflects a trade, this
recalculates automatically on the next pipeline run.

LIMITS (stay honest about these):
- Only weights sacks and interceptions right now (see build_defense_stats.py)
  - a real defensive value model would include pressures, tackles for loss,
  passes defended, and ideally snap-share-adjusted rates. This is a v1.
- Coordinator/scheme changes aren't captured at all - a great personnel group
  playing a worse scheme won't be caught by this. team_overrides.py remains
  the place for judgment calls like that.
- Adjustments are capped (see MAX_ADJUSTMENT) to avoid small-sample noise
  in defensive_value swinging a team's rating wildly.
"""

import pandas as pd
import numpy as np
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# How much a 2x (or 0.5x) swing in personnel value translates to in EPA terms.
# Kept conservative - see team_overrides.py for the "how big should a real,
# researched change be" reference point (~0.02-0.035 for a historic move).
ADJUSTMENT_SCALE = 0.03
MAX_ADJUSTMENT = 0.04  # hard cap either direction, regardless of how extreme the ratio gets
MIN_HISTORICAL_VALUE = 3.0  # floor to avoid divide-by-near-zero blowing up the ratio for low-event teams

DEFENSIVE_POSITIONS = {"DL", "DE", "DT", "EDGE", "LB", "ILB", "OLB", "CB", "DB", "S", "FS", "SS"}

def load_data():
    player_value = pd.read_csv(os.path.join(PROCESSED_DIR, "player_defense_value.csv"))
    team_historical = pd.read_csv(os.path.join(PROCESSED_DIR, "team_historical_defense_value.csv"))
    current_roster = pd.read_csv(os.path.join(RAW_DIR, "current_roster.csv"), low_memory=False)
    return player_value, team_historical, current_roster

def compute_current_roster_value(current_roster, player_value):
    """Sum each team's current defensive players' portable career value."""
    if "position" in current_roster.columns:
        defense_roster = current_roster[current_roster["position"].isin(DEFENSIVE_POSITIONS)]
    else:
        defense_roster = current_roster  # fallback: no position filter available

    merged = defense_roster.merge(player_value, left_on="gsis_id", right_on="player_id", how="left")
    merged["career_defensive_value"] = merged["career_defensive_value"].fillna(0.0)

    return merged.groupby("team")["career_defensive_value"].sum().reset_index().rename(
        columns={"career_defensive_value": "current_roster_value"})

def compute_adjustments(current_roster, player_value, team_historical):
    current_value = compute_current_roster_value(current_roster, player_value)
    merged = current_value.merge(team_historical, on="team", how="outer").fillna(0.0)

    adjustments = {}
    details = []
    for _, row in merged.iterrows():
        team = row["team"]
        historical = max(row["historical_defensive_value"], MIN_HISTORICAL_VALUE)
        current = row["current_roster_value"]

        ratio = current / historical
        # ratio > 1 means stronger personnel now than what generated last year's
        # number -> defense should be BETTER -> def_epa_per_play_allowed should
        # go DOWN (negative = fewer expected points allowed per play).
        raw_adjustment = (1 - ratio) * ADJUSTMENT_SCALE
        adjustment = float(np.clip(raw_adjustment, -MAX_ADJUSTMENT, MAX_ADJUSTMENT))

        if abs(adjustment) > 0.002:  # skip reporting negligible noise
            adjustments[team] = adjustment
            details.append((team, current, historical, ratio, adjustment))

    return adjustments, details

def apply_auto_adjustments(team_stats):
    """Same interface as team_overrides.apply_overrides - applies in place and returns."""
    player_value_path = os.path.join(PROCESSED_DIR, "player_defense_value.csv")
    team_historical_path = os.path.join(PROCESSED_DIR, "team_historical_defense_value.csv")

    if not (os.path.exists(player_value_path) and os.path.exists(team_historical_path)):
        return team_stats  # defense stats haven't been built yet - skip gracefully

    player_value, team_historical, current_roster = load_data()
    adjustments, details = compute_adjustments(current_roster, player_value, team_historical)

    if details:
        print("  Applied automatic defense personnel adjustments:")
        for team, current, historical, ratio, adj in details:
            print(f"    {team}: roster value {current:.1f} vs historical {historical:.1f} (ratio {ratio:.2f}) -> {adj:+.4f} EPA/play")

    for team, adj in adjustments.items():
        if team not in team_stats.index:
            continue
        for col in ["def_epa_per_play_allowed", "def_epa_per_play_allowed_season_avg"]:
            if col in team_stats.columns:
                team_stats.loc[team, col] += adj

    return team_stats
