"""
team_overrides.py
Manual adjustments for offseason changes our stats-only model can't see yet:
trades, major free agent signings, coordinator changes, etc. These apply as
small nudges to a team's EPA-based ratings before real 2026 data exists to
correct it automatically (our recency weighting handles this naturally once
a team has played 4-5 real games - this file is a bridge for the first few
weeks of the season).

HOW TO USE:
Add an entry below for any team with a real, meaningful offseason change.
Keep adjustments modest and conservative - a single elite player rarely
swings a team's EPA/play by more than ~0.02-0.04, and a coordinator change is
even harder to size confidently before a snap has been played.

REMOVE entries as the season provides real data on a team - a couple of
lines in build_features.py's recency weighting will naturally take over
by week 4-5, and a stale manual override at that point does more harm
than good.
"""

TEAM_ADJUSTMENTS = {
    # Format: team abbreviation -> dict of stat adjustments (additive, not multiplicative)
    # "def_epa_per_play_allowed": negative = defense expected to be BETTER
    #   (allowing fewer expected points) than last year's numbers suggest.
    # "epa_per_play": positive = offense expected to be better.

    "LA": {
        "def_epa_per_play_allowed": -0.025,
        "reason": "Traded for Myles Garrett (reigning DPOY, coming off a record sack season) - "
                  "major upgrade to an already-strong pass rush.",
    },
    "CLE": {
        "def_epa_per_play_allowed": 0.02,
        "reason": "Traded away Myles Garrett - real downgrade to their pass rush.",
    },
    "ATL": {
        "epa_per_play": 0.0,  # Tua vs Penix - not adjusting yet, genuinely uncertain which QB
                               # will outperform our stats-only projection; leaving neutral
                               # rather than guessing.
        "reason": "Tua Tagovailoa confirmed Week 1 starter over Penix - flagged for awareness, "
                  "no adjustment applied (too uncertain to size confidently).",
    },

    # Add more entries here as you identify meaningful offseason changes.
    # A move you're not confident sizing is fine to list with 0.0 adjustments -
    # it keeps the reasoning documented even if we don't act on it yet.
}

def apply_overrides(team_stats):
    """
    Applies TEAM_ADJUSTMENTS to a team_stats DataFrame (indexed by team).
    Adjusts both the recency-weighted "current form" columns and the
    season-average columns, since game_predictions.py can use either.
    Returns the same DataFrame with adjustments applied in place.
    """
    applied = []
    for team, adj in TEAM_ADJUSTMENTS.items():
        if team not in team_stats.index:
            continue
        for stat, value in adj.items():
            if stat == "reason":
                continue
            if value == 0.0:
                continue
            for col in [stat, f"{stat}_season_avg"]:
                if col in team_stats.columns:
                    team_stats.loc[team, col] += value
        if any(v != 0.0 for k, v in adj.items() if k != "reason"):
            applied.append((team, adj.get("reason", "no reason given")))

    if applied:
        print("  Applied manual team overrides:")
        for team, reason in applied:
            print(f"    {team}: {reason}")

    return team_stats
