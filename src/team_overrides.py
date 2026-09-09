"""
team_overrides.py
Manual adjustments for offseason changes our AUTOMATIC systems can't see.

As of this version, auto_defense_adjustments.py automatically detects
defensive personnel changes (trades, signings) by following each player's
real sack/interception history via current_roster.csv - no manual research
needed for that category anymore. That's why entries like the Myles Garrett
trade (LA/CLE) and the McDuffie/Watson move (LA/KC) were REMOVED from here -
the automatic system now handles them on its own, and leaving them here too
would double-count the adjustment.

What's left here is what automation genuinely can't capture yet:
- Offensive skill-position team-strength effects (our player-continuity
  system correctly projects an individual player's OWN stats on their new
  team, but doesn't yet roll that up into the team's overall offensive EPA
  rating the way we do for defense)
- Judgment calls with no clean statistical proxy (a QB competition, a
  coordinator change, "this team's identity changed" type calls)

HOW TO USE:
Add an entry below only for changes that don't already flow through
current_roster.csv + our automatic systems. Keep adjustments modest -
~0.01-0.02 for a real, material offensive addition; reserve anything larger
for something historically significant.

REMOVE entries as the season provides real data on a team - by week 4-5 our
recency weighting naturally takes over.

Last reviewed Sept 9, 2026 (Week 1).
"""

TEAM_ADJUSTMENTS = {
    "BUF": {
        "epa_per_play": 0.015,
        "reason": "Added WR D.J. Moore - Josh Allen hasn't had a receiver top 821 receiving yards since "
                  "2023. Not caught by the automatic system since that only models defensive personnel "
                  "value, not offensive skill-position impact on team-level offensive EPA.",
    },
    "NO": {
        "epa_per_play": 0.012,
        "reason": "Signed RB Travis Etienne (three 1,000-yard seasons in Jacksonville) after finishing "
                  "28th in rushing last season. Same reasoning as BUF above.",
    },
    "ATL": {
        "epa_per_play": 0.0,  # intentionally not adjusted - see reason
        "reason": "Tua Tagovailoa confirmed Week 1 starter over Penix - flagged for awareness, no "
                  "adjustment applied (too uncertain to size confidently which QB outperforms our "
                  "stats-only projection before either has taken a real 2026 snap). No statistical "
                  "proxy could catch this either, since it's a competition outcome, not a personnel change.",
    },

    # Add more entries here only for offense-side moves or judgment calls -
    # defensive personnel changes should be caught automatically now.
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
