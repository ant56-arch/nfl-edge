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
swings a team's EPA/play by more than ~0.02-0.03, and a role-player addition
is worth much less than that (~0.005-0.01). Bigger adjustments are reserved
for genuinely historic moves (a reigning DPOY changing teams, for example).

REMOVE entries as the season provides real data on a team - by week 4-5 our
recency weighting naturally takes over, and a stale manual override at that
point does more harm than good.

Researched as of Sept 8, 2026 (day before Week 1). Not exhaustive - these
are the moves with clear, well-reported, material positional impact. Smaller
depth-chart moves are left out since they're hard to size confidently.
"""

TEAM_ADJUSTMENTS = {
    "LA": {
        "def_epa_per_play_allowed": -0.035,
        "reason": "Traded for Myles Garrett (reigning DPOY, set the single-season sack record) AND "
                  "traded for/extended All-Pro CB Trent McDuffie AND signed CB Jaylen Watson (McDuffie's "
                  "ex-Chiefs running mate) - a full defensive overhaul. Partially offset by trading away "
                  "Jared Verse (a 2x Pro Bowl edge rusher) as part of the Garrett deal.",
    },
    "CLE": {
        "def_epa_per_play_allowed": 0.02,
        "reason": "Traded away Myles Garrett - real downgrade to their pass rush.",
    },
    "KC": {
        "def_epa_per_play_allowed": 0.02,
        "reason": "Lost both starting corners (Trent McDuffie traded, Jaylen Watson left in free agency) "
                  "to the Rams. Drafted a rookie CB (Delane) as a replacement, but rookies rarely replace "
                  "lost All-Pro production in year one.",
    },
    "BUF": {
        "epa_per_play": 0.015,
        "reason": "Added WR D.J. Moore - Josh Allen hasn't had a receiver top 821 receiving yards since "
                  "2023, so this fills a real offensive gap.",
    },
    "NO": {
        "epa_per_play": 0.012,
        "reason": "Signed RB Travis Etienne (three 1,000-yard seasons in Jacksonville) after finishing "
                  "28th in rushing last season - real running-game upgrade for 2nd-year QB Tyler Shough.",
    },
    "CIN": {
        "def_epa_per_play_allowed": -0.015,
        "reason": "Defensive line overhaul: signed DL Jonathan Allen and Boye Mafe, traded for an "
                  "interior lineman (Lawrence) - meaningful pass rush upgrade.",
    },
    "WAS": {
        "def_epa_per_play_allowed": -0.008,
        "reason": "Signed pass rushers Jadeveon Oweh and K'Lavon Chaisson to address a subpar pass rush - "
                  "real but more modest upgrade than the moves above.",
    },
    "ATL": {
        "epa_per_play": 0.0,  # intentionally not adjusted - see reason
        "reason": "Tua Tagovailoa confirmed Week 1 starter over Penix - flagged for awareness, no "
                  "adjustment applied (too uncertain to size confidently which QB outperforms our "
                  "stats-only projection before either has taken a real 2026 snap).",
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
