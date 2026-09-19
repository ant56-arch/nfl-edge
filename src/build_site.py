"""
build_site.py
Generates the static NFL Edge website (dist/) from the same processed data
the email used to read - the email is retired; this is the sole output now.

Pages:
  index.html    - Home: notable model-vs-market gaps, track record, and this
                  week's slate only (kept short - see teams.html for the rest
                  of the season)
  teams.html    - every 2026 game, week 1 through the postseason, browsable
                  via a dropdown - past weeks graded, future weeks straight
                  from game_predictions.csv (which already forecasts the
                  whole season, not just the imminent week)
  players.html  - player projections by category (Passing/Rushing/Receiving),
                  switchable via in-page tabs so it isn't one long scroll
  history.html  - every graded week across all seasons (backfill included),
                  browsable via a dropdown (client-side, no per-week routing
                  needed for a site this size)
  accuracy.html - trend charts of our model's own accuracy vs Vegas's over time

Published to GitHub Pages by weekly-picks.yml. This script only builds the
dist/ directory - the workflow's own steps upload and deploy it.
"""

import pandas as pd
import numpy as np
import json
import os
import shutil
import hashlib
from datetime import datetime, timezone

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TRACKING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tracking")
WEB_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "web")
DIST_DIR = os.path.join(os.path.dirname(__file__), "..", "dist")

DASH = "—"

# The 2024-2025 tracking history was backfilled after the fact (see
# backfill_tracking.py's BACKFILL_SEASONS) by running a holdout-trained model
# against seasons that already happened - useful as an archive of what the
# system would have called, but not real picks made before kickoff. The
# accuracy trend charts are meant to show actual live performance, so they
# start at the first season the tracker ran for real.
LIVE_TRACKING_START_SEASON = 2026

# Minimal hand-drawn line icons (24x24, currentColor stroke) for card headers -
# avoids pulling in an icon font/library for five glyphs.
ICONS = {
    "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="8" y1="3" x2="8" y2="7"/><line x1="16" y1="3" x2="16" y2="7"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/>',
    "bars": '<line x1="5" y1="20" x2="5" y2="12"/><line x1="12" y1="20" x2="12" y2="7"/><line x1="19" y1="20" x2="19" y2="14"/><line x1="3" y1="20.5" x2="21" y2="20.5"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><polyline points="12,7 12,12 16,14"/>',
    "trend": '<polyline points="4,17 10,11 14,15 20,7"/><polyline points="14,7 20,7 20,13"/>',
    "shield": '<path d="M12 3l7 3v6c0 4.5-3 8-7 9-4-1-7-4.5-7-9V6z"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.5 3.5-7 8-7s8 2.5 8 7"/>',
}

# Well-known primary team colors (public branding facts, not logos/trademarks),
# used only as a thin left-border accent so rows are scannable at a glance
# without reading every cell - not an attempt at official team branding.
TEAM_COLORS = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#FF3C00",
    "DAL": "#041E42", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LA": "#003594", "LAC": "#0080C6", "LV": "#A5ACAF", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#69BE28",
    "SF": "#AA0000", "TB": "#D50A0A", "TEN": "#4B92DB", "WAS": "#5A1414",
}

def team_color(abbr):
    return TEAM_COLORS.get(abbr, "#94a3b8")

def model_pick(row):
    """The pure, unblended model's own call for a game - favored team, margin,
    and win probability - as opposed to the market-blended 'sharp' numbers.
    Used everywhere the site claims something is genuinely OUR pick, since
    the blended line currently has a 0.0 weight on the model for spreads
    (see fitted_coefficients.json) and is therefore identical to Vegas."""
    home_favored = row["model_spread"] > 0
    favored_team = row["home_team"] if home_favored else row["away_team"]
    win_pct = row["model_home_win_prob"] if home_favored else 1 - row["model_home_win_prob"]
    return {
        "favored_team": favored_team,
        "favored_by": abs(row["model_spread"]),
        "win_pct": win_pct,
        "total": row["model_total"],
    }

def icon(name):
    return (f'<svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')

_ASSET_VERSION = None

def asset_version():
    """A content hash of style.css/site.js, appended as a ?v= query string so
    a redeploy always busts stale browser/CDN caches of these files - without
    it, an HTML page can update (new markup, new classes) while a visitor's
    browser keeps serving its old cached stylesheet that doesn't know about
    them yet, making the site look broken until a hard refresh."""
    global _ASSET_VERSION
    if _ASSET_VERSION is None:
        h = hashlib.md5()
        for name in ("style.css", "site.js"):
            with open(os.path.join(WEB_SRC_DIR, name), "rb") as f:
                h.update(f.read())
        _ASSET_VERSION = h.hexdigest()[:10]
    return _ASSET_VERSION

def load_data():
    games = pd.read_csv(os.path.join(PROCESSED_DIR, "game_predictions.csv"))
    props = pd.read_csv(os.path.join(PROCESSED_DIR, "player_props.csv"))

    comparison = None
    comparison_path = os.path.join(PROCESSED_DIR, "vegas_comparison.csv")
    if os.path.exists(comparison_path):
        comparison = pd.read_csv(comparison_path)

    accuracy_summary = None
    summary_path = os.path.join(TRACKING_DIR, "accuracy_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            accuracy_summary = json.load(f)
        if accuracy_summary.get("n_graded_games", 0) == 0:
            accuracy_summary = None

    log_path = os.path.join(TRACKING_DIR, "predictions_log.csv")
    log = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()

    return games, props, comparison, accuracy_summary, log

def next_week_games(games):
    if games.empty:
        return games
    nxt = games.sort_values(["season", "week"]).iloc[0][["season", "week"]]
    return games[(games["season"] == nxt["season"]) & (games["week"] == nxt["week"])]

def format_kickoff(weekday, gametime):
    if not weekday or pd.isna(weekday) or not gametime or pd.isna(gametime):
        return ""
    try:
        hour, minute = (int(x) for x in str(gametime).split(":"))
    except ValueError:
        return ""
    period = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12 or 12
    return str(weekday)[:3] + " " + str(hour_12) + ":" + format(minute, "02d") + " " + period

def pill(text, style):
    return f'<span class="pill pill-{style}">{text}</span>'

FAVICON = ('data:image/svg+xml,'
    '%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 32 32%22%3E'
    '%3Crect width=%2232%22 height=%2232%22 rx=%227%22 fill=%22%23111827%22/%3E'
    '%3Cpath d=%22M9 23V9h3.4l6.6 9.3V9H22v14h-3.4L12 13.6V23z%22 fill=%22%23c2410c%22/%3E'
    '%3C/svg%3E')

def page_shell(title, active_tab, body_html):
    tabs = [
        ("index.html", "index", "Home"),
        ("teams.html", "teams", "Teams"),
        ("players.html", "players", "Players"),
        ("history.html", "history", "History"),
        ("accuracy.html", "accuracy", "Accuracy"),
    ]
    nav = "".join(
        f'<a href="{href}" class="{"active" if tab == active_tab else ""}">{label}</a>'
        for href, tab, label in tabs
    )
    now = datetime.now(timezone.utc)
    generated = now.strftime("%b %d, %Y %H:%M UTC")
    ver = asset_version()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} - NFL Edge</title>
<meta name="description" content="Model-driven NFL spreads, totals and player props, validated against the closing Vegas line every week.">
<link rel="icon" href="{FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css?v={ver}">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
</head>
<body>
<div class="topbar"></div>
<div class="hero-glow" aria-hidden="true"></div>
<div class="wrap">
  <header class="masthead">
    <div class="masthead-row">
      <div>
        <div class="wordmark">NFL <span>EDGE</span></div>
        <div class="tagline">Model-driven spreads, totals &amp; props - graded against the closing line every week.</div>
      </div>
      <div class="updated-chip">Updated {generated}</div>
    </div>
  </header>
  <nav class="tabs">{nav}</nav>
  {body_html}
  <footer class="site-footer">
    <div class="footer-grid">
      <div class="footer-col">
        <div class="footer-heading">The Model</div>
        <p>A coefficients-fit EPA model blended with the live market line, weights validated on held-out seasons - not a gut feeling with a spreadsheet attached.</p>
      </div>
      <div class="footer-col">
        <div class="footer-heading">Data &amp; Sources</div>
        <p>Play-by-play and schedules via nflverse. Vegas lines via DraftKings, through the-odds-api.com, where available.</p>
      </div>
      <div class="footer-col">
        <div class="footer-heading">Disclaimer</div>
        <p>For entertainment and research only. Not betting advice - past accuracy does not guarantee future results.</p>
      </div>
    </div>
    <div class="footer-bottom">
      <span class="footer-brand">NFL <span>EDGE</span></span>
      <span>&copy; {now.year} - rebuilt from real results every week.</span>
    </div>
  </footer>
</div>
<script src="site.js?v={ver}"></script>
</body>
</html>"""

def card(title, subtitle, body_html, icon_name=None):
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    badge = f'<div class="card-icon-badge">{icon(icon_name)}</div>' if icon_name else ""
    return f"""<div class="card">
    <div class="card-header">{badge}<div><h2>{title}</h2>{sub}</div></div>
    <div class="card-body">{body_html}</div>
  </div>"""

def build_slate_table(games, comparison):
    if games.empty:
        return '<div class="empty-state">No upcoming games found.</div>'

    merged = games.copy()
    if comparison is not None and not comparison.empty:
        vegas_cols = comparison[["home_team", "away_team", "vegas_favored_team", "vegas_home_favored_by", "total_line", "vegas_home_win_prob"]]
        merged = merged.merge(vegas_cols, on=["home_team", "away_team"], how="left")

    sort_cols = [c for c in ["gameday", "gametime"] if c in merged.columns]
    merged = merged.sort_values(sort_cols if sort_cols else "model_spread").reset_index(drop=True)

    rows = ""
    last_day = None
    for _, g in merged.iterrows():
        pick = model_pick(g)
        has_vegas = pd.notna(g.get("vegas_home_favored_by"))

        if has_vegas:
            vegas_favored_team = g["vegas_favored_team"]
            vegas_line = f"{vegas_favored_team} -{abs(g['vegas_home_favored_by']):.1f}"
            vegas_total = f"{g['total_line']:.1f}" if pd.notna(g.get("total_line")) else None
            disagree = pick["favored_team"] != vegas_favored_team
        else:
            vegas_line, vegas_total, disagree = None, None, False

        game_day = g.get("gameday")
        if pd.notna(game_day) and game_day != last_day:
            last_day = game_day
            day_label = str(g.get("weekday")) if pd.notna(g.get("weekday")) else str(game_day)
            rows += f'<tr class="day-header"><td colspan="6">{day_label}</td></tr>'

        kickoff = format_kickoff(g.get("weekday"), g.get("gametime"))
        kickoff_sort = f"{game_day} {g.get('gametime', '')}"
        matchup = f"{g['away_team']} @ {g['home_team']}"
        our_line = f"{pick['favored_team']} -{pick['favored_by']:.1f}"
        vegas_html = f'<span class="market-color">{vegas_line}</span>' if vegas_line else f'<span class="faint">{DASH}</span>'
        total_html = f"{pick['total']:.1f} <span class='faint'>/</span> " + (f'<span class="market-color">{vegas_total}</span>' if vegas_total else f'<span class="faint">{DASH}</span>')
        flag = ' row-flag' if disagree else ""
        flip_note = f' {pill("DIFFERENT PICK", "danger")}' if disagree else ""
        border = f"border-left:4px solid {team_color(pick['favored_team'])};"

        rows += f"""<tr class="{flag.strip()}">
          <td data-key="matchup" data-value="{matchup}" style="{border}">{matchup}{flip_note}</td>
          <td data-key="kickoff" data-value="{kickoff_sort}" class="num mono">{kickoff}</td>
          <td data-key="ourline" data-value="{pick['favored_by']:.2f}" class="num mono accent">{our_line}</td>
          <td data-key="vegas" data-value="{abs(g['vegas_home_favored_by']) if has_vegas else -1:.2f}" class="num mono">{vegas_html}</td>
          <td data-key="total" data-value="{pick['total']:.2f}" class="num mono">{total_html}</td>
          <td data-key="winpct" data-value="{pick['win_pct']:.3f}" class="num mono">{pick['win_pct']:.0%}</td>
        </tr>"""

    return f"""<table class="data" data-sortable>
      <thead><tr>
        <th data-sort-key="matchup">Matchup</th>
        <th data-sort-key="kickoff" class="num">Kickoff</th>
        <th data-sort-key="ourline" class="num">Our Pick</th>
        <th data-sort-key="vegas" class="num">Vegas</th>
        <th data-sort-key="total" class="num">Total (us / vegas)</th>
        <th data-sort-key="winpct" class="num">Win%</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="table-footnote muted">{pill("DIFFERENT PICK", "danger")} = we favor a different team than Vegas entirely. The colored bar is the team we favor. Click a column header to sort.</div>"""

def build_edge_cards(comparison, week_games, max_cards=3):
    if comparison is None or comparison.empty:
        return ""
    week_comparison = comparison.merge(week_games[["home_team", "away_team"]], on=["home_team", "away_team"], how="inner")
    notable = week_comparison[week_comparison["has_notable_edge"]].copy()
    if notable.empty:
        return ""
    notable["sort_key"] = notable["spread_edge"].abs() + notable["picks_flip"].astype(int) * 10
    top = notable.sort_values("sort_key", ascending=False).head(max_cards)

    cards = ""
    for _, g in top.iterrows():
        flip = g["model_favored_team"] != g["vegas_favored_team"]
        flip_line = f'<div class="edge-flip">{pill("DIFFERENT TEAM FAVORED", "danger")}</div>' if flip else ""
        cards += f"""<div class="edge-card{' edge-card-flip' if flip else ''}">
          <div class="edge-matchup">{g['away_team']} @ {g['home_team']}</div>
          <div class="edge-our-line mono">{g['model_favored_team']} -{g['model_favored_by']:.1f} <span class="muted">our model</span></div>
          <div class="edge-vegas-line mono market-color">{g['vegas_favored_team']} -{abs(g['vegas_home_favored_by']):.1f} <span class="muted">vegas</span></div>
          {flip_line}
        </div>"""

    return f"""<div class="edge-section">
      <div class="edge-kicker">Notable Model vs. Market Gaps</div>
      <div class="edge-grid">{cards}</div>
    </div>"""

def build_track_record_section(summary):
    if summary is None:
        return card("Track Record", "Graded against real final scores, not vibes",
                     '<div class="empty-state">No games graded yet. Check back once the first week wraps.</div>', "target")

    year = summary.get("current_season_year", "")
    # "model" = the pure, unblended model's own straight-up pick. NOT "sharp" -
    # sharp_spread currently has a 0.0 blend weight on the model (see
    # fitted_coefficients.json), so it is mathematically identical to Vegas's
    # line for every graded game. Showing "sharp" here as "our" record would
    # just be Vegas's own record relabeled as ours. "model" is the number that
    # actually reflects independent model skill, for better or worse.
    current = summary.get("current_season", {}).get("model")
    all_time = summary.get("all_time", {}).get("model")
    all_time_vegas = summary.get("all_time", {}).get("vegas")

    if not current:
        body = f'<div class="empty-state">No {year} games graded yet.</div>'
        return card("Track Record", "Graded against real final scores, not vibes", body, "target")

    # A bento grid, not a uniform row of tiles: one big square carries the
    # headline record, two wide bars carry the all-time records, and four
    # small tiles fill in the supporting numbers - mixed tile sizes read as
    # a hierarchy (this number matters most) instead of a flat stat wall.
    hero_tile = f"""<div class="bento-tile bento-hero">
      <div class="hero-label">{year} Season Straight-Up Record (Our Model)</div>
      <div class="hero-number">{current['record']}</div>
      {pill(f"{current['pick_accuracy']:.0%} HIT RATE", "primary")}
    </div>"""

    wide_tiles = ""
    if all_time:
        wide_tiles += f'<div class="bento-tile bento-wide tile-primary"><div class="label">All-Time Record (Model)</div><div class="value accent">{all_time["record"]}</div></div>'
    if all_time_vegas:
        wide_tiles += f'<div class="bento-tile bento-wide tile-market"><div class="label">Vegas All-Time</div><div class="value market-color">{all_time_vegas["record"]}</div></div>'

    small_tiles = ""
    if all_time:
        small_tiles += f'<div class="bento-tile tile-primary"><div class="value">&plusmn;{all_time["spread_mae"]:.1f}</div><div class="label">Model Spread Error</div></div>'
    if all_time_vegas:
        small_tiles += f'<div class="bento-tile tile-market"><div class="value">&plusmn;{all_time_vegas["spread_mae"]:.1f}</div><div class="label">Vegas Spread Error</div></div>'
    small_tiles += f"""<div class="bento-tile"><div class="value">{summary['n_graded_games']}</div><div class="label">Games Graded</div></div>
    <div class="bento-tile"><div class="value">{LIVE_TRACKING_START_SEASON}</div><div class="label">Live Tracking Since</div></div>"""

    note = ("""<div class="table-footnote muted">These are the pure model's own straight-up picks, not the market-blended
      line shown in the Slate below. Backtesting found no edge in overriding Vegas on who wins straight-up
      (0% model weight there), so the blended pick defers fully to the market for that call - the model's
      independent record is tracked here for transparency, not because it beats the market.</div>""")

    body = f'<div class="bento-grid">{hero_tile}{wide_tiles}{small_tiles}</div>' + note
    return card("Track Record", "The model's own picks, graded against real final scores " + f"({summary['n_graded_games']} games all-time)", body, "target")

def build_players_page(props):
    def table(stat_cols, cat_id, active):
        if props.empty or stat_cols["sort"] not in props.columns:
            return ""
        top = props.dropna(subset=[stat_cols["sort"]]).sort_values(stat_cols["sort"], ascending=False).head(20).reset_index(drop=True)
        if top.empty:
            return ""
        headers = "".join(f'<th data-sort-key="{c}" class="num">{h}</th>' for c, h in zip(stat_cols["display"], stat_cols["headers"]))
        rows = ""
        for _, p in top.iterrows():
            tag = ""
            if p.get("injury_status") and pd.notna(p.get("injury_status")) and p["injury_status"]:
                style = "primary" if p["injury_status"] == "Questionable" else "danger"
                tag = " " + pill(str(p["injury_status"]).upper(), style)
            mult = p.get(stat_cols.get("matchup_col", ""), np.nan)
            badge = ""
            if pd.notna(mult):
                if mult >= 1.08:
                    badge = " " + pill("SOFT MATCHUP", "positive")
                elif mult <= 0.92:
                    badge = " " + pill("TOUGH MATCHUP", "danger")
            cells = "".join(f'<td data-key="{c}" data-value="{p[c]}" class="num mono">{p[c]}</td>' for c in stat_cols["display"])
            border = f"border-left:4px solid {team_color(p['team'])};"
            rows += f"""<tr>
              <td data-key="player" data-value="{p['player_name']}" style="{border}"><b>{p['player_name']}</b>{tag}<div class="muted" style="font-size:11px;">{p['team']} vs {p['opponent']}</div>{badge}</td>
              {cells}
            </tr>"""
        hidden = "" if active else " hidden"
        return f"""<div class="cat-panel" id="cat-{cat_id}"{hidden}>
        <table class="data" data-sortable>
          <thead><tr><th data-sort-key="player">Player</th>{headers}</tr></thead>
          <tbody>{rows}</tbody>
        </table></div>"""

    passing = table({"sort": "proj_pass_yards", "display": ["proj_pass_attempts", "proj_completions", "proj_pass_yards", "proj_pass_tds"],
                      "headers": ["Att", "Comp", "Yds", "TDs"], "matchup_col": "matchup_mult_pass"}, "passing", True)
    rushing = table({"sort": "proj_rush_yards", "display": ["proj_carries", "proj_rush_yards", "proj_rush_tds"],
                      "headers": ["Car", "Yds", "TDs"], "matchup_col": "matchup_mult_rush"}, "rushing", False)
    receiving = table({"sort": "proj_rec_yards", "display": ["proj_targets", "proj_receptions", "proj_rec_yards", "proj_rec_tds"],
                        "headers": ["Tgt", "Rec", "Yds", "TDs"], "matchup_col": "matchup_mult_rec"}, "receiving", False)

    if not (passing or rushing or receiving):
        body = '<div class="empty-state">No player projections available yet.</div>'
        return page_shell("Players", "players", card("Player Projections", "Top 20 per category by projected yards", body, "user"))

    subtabs = f"""<div class="subtabs">
      <button class="subtab active" data-target="cat-passing">Passing</button>
      <button class="subtab" data-target="cat-rushing">Rushing</button>
      <button class="subtab" data-target="cat-receiving">Receiving</button>
    </div>"""
    body = subtabs + passing + rushing + receiving
    card_html = card("Player Projections", "Top 20 per category by projected yards - the colored bar is the player's team, click a column to sort", body, "user")
    return page_shell("Players", "players", card_html)

def build_index_page(games, props, comparison, accuracy_summary):
    week_games = next_week_games(games)
    week_label = f"Week {int(week_games.iloc[0]['week'])}" if not week_games.empty else "Upcoming"
    week_comparison = None
    if comparison is not None and not comparison.empty:
        week_comparison = comparison.merge(week_games[["home_team", "away_team"]], on=["home_team", "away_team"], how="inner")

    edge_html = build_edge_cards(comparison, week_games) if comparison is not None else ""
    slate_html = build_slate_table(week_games, week_comparison)
    track_html = build_track_record_section(accuracy_summary)

    body = edge_html + card(f"This Week's Slate - {week_label}", "Our pick vs. the market, every game - see the Teams tab for the full season", slate_html, "calendar") + track_html
    return page_shell("Home", "index", body)

WEEK_TYPE_LABELS = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conf. Championship", "SB": "Super Bowl"}

def week_label(week, game_type=None):
    if game_type in WEEK_TYPE_LABELS:
        return WEEK_TYPE_LABELS[game_type]
    return f"Week {int(week)}"

def build_teams_page(games, log, comparison):
    """Every 2026 matchup, week 1 through the postseason, browsable by week.
    Already-played weeks come from the tracking log (graded); weeks still
    ahead come straight from game_predictions.csv, which already forecasts
    the rest of the season, not just the imminent week - the site just never
    surfaced that beyond 'next week' before. Uses the pure model's own picks
    throughout (see model_pick()), not the market-mirroring blended line."""
    season = LIVE_TRACKING_START_SEASON
    weeks = {}

    # A week's games can be split across "already played" and "still upcoming"
    # (e.g. Thursday night graded, Sunday/Monday not yet) - append into the
    # same week bucket rather than letting one loop overwrite the other's rows.
    def week_bucket(wk, game_type):
        key = f"w{int(wk)}"
        if key not in weeks:
            weeks[key] = {"label": week_label(wk, game_type), "week": int(wk), "games": []}
        return weeks[key]

    graded = log[(log["actual_margin"].notna()) & (log["season"] == season)].copy() if not log.empty else pd.DataFrame()
    already_graded = set(zip(graded["week"], graded["home_team"], graded["away_team"]))
    for wk, g in graded.groupby("week"):
        bucket = week_bucket(wk, g["game_type"].iloc[0] if "game_type" in g.columns else None)
        for _, r in g.sort_values("home_team").iterrows():
            pick = model_pick(r)
            has_vegas = pd.notna(r.get("vegas_home_favored_by"))
            vegas_favored = (r["home_team"] if r["vegas_home_favored_by"] > 0 else r["away_team"]) if has_vegas else None
            bucket["games"].append({
                "away_team": r["away_team"], "home_team": r["home_team"], "kickoff": None,
                "favored_team": pick["favored_team"], "favored_by": round(float(pick["favored_by"]), 1),
                "win_pct": round(float(pick["win_pct"]), 3), "total": round(float(pick["total"]), 1),
                "vegas_favored_team": vegas_favored,
                "vegas_favored_by": round(float(abs(r["vegas_home_favored_by"])), 1) if has_vegas else None,
                "vegas_total": round(float(r["vegas_total"]), 1) if pd.notna(r.get("vegas_total")) else None,
                "graded": True,
                "home_score": int(r["home_score"]) if pd.notna(r.get("home_score")) else None,
                "away_score": int(r["away_score"]) if pd.notna(r.get("away_score")) else None,
                "correct": bool(r["model_correct_pick"]) if pd.notna(r.get("model_correct_pick")) else None,
                "color": team_color(pick["favored_team"]),
            })

    upcoming = games[games["season"] == season].copy() if not games.empty else pd.DataFrame()
    vegas_cols = None
    if comparison is not None and not comparison.empty:
        vegas_cols = comparison[["home_team", "away_team", "vegas_favored_team", "vegas_home_favored_by", "total_line"]]
    for wk, g in upcoming.groupby("week"):
        bucket = week_bucket(wk, g["game_type"].iloc[0] if "game_type" in g.columns else None)
        merged = g.merge(vegas_cols, on=["home_team", "away_team"], how="left") if vegas_cols is not None else g
        sort_cols = [c for c in ["gameday", "gametime"] if c in merged.columns]
        for _, r in merged.sort_values(sort_cols if sort_cols else "home_team").iterrows():
            if (r["week"], r["home_team"], r["away_team"]) in already_graded:
                continue  # stale/duplicate row - the tracking log already has the final result
            pick = model_pick(r)
            has_vegas = pd.notna(r.get("vegas_home_favored_by"))
            bucket["games"].append({
                "away_team": r["away_team"], "home_team": r["home_team"],
                "kickoff": format_kickoff(r.get("weekday"), r.get("gametime")),
                "favored_team": pick["favored_team"], "favored_by": round(float(pick["favored_by"]), 1),
                "win_pct": round(float(pick["win_pct"]), 3), "total": round(float(pick["total"]), 1),
                "vegas_favored_team": r["vegas_favored_team"] if has_vegas else None,
                "vegas_favored_by": round(float(abs(r["vegas_home_favored_by"])), 1) if has_vegas else None,
                "vegas_total": round(float(r["total_line"]), 1) if has_vegas and pd.notna(r.get("total_line")) else None,
                "graded": False, "home_score": None, "away_score": None, "correct": None,
                "color": team_color(pick["favored_team"]),
            })

    if not weeks:
        body = card("Teams", f"Every {season} matchup, picked and graded", '<div class="empty-state">No games available yet.</div>', "shield")
        return page_shell("Teams", "teams", body)

    week_order = [k for k, _ in sorted(weeks.items(), key=lambda kv: kv[1]["week"])]
    teams_json = json.dumps({"week_order": week_order, "weeks": weeks})
    body = card("Teams", f"Every {season} matchup, week 1 through the postseason - picked with our own model, graded once final",
                '<select id="teams-week-select" class="week-picker"></select><div id="teams-week-content" style="margin-top:16px;"></div>'
                f'<script>const TEAMS_DATA = {teams_json};</script>', "shield")
    return page_shell("Teams", "teams", body)

def build_history_page(log):
    graded = log[log["actual_margin"].notna()].copy() if not log.empty else log
    if graded.empty:
        body = card("History", "Every graded week, once there's one to show", '<div class="empty-state">No games graded yet.</div>', "clock")
        return page_shell("History", "history", body)

    graded = graded.sort_values(["season", "week"])
    weeks = {}
    week_order = []
    for (season, week), g in graded.groupby(["season", "week"]):
        key = f"{int(season)}-{int(week)}"
        week_order.append(key)
        games_list = []
        for _, r in g.iterrows():
            def pick_str(spread_col):
                spread = r[spread_col]
                if pd.isna(spread):
                    return DASH
                team = r["home_team"] if spread > 0 else r["away_team"]
                return f"{team} -{abs(spread):.1f}"
            games_list.append({
                "away_team": r["away_team"], "home_team": r["home_team"],
                "away_score": int(r["away_score"]) if pd.notna(r.get("away_score")) else None,
                "home_score": int(r["home_score"]) if pd.notna(r.get("home_score")) else None,
                "model_pick": pick_str("model_spread"), "model_correct": bool(r["model_correct_pick"]) if pd.notna(r.get("model_correct_pick")) else None,
                "vegas_pick": pick_str("vegas_home_favored_by"), "vegas_correct": bool(r["vegas_correct_pick"]) if pd.notna(r.get("vegas_correct_pick")) else None,
            })
        weeks[key] = {"label": f"{int(season)} - Week {int(week)}", "games": games_list}

    history_json = json.dumps({"week_order": week_order, "weeks": weeks})
    body = card("History", "Every graded week - pick a week to see how we did",
                f'<select id="week-select" class="week-picker"></select><div id="week-content" style="margin-top:16px;"></div>'
                f'<script>const HISTORY_DATA = {history_json};</script>', "clock")
    return page_shell("History", "history", body)

def build_accuracy_page(log):
    graded = log[log["actual_margin"].notna()].copy() if not log.empty else log
    if not graded.empty:
        graded = graded[graded["season"] >= LIVE_TRACKING_START_SEASON]
    if graded.empty:
        body = card("Accuracy Over Time", "Weekly trend, us vs. the market",
                     '<div class="empty-state">No live-tracked games graded yet - check back once the '
                     f'{LIVE_TRACKING_START_SEASON} season kicks off.</div>', "trend")
        return page_shell("Accuracy", "accuracy", body)

    graded = graded.sort_values(["season", "week"])
    # "model_*" here too, not "sharp_*" - sharp is the market-blended line,
    # which for spreads has a 0.0 model weight and is therefore identical to
    # Vegas every week. Charting it as "Us" would just plot Vegas twice.
    weekly = graded.groupby(["season", "week"]).agg(
        model_accuracy=("model_correct_pick", "mean"), vegas_accuracy=("vegas_correct_pick", "mean"),
        model_spread_mae=("model_spread_error", "mean"), vegas_spread_mae=("vegas_spread_error", "mean"),
        model_brier=("model_brier", "mean"), vegas_brier=("vegas_brier", "mean"),
        n=("model_correct_pick", "size"),
    ).reset_index()

    labels = [f"{int(s)} Wk{int(w)}" for s, w in zip(weekly["season"], weekly["week"])]
    data = {
        "labels": labels,
        "us_accuracy": weekly["model_accuracy"].round(3).tolist(), "vegas_accuracy": weekly["vegas_accuracy"].round(3).tolist(),
        "us_spread_mae": weekly["model_spread_mae"].round(2).tolist(), "vegas_spread_mae": weekly["vegas_spread_mae"].round(2).tolist(),
        "us_brier": weekly["model_brier"].round(4).tolist(), "vegas_brier": weekly["vegas_brier"].round(4).tolist(),
    }
    charts_html = "".join(
        f'<div class="chart-card" style="margin-bottom:28px;"><canvas id="{cid}" height="90"></canvas></div>'
        for cid in ["chart-accuracy", "chart-spread-mae", "chart-brier"]
    )
    body = card("Accuracy Over Time",
                f"Weekly trend across {int(weekly['n'].sum())} live-picked games ({LIVE_TRACKING_START_SEASON} season onward) - our model's own picks vs. the market",
                charts_html + f'<script>const ACCURACY_DATA = {json.dumps(data)};</script>', "trend")
    return page_shell("Accuracy", "accuracy", body)

def build_404_page():
    body = """<div class="card">
      <div class="card-body error-body">
        <div class="error-code mono">404</div>
        """ + pill("PENALTY - LOSS OF PAGE", "danger") + """
        <h2>This one got called back.</h2>
        <p class="muted">The page you're looking for doesn't exist - it might have been moved, renamed,
        or never existed to begin with.</p>
        <a class="btn-primary" href="index.html">Back to Home</a>
      </div>
    </div>"""
    return page_shell("Page Not Found", None, body)

def main():
    print("Loading data...")
    games, props, comparison, accuracy_summary, log = load_data()

    print("Building pages...")
    os.makedirs(DIST_DIR, exist_ok=True)
    pages = {
        "index.html": build_index_page(games, props, comparison, accuracy_summary),
        "teams.html": build_teams_page(games, log, comparison),
        "players.html": build_players_page(props),
        "history.html": build_history_page(log),
        "accuracy.html": build_accuracy_page(log),
        "404.html": build_404_page(),
    }
    for filename, html in pages.items():
        with open(os.path.join(DIST_DIR, filename), "w") as f:
            f.write(html)
        print(f"  Wrote {filename}")

    for asset in ["style.css", "site.js"]:
        shutil.copy(os.path.join(WEB_SRC_DIR, asset), os.path.join(DIST_DIR, asset))
    print(f"  Copied static assets")

    print(f"\nSite built in {DIST_DIR}")

if __name__ == "__main__":
    main()
