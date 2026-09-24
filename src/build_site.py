"""
build_site.py
Generates the static Edge website (dist/) from the same processed data the
email used to read - the email is retired; this is the sole output now.

Two sports, same page structure, kept in separate subdirectories so each is
independently browsable and linkable:
  dist/nfl/...  - NFL Edge (unchanged behavior/content from before this
                  became multi-sport)
  dist/cfb/...  - College Football Edge, Power-conference + independent FBS
                  teams only, same modeling approach (see fit_cfb_model.py)
  dist/index.html - a plain redirect to nfl/index.html so old bookmarks/links
                  to the site root keep working with NFL as the default sport
  dist/<sport>/summary.json - this week's top picks and the season record,
                  read by the home page linking every site (ant56-arch.github.io)

Pages per sport (see SPORTS below for which apply to which sport):
  index.html    - Home: notable model-vs-market gaps, track record, and this
                  week's slate only (kept short - see teams.html for the rest
                  of the season)
  teams.html    - every game this season, week 1 through the postseason,
                  browsable via a dropdown - past weeks graded, future weeks
                  straight from the season's game_predictions.csv
  players.html  - player projections by category (Passing/Rushing/Receiving)
                  - NFL only, no college football player props in this pass
  history.html  - every graded week across all seasons (backfill included),
                  browsable via a dropdown (client-side, no per-week routing
                  needed for a site this size)
  accuracy.html - trend charts of our model's own accuracy vs Vegas's over time

Published to GitHub Pages by weekly-picks.yml. This script only builds the
dist/ directory - the workflow's own steps upload and deploy it.
"""

import pandas as pd
import numpy as np
import games as games_mod
import model_page
import moneyline
import json
import os
import shutil
import hashlib
from datetime import datetime, timezone

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TRACKING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tracking")
WEB_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "web")
DIST_DIR = os.path.join(os.path.dirname(__file__), "..", "dist")

DASH = "-"

# Per-sport configuration - every page-building function below takes a
# `sport` dict as its first argument and reads file paths / display text
# from it, instead of the module-level constants this file used before it
# covered more than one sport.
SPORTS = {
    "nfl": {
        "slug": "nfl",
        "wordmark": "NFL",
        "tagline": "Model-driven NFL spreads, totals and player props, graded against the closing line every week.",
        "meta_description": "Model-driven NFL spreads, totals and player props, validated against the closing Vegas line every week.",
        "team_csv": "teams.csv",
        "team_abbr_col": "team_abbr", "team_color_col": "team_color", "team_color2_col": "team_color2", "team_logo_col": "team_logo_espn",
        "game_predictions_csv": "game_predictions.csv",
        "player_props_csv": "player_props.csv",
        "vegas_comparison_csv": "vegas_comparison.csv",
        "predictions_log_csv": "predictions_log.csv",
        "accuracy_summary_json": "accuracy_summary.json",
        "top25_summary_json": None,
        "coefficients_json": "fitted_coefficients.json",
        "live_tracking_start_season": 2026,
        "ats_since_year": 2024,
        "data_source_text": "Play-by-play and schedules via nflverse. Vegas lines via DraftKings, through the-odds-api.com, where available.",
        "no_games_note": "",
    },
    "cfb": {
        "slug": "cfb",
        "wordmark": "CFB",
        "tagline": "Model-driven spreads and totals for Power-conference college football, graded against the closing line every week.",
        "meta_description": "Model-driven college football spreads and totals for the Power conferences, validated against the closing Vegas line every week.",
        "team_csv": "cfb_teams.csv",
        "team_abbr_col": "team", "team_short_col": "abbreviation", "team_color_col": "color", "team_color2_col": "alt_color", "team_logo_col": "logo",
        "game_predictions_csv": "cfb_game_predictions.csv",
        "player_props_csv": None,
        "vegas_comparison_csv": "cfb_vegas_comparison.csv",
        "predictions_log_csv": "cfb_predictions_log.csv",
        "accuracy_summary_json": "cfb_accuracy_summary.json",
        "top25_summary_json": "cfb_top25_summary.json",
        "coefficients_json": "fitted_cfb_coefficients.json",
        "live_tracking_start_season": 2026,
        "ats_since_year": 2026,
        "data_source_text": "Team efficiency (PPA, success rate, explosiveness) via CollegeFootballData.com. Vegas lines via the-odds-api.com, where available. Covers SEC, Big Ten, Big 12, ACC and FBS independent teams.",
        "no_games_note": "Covers Power-conference and independent FBS teams only.",
    },
}

# Fallback only, used if a sport's team CSV (fetched fresh each run) isn't
# there for some reason.
_FALLBACK_TEAM_COLOR = "#94a3b8"
_TEAM_INFO = {}

def team_info(sport):
    """Official team colors + logo URLs for this sport, from whatever CSV
    that sport's fetch step produced (data/raw/teams.csv for NFL via
    nflverse, data/raw/cfb_teams.csv for CFB via CollegeFootballData.com)."""
    slug = sport["slug"]
    if slug not in _TEAM_INFO:
        info = {}
        path = os.path.join(RAW_DIR, sport["team_csv"])
        if os.path.exists(path):
            df = pd.read_csv(path)
            short_col = sport.get("team_short_col")
            for _, r in df.iterrows():
                key = r[sport["team_abbr_col"]]
                info[key] = {
                    "color": r[sport["team_color_col"]], "color2": r[sport["team_color2_col"]], "logo": r[sport["team_logo_col"]],
                    "short": r[short_col] if short_col else key,
                }
        _TEAM_INFO[slug] = info
    return _TEAM_INFO[slug]

def team_color(sport, abbr):
    return team_info(sport).get(abbr, {}).get("color", _FALLBACK_TEAM_COLOR)

def team_logo(sport, abbr):
    return team_info(sport).get(abbr, {}).get("logo", "")

def team_short(sport, name):
    """Short display name for a team - the identifier itself for NFL (its
    join key is already a 2-3 letter code), a real abbreviation for CFB
    (whose join key is the full school name, e.g. 'Mississippi State' -
    long full names in every stat table column made game tables overflow
    even on desktop, not just mobile)."""
    return team_info(sport).get(name, {}).get("short", name)

def matchup_bar(sport, away, home, right_html):
    """A single cell replacing separate Matchup/Kickoff columns: both teams'
    logos and short names, with `right_html` (kickoff time, or the final
    score once graded) anchored right."""
    away_logo, home_logo = team_logo(sport, away), team_logo(sport, home)
    away_label, home_label = team_short(sport, away), team_short(sport, home)
    away_img = f'<img class="team-logo" src="{away_logo}" alt="" loading="lazy" onerror="this.style.display=\'none\'">' if away_logo else ""
    home_img = f'<img class="team-logo" src="{home_logo}" alt="" loading="lazy" onerror="this.style.display=\'none\'">' if home_logo else ""
    return f"""<div class="matchup-content">
        <span class="matchup-team">{away_img}<span>{away_label}</span></span>
        <span class="matchup-at">@</span>
        <span class="matchup-team">{home_img}<span>{home_label}</span></span>
        <span class="matchup-right">{right_html}</span>
      </div>"""

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

def ml_view(sport, r):
    """A game's moneyline pick for display (see src/moneyline.py), from any
    row carrying the ml_* columns - a tracking-log row (locked at kickoff,
    graded once final) or a vegas_comparison row. None when the game has no
    book moneyline."""
    if r is None or pd.isna(r.get("ml_pick_side")) or pd.isna(r.get("ml_pick_price")):
        return None
    won, push = r.get("ml_won"), r.get("ml_push")
    result = None
    if pd.notna(won):
        result = "W" if won == 1 else "L"
    elif pd.notna(push) and push == 1:
        result = "P"
    return {
        "text": f"{team_short(sport, r['ml_pick_team'])} {moneyline.format_price(r['ml_pick_price'])}",
        "our": round(float(r["ml_our_prob"]), 3), "book": round(float(r["ml_book_prob"]), 3),
        "edge": round(float(r["ml_edge"]), 3), "value": moneyline.is_value(r.get("ml_value")),
        "result": result, "units": round(float(r["ml_units"]), 2) if pd.notna(r.get("ml_units")) else None,
    }

def ml_cell_html(ml):
    """The Moneyline cell: pick and price, our win % vs the no-vig book %,
    the edge, a VALUE tag at 3+ points, and W/L with units once graded."""
    if not ml:
        return f'<span class="faint">{DASH}</span>'
    tags = f' {pill("VALUE", "positive")}' if ml["value"] else ""
    if ml["result"] == "W":
        tags += f' {pill("W " + moneyline.fmt_units(ml["units"]), "positive")}'
    elif ml["result"] == "L":
        tags += f' {pill("L " + moneyline.fmt_units(ml["units"]), "danger")}'
    elif ml["result"] == "P":
        tags += f' {pill("NO DECISION", "market")}'
    return (f'<span class="ml-pick"><span class="ml-line"><span class="accent">{ml["text"]}</span>{tags}</span>'
            f'<span class="ml-sub">our {ml["our"]:.0%} vs book {ml["book"]:.0%}, {ml["edge"] * 100:+.0f}</span></span>')

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

def load_data(sport):
    games_path = os.path.join(PROCESSED_DIR, sport["game_predictions_csv"])
    games = pd.read_csv(games_path) if os.path.exists(games_path) else pd.DataFrame()

    props = pd.DataFrame()
    if sport["player_props_csv"]:
        props_path = os.path.join(PROCESSED_DIR, sport["player_props_csv"])
        if os.path.exists(props_path):
            props = pd.read_csv(props_path)

    comparison = None
    comparison_path = os.path.join(PROCESSED_DIR, sport["vegas_comparison_csv"])
    if os.path.exists(comparison_path):
        comparison = pd.read_csv(comparison_path)

    accuracy_summary = None
    summary_path = os.path.join(TRACKING_DIR, sport["accuracy_summary_json"])
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            accuracy_summary = json.load(f)
        if accuracy_summary.get("n_graded_games", 0) == 0:
            accuracy_summary = None

    log_path = os.path.join(TRACKING_DIR, sport["predictions_log_csv"])
    log = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()

    top25_summary = None
    if sport.get("top25_summary_json"):
        top25_path = os.path.join(TRACKING_DIR, sport["top25_summary_json"])
        if os.path.exists(top25_path):
            with open(top25_path) as f:
                top25_summary = json.load(f)

    return games, props, comparison, accuracy_summary, log, top25_summary

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
    '%3Crect width=%2232%22 height=%2232%22 fill=%22%23121314%22/%3E'
    '%3Cpath d=%22M9 23V9h3.4l6.6 9.3V9H22v14h-3.4L12 13.6V23z%22 fill=%22%23e5793b%22/%3E'
    '%3C/svg%3E')

# MLB Edge and NBA Edge are separate sites (both built in
# github.com/ant56-arch/mlb-hit-predictor) that share this look; the sport
# switcher links out to them after the NFL and CFB tabs.
MLB_EDGE_URL = "https://ant56-arch.github.io/mlb-hit-predictor/"
NBA_EDGE_URL = "https://ant56-arch.github.io/mlb-hit-predictor/nba/index.html"
# Every sport's games as a TV grid, on the home site (schedule.js there reads
# the games.json each sport publishes).
SCHEDULE_URL = "https://ant56-arch.github.io/schedule.html"
OTHER_SPORT_TABS = (f'<a class="sport-tab" href="{MLB_EDGE_URL}">MLB</a>'
                 f'<a class="sport-tab" href="{NBA_EDGE_URL}">NBA</a>'
                 f'<a class="sport-tab" href="{SCHEDULE_URL}">Schedule</a>')
# The home page (github.com/ant56-arch/ant56-arch.github.io) links every site
# and shows each one's summary.json; the switcher's first tab goes back to it.
HOME_URL = "https://ant56-arch.github.io/"
HOME_SPORT_TAB = f'<a class="sport-tab" href="{HOME_URL}">All</a>'
# The Sports Edge brand mark in the top bar, same on every Edge site.
BRAND_MARK = ('<svg class="brand-mark" viewBox="0 0 32 32" aria-hidden="true"><path d="M9 3h22l-8 26H1z" fill="#e5793b"/>'
              '<path transform="translate(4.3 0) skewX(-15)" d="M10 9h12v3.2h-8.4v2.3h7.4v3h-7.4v2.3H22V23H10z" '
              'fill="#121314"/></svg>')

def top_bar(sport_switcher):
    """The black network bar (brand + sport tabs) and the scoreboard strip
    under it, which site.js fills from every Edge site's summary.json."""
    return f"""<header class="topbar">
  <div class="topbar-inner">
    <a class="brand" href="{HOME_URL}">{BRAND_MARK}<span class="brand-name">Sports <span>Edge</span></span></a>
    <nav class="sport-switcher" aria-label="Sport">{sport_switcher}</nav>
  </div>
</header>
<aside class="scoreboard" aria-label="Latest top picks" hidden></aside>"""

def page_shell(sport, title, active_tab, body_html):
    tabs = [
        ("index.html", "index", "Home"),
        ("teams.html", "teams", "Teams"),
    ]
    if sport["player_props_csv"]:
        tabs.append(("players.html", "players", "Players"))
    tabs += [
        ("history.html", "history", "History"),
        ("accuracy.html", "accuracy", "Accuracy"),
        ("model.html", "model", "Model"),
    ]
    nav = "".join(
        f'<a href="{href}" class="active" aria-current="page">{label}</a>' if tab == active_tab
        else f'<a href="{href}">{label}</a>'
        for href, tab, label in tabs
    )

    other_slug = "cfb" if sport["slug"] == "nfl" else "nfl"
    other_page = active_tab + ".html" if active_tab else "index.html"
    sport_switcher = HOME_SPORT_TAB + "".join(
        f'<a href="{"../" + s["slug"] + "/" + other_page if s["slug"] != sport["slug"] else "#"}" '
        + ('class="sport-tab active" aria-current="page">' if s["slug"] == sport["slug"] else 'class="sport-tab">')
        + f'{s["wordmark"]}</a>'
        for s in SPORTS.values()
    ) + OTHER_SPORT_TABS

    now = datetime.now(timezone.utc)
    generated = now.strftime("%b %d, %Y %H:%M UTC")
    ver = asset_version()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} | {sport["wordmark"]} Edge</title>
<meta name="description" content="{sport["meta_description"]}">
<link rel="icon" href="{FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:ital,wght@0,600;0,700;0,800;1,700;1,800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../style.css?v={ver}">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
</head>
<body>
<a class="skip-link" href="#main-content">Skip to main content</a>
{top_bar(sport_switcher)}
<header class="masthead" data-sport="{sport["wordmark"]}">
  <div class="masthead-inner">
    <h1 class="wordmark">{sport["wordmark"]} <span>EDGE</span></h1>
    <div class="tagline">{sport["tagline"]}</div>
    <div class="updated-chip">Updated {generated}</div>
  </div>
</header>
<nav class="tabs" aria-label="Sections"><div class="tabs-inner">{nav}</div></nav>
<div class="wrap">
  <main id="main-content">
  {body_html}
  </main>
  <footer class="site-footer">
    <div class="footer-brand">{sport["wordmark"]} <span>Edge</span></div>
    <p class="footer-text">Our model's lines come from team efficiency stats, with weights fit on past seasons and
      checked against seasons it wasn't fit on. {sport["data_source_text"]}</p>
    <p class="footer-text">For entertainment and research only. This is not betting advice, and past results don't
      predict future ones. If gambling is a problem for you or someone you know, call 1-800-GAMBLER.</p>
    <nav class="footer-links" aria-label="Site">
      <a href="https://ant56-arch.github.io/terms.html">Terms of Use</a>
      <a href="https://ant56-arch.github.io/privacy.html">Privacy Policy</a>
      <a href="{HOME_URL}">All sites</a>
      <a href="https://github.com/ant56-arch/nfl-edge">Source code</a>
      <span>&copy; {now.year} {sport["wordmark"]} Edge. Updated from final scores every week.</span>
    </nav>
  </footer>
</div>
<script src="../site.js?v={ver}"></script>
</body>
</html>"""

def card(title, subtitle, body_html):
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    return f"""<section class="card">
    <div class="card-header"><h2>{title}</h2>{sub}</div>
    <div class="card-body">{body_html}</div>
  </section>"""

def render_week_table(week_games):
    """Renders one week's worth of normalized game dicts (the shape produced
    by assemble_season_weeks) as a table - shared by the Home page (current
    week only) and Teams' per-week view. Each row shows the real matchup bar
    (logos + short names) and, once a game is graded, a HIT/MISS
    result instead of just a kickoff time - so "did we get it right" is
    visible for the games in this week that have already been played."""
    if not week_games:
        return '<div class="empty-state">No games this week.</div>'

    rows = ""
    for g in week_games:
        disagree = g["vegas_favored_team"] is not None and g["favored_team"] != g["vegas_favored_team"]
        vegas_html = (f'<span class="market-color">{g["vegas_favored_team"]} -{g["vegas_favored_by"]:.1f}</span>'
                      if g["vegas_favored_team"] else f'<span class="faint">{DASH}</span>')
        vegas_total_html = (f'<span class="market-color">{g["vegas_total"]:.1f}</span>'
                             if g["vegas_total"] is not None else f'<span class="faint">{DASH}</span>')
        our_line = f"{g['favored_team']} -{g['favored_by']:.1f}"
        flip_note = f' {pill("DIFFERENT PICK", "danger")}' if disagree else ""

        if g["graded"]:
            result_pill = pill("HIT", "positive") if g["correct"] else pill("MISS", "danger")
            result_html = f'{g["away_score"]}-{g["home_score"]} {result_pill}'
        else:
            result_html = f'<span class="faint">{DASH}</span>'

        matchup_sort_value = f"{g['away_team']} @ {g['home_team']}"

        rows += f"""<tr>
          <td data-key="matchup" data-value="{matchup_sort_value}">{g['matchup_html']}{flip_note}</td>
          <td data-key="ourline" data-value="{g['favored_by']:.2f}" data-label="Our Pick" class="num accent">{our_line}</td>
          <td data-key="vegas" data-value="{g['vegas_favored_by'] if g['vegas_favored_by'] is not None else -1:.2f}" data-label="Vegas" class="num">{vegas_html}</td>
          <td data-key="total" data-value="{g['total']:.2f}" data-label="Total" class="num"><span>{g['total']:.1f} <span class="faint">/</span> {vegas_total_html}</span></td>
          <td data-key="winpct" data-value="{g['win_pct']:.3f}" data-label="Win%" class="num">{g['win_pct']:.0%}</td>
          <td data-key="ml" data-value="{g['ml']['edge'] if g.get('ml') else -1:.3f}" data-label="Moneyline" class="num ml-cell">{ml_cell_html(g.get('ml'))}</td>
          <td data-label="Result" class="num"><span>{result_html}</span></td>
        </tr>"""

    return f"""<table class="data responsive-stack" data-sortable>
      <thead><tr>
        <th data-sort-key="matchup">Matchup</th>
        <th data-sort-key="ourline" class="num">Our Pick</th>
        <th data-sort-key="vegas" class="num">Vegas</th>
        <th data-sort-key="total" class="num">Total (us / vegas)</th>
        <th data-sort-key="winpct" class="num">Win%</th>
        <th data-sort-key="ml" class="num">Moneyline</th>
        <th class="num">Result</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="table-footnote muted">{pill("DIFFERENT PICK", "danger")} means our model favors a different team than Vegas does.
      Moneyline is the side where our win chance beats the book's (vig removed) by more, at the book's price;
      {pill("VALUE", "positive")} means an edge of 6 points or more. Locked at kickoff and graded at 1 unit. Select a column header to sort.</div>"""

def build_edge_cards(sport, comparison, week_games, max_cards=3):
    if comparison is None or comparison.empty:
        return ""
    week_comparison = comparison.merge(week_games[["home_team", "away_team"]], on=["home_team", "away_team"], how="inner")
    notable = week_comparison[week_comparison["has_notable_edge"]].copy()
    if notable.empty:
        return ""
    notable["sort_key"] = notable["spread_edge"].abs() + notable["picks_flip"].astype(int) * 10
    top = notable.sort_values("sort_key", ascending=False).head(max_cards)

    rows = ""
    for _, g in top.iterrows():
        flip = g["model_favored_team"] != g["vegas_favored_team"]
        flip_tag = f' {pill("DIFFERENT PICK", "danger")}' if flip else ""
        away_label, home_label = team_short(sport, g["away_team"]), team_short(sport, g["home_team"])
        model_label, vegas_label = team_short(sport, g["model_favored_team"]), team_short(sport, g["vegas_favored_team"])
        rows += f"""<tr>
          <td><b>{away_label} @ {home_label}</b>{flip_tag}</td>
          <td data-label="Our model" class="num accent">{model_label} -{g['model_favored_by']:.1f}</td>
          <td data-label="Vegas" class="num market-color">{vegas_label} -{abs(g['vegas_home_favored_by']):.1f}</td>
          <td data-label="Gap" class="num">{abs(g['spread_edge']):.1f} pts</td>
        </tr>"""

    table = f"""<table class="data responsive-stack">
      <thead><tr><th>Matchup</th><th class="num">Our model</th><th class="num">Vegas</th><th class="num">Gap</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""
    return card("Biggest Gaps vs. Vegas", "This week's games where our model's line is furthest from the market's", table)

def pct(x):
    return f"{x:.0%}" if x is not None else DASH

def record_row(label, rec):
    """One row of a straight-up / against-the-spread record table."""
    ats = rec.get("ats_record") if rec.get("ats_accuracy") is not None else None
    return f"""<tr>
      <td class="row-label">{label}</td>
      <td data-label="Straight-up" class="num">{rec['record']}</td>
      <td data-label="Win %" class="num">{pct(rec.get('pick_accuracy'))}</td>
      <td data-label="Against the spread" class="num">{ats or DASH}</td>
      <td data-label="Cover %" class="num">{pct(rec.get('ats_accuracy'))}</td>
      <td data-label="Avg. spread miss" class="num">{rec['spread_mae']:.1f} pts</td>
    </tr>"""

def record_table(rows_html):
    return f"""<table class="data record-table responsive-stack">
      <thead><tr><th></th><th class="num">Straight-up</th><th class="num">Win %</th>
        <th class="num">Against the spread</th><th class="num">Cover %</th><th class="num">Avg. spread miss</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""

def build_top25_block(top25_summary):
    """AP Top 25 teams' Vegas closing-line record since 2024 - a separate,
    much larger historical sample than the CFB tracker's own live history
    (which only starts whenever CFBD_API_KEY was added this season), seeded
    by backfill_cfb_top25.py. Vegas's record, not ours, disclosed as such -
    same framing as the main Track Record section."""
    if not top25_summary:
        return ""
    since = top25_summary["since_year"]
    poll_note = f" (as of the {top25_summary['poll_season']} week {top25_summary['poll_week']} poll)" if top25_summary.get("poll_week") else ""
    table = record_table(record_row(f"{since} to now, {top25_summary['n_games']} games", top25_summary))
    return f"""<div class="section-label">AP Top 25: Vegas record, {since} to now</div>
    {table}
    <div class="table-footnote">Vegas's closing-line record in games involving this week's AP Top 25 teams{poll_note}.
      This is the betting market's record, not our model's, and it covers more seasons than our live tracking.
      Straight-up means the favorite won. Against the spread means the favorite won by more than the line.</div>"""

ML_EMPTY = "No moneyline picks graded yet this season."

def ml_record(sport, games, log):
    """(season, record) for the moneyline: this season's live picks only,
    graded from the tracking log - never backfill or past seasons."""
    season = display_season(sport, games, log)
    return season, moneyline.summarize(log, season)

def build_moneyline_block(season, ml, with_label=True):
    """Moneyline record tiles (W-L, units, ROI, Value picks), or the empty
    state until a pick this season has been graded."""
    label = f'<div class="section-label">Moneyline, {season}</div>' if with_label else ""
    if not ml:
        return label + f'<div class="empty-state">{ML_EMPTY}</div>'
    pushes = f", {ml['pushes']} no decision" if ml["pushes"] else ""
    v = ml["value"]
    stats = [
        (ml["record"], f"{season} moneyline", f"{ml['n_decided']} picks graded{pushes}"),
        (moneyline.fmt_units(ml["units"]), "Units", "1 unit risked per pick"),
        (moneyline.fmt_roi(ml["roi"]), "ROI", "return per unit risked"),
        (v["record"] if v["n_decided"] else DASH, "Value picks",
         f"{moneyline.fmt_units(v['units'])}, ROI {moneyline.fmt_roi(v['roi'])}" if v["n_decided"] else "edge of 3+ points"),
    ]
    statline = '<div class="statline">' + "".join(
        f'<div class="stat"><div class="stat-value">{val}</div><div class="stat-label">{lab}</div>'
        f'<div class="stat-sub">{sub}</div></div>' for val, lab, sub in stats) + "</div>"
    note = ('<div class="table-footnote">Our moneyline pick in every game with a book moneyline, graded at 1 unit '
            'risked at the book price and locked at kickoff. A win at +135 pays 1.35 units, a win at -150 pays '
            '0.67, a loss costs 1. Ties count as no decision. This season\'s live picks only.</div>')
    return label + statline + note

def build_track_record_section(sport, summary, top25_summary=None, ml_html=""):
    top25_html = ml_html + build_top25_block(top25_summary)
    live_start = sport["live_tracking_start_season"]
    if summary is None:
        body = '<div class="empty-state">No games graded yet. Results appear here after the first week is played.</div>' + top25_html
        return card("Track Record", "Picks graded against final scores", body)

    year = summary.get("current_season_year", "")
    since = sport["ats_since_year"]
    # Vegas's own record, not the model's. The site's displayed pick defers
    # fully to the market for the straight-up spread call (0% model weight
    # there - see fitted_coefficients.json), so this IS what "our pick"
    # actually follows; showing the model's separate, weaker record next to
    # it read as confusing/misleading rather than transparent.
    current = summary.get("current_season", {}).get("vegas")
    all_time = summary.get("all_time", {}).get("vegas")

    if not current:
        body = f'<div class="empty-state">No {year} games graded yet.</div>' + top25_html
        return card("Track Record", "Picks graded against final scores", body)

    # A box-score style stat line for this season's headline numbers, then a
    # plain table comparing this season with the full graded history.
    stats = [(current["record"], f"{year} straight-up", pct(current.get("pick_accuracy")))]
    if current.get("ats_accuracy") is not None:
        stats.append((current["ats_record"], f"{year} against the spread", pct(current.get("ats_accuracy"))))
    stats.append((str(summary["n_graded_games"]), "Games graded", f"since {since}"))
    stats.append((str(live_start), "Live tracking since", f"{since}-{live_start - 1} backfilled" if since < live_start else ""))
    statline = '<div class="statline">' + "".join(
        f'<div class="stat"><div class="stat-value">{v}</div><div class="stat-label">{label}</div>'
        f'<div class="stat-sub">{sub}</div></div>'
        for v, label, sub in stats) + "</div>"

    rows = record_row(f"{year} season", current)
    if all_time:
        rows += record_row(f"Since {since}", all_time)

    note = f"""<div class="table-footnote">These are Vegas's closing-line results from {since} on, not a separate
      in-house number. For the straight-up call our displayed pick follows the market, because backtesting found
      no edge in overriding it. Against the spread is the harder test: the line is set so each side should
      cover about half the time.</div>"""

    body = statline + record_table(rows) + note + top25_html
    return card("Track Record", f"Vegas's closing-line record, {since} to now", body)

def build_players_page(sport, props):
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
            cells = "".join(f'<td data-key="{c}" data-value="{p[c]}" data-label="{h}" class="num">{p[c]}</td>'
                            for c, h in zip(stat_cols["display"], stat_cols["headers"]))
            rows += f"""<tr>
              <td data-key="player" data-value="{p['player_name']}"><b>{p['player_name']}</b>{tag}<div class="muted" style="font-size:13px;">{p['team']} vs {p['opponent']}</div>{badge}</td>
              {cells}
            </tr>"""
        hidden = "" if active else " hidden"
        return f"""<div class="cat-panel" id="cat-{cat_id}"{hidden}>
        <table class="data responsive-stack" data-sortable>
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
        return page_shell(sport, "Players", "players", card("Player Projections", "Top 20 per category by projected yards", body))

    subtabs = f"""<div class="subtabs">
      <button type="button" class="subtab active" aria-pressed="true" data-target="cat-passing">Passing</button>
      <button type="button" class="subtab" aria-pressed="false" data-target="cat-rushing">Rushing</button>
      <button type="button" class="subtab" aria-pressed="false" data-target="cat-receiving">Receiving</button>
    </div>"""
    body = subtabs + passing + rushing + receiving
    card_html = card("Player Projections", "Top 20 per category by projected yards. Select a column header to sort.", body)
    return page_shell(sport, "Players", "players", card_html)

def display_season(sport, games, log):
    """The season the Home and Teams pages show, so the site rolls over to a
    new season on its own: the season of the next game still to play (ignoring
    stale rows for old games that never got a result, like a cancelled game),
    else the latest season with a graded game."""
    if not games.empty and "season" in games.columns:
        upcoming = games
        if "gameday" in games.columns:
            cutoff = (datetime.now(timezone.utc) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            upcoming = games[games["gameday"].astype(str).str[:10] >= cutoff]
        if not upcoming.empty:
            return int(upcoming["season"].min())
    if not log.empty and "actual_margin" in log.columns and log["actual_margin"].notna().any():
        return int(log.loc[log["actual_margin"].notna(), "season"].max())
    return sport["live_tracking_start_season"]

def build_index_page(sport, games, props, comparison, accuracy_summary, log, top25_summary=None):
    season = display_season(sport, games, log)
    weeks = assemble_season_weeks(sport, games, log, comparison, season)
    this_week_key = current_week_key(weeks)
    this_week = weeks.get(this_week_key) if this_week_key else None
    week_title = this_week["label"] if this_week else "Upcoming"

    week_games_df = next_week_games(games)
    edge_html = build_edge_cards(sport, comparison, week_games_df) if comparison is not None else ""
    slate_html = render_week_table(this_week["games"] if this_week else [])
    ml_season, ml = ml_record(sport, games, log)
    track_html = build_track_record_section(sport, accuracy_summary, top25_summary, build_moneyline_block(ml_season, ml))

    subtitle = "Our model's pick and the Vegas line for every game this week, graded once played. The Teams tab has the full season."
    if not weeks and sport.get("no_games_note"):
        slate_html = f'<div class="empty-state">No games available yet. {sport["no_games_note"]}</div>'
    body = edge_html + card(f"This Week's Slate: {week_title}", subtitle, slate_html) + track_html
    return page_shell(sport, "Home", "index", body)

WEEK_TYPE_LABELS = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conf. Championship", "SB": "Super Bowl", "POST": "Postseason"}

def week_label(week, game_type=None):
    if game_type in WEEK_TYPE_LABELS:
        return WEEK_TYPE_LABELS[game_type]
    return f"Week {int(week)}"

def assemble_season_weeks(sport, games, log, comparison, season):
    """Every game for a season, grouped by week, already-played weeks merged
    with weeks still ahead. Already-played comes from the tracking log
    (graded); weeks still ahead come straight from that sport's
    game_predictions.csv, which already forecasts the rest of the season, not
    just the imminent week. Uses the pure model's own picks throughout (see
    model_pick()), not the market-mirroring blended line. Shared by both the
    Teams page (every week) and the Home page (just the current week)."""
    weeks = {}

    # A week's games can be split across "already played" and "still upcoming"
    # (e.g. Thursday night graded, Sunday/Monday not yet) - append into the
    # same week bucket rather than letting one loop overwrite the other's rows.
    def week_bucket(wk, game_type):
        key = f"w{int(wk)}"
        if key not in weeks:
            weeks[key] = {"label": week_label(wk, game_type), "week": int(wk), "games": []}
        return weeks[key]

    if not log.empty:
        graded = log[(log["actual_margin"].notna()) & (log["season"] == season)].copy()
    else:
        graded = pd.DataFrame(columns=["week", "home_team", "away_team"])
    already_graded = set(zip(graded["week"], graded["home_team"], graded["away_team"]))
    for wk, g in graded.groupby("week"):
        bucket = week_bucket(wk, g["game_type"].iloc[0] if "game_type" in g.columns else None)
        for _, r in g.sort_values("home_team").iterrows():
            pick = model_pick(r)
            has_vegas = pd.notna(r.get("vegas_home_favored_by"))
            vegas_favored = (r["home_team"] if r["vegas_home_favored_by"] > 0 else r["away_team"]) if has_vegas else None
            away_score = int(r["away_score"]) if pd.notna(r.get("away_score")) else None
            home_score = int(r["home_score"]) if pd.notna(r.get("home_score")) else None
            bucket["games"].append({
                "away_team": r["away_team"], "home_team": r["home_team"],
                "matchup_html": matchup_bar(sport, r["away_team"], r["home_team"], f"{away_score}-{home_score}"),
                "favored_team": team_short(sport, pick["favored_team"]), "favored_by": round(float(pick["favored_by"]), 1),
                "win_pct": round(float(pick["win_pct"]), 3), "total": round(float(pick["total"]), 1),
                "vegas_favored_team": team_short(sport, vegas_favored) if vegas_favored else None,
                "vegas_favored_by": round(float(abs(r["vegas_home_favored_by"])), 1) if has_vegas else None,
                "vegas_total": round(float(r["vegas_total"]), 1) if pd.notna(r.get("vegas_total")) else None,
                "graded": True, "home_score": home_score, "away_score": away_score,
                "correct": bool(r["model_correct_pick"]) if pd.notna(r.get("model_correct_pick")) else None,
                "ml": ml_view(sport, r),
            })

    if not games.empty:
        upcoming = games[games["season"] == season].copy()
    else:
        upcoming = pd.DataFrame(columns=["week", "home_team", "away_team"])
    vegas_cols = None
    if comparison is not None and not comparison.empty:
        vegas_cols = comparison[["home_team", "away_team", "vegas_favored_team", "vegas_home_favored_by", "total_line"]]
    # Moneyline picks for games not yet graded: the tracking log's copy first
    # (frozen at kickoff, so a game in progress keeps its pre-game price),
    # else this run's comparison.
    ml_rows = {}
    if comparison is not None and not comparison.empty and "ml_pick_side" in comparison.columns:
        for _, c in comparison[comparison["ml_pick_side"].notna()].iterrows():
            ml_rows[(int(c["week"]), c["home_team"], c["away_team"])] = c
    if not log.empty and "ml_pick_side" in log.columns:
        for _, c in log[(log["season"] == season) & log["ml_pick_side"].notna()].iterrows():
            ml_rows[(int(c["week"]), c["home_team"], c["away_team"])] = c
    for wk, g in upcoming.groupby("week"):
        bucket = week_bucket(wk, g["game_type"].iloc[0] if "game_type" in g.columns else None)
        merged = g.merge(vegas_cols, on=["home_team", "away_team"], how="left") if vegas_cols is not None else g
        sort_cols = [c for c in ["gameday", "gametime"] if c in merged.columns]
        for _, r in merged.sort_values(sort_cols if sort_cols else "home_team").iterrows():
            if (r["week"], r["home_team"], r["away_team"]) in already_graded:
                continue  # stale/duplicate row - the tracking log already has the final result
            pick = model_pick(r)
            has_vegas = pd.notna(r.get("vegas_home_favored_by"))
            kickoff = format_kickoff(r.get("weekday"), r.get("gametime"))
            bucket["games"].append({
                "away_team": r["away_team"], "home_team": r["home_team"],
                "matchup_html": matchup_bar(sport, r["away_team"], r["home_team"], kickoff or DASH),
                "favored_team": team_short(sport, pick["favored_team"]), "favored_by": round(float(pick["favored_by"]), 1),
                "win_pct": round(float(pick["win_pct"]), 3), "total": round(float(pick["total"]), 1),
                "vegas_favored_team": team_short(sport, r["vegas_favored_team"]) if has_vegas else None,
                "vegas_favored_by": round(float(abs(r["vegas_home_favored_by"])), 1) if has_vegas else None,
                "vegas_total": round(float(r["total_line"]), 1) if has_vegas and pd.notna(r.get("total_line")) else None,
                "graded": False, "home_score": None, "away_score": None, "correct": None,
                "ml": ml_view(sport, ml_rows.get((int(r["week"]), r["home_team"], r["away_team"]))),
            })

    # Pre-rendered for the Teams page, whose table site.js draws client-side.
    for wk in weeks.values():
        for g in wk["games"]:
            g["ml_html"] = ml_cell_html(g["ml"])
    return weeks

def current_week_key(weeks):
    """The week to default to: the earliest one with a game not yet played.
    Falls back to the earliest week overall if the whole season is graded."""
    ordered = sorted(weeks.items(), key=lambda kv: kv[1]["week"])
    for key, wk in ordered:
        if any(not g["graded"] for g in wk["games"]):
            return key
    return ordered[0][0] if ordered else None

def build_teams_page(sport, games, log, comparison):
    season = display_season(sport, games, log)
    weeks = assemble_season_weeks(sport, games, log, comparison, season)

    if not weeks:
        empty_note = f' {sport["no_games_note"]}' if sport.get("no_games_note") else ""
        body = card("Teams", f"Every {season} matchup, picked and graded", f'<div class="empty-state">No games available yet.{empty_note}</div>')
        return page_shell(sport, "Teams", "teams", body)

    week_order = [k for k, _ in sorted(weeks.items(), key=lambda kv: kv[1]["week"])]
    teams_json = json.dumps({"week_order": week_order, "weeks": weeks, "default_week": current_week_key(weeks)})
    body = card("Teams", f"Every {season} matchup from week 1 through the postseason, picked by our model and graded once final",
                '<select id="teams-week-select" class="week-picker"></select><div id="teams-week-content" style="margin-top:16px;"></div>'
                f'<script>const TEAMS_DATA = {teams_json};</script>')
    return page_shell(sport, "Teams", "teams", body)

def build_history_page(sport, log):
    graded = log[log["actual_margin"].notna()].copy() if not log.empty else log
    if graded.empty:
        body = card("History", "Every graded week, once there's one to show", '<div class="empty-state">No games graded yet.</div>')
        return page_shell(sport, "History", "history", body)

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
                return f"{team_short(sport, team)} -{abs(spread):.1f}"
            games_list.append({
                "away_team": team_short(sport, r["away_team"]), "home_team": team_short(sport, r["home_team"]),
                "away_score": int(r["away_score"]) if pd.notna(r.get("away_score")) else None,
                "home_score": int(r["home_score"]) if pd.notna(r.get("home_score")) else None,
                "model_pick": pick_str("model_spread"), "model_correct": bool(r["model_correct_pick"]) if pd.notna(r.get("model_correct_pick")) else None,
                "vegas_pick": pick_str("vegas_home_favored_by"), "vegas_correct": bool(r["vegas_correct_pick"]) if pd.notna(r.get("vegas_correct_pick")) else None,
            })
        weeks[key] = {"label": f"{int(season)}, Week {int(week)}", "games": games_list}

    history_json = json.dumps({"week_order": week_order, "weeks": weeks})
    body = card("History", "Every graded week. Choose a week to see how the picks did.",
                f'<select id="week-select" class="week-picker"></select><div id="week-content" style="margin-top:16px;"></div>'
                f'<script>const HISTORY_DATA = {history_json};</script>')
    return page_shell(sport, "History", "history", body)

def build_moneyline_card(sport, games, log):
    """Accuracy tab: this season's moneyline record and every graded pick."""
    season, ml = ml_record(sport, games, log)
    body = build_moneyline_block(season, ml, with_label=False)
    if ml:
        picks = log[(log["season"] == season) & log["ml_pick_side"].notna()
                    & (log["ml_won"].notna() | (log["ml_push"] == 1))].sort_values(["week", "gameday"], ascending=False)
        rows = ""
        for _, r in picks.iterrows():
            view = ml_view(sport, r)
            rows += f"""<tr>
          <td>{week_label(r["week"], r.get("game_type"))}<div class="faint" style="font-size:13px;">{team_short(sport, r["away_team"])} @ {team_short(sport, r["home_team"])}, final {int(r["away_score"])}-{int(r["home_score"])}</div></td>
          <td data-label="Pick" class="num ml-cell">{ml_cell_html(view)}</td>
        </tr>"""
        body += f"""<table class="data responsive-stack">
      <thead><tr><th>Game</th><th class="num">Pick</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""
    return card("Moneyline Record", f"{season} moneyline picks, graded at 1 unit each", body)

def build_accuracy_page(sport, log, games=None):
    games = games if games is not None else pd.DataFrame()
    ml_card = build_moneyline_card(sport, games, log)
    live_start = sport["live_tracking_start_season"]
    graded = log[log["actual_margin"].notna()].copy() if not log.empty else log
    if not graded.empty:
        graded = graded[graded["season"] >= live_start]
    if graded.empty:
        body = card("Accuracy Over Time", "Weekly trend, us vs. the market",
                     '<div class="empty-state">No live-tracked games graded yet. Check back once the '
                     f'{live_start} season starts.</div>')
        return page_shell(sport, "Accuracy", "accuracy", ml_card + body)

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
        f'<div class="chart-card" data-state="loading"><canvas id="{cid}" height="90"></canvas></div>'
        for cid in ["chart-accuracy", "chart-spread-mae", "chart-brier"]
    )
    body = card("Accuracy Over Time",
                f"Week-by-week results for {int(weekly['n'].sum())} live-picked games since the start of {live_start}: our model's picks against the market",
                charts_html + f'<script>const ACCURACY_DATA = {json.dumps(data)};</script>')
    return page_shell(sport, "Accuracy", "accuracy", ml_card + body)

def root_page_shell(title, body_html):
    """Shell for the pages that live at the site root rather than under a
    sport (404, Terms, Privacy) - same top bar and footer links, no sport
    hero or section tabs."""
    ver = asset_version()
    now = datetime.now(timezone.utc)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} | NFL Edge</title>
<link rel="icon" href="{FAVICON}">
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:ital,wght@0,600;0,700;0,800;1,700;1,800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css?v={ver}">
</head>
<body>
<a class="skip-link" href="#main-content">Skip to main content</a>
{top_bar(HOME_SPORT_TAB + '<a class="sport-tab" href="nfl/index.html">NFL</a><a class="sport-tab" href="cfb/index.html">CFB</a>' + OTHER_SPORT_TABS)}
<div class="wrap">
  <main id="main-content">
  {body_html}
  </main>
  <footer class="site-footer">
    <nav class="footer-links" aria-label="Site" style="border-top:none;margin-top:0;padding-top:0;">
      <a href="https://ant56-arch.github.io/terms.html">Terms of Use</a>
      <a href="https://ant56-arch.github.io/privacy.html">Privacy Policy</a>
      <a href="{HOME_URL}">All sites</a>
      <a href="https://github.com/ant56-arch/nfl-edge">Source code</a>
      <span>&copy; {now.year} NFL Edge</span>
    </nav>
  </footer>
</div>
</body>
</html>"""

def build_404_page():
    body = """<div class="error-body">
        <div class="error-code">404</div>
        <h1 class="error-title">Page not found</h1>
        <p>This page doesn't exist. It may have been moved or renamed.</p>
        <a class="btn-primary" href="nfl/index.html">Go to NFL Edge</a>
      </div>"""
    return root_page_shell("Page Not Found", body)

def build_redirect_page():
    """dist/index.html - a plain redirect to the default sport (NFL) so old
    bookmarks/links to the site root keep working now that content lives
    under nfl/ and cfb/ subdirectories."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=nfl/index.html">
<link rel="canonical" href="nfl/index.html">
<title>NFL Edge</title>
</head>
<body>
<p>Redirecting to <a href="nfl/index.html">NFL Edge</a>&hellip;</p>
</body>
</html>"""

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Factor labels for the Model tab: (label, note, scale, format). Rate gaps are
# fractions in the fit, so they're shown per 0.1 (ten percentage points).
FACTOR_LABELS = {
    "epa_diff_coef": ("Play efficiency gap (EPA)", "points of margin per 0.1 EPA per play", 0.1, "pts"),
    "ppa_diff_coef": ("Play efficiency gap (PPA)", "points of margin per 0.1 PPA per play", 0.1, "pts"),
    "third_down_weight": ("Third-down conversion gap", "points per 10-point gap in third-down rate", 0.1, "pts"),
    "redzone_weight": ("Red-zone touchdown gap", "points per 10-point gap in red-zone TD rate", 0.1, "pts"),
    "explosive_weight": ("Explosive-play gap", "points per 10-point gap in explosive-play rate", 0.1, "pts"),
    "sack_weight": ("Sack rate gap", "points per 10-point gap in sack rate", 0.1, "pts"),
    "success_rate_weight": ("Success rate gap", "points per 10-point gap in success rate", 0.1, "pts"),
    "explosiveness_weight": ("Explosiveness gap", "points per 0.1 gap in explosiveness", 0.1, "pts"),
    "home_field_advantage": ("Home field", "points for the home team", 1, "pts"),
    "margin_std_dev": ("Game-to-game swing", "how many points results typically miss by", 1, "plain"),
    "blend_weight_on_model_winprob": ("Model's say in the win chance", "the rest comes from the Vegas line", 1, "share"),
    "blend_weight_on_model_spread": ("Model's say in the spread", "the rest comes from the Vegas line", 1, "share"),
    "blend_weight_on_model_total": ("Model's say in the total", "the rest comes from the Vegas line", 1, "share"),
}

def next_weekly(weekday, hour_utc):
    """The next time a weekly UTC cron (weekday: Monday=0) fires, as UTC."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    d = now.replace(hour=hour_utc, minute=0, second=0, microsecond=0)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d if d > now else d + timedelta(days=7)

def load_model_runs(sport):
    path = os.path.join(TRACKING_DIR, "model_history.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [r for r in json.load(f).get("runs", []) if r.get("sport") == sport["slug"]]

def load_coefficients(sport):
    path = os.path.join(MODELS_DIR, sport["coefficients_json"])
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

def build_model_page(sport):
    model = load_coefficients(sport)
    runs = list(reversed(load_model_runs(sport)))
    rows = []
    for r in runs:
        label, tone = model_page.decision(r)
        chosen = r["best_candidate"] if r.get("switched_recipe") else (r.get("current_recipe_refit_on_holdout") or r["best_candidate"])
        rows.append({
            "date": model_page.short_date(r["run_at"]), "data_through": model_page.short_date(r.get("data_through")),
            "tested": len(r.get("candidates", [])), "decision": label, "tone": tone, "reason": r["reason"].capitalize() + ".",
            "before": (r.get("live_model_on_holdout") or {}).get("accuracy"),
            "after": chosen.get("accuracy") if r.get("deployed") else None,
        })
    last = runs[0] if runs else {}
    now_w = model_page_weights(model)
    before_w = last.get("weights_before") or {}
    factors = []
    for key, (label, note, scale, kind) in FACTOR_LABELS.items():
        if key not in now_w:
            continue
        fmt = ((lambda v: f"{v:.0%}") if kind == "share" else (lambda v: f"{v:+.1f} pts") if kind == "pts"
               else (lambda v: f"{v:.1f} pts"))
        factors.append({"label": label, "note": note, "now": now_w[key] * scale,
                        "before": before_w[key] * scale if key in before_w else None, "fmt": fmt})
    recipe = model.get("recipe", {})
    seasons = model.get("train_seasons", [])
    since = (f"every season since {seasons[0]}" if recipe.get("seasons") == "all" and seasons
             else f"the last {recipe.get('seasons')} seasons" if recipe.get("seasons") else "all past seasons")
    half = recipe.get("team_half_life")
    nxt = next_weekly(1, 10)
    trained = model.get("fitted_at")
    spec = {
        "intro": "Every Tuesday it checks itself against the newest games and only changes when a new version "
                 "clearly predicts better.",
        "tiles": [
            (model_page.short_date(trained)[:-6] if trained else DASH, "Last retrained",
             f"games through {model_page.short_date(model.get('trained_through'))}" if model.get("trained_through") else ""),
            (f"{model.get('n_games_used', 0):,}", "Games learned from", f"{seasons[0]} to {seasons[-1]}" if seasons else ""),
            (nxt.strftime("%b %-d"), "Next check", "Tuesday morning, every week"),
            (rows[0]["decision"].split(" ")[0] if rows else DASH, "Last decision",
             f"{rows[0]['tested']} versions tested" if rows else "no retrains yet"),
        ],
        "setup": [
            ("Learns from:", since),
            ("Team form:", f"recent games count most - a game's weight halves every {half} games" if half
             else "every game this season counts the same"),
            ("Sharp line:", "blends the model with the Vegas line; the last rows below show how much say the model gets"),
            ("Retrains:", "Tuesdays, only when new games have been played; in the offseason it waits"),
        ],
        "runs": rows,
        "score_name": "Picked right",
        "score_fmt": lambda v: f"{v:.1%}",
        "higher_better": True,
        "factors": factors,
        "factors_note": "Points are how much each factor moves the predicted margin toward the team that has the "
                        "edge in it. Before is the model that was live until the last retrain.",
        "empty": "No retrains logged yet. The first one runs on the next Tuesday after new games.",
    }
    return page_shell(sport, "Model", "model", model_page.render(spec))

def model_page_weights(model):
    out = dict(model.get("coefficients", {}))
    out.update({k: v for k, v in model.items() if k.startswith("blend_weight_on_")})
    return out

def build_summary(sport, games, log, comparison, accuracy_summary):
    """<sport>/summary.json - the current week's three most confident model
    picks (games not yet played first) and the same season record the Track
    Record card leads with, for this sport's card on the home page."""
    weeks = assemble_season_weeks(sport, games, log, comparison, display_season(sport, games, log))
    key = current_week_key(weeks)
    summary = {"updated": datetime.now(timezone.utc).isoformat(), "heading": None, "picks": [], "record": None,
               "empty": ("No games available yet. " + sport["no_games_note"]).strip(),
               "retrained": load_coefficients(sport).get("fitted_at"), "model_url": "model.html"}
    # Offseason (every game graded): no picks, rather than last season's.
    if key and any(not g["graded"] for g in weeks[key]["games"]):
        week = weeks[key]
        summary["heading"] = week["label"]
        top = sorted(week["games"], key=lambda g: (g["graded"], -g["win_pct"]))[:3]
        summary["picks"] = [{
            "label": f"{team_short(sport, g['away_team'])} @ {team_short(sport, g['home_team'])}",
            "sub": f"{g['win_pct']:.0%} to win",
            "value": f"{g['favored_team']} -{g['favored_by']:.1f}",
            "result": g["correct"] if g["graded"] else None,
        } for g in top]
    current = (accuracy_summary or {}).get("current_season", {}).get("vegas")
    if current:
        summary["record"] = {"value": current["record"],
                             "label": f"{accuracy_summary.get('current_season_year', '')} straight-up".strip(),
                             "sub": pct(current.get("pick_accuracy"))}
    # Optional: this season's moneyline record, same shape as "record".
    ml_season, ml = ml_record(sport, games, log)
    summary["moneyline_record"] = ({"value": ml["record"], "label": f"{ml_season} moneyline",
                                    "sub": f"{moneyline.fmt_units(ml['units'])}, ROI {moneyline.fmt_roi(ml['roi'])}"}
                                   if ml else None)
    return summary

# nflverse team codes that differ from ESPN's.
ESPN_NFL_CODES = {"LA": "LAR", "WAS": "WSH"}

def attach_game_picks(sport, slate, games, log, comparison):
    """Each ESPN game gets the model's pick from this season's weeks, matched
    on both teams (ESPN abbreviation, or school name for CFB)."""
    weeks = assemble_season_weeks(sport, games, log, comparison, display_season(sport, games, log))

    def keys(code):
        short = team_short(sport, code)
        return {str(code).upper(), str(short).upper(), ESPN_NFL_CODES.get(code, code).upper()}

    picks = []
    for wk in weeks.values():
        if slate.get("week") and wk["week"] != slate["week"]:
            continue
        for g in wk["games"]:
            picks.append((keys(g["away_team"]), keys(g["home_team"]), g))

    def espn_keys(t):
        return {t["abbr"].upper(), t["location"].upper(), t["short"].upper()}

    for eg in slate["games"]:
        a, h = espn_keys(eg["away"]), espn_keys(eg["home"])
        match = next((g for ak, hk, g in picks if ak & a and hk & h), None)
        if match:
            eg["pick"] = {"text": f"{match['favored_team']} -{match['favored_by']:.1f}, {match['win_pct']:.0%}",
                          "result": match["correct"] if match["graded"] and eg["state"] == "post" else None}
    return slate

def build_sport_pages(sport):
    print(f"Loading {sport['wordmark']} data...")
    games, props, comparison, accuracy_summary, log, top25_summary = load_data(sport)
    slate = attach_game_picks(sport, games_mod.load(sport["slug"]), games, log, comparison)

    pages = {
        "index.html": build_index_page(sport, games, props, comparison, accuracy_summary, log, top25_summary),
        "teams.html": build_teams_page(sport, games, log, comparison),
        "history.html": build_history_page(sport, log),
        "accuracy.html": build_accuracy_page(sport, log, games),
        "model.html": build_model_page(sport),
        "schedule.html": games_mod.schedule_redirect(sport["slug"]),
    }
    if sport["player_props_csv"]:
        pages["players.html"] = build_players_page(sport, props)

    out_dir = os.path.join(DIST_DIR, sport["slug"])
    os.makedirs(out_dir, exist_ok=True)
    for filename, html in pages.items():
        with open(os.path.join(out_dir, filename), "w") as f:
            f.write(html)
        print(f"  Wrote {sport['slug']}/{filename}")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(build_summary(sport, games, log, comparison, accuracy_summary), f, indent=1)
    print(f"  Wrote {sport['slug']}/summary.json")
    games_mod.write_json(os.path.join(out_dir, "games.json"), sport["slug"], slate,
                         datetime.now(timezone.utc).isoformat())

def main():
    print("Building site...")
    os.makedirs(DIST_DIR, exist_ok=True)

    for sport in SPORTS.values():
        build_sport_pages(sport)

    with open(os.path.join(DIST_DIR, "index.html"), "w") as f:
        f.write(build_redirect_page())
    print("  Wrote index.html (redirect to nfl/)")

    for filename, html in [("404.html", build_404_page()), ("terms.html", games_mod.legal_redirect("terms")),
                           ("privacy.html", games_mod.legal_redirect("privacy"))]:
        with open(os.path.join(DIST_DIR, filename), "w") as f:
            f.write(html)
        print(f"  Wrote {filename}")

    for asset in ["style.css", "site.js"]:
        shutil.copy(os.path.join(WEB_SRC_DIR, asset), os.path.join(DIST_DIR, asset))
    print("  Copied static assets")

    print(f"\nSite built in {DIST_DIR}")

if __name__ == "__main__":
    main()
