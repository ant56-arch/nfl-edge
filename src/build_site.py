"""
build_site.py
Generates the static NFL Edge website (dist/) from the same processed data
the email used to read - the email is retired; this is the sole output now.

Pages:
  index.html    - this week's slate (sorted by kickoff time, day-grouped),
                  notable model-vs-market gaps, track record, player props
  history.html  - every graded week, browsable via a dropdown (client-side,
                  no per-week routing needed for a site this size)
  accuracy.html - trend charts of our accuracy vs Vegas's over time

Published to GitHub Pages by weekly-picks.yml. This script only builds the
dist/ directory - the workflow's own steps upload and deploy it.
"""

import pandas as pd
import numpy as np
import json
import os
import shutil
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

def page_shell(title, active_tab, body_html):
    tabs = [("index.html", "index", "This Week"), ("history.html", "history", "History"), ("accuracy.html", "accuracy", "Accuracy")]
    nav = "".join(
        f'<a href="{href}" class="{"active" if tab == active_tab else ""}">{label}</a>'
        for href, tab, label in tabs
    )
    generated = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} - NFL Edge</title>
<link rel="stylesheet" href="style.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
</head>
<body>
<div class="wrap">
  <header class="masthead">
    <div class="wordmark">NFL <span>EDGE</span></div>
    <div class="accent-bar"></div>
    <div class="subline">Updated {generated}</div>
    <nav class="tabs">{nav}</nav>
  </header>
  {body_html}
  <footer class="site-footer">
    Our line blends a coefficients-fit EPA model with the live market line (weights validated on
    held-out seasons). Vegas lines via DraftKings (the-odds-api.com) where available. Not betting advice.
  </footer>
</div>
<script src="site.js"></script>
</body>
</html>"""

def card(title, subtitle, body_html):
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    return f"""<div class="card">
    <div class="card-header"><div class="bar"></div><div><h2>{title}</h2>{sub}</div></div>
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
    merged = merged.sort_values(sort_cols if sort_cols else "favored_by").reset_index(drop=True)

    rows = ""
    last_day = None
    for _, g in merged.iterrows():
        win_pct = g["home_win_prob"] if g["favored_team"] == g["home_team"] else g["away_win_prob"]
        has_vegas = pd.notna(g.get("vegas_home_favored_by"))

        if has_vegas:
            vegas_favored_team = g["vegas_favored_team"]
            vegas_line = f"{vegas_favored_team} -{abs(g['vegas_home_favored_by']):.1f}"
            vegas_total = f"{g['total_line']:.1f}" if pd.notna(g.get("total_line")) else None
            disagree = g["favored_team"] != vegas_favored_team
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
        our_line = f"{g['favored_team']} -{g['favored_by']:.1f}"
        vegas_html = f'<span class="market-color">{vegas_line}</span>' if vegas_line else f'<span class="faint">{DASH}</span>'
        total_html = f"{g['projected_total']:.1f} <span class='faint'>/</span> " + (f'<span class="market-color">{vegas_total}</span>' if vegas_total else f'<span class="faint">{DASH}</span>')
        flag = ' row-flag' if disagree else ""
        flip_note = f' {pill("DIFFERENT PICK", "danger")}' if disagree else ""

        rows += f"""<tr class="{flag.strip()}">
          <td data-key="matchup" data-value="{matchup}">{matchup}{flip_note}</td>
          <td data-key="kickoff" data-value="{kickoff_sort}" class="num mono">{kickoff}</td>
          <td data-key="ourline" data-value="{g['favored_by']:.2f}" class="num mono accent">{our_line}</td>
          <td data-key="vegas" data-value="{abs(g['vegas_home_favored_by']) if has_vegas else -1:.2f}" class="num mono">{vegas_html}</td>
          <td data-key="total" data-value="{g['projected_total']:.2f}" class="num mono">{total_html}</td>
          <td data-key="winpct" data-value="{win_pct:.3f}" class="num mono">{win_pct:.0%}</td>
        </tr>"""

    return f"""<table class="data" data-sortable>
      <thead><tr>
        <th data-sort-key="matchup">Matchup</th>
        <th data-sort-key="kickoff" class="num">Kickoff</th>
        <th data-sort-key="ourline" class="num">Our Line</th>
        <th data-sort-key="vegas" class="num">Vegas</th>
        <th data-sort-key="total" class="num">Total (us / vegas)</th>
        <th data-sort-key="winpct" class="num">Win%</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="muted" style="font-size:11px; margin-top:12px;">{pill("DIFFERENT PICK", "danger")} = we favor a different team than Vegas entirely. Click a column header to sort.</div>"""

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
        border = "var(--danger)" if flip else "var(--primary)"
        flip_line = f'<div style="margin-top:6px;">{pill("DIFFERENT TEAM FAVORED", "danger")}</div>' if flip else ""
        cards += f"""<div style="border:1px solid var(--card-border); border-top:4px solid {border}; border-radius:10px; padding:14px; flex:1; min-width:180px;">
          <div class="muted" style="font-size:12px; font-weight:600;">{g['away_team']} @ {g['home_team']}</div>
          <div class="mono" style="font-size:17px; font-weight:700; color:var(--primary); margin-top:6px;">{g['model_favored_team']} -{g['model_favored_by']:.1f} <span class="muted" style="font-size:11px; font-weight:400;">our model</span></div>
          <div class="mono market-color" style="font-size:13px; margin-top:2px;">{g['vegas_favored_team']} -{abs(g['vegas_home_favored_by']):.1f} <span class="muted" style="font-size:11px;">vegas</span></div>
          {flip_line}
        </div>"""

    return f"""<div style="margin-bottom:20px;">
      <div class="muted" style="font-size:11px; font-weight:800; letter-spacing:0.8px; text-transform:uppercase; margin-bottom:10px;">Notable Model vs. Market Gaps</div>
      <div style="display:flex; gap:12px; flex-wrap:wrap;">{cards}</div>
    </div>"""

def build_track_record_section(summary):
    if summary is None:
        return card("Track Record", "Graded against real final scores, not vibes",
                     '<div class="empty-state">No games graded yet. Check back once the first week wraps.</div>')

    year = summary.get("current_season_year", "")
    current = summary.get("current_season", {}).get("sharp")
    all_time = summary.get("all_time", {}).get("sharp")
    all_time_vegas = summary.get("all_time", {}).get("vegas")

    hero = ""
    if current:
        hero = f"""<div class="hero-label">{year} SEASON STRAIGHT-UP RECORD</div>
        <div class="hero-number">{current['record']} <span class="muted" style="font-size:16px; font-weight:400;">({current['pick_accuracy']:.0%})</span></div>"""
    else:
        hero = f'<div class="empty-state">No {year} games graded yet.</div>'

    tiles = ""
    if all_time:
        tiles += f"""<div class="stat-tile"><div class="value accent">{all_time['record']}</div><div class="label">All-Time Record</div></div>
        <div class="stat-tile"><div class="value">&plusmn;{all_time['spread_mae']:.1f}</div><div class="label">Our Spread Error</div></div>"""
    if all_time_vegas:
        tiles += f"""<div class="stat-tile"><div class="value market-color">{all_time_vegas['record']}</div><div class="label">Vegas All-Time</div></div>
        <div class="stat-tile"><div class="value">&plusmn;{all_time_vegas['spread_mae']:.1f}</div><div class="label">Vegas Spread Error</div></div>"""

    body = hero + (f'<div class="stat-grid">{tiles}</div>' if tiles else "")
    return card("Track Record", "Graded against real final scores, not vibes " + f"({summary['n_graded_games']} games all-time)", body)

def build_props_section(props):
    def table(stat_cols, title):
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
            rows += f"""<tr>
              <td data-key="player" data-value="{p['player_name']}"><b>{p['player_name']}</b>{tag}<div class="muted" style="font-size:11px;">{p['team']} vs {p['opponent']}</div>{badge}</td>
              {cells}
            </tr>"""
        return f"""<h3 style="font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:0.6px; margin:20px 0 10px 0;">{title}</h3>
        <table class="data" data-sortable>
          <thead><tr><th data-sort-key="player">Player</th>{headers}</tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    passing = table({"sort": "proj_pass_yards", "display": ["proj_pass_attempts", "proj_completions", "proj_pass_yards", "proj_pass_tds"],
                      "headers": ["Att", "Comp", "Yds", "TDs"], "matchup_col": "matchup_mult_pass"}, "Passing")
    rushing = table({"sort": "proj_rush_yards", "display": ["proj_carries", "proj_rush_yards", "proj_rush_tds"],
                      "headers": ["Car", "Yds", "TDs"], "matchup_col": "matchup_mult_rush"}, "Rushing")
    receiving = table({"sort": "proj_rec_yards", "display": ["proj_targets", "proj_receptions", "proj_rec_yards", "proj_rec_tds"],
                        "headers": ["Tgt", "Rec", "Yds", "TDs"], "matchup_col": "matchup_mult_rec"}, "Receiving")
    return card("Player Projections", "Top 20 per category by projected yards - click a column to sort", passing + rushing + receiving)

def build_index_page(games, props, comparison, accuracy_summary):
    week_games = next_week_games(games)
    week_label = f"Week {int(week_games.iloc[0]['week'])}" if not week_games.empty else "Upcoming"
    week_comparison = None
    if comparison is not None and not comparison.empty:
        week_comparison = comparison.merge(week_games[["home_team", "away_team"]], on=["home_team", "away_team"], how="inner")

    edge_html = build_edge_cards(comparison, week_games) if comparison is not None else ""
    slate_html = build_slate_table(week_games, week_comparison)
    track_html = build_track_record_section(accuracy_summary)
    props_html = build_props_section(props)

    body = edge_html + card(f"Slate - {week_label}", "Our line vs. the market, every game", slate_html) + track_html + props_html
    return page_shell(week_label, "index", body)

def build_history_page(log):
    graded = log[log["actual_margin"].notna()].copy() if not log.empty else log
    if graded.empty:
        body = card("History", "Every graded week, once there's one to show", '<div class="empty-state">No games graded yet.</div>')
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
                "sharp_pick": pick_str("sharp_spread"), "sharp_correct": bool(r["sharp_correct_pick"]) if pd.notna(r.get("sharp_correct_pick")) else None,
                "vegas_pick": pick_str("vegas_home_favored_by"), "vegas_correct": bool(r["vegas_correct_pick"]) if pd.notna(r.get("vegas_correct_pick")) else None,
            })
        weeks[key] = {"label": f"{int(season)} - Week {int(week)}", "games": games_list}

    history_json = json.dumps({"week_order": week_order, "weeks": weeks})
    body = card("History", "Every graded week - pick a week to see how we did",
                f'<select id="week-select" class="week-picker"></select><div id="week-content" style="margin-top:16px;"></div>'
                f'<script>const HISTORY_DATA = {history_json};</script>')
    return page_shell("History", "history", body)

def build_accuracy_page(log):
    graded = log[log["actual_margin"].notna()].copy() if not log.empty else log
    if not graded.empty:
        graded = graded[graded["season"] >= LIVE_TRACKING_START_SEASON]
    if graded.empty:
        body = card("Accuracy Over Time", "Weekly trend, us vs. the market",
                     '<div class="empty-state">No live-tracked games graded yet - check back once the '
                     f'{LIVE_TRACKING_START_SEASON} season kicks off.</div>')
        return page_shell("Accuracy", "accuracy", body)

    graded = graded.sort_values(["season", "week"])
    weekly = graded.groupby(["season", "week"]).agg(
        sharp_accuracy=("sharp_correct_pick", "mean"), vegas_accuracy=("vegas_correct_pick", "mean"),
        sharp_spread_mae=("sharp_spread_error", "mean"), vegas_spread_mae=("vegas_spread_error", "mean"),
        sharp_brier=("sharp_brier", "mean"), vegas_brier=("vegas_brier", "mean"),
        n=("sharp_correct_pick", "size"),
    ).reset_index()

    labels = [f"{int(s)} Wk{int(w)}" for s, w in zip(weekly["season"], weekly["week"])]
    data = {
        "labels": labels,
        "us_accuracy": weekly["sharp_accuracy"].round(3).tolist(), "vegas_accuracy": weekly["vegas_accuracy"].round(3).tolist(),
        "us_spread_mae": weekly["sharp_spread_mae"].round(2).tolist(), "vegas_spread_mae": weekly["vegas_spread_mae"].round(2).tolist(),
        "us_brier": weekly["sharp_brier"].round(4).tolist(), "vegas_brier": weekly["vegas_brier"].round(4).tolist(),
    }
    charts_html = "".join(
        f'<div class="chart-card" style="margin-bottom:28px;"><canvas id="{cid}" height="90"></canvas></div>'
        for cid in ["chart-accuracy", "chart-spread-mae", "chart-brier"]
    )
    body = card("Accuracy Over Time",
                f"Weekly trend across {int(weekly['n'].sum())} live-picked games ({LIVE_TRACKING_START_SEASON} season onward), us vs. the market",
                charts_html + f'<script>const ACCURACY_DATA = {json.dumps(data)};</script>')
    return page_shell("Accuracy", "accuracy", body)

def main():
    print("Loading data...")
    games, props, comparison, accuracy_summary, log = load_data()

    print("Building pages...")
    os.makedirs(DIST_DIR, exist_ok=True)
    pages = {
        "index.html": build_index_page(games, props, comparison, accuracy_summary),
        "history.html": build_history_page(log),
        "accuracy.html": build_accuracy_page(log),
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
