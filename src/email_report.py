"""
email_report.py
Formats game predictions + player props + Vegas lines into a styled HTML
email and sends it via Resend's API.

DESIGN: clean light "analytics report" theme - white cards on a soft neutral
background, a single burnt-orange brand accent for "our" numbers, teal for
market/Vegas data, green/red reserved for genuinely good/bad signals (a
favorable matchup, a blown call) rather than spread across everything.
Numeric columns are right-aligned and zebra-striped for scannability, the
way an actual research report reads rather than a wall of stats.

WHY LIGHT, NOT DARK: Gmail's automatic dark-mode recoloring can partially
override a custom dark theme's own colors even with the [data-ogsc] fix
below - a known, frustrating limitation that behaves differently on web vs.
iOS vs. Android. Light is the reliable choice across clients.

Required GitHub Secrets:
  RESEND_API_KEY   - your Resend API key
  RECIPIENT_EMAIL  - where picks get sent
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TRACKING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tracking")

THEME = "light"  # "dark" is kept below for reference, but see the docstring - light is the reliable choice.

if THEME == "dark":
    PAGE_BG = "#0b0d10"
    CARD_BG = "#151920"
    CARD_BORDER = "#252b35"
    ROW_ALT_BG = "#1a1f28"
    TEXT_PRIMARY = "#f5f7fa"
    TEXT_MUTED = "#b9c2cf"
    TEXT_FAINT = "#6b7280"
    ACCENT_PRIMARY = "#f5a623"
    ACCENT_PRIMARY_TINT = "#3a2f18"
    ACCENT_MARKET = "#3ec9d6"
    ACCENT_MARKET_TINT = "#173238"
    ACCENT_DANGER = "#e5484d"
    ACCENT_DANGER_TINT = "#3a1e1f"
    ACCENT_POSITIVE = "#3dd68c"
    ACCENT_POSITIVE_TINT = "#173328"
else:
    PAGE_BG = "#eef1f6"
    CARD_BG = "#ffffff"
    CARD_BORDER = "#e2e6ed"
    ROW_ALT_BG = "#f7f9fc"
    TEXT_PRIMARY = "#111827"
    TEXT_MUTED = "#6b7280"
    TEXT_FAINT = "#a3aab5"
    ACCENT_PRIMARY = "#c2410c"
    ACCENT_PRIMARY_TINT = "#fdece2"
    ACCENT_MARKET = "#0e7490"
    ACCENT_MARKET_TINT = "#e3f2f3"
    ACCENT_DANGER = "#b91c1c"
    ACCENT_DANGER_TINT = "#fbe9e9"
    ACCENT_POSITIVE = "#15803d"
    ACCENT_POSITIVE_TINT = "#e8f5ec"

FONT_DISPLAY = "'Helvetica Neue', Helvetica, Arial, sans-serif"
FONT_MONO = "'Courier New', Courier, monospace"

DASH = "—"
MIDDOT = "·"

def load_predictions():
    games = pd.read_csv(os.path.join(PROCESSED_DIR, "game_predictions.csv"))
    props = pd.read_csv(os.path.join(PROCESSED_DIR, "player_props.csv"))
    return games, props

def next_week_games(games):
    if games.empty:
        return games
    next_week = games.sort_values(["season", "week"]).iloc[0][["season", "week"]]
    return games[(games["season"] == next_week["season"]) & (games["week"] == next_week["week"])]

def pill(text, color, tint):
    return ('<span style="display:inline-block; color:' + color + '; background:' + tint + '; font-family:' + FONT_DISPLAY
            + '; font-size:9px; font-weight:800; letter-spacing:0.4px; border-radius:10px; padding:2px 7px; white-space:nowrap;">'
            + text + '</span>')

def card_open(title=None, subtitle=None):
    header = ""
    if title:
        subtitle_html = ""
        if subtitle:
            subtitle_html = "<div style=\"font-family:" + FONT_DISPLAY + "; font-size:12px; color:" + TEXT_MUTED + "; margin-top:3px;\">" + subtitle + "</div>"
        header = """
        <tr>
          <td style="padding:20px 22px 12px 22px; border-bottom:1px solid """ + CARD_BORDER + """;">
            <table role="presentation" cellpadding="0" cellspacing="0"><tr>
              <td style="width:4px; background:""" + ACCENT_PRIMARY + """; border-radius:2px; font-size:0; line-height:0;">&nbsp;</td>
              <td style="padding-left:10px;">
                <div style="font-family:""" + FONT_DISPLAY + """; font-size:15px; font-weight:800; color:""" + TEXT_PRIMARY + """;">""" + title + """</div>
                """ + subtitle_html + """
              </td>
            </tr></table>
          </td>
        </tr>"""
    return """
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:""" + CARD_BG + """; border:1px solid """ + CARD_BORDER + """; border-radius:12px; margin-bottom:18px;">
      """ + header + """
      <tr><td style="padding:16px 22px 22px 22px;">"""

def card_close():
    return "</td></tr></table>"

def format_kickoff(weekday, gametime):
    """'Thursday' + '20:20' -> 'Thu 8:20 PM'. Falls back to nothing if either
    piece is missing rather than showing a half-formed label."""
    if not weekday or pd.isna(weekday) or not gametime or pd.isna(gametime):
        return ""
    try:
        hour, minute = (int(x) for x in str(gametime).split(":"))
    except ValueError:
        return ""
    period = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12 or 12
    return str(weekday)[:3] + " " + str(hour_12) + ":" + format(minute, "02d") + " " + period

def build_games_table(games, comparison):
    if games.empty:
        return "<p style=\"color:" + TEXT_MUTED + "; font-family:" + FONT_DISPLAY + ";\">No upcoming games found.</p>"

    merged = games.copy()
    if comparison is not None and not comparison.empty:
        vegas_cols = comparison[["home_team", "away_team", "vegas_favored_team", "vegas_home_favored_by", "total_line", "vegas_home_win_prob"]]
        merged = merged.merge(vegas_cols, on=["home_team", "away_team"], how="left")

    # Kickoff order, not confidence order - "most confident pick" is
    # interesting but not how anyone actually plans their Sunday.
    sort_cols = [c for c in ["gameday", "gametime"] if c in merged.columns]
    merged = merged.sort_values(sort_cols if sort_cols else "favored_by", ascending=True).reset_index(drop=True)

    rows = ""
    last_day_seen = None
    day_row_idx = 0
    for i, g in merged.iterrows():
        win_pct = g["home_win_prob"] if g["favored_team"] == g["home_team"] else g["away_win_prob"]
        has_vegas = pd.notna(g.get("vegas_home_favored_by"))
        kickoff = format_kickoff(g.get("weekday"), g.get("gametime"))

        game_day = g.get("gameday")
        if pd.notna(game_day) and game_day != last_day_seen:
            last_day_seen = game_day
            day_row_idx = 0
            day_label = str(g.get("weekday")) if pd.notna(g.get("weekday")) else str(game_day)
            rows += ("<tr><td colspan=\"2\" style=\"padding:14px 10px 4px 10px; font-family:" + FONT_DISPLAY
                     + "; font-size:10px; font-weight:800; letter-spacing:0.8px; text-transform:uppercase; color:"
                     + TEXT_FAINT + ";\">" + day_label + "</td></tr>")

        if has_vegas:
            vegas_favored_team = g["vegas_favored_team"]
            vegas_line = vegas_favored_team + " -" + format(abs(g['vegas_home_favored_by']), ".1f")
            vegas_total = format(g['total_line'], ".1f") if pd.notna(g.get("total_line")) else None
            disagree = g["favored_team"] != vegas_favored_team
        else:
            vegas_line = None
            vegas_total = None
            disagree = False

        vegas_line_html = ('<span style="color:' + ACCENT_MARKET + ';">Vegas ' + vegas_line + '</span>') if vegas_line else ('<span style="color:' + TEXT_FAINT + ';">Vegas ' + DASH + '</span>')
        total_line = "O/U " + format(g['projected_total'], ".1f") + " <span style='color:" + TEXT_FAINT + ";'>(Vegas " + (vegas_total if vegas_total else DASH) + ")</span>"

        row_bg = ACCENT_DANGER_TINT if disagree else (ROW_ALT_BG if day_row_idx % 2 else CARD_BG)
        day_row_idx += 1
        flip_note = ('<div style="margin-top:4px;">' + pill("DIFFERENT PICK", ACCENT_DANGER, CARD_BG) + '</div>') if disagree else ""
        cell_style = "padding:12px 10px; background:" + row_bg + "; border-bottom:1px solid " + CARD_BORDER + ";"

        # Two columns, not five - a rigid 5-column grid overflows and gets
        # clipped on a phone-width inbox. Details stack vertically inside
        # each column instead, which degrades gracefully at any width.
        rows += """
        <tr>
          <td style=\"""" + cell_style + """ font-family:""" + FONT_DISPLAY + """; font-size:13px; color:""" + TEXT_PRIMARY + """; font-weight:600;\">
            """ + g['away_team'] + """ <span style="color:""" + TEXT_MUTED + """; font-weight:400;">@</span> """ + g['home_team'] + """
            <div style="font-family:""" + FONT_MONO + """; font-size:11px; color:""" + TEXT_MUTED + """; font-weight:400; margin-top:4px;">Win """ + format(win_pct, ".0%") + """ &middot; """ + total_line + """</div>
            """ + flip_note + """
          </td>
          <td style=\"""" + cell_style + """ text-align:right;\">
            """ + (('<div style="font-family:' + FONT_MONO + '; font-size:10px; color:' + TEXT_FAINT + ';">' + kickoff + '</div>') if kickoff else "") + """
            <div style="font-family:""" + FONT_MONO + """; font-size:14px; color:""" + ACCENT_PRIMARY + """; font-weight:700; margin-top:2px;">""" + g['favored_team'] + " -" + format(g['favored_by'], ".1f") + """</div>
            <div style="font-family:""" + FONT_MONO + """; font-size:11px; margin-top:4px;">""" + vegas_line_html + """</div>
          </td>
        </tr>"""

    return """
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="table-layout:fixed;">
      <tr style="font-family:""" + FONT_DISPLAY + """; font-size:10px; font-weight:800; letter-spacing:0.8px; text-transform:uppercase; color:""" + TEXT_MUTED + """;">
        <td width="58%" style="padding:0 10px 8px 10px; border-bottom:2px solid """ + CARD_BORDER + """;">Matchup</td>
        <td width="42%" style="padding:0 10px 8px 10px; border-bottom:2px solid """ + CARD_BORDER + """; text-align:right;">Line</td>
      </tr>
      """ + rows + """
    </table>
    <div style="font-family:""" + FONT_DISPLAY + """; font-size:11px; color:""" + TEXT_MUTED + """; margin-top:12px;">""" + pill("DIFFERENT PICK", ACCENT_DANGER, CARD_BG) + """ = we favor a different team than Vegas entirely.</div>"""

def build_edge_highlight_cards(comparison, max_cards=3):
    if comparison is None or comparison.empty:
        return ""
    notable = comparison[comparison["has_notable_edge"]].copy()
    if notable.empty:
        return ""
    notable["sort_key"] = notable["spread_edge"].abs() + notable["picks_flip"].astype(int) * 10
    top = notable.sort_values("sort_key", ascending=False).head(max_cards)

    # Stacked full-width cards, not a side-by-side grid - 3 columns
    # crammed into a phone-width inbox is exactly the overflow problem the
    # other tables had, just harder to fix with in-cell wrapping since these
    # are already small, dense cards rather than table rows.
    cards = ""
    for _, g in top.iterrows():
        flip = g["model_favored_team"] != g["vegas_favored_team"]
        bar_color = ACCENT_DANGER if flip else ACCENT_PRIMARY
        flip_pill = pill("DIFFERENT TEAM FAVORED", ACCENT_DANGER, ACCENT_DANGER_TINT) if flip else ""
        cards += """
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:""" + CARD_BG + """; border:1px solid """ + CARD_BORDER + """; border-radius:10px; margin-bottom:10px;">
          <tr>
            <td width="4" style="background:""" + bar_color + """; border-radius:10px 0 0 10px; font-size:0; line-height:0;">&nbsp;</td>
            <td style="padding:13px 15px;">
              <div style="font-family:""" + FONT_DISPLAY + """; font-size:12px; color:""" + TEXT_MUTED + """; font-weight:600;">""" + g['away_team'] + " @ " + g['home_team'] + """</div>
              """ + (('<div style="margin-top:6px;">' + flip_pill + '</div>') if flip_pill else "") + """
              <div style="font-family:""" + FONT_MONO + """; font-size:16px; font-weight:700; color:""" + ACCENT_PRIMARY + """; margin-top:6px;">""" + g['model_favored_team'] + " -" + format(g['model_favored_by'], ".1f") + """ <span style="font-family:""" + FONT_DISPLAY + """; font-size:11px; color:""" + TEXT_MUTED + """; font-weight:400;">our model</span></div>
              <div style="font-family:""" + FONT_MONO + """; font-size:13px; color:""" + ACCENT_MARKET + """; margin-top:2px;">""" + g['vegas_favored_team'] + " -" + format(abs(g['vegas_home_favored_by']), ".1f") + """ <span style="font-family:""" + FONT_DISPLAY + """; font-size:11px; color:""" + TEXT_MUTED + """; font-weight:400;">vegas</span></div>
            </td>
          </tr>
        </table>"""

    return """
    <div style="font-family:""" + FONT_DISPLAY + """; font-size:11px; font-weight:800; letter-spacing:0.8px; text-transform:uppercase; color:""" + TEXT_MUTED + """; margin-bottom:10px;">Notable Model vs. Market Gaps</div>
    """ + cards

def load_accuracy_summary():
    path = os.path.join(TRACKING_DIR, "accuracy_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        summary = json.load(f)
    return summary if summary.get("n_graded_games", 0) > 0 else None

def build_accuracy_scorecard(summary):
    """A running 'are we actually sharp' scorecard for the CURRENT season only
    - the 2024-2025 backfill exists so the model could be validated before
    launch, but the email only shows how the live system is doing this year.
    (Full history stays in data/tracking/accuracy_summary.json's "all_time"
    key for backend reference.)"""
    if summary is None:
        return ""

    year = summary.get("current_season_year", "")
    current = summary.get("current_season", {})
    sharp_current = current.get("sharp", {})

    if not sharp_current:
        return """
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border:1px dashed """ + CARD_BORDER + """; border-radius:8px;">
          <tr><td align="center" style="padding:22px;">
            <div style="font-family:""" + FONT_DISPLAY + """; font-size:13px; color:""" + TEXT_MUTED + """; line-height:1.6;">
              Tracking started for the """ + str(year) + """ season - no games graded yet.<br>Check back once the first week's games wrap.
            </div>
          </td></tr>
        </table>"""

    headline = """
        <div style="font-family:""" + FONT_DISPLAY + """; font-size:11px; font-weight:700; color:""" + TEXT_MUTED + """; letter-spacing:0.6px; text-transform:uppercase;">""" + str(year) + """ Season Straight-Up Record</div>
        <div style="font-family:""" + FONT_MONO + """; font-size:30px; font-weight:700; color:""" + ACCENT_PRIMARY + """; margin-top:4px;">""" + sharp_current['record'] + """ <span style="font-family:""" + FONT_DISPLAY + """; font-size:15px; color:""" + TEXT_MUTED + """; font-weight:400;">(""" + format(sharp_current['pick_accuracy'], ".0%") + """)</span></div>
        """

    def stat_row(window_key, window_label, i):
        window = summary.get(window_key)
        if not window:
            return ""
        sharp, vegas = window.get("sharp", {}), window.get("vegas", {})
        if not sharp or not vegas:
            return ""
        row_bg = ROW_ALT_BG if i % 2 else CARD_BG
        cell = "padding:11px 10px; background:" + row_bg + "; border-bottom:1px solid " + CARD_BORDER + ";"
        return """
        <tr>
          <td style=\"""" + cell + """ font-family:""" + FONT_DISPLAY + """; font-size:12px; color:""" + TEXT_MUTED + """;\">""" + window_label + """ <span style="color:""" + TEXT_FAINT + """;">(""" + str(sharp.get('n_games', 0)) + """ gm)</span></td>
          <td style=\"""" + cell + """ text-align:right;\">
            <div style="font-family:""" + FONT_MONO + """; font-size:13px; color:""" + ACCENT_PRIMARY + """; font-weight:700;">Us """ + sharp['record'] + """ <span style="color:""" + TEXT_FAINT + """; font-weight:400;">&plusmn;""" + format(sharp['spread_mae'], ".1f") + """</span></div>
            <div style="font-family:""" + FONT_MONO + """; font-size:11px; color:""" + ACCENT_MARKET + """; margin-top:3px;">Vegas """ + vegas['record'] + """ <span style="color:""" + TEXT_FAINT + """;">&plusmn;""" + format(vegas['spread_mae'], ".1f") + """</span></div>
          </td>
        </tr>"""

    rows = stat_row("current_season", str(year), 0) + stat_row("last_4_weeks", "Last 4 wks", 1)
    if not rows:
        return ""

    return headline + """
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="table-layout:fixed; margin-top:16px;">
      <tr style="font-family:""" + FONT_DISPLAY + """; font-size:10px; font-weight:800; letter-spacing:0.8px; text-transform:uppercase; color:""" + TEXT_MUTED + """;">
        <td width="46%" style="padding:0 10px 8px 10px; border-bottom:2px solid """ + CARD_BORDER + """;">Window</td>
        <td width="54%" style="padding:0 10px 8px 10px; border-bottom:2px solid """ + CARD_BORDER + """; text-align:right;">Record (spread err)</td>
      </tr>
      """ + rows + """
    </table>
    <div style="font-family:""" + FONT_DISPLAY + """; font-size:11px; color:""" + TEXT_MUTED + """; margin-top:12px; line-height:1.5;">Record = correct straight-up picks, graded against actual final scores. Spread err = avg points off the actual margin (lower is sharper).</div>"""

def injury_tag(status):
    if not status or pd.isna(status) or status == "":
        return ""
    color, tint = (ACCENT_PRIMARY, ACCENT_PRIMARY_TINT) if status == "Questionable" else (ACCENT_DANGER, ACCENT_DANGER_TINT)
    return ' ' + pill(status.upper(), color, tint)

def matchup_badge(mult):
    """Small colored pill flagging a real matchup edge - soft (favorable) or
    tough - so the reader doesn't have to mentally compare multipliers.
    Silent (no badge) for anything close to a neutral matchup."""
    if mult is None or pd.isna(mult):
        return ""
    if mult >= 1.08:
        return ' ' + pill("SOFT MATCHUP", ACCENT_POSITIVE, ACCENT_POSITIVE_TINT)
    if mult <= 0.92:
        return ' ' + pill("TOUGH MATCHUP", ACCENT_DANGER, ACCENT_DANGER_TINT)
    return ""

def sample_size_tag(games_played):
    if pd.isna(games_played) or games_played >= 6:
        return ""
    return ' <span style="color:' + TEXT_FAINT + '; font-size:10px; font-style:italic;">(' + str(int(games_played)) + ' gm sample)</span>'

def build_props_table(props, stat_cols, title, n=5):
    """Two columns, not one-per-stat: a rigid 4-5 column stat grid overflows
    on a phone-width inbox exactly like the games table did. The headline
    stat (yards) stands alone as a big number; everything else (attempts,
    completions, TDs) stacks underneath it as a single muted line."""
    if props.empty or stat_cols["sort"] not in props.columns:
        return ""
    top = props.dropna(subset=[stat_cols["sort"]]).sort_values(stat_cols["sort"], ascending=False).head(n).reset_index(drop=True)
    if top.empty:
        return ""

    headline_col = stat_cols["sort"]
    secondary_cols = [c for c in stat_cols["display"] if c != headline_col]
    secondary_labels = [h for c, h in zip(stat_cols["display"], stat_cols["headers"]) if c != headline_col]
    headline_label = stat_cols["headers"][stat_cols["display"].index(headline_col)]

    rows = ""
    for i, p in top.iterrows():
        row_bg = ROW_ALT_BG if i % 2 else CARD_BG
        cell_style = "padding:11px 10px; background:" + row_bg + "; border-bottom:1px solid " + CARD_BORDER + ";"

        secondary_line = " &middot; ".join(
            format(p[c], ".2f" if abs(p[c]) < 3 else ".1f") + " " + label.lower()
            for c, label in zip(secondary_cols, secondary_labels)
        )

        tag = injury_tag(p.get("injury_status")) + sample_size_tag(p.get("games_played"))
        badge = matchup_badge(p.get(stat_cols["matchup_col"])) if stat_cols.get("matchup_col") else ""
        rows += "<tr>"
        rows += ("<td style=\"" + cell_style + " font-family:" + FONT_DISPLAY + "; font-size:13px; color:"
                 + TEXT_PRIMARY + ";\"><b>" + str(p['player_name']) + "</b>" + tag
                 + "<div style='color:" + TEXT_MUTED + "; font-size:11px; margin-top:2px;'>" + str(p['team']) + " vs " + str(p['opponent'])
                 + "</div>" + ("<div style='margin-top:4px;'>" + badge.strip() + "</div>" if badge else "") + "</td>")
        rows += ("<td style='" + cell_style + " text-align:right;'>"
                 + "<div style=\"font-family:" + FONT_MONO + "; font-size:15px; color:" + ACCENT_PRIMARY + "; font-weight:700;\">"
                 + str(p[headline_col]) + " <span style=\"font-family:" + FONT_DISPLAY + "; font-size:11px; color:" + TEXT_MUTED + "; font-weight:400;\">" + headline_label.lower() + "</span></div>"
                 + "<div style=\"font-family:" + FONT_MONO + "; font-size:11px; color:" + TEXT_MUTED + "; margin-top:3px;\">" + secondary_line + "</div></td>")
        rows += "</tr>"

    return """
    <div style="font-family:""" + FONT_DISPLAY + """; font-size:12px; font-weight:800; letter-spacing:0.6px; text-transform:uppercase; color:""" + TEXT_PRIMARY + """; margin:20px 0 10px 0;">""" + title + """</div>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="table-layout:fixed;">
      <tr style="font-family:""" + FONT_DISPLAY + """; font-size:10px; font-weight:800; letter-spacing:0.8px; text-transform:uppercase; color:""" + TEXT_MUTED + """;">
        <td width="48%" style="padding:0 10px 8px 10px; border-bottom:2px solid """ + CARD_BORDER + """;">Player</td>
        <td width="52%" style="padding:0 10px 8px 10px; border-bottom:2px solid """ + CARD_BORDER + """; text-align:right;">Projection</td>
      </tr>
      """ + rows + """
    </table>"""

def build_email_html(games, props, comparison=None, accuracy_summary=None):
    week_games = next_week_games(games)
    week_label = "WEEK " + str(int(week_games.iloc[0]['week'])) if not week_games.empty else "UPCOMING"
    week_comparison = None
    if comparison is not None and not comparison.empty:
        week_comparison = comparison.merge(week_games[["home_team", "away_team"]], on=["home_team", "away_team"], how="inner")

    passing_html = build_props_table(props, {
        "sort": "proj_pass_yards",
        "display": ["proj_pass_attempts", "proj_completions", "proj_pass_yards", "proj_pass_tds"],
        "headers": ["Att", "Comp", "Yds", "TDs"],
        "matchup_col": "matchup_mult_pass",
    }, "Passing")

    rushing_html = build_props_table(props, {
        "sort": "proj_rush_yards",
        "display": ["proj_carries", "proj_rush_yards", "proj_rush_tds"],
        "headers": ["Car", "Yds", "TDs"],
        "matchup_col": "matchup_mult_rush",
    }, "Rushing")

    receiving_html = build_props_table(props, {
        "sort": "proj_rec_yards",
        "display": ["proj_targets", "proj_receptions", "proj_rec_yards", "proj_rec_tds"],
        "headers": ["Tgt", "Rec", "Yds", "TDs"],
        "matchup_col": "matchup_mult_rec",
    }, "Receiving")

    edge_cards = build_edge_highlight_cards(week_comparison)
    edge_section = ('<tr><td style="padding-bottom:18px;">' + edge_cards + '</td></tr>') if edge_cards else ""

    scorecard_html = build_accuracy_scorecard(accuracy_summary)
    scorecard_section = ('<tr><td>' + card_open("Track Record", "Graded against real final scores, not vibes") + scorecard_html + card_close() + '</td></tr>') if scorecard_html else ""

    props_subtitle = "(Q)/(D) = injury-discounted " + MIDDOT + " Out players excluded"
    generated_line = datetime.now().strftime('%b %d, %Y %I:%M %p')
    masthead_sub = week_label + " " + MIDDOT + " " + generated_line

    dark_mode_fix = ""
    if THEME == "dark":
        dark_mode_fix = """
      <style>
        [data-ogsc] { background-color: """ + PAGE_BG + """ !important; }
        [data-ogsb] { background-color: """ + PAGE_BG + """ !important; }
        [data-ogsc] td, [data-ogsc] div, [data-ogsc] span { color: """ + TEXT_PRIMARY + """ !important; }
      </style>"""

    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <meta name="color-scheme" content=\"""" + THEME + """\">
      <meta name="supported-color-schemes" content=\"""" + THEME + """\">""" + dark_mode_fix + """
    </head>
    <body class="body" style="margin:0; padding:0; background:""" + PAGE_BG + """;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:""" + PAGE_BG + """;" bgcolor=\"""" + PAGE_BG + """\">
        <tr><td align="center" style="padding:32px 12px;">
          <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px; width:100%;">

            <tr><td style="padding-bottom:24px;">
              <div style="font-family:""" + FONT_DISPLAY + """; font-size:26px; font-weight:800; letter-spacing:-0.5px; color:""" + TEXT_PRIMARY + """;">NFL <span style="color:""" + ACCENT_PRIMARY + """;">EDGE</span></div>
              <div style="height:3px; width:48px; background:""" + ACCENT_PRIMARY + """; border-radius:2px; margin:8px 0 10px 0; font-size:0; line-height:0;">&nbsp;</div>
              <div style="font-family:""" + FONT_DISPLAY + """; font-size:12px; font-weight:600; color:""" + TEXT_MUTED + """; letter-spacing:0.3px;">""" + masthead_sub + """</div>
            </td></tr>

            """ + edge_section + """

            <tr><td>""" + card_open("Slate", "Our line vs. the market, every game") + """
              """ + build_games_table(week_games, week_comparison) + """
            """ + card_close() + """</td></tr>

            """ + scorecard_section + """

            <tr><td>""" + card_open("Player Projections", props_subtitle) + """
              """ + passing_html + """
              """ + rushing_html + """
              """ + receiving_html + """
            """ + card_close() + """</td></tr>

            <tr><td style="padding-top:6px; border-top:1px solid """ + CARD_BORDER + """;">
              <div style="font-family:""" + FONT_DISPLAY + """; font-size:11px; color:""" + TEXT_MUTED + """; line-height:1.6; padding-top:14px;">
                Our line blends a coefficients-fit EPA model with the live market line (weights validated on
                held-out seasons, see src/fit_model.py). Vegas lines via DraftKings (the-odds-api.com) where
                available. Not betting advice.
              </div>
            </td></tr>

          </table>
        </td></tr>
      </table>
    </body>
    </html>"""
    return html

def send_email(html_content, week_label):
    api_key = os.environ["RESEND_API_KEY"]
    recipient = os.environ["RECIPIENT_EMAIL"]

    response = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        json={
            "from": "onboarding@resend.dev",
            "to": [recipient],
            "subject": "NFL Edge - " + week_label + " - " + datetime.now().strftime('%b %d, %Y'),
            "html": html_content,
        },
        timeout=30,
    )
    response.raise_for_status()
    print("Email sent to " + recipient + " (Resend id: " + str(response.json().get('id')) + ")")

def main():
    print("Loading predictions...")
    games, props = load_predictions()

    comparison = None
    comparison_path = os.path.join(PROCESSED_DIR, "vegas_comparison.csv")
    if os.path.exists(comparison_path):
        comparison = pd.read_csv(comparison_path)
        print("  Loaded Vegas comparison (" + str(len(comparison)) + " games)")
    else:
        print("  No Vegas comparison found - email will show a dash for Vegas lines")

    accuracy_summary = load_accuracy_summary()
    if accuracy_summary:
        print("  Loaded accuracy track record (" + str(accuracy_summary["n_graded_games"]) + " graded games)")
    else:
        print("  No graded prediction history yet - track record section will be omitted")

    print("Building email...")
    week_games = next_week_games(games)
    week_label = "Week " + str(int(week_games.iloc[0]['week'])) if not week_games.empty else "Upcoming"
    html = build_email_html(games, props, comparison, accuracy_summary)

    print("Sending email via Resend...")
    send_email(html, week_label)

if __name__ == "__main__":
    main()
