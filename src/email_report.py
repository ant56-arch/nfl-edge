"""
email_report.py
Formats game predictions + player props + Vegas lines into a styled HTML
email and sends it via Resend's API.

DESIGN: dark "broadcast terminal" theme. Bold condensed headers for
structure, monospace for every number (scores, odds, stat lines), amber for
our own picks, cyan for market/Vegas data.

Required GitHub Secrets:
  RESEND_API_KEY   - your Resend API key
  RECIPIENT_EMAIL  - where picks get sent
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

BG = "#0b0d10"
CARD_BG = "#151920"
CARD_BORDER = "#252b35"
TEXT_PRIMARY = "#e8eaed !important"
TEXT_MUTED = "#8b93a1 !important"
ACCENT_AMBER = "#f5a623 !important"
ACCENT_CYAN = "#3ec9d6 !important"
ACCENT_RED = "#e5484d !important"
FONT_DISPLAY = "'Helvetica Neue', Helvetica, Arial, sans-serif"
FONT_MONO = "'Courier New', Courier, monospace"

DASH = "\u2014"
MIDDOT = "\u00b7"

def load_predictions():
    games = pd.read_csv(os.path.join(PROCESSED_DIR, "game_predictions.csv"))
    props = pd.read_csv(os.path.join(PROCESSED_DIR, "player_props.csv"))
    return games, props

def next_week_games(games):
    if games.empty:
        return games
    next_week = games.sort_values(["season", "week"]).iloc[0][["season", "week"]]
    return games[(games["season"] == next_week["season"]) & (games["week"] == next_week["week"])]

def card_open(title=None, subtitle=None):
    header = ""
    if title:
        subtitle_html = ""
        if subtitle:
            subtitle_html = "<div style=\"font-family:" + FONT_DISPLAY + "; font-size:12px; color:" + TEXT_MUTED + "; margin-top:2px;\">" + subtitle + "</div>"
        header = """
        <tr>
          <td style="padding:18px 20px 4px 20px;">
            <div style="font-family:""" + FONT_DISPLAY + """; font-size:11px; font-weight:800; letter-spacing:2px; text-transform:uppercase; color:""" + ACCENT_AMBER + """;">""" + title + """</div>
            """ + subtitle_html + """
          </td>
        </tr>"""
    return """
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:""" + CARD_BG + """; border:1px solid """ + CARD_BORDER + """; border-radius:10px; margin-bottom:16px;">
      """ + header + """
      <tr><td style="padding:8px 20px 20px 20px;">"""

def card_close():
    return "</td></tr></table>"

def build_games_table(games, comparison):
    if games.empty:
        return "<p style='color:" + TEXT_MUTED + "; font-family:" + FONT_DISPLAY + ";'>No upcoming games found.</p>"

    merged = games.copy()
    if comparison is not None and not comparison.empty:
        vegas_cols = comparison[["home_team", "away_team", "vegas_favored_team", "vegas_home_favored_by", "total_line", "vegas_home_win_prob"]]
        merged = merged.merge(vegas_cols, on=["home_team", "away_team"], how="left")

    merged = merged.sort_values("favored_by", ascending=False)

    rows = ""
    for i, g in merged.iterrows():
        win_pct = g["home_win_prob"] if g["favored_team"] == g["home_team"] else g["away_win_prob"]
        has_vegas = pd.notna(g.get("vegas_home_favored_by"))

        if has_vegas:
            vegas_favored_team = g["vegas_favored_team"]
            vegas_line = vegas_favored_team + " -" + format(abs(g['vegas_home_favored_by']), ".1f")
            vegas_total = format(g['total_line'], ".1f") if pd.notna(g.get("total_line")) else DASH
            disagree = g["favored_team"] != vegas_favored_team
        else:
            vegas_line = DASH
            vegas_total = DASH
            disagree = False

        row_bg = "#1a1f27" if i % 2 == 0 else CARD_BG
        flip_dot = ('<span style="color:' + ACCENT_RED + '; font-weight:800;">&#9679;</span> ') if disagree else ""

        rows += """
        <tr style="background:""" + row_bg + """;">
          <td style="padding:10px 8px; font-family:""" + FONT_DISPLAY + """; font-size:13px; color:""" + TEXT_PRIMARY + """; border-bottom:1px solid """ + CARD_BORDER + """;">""" + g['away_team'] + """ <span style="color:""" + TEXT_MUTED + """;">@</span> """ + g['home_team'] + """</td>
          <td style="padding:10px 8px; font-family:""" + FONT_MONO + """; font-size:13px; color:""" + ACCENT_AMBER + """; font-weight:700; border-bottom:1px solid """ + CARD_BORDER + """;">""" + g['favored_team'] + " -" + format(g['favored_by'], ".1f") + """</td>
          <td style="padding:10px 8px; font-family:""" + FONT_MONO + """; font-size:13px; color:""" + ACCENT_CYAN + """; border-bottom:1px solid """ + CARD_BORDER + """;">""" + flip_dot + vegas_line + """</td>
          <td style="padding:10px 8px; font-family:""" + FONT_MONO + """; font-size:13px; color:""" + TEXT_PRIMARY + """; border-bottom:1px solid """ + CARD_BORDER + """;">""" + format(g['projected_total'], ".1f") + """ <span style="color:""" + TEXT_MUTED + """;">/</span> <span style="color:""" + ACCENT_CYAN + """;">""" + vegas_total + """</span></td>
          <td style="padding:10px 8px; font-family:""" + FONT_MONO + """; font-size:13px; color:""" + TEXT_PRIMARY + """; border-bottom:1px solid """ + CARD_BORDER + """;">""" + format(win_pct, ".0%") + """</td>
        </tr>"""

    return """
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
      <tr style="font-family:""" + FONT_DISPLAY + """; font-size:10px; font-weight:800; letter-spacing:1px; text-transform:uppercase; color:""" + TEXT_MUTED + """;">
        <td style="padding:0 8px 8px 8px;">Matchup</td>
        <td style="padding:0 8px 8px 8px;">Our Line</td>
        <td style="padding:0 8px 8px 8px;">Vegas</td>
        <td style="padding:0 8px 8px 8px;">Total (us / vegas)</td>
        <td style="padding:0 8px 8px 8px;">Win%</td>
      </tr>
      """ + rows + """
    </table>
    <div style="font-family:""" + FONT_DISPLAY + """; font-size:11px; color:""" + TEXT_MUTED + """; margin-top:10px;"><span style="color:""" + ACCENT_RED + """;">&#9679;</span> = we favor a different team than Vegas entirely.</div>"""

def build_edge_highlight_cards(comparison, max_cards=3):
    if comparison is None or comparison.empty:
        return ""
    notable = comparison[comparison["has_notable_edge"]].copy()
    if notable.empty:
        return ""
    notable["sort_key"] = notable["spread_edge"].abs() + notable["picks_flip"].astype(int) * 10
    top = notable.sort_values("sort_key", ascending=False).head(max_cards)

    cells = ""
    for _, g in top.iterrows():
        flip = g["favored_team"] != g["vegas_favored_team"]
        border_color = ACCENT_RED if flip else ACCENT_AMBER
        flip_line = ""
        if flip:
            flip_line = '<div style="font-family:' + FONT_DISPLAY + '; font-size:10px; font-weight:800; color:' + ACCENT_RED + '; letter-spacing:1px; margin-top:6px;">DIFFERENT TEAM FAVORED</div>'
        cells += """
        <td width=\"""" + str(100 // max_cards) + """%\" valign="top" style="padding:0 6px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:""" + BG + """; border:1px solid """ + border_color + """; border-radius:8px;">
            <tr><td style="padding:12px;">
              <div style="font-family:""" + FONT_DISPLAY + """; font-size:12px; color:""" + TEXT_MUTED + """;">""" + g['away_team'] + " @ " + g['home_team'] + """</div>
              <div style="font-family:""" + FONT_MONO + """; font-size:16px; font-weight:700; color:""" + ACCENT_AMBER + """; margin-top:4px;">""" + g['favored_team'] + " -" + format(g['favored_by'], ".1f") + """</div>
              <div style="font-family:""" + FONT_MONO + """; font-size:12px; color:""" + ACCENT_CYAN + """; margin-top:2px;">Vegas: """ + g['vegas_favored_team'] + " -" + format(abs(g['vegas_home_favored_by']), ".1f") + """</div>
              """ + flip_line + """
            </td></tr>
          </table>
        </td>"""

    return """
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
      <tr>""" + cells + """</tr>
    </table>"""

def injury_tag(status):
    if not status or pd.isna(status) or status == "":
        return ""
    color = ACCENT_AMBER if status == "Questionable" else ACCENT_RED
    return ' <span style="color:' + color + '; font-size:10px; font-weight:800; letter-spacing:0.5px;">' + status.upper() + '</span>'

def build_props_table(props, stat_cols, title, n=5):
    if props.empty or stat_cols["sort"] not in props.columns:
        return ""
    top = props.dropna(subset=[stat_cols["sort"]]).sort_values(stat_cols["sort"], ascending=False).head(n)
    if top.empty:
        return ""

    rows = ""
    for i, p in top.iterrows():
        cells = "".join(
            "<td style='padding:9px 8px; font-family:" + FONT_MONO + "; font-size:13px; color:" + TEXT_PRIMARY + "; border-bottom:1px solid " + CARD_BORDER + ";'>" + str(p[c]) + "</td>"
            for c in stat_cols["display"]
        )
        tag = injury_tag(p.get("injury_status"))
        row_bg = "#1a1f27" if i % 2 == 0 else CARD_BG
        rows += "<tr style=\"background:" + row_bg + ";\">"
        rows += "<td style='padding:9px 8px; font-family:" + FONT_DISPLAY + "; font-size:13px; color:" + TEXT_PRIMARY + "; border-bottom:1px solid " + CARD_BORDER + ";'><b>" + str(p['player_name']) + "</b>" + tag + "<br><span style='color:" + TEXT_MUTED + "; font-size:11px;'>" + str(p['team']) + " vs " + str(p['opponent']) + "</span></td>"
        rows += cells
        rows += "</tr>"

    headers = "".join("<td style='padding:0 8px 6px 8px;'>" + h + "</td>" for h in stat_cols["headers"])
    return """
    <div style="font-family:""" + FONT_DISPLAY + """; font-size:12px; font-weight:800; color:""" + TEXT_PRIMARY + """; letter-spacing:0.5px; margin:16px 0 8px 0;">""" + title + """</div>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
      <tr style="font-family:""" + FONT_DISPLAY + """; font-size:10px; font-weight:800; letter-spacing:1px; text-transform:uppercase; color:""" + TEXT_MUTED + """;">
        <td style="padding:0 8px 6px 8px;">Player</td>
        """ + headers + """
      </tr>
      """ + rows + """
    </table>"""

def build_email_html(games, props, comparison=None):
    week_games = next_week_games(games)
    week_label = "WEEK " + str(int(week_games.iloc[0]['week'])) if not week_games.empty else "UPCOMING"
    week_comparison = None
    if comparison is not None and not comparison.empty:
        week_comparison = comparison.merge(week_games[["home_team", "away_team"]], on=["home_team", "away_team"], how="inner")

    passing_html = build_props_table(props, {
        "sort": "proj_pass_yards",
        "display": ["proj_pass_attempts", "proj_completions", "proj_pass_yards", "proj_pass_tds"],
        "headers": ["Att", "Comp", "Yds", "TDs"],
    }, "PASSING")

    rushing_html = build_props_table(props, {
        "sort": "proj_rush_yards",
        "display": ["proj_carries", "proj_rush_yards", "proj_rush_tds"],
        "headers": ["Car", "Yds", "TDs"],
    }, "RUSHING")

    receiving_html = build_props_table(props, {
        "sort": "proj_rec_yards",
        "display": ["proj_targets", "proj_receptions", "proj_rec_yards", "proj_rec_tds"],
        "headers": ["Tgt", "Rec", "Yds", "TDs"],
    }, "RECEIVING")

    edge_cards = build_edge_highlight_cards(week_comparison)
    edge_section = ('<tr><td style="padding-bottom:16px;">' + edge_cards + '</td></tr>') if edge_cards else ""

    props_subtitle = "(Q)/(D) = injury-discounted " + MIDDOT + " Out players excluded"
    generated_line = datetime.now().strftime('%b %d, %Y %I:%M %p')
    masthead_sub = week_label + " " + MIDDOT + " " + generated_line

    html = """
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <meta name="color-scheme" content="dark">
      <meta name="supported-color-schemes" content="dark">
    </head>
    <body style="margin:0; padding:0; background:""" + BG + """;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:""" + BG + """;" bgcolor=\"""" + BG + """\">
        <tr><td align="center" style="padding:24px 12px;">
          <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px; width:100%;">

            <tr><td style="padding-bottom:20px;">
              <div style="font-family:""" + FONT_DISPLAY + """; font-size:28px; font-weight:800; letter-spacing:-0.5px; color:""" + TEXT_PRIMARY + """;">NFL <span style="color:""" + ACCENT_AMBER + """;">EDGE</span></div>
              <div style="font-family:""" + FONT_MONO + """; font-size:12px; color:""" + TEXT_MUTED + """; margin-top:2px;">""" + masthead_sub + """</div>
            </td></tr>

            """ + edge_section + """

            <tr><td>""" + card_open("Slate", "Our line vs. the market, every game") + """
              """ + build_games_table(week_games, week_comparison) + """
            """ + card_close() + """</td></tr>

            <tr><td>""" + card_open("Player Projections", props_subtitle) + """
              """ + passing_html + """
              """ + rushing_html + """
              """ + receiving_html + """
            """ + card_close() + """</td></tr>

            <tr><td style="padding-top:8px;">
              <div style="font-family:""" + FONT_DISPLAY + """; font-size:11px; color:""" + TEXT_MUTED + """; line-height:1.5;">
                Model-based projections from 2 years of play-by-play data plus current-season adjustments.
                Vegas lines via DraftKings (the-odds-api.com) where available. Not betting advice.
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

    print("Building email...")
    week_games = next_week_games(games)
    week_label = "Week " + str(int(week_games.iloc[0]['week'])) if not week_games.empty else "Upcoming"
    html = build_email_html(games, props, comparison)

    print("Sending email via Resend...")
    send_email(html, week_label)

if __name__ == "__main__":
    main()
