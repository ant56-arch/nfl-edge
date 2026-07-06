"""
email_report.py
Formats game predictions + player props into a clean HTML email and sends
it via Resend's API - same delivery mechanism as BTS Edge.

Required GitHub Secrets (this repo, matching bts-edge's naming):
  RESEND_API_KEY   - your Resend API key
  RECIPIENT_EMAIL  - where picks get sent

Note: since we're using the default onboarding@resend.dev sending address
(no custom domain verified), Resend only allows sending TO the email address
associated with your Resend account. That's fine here since you're the only
recipient - but if you ever want to send to a different inbox, you'd need to
verify a domain in Resend first.
"""

import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def load_predictions():
    games = pd.read_csv(os.path.join(PROCESSED_DIR, "game_predictions.csv"))
    props = pd.read_csv(os.path.join(PROCESSED_DIR, "player_props.csv"))
    return games, props

def next_week_games(games):
    """Only show the soonest upcoming week, not the entire season."""
    if games.empty:
        return games
    next_week = games.sort_values(["season", "week"]).iloc[0][["season", "week"]]
    return games[(games["season"] == next_week["season"]) & (games["week"] == next_week["week"])]

def build_games_table(games):
    if games.empty:
        return "<p>No upcoming games found.</p>"
    games = games.sort_values("favored_by", ascending=False)
    rows = ""
    for _, g in games.iterrows():
        win_pct = g["home_win_prob"] if g["favored_team"] == g["home_team"] else g["away_win_prob"]
        rows += f"""
        <tr>
            <td>{g['away_team']} @ {g['home_team']}</td>
            <td><b>{g['favored_team']}</b> -{g['favored_by']}</td>
            <td>{win_pct:.0%}</td>
            <td>{g['projected_total']}</td>
        </tr>"""
    return f"""
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background:#222; color:#fff;">
            <th style="padding:8px; text-align:left;">Matchup</th>
            <th style="padding:8px; text-align:left;">Favorite</th>
            <th style="padding:8px; text-align:left;">Win%</th>
            <th style="padding:8px; text-align:left;">Total</th>
        </tr>
        {rows}
    </table>"""

def build_props_table(props, stat_cols, title, n=5):
    if props.empty or stat_cols["sort"] not in props.columns:
        return ""
    top = props.dropna(subset=[stat_cols["sort"]]).sort_values(stat_cols["sort"], ascending=False).head(n)
    if top.empty:
        return ""
    rows = ""
    for _, p in top.iterrows():
        cells = "".join(f"<td style='padding:6px;'>{p[c]}</td>" for c in stat_cols["display"])
        rows += f"<tr><td style='padding:6px;'><b>{p['player_name']}</b> ({p['team']} vs {p['opponent']})</td>{cells}</tr>"
    headers = "".join(f"<th style='padding:6px; text-align:left;'>{h}</th>" for h in stat_cols["headers"])
    return f"""
    <h3>{title}</h3>
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background:#222; color:#fff;">
            <th style="padding:6px; text-align:left;">Player</th>
            {headers}
        </tr>
        {rows}
    </table>"""

def build_vegas_comparison_table(comparison):
    if comparison is None or comparison.empty:
        return ""
    notable = comparison[comparison["has_notable_edge"]].sort_values("spread_edge", key=abs, ascending=False)
    if notable.empty:
        return "<h2>Vs. Vegas</h2><p>No notable disagreements with the market this week.</p>"

    rows = ""
    for _, g in notable.iterrows():
        flags = []
        if g["notable_spread_edge"]:
            flags.append(f"Spread: us {g['projected_spread']:+.1f} vs Vegas {g['vegas_home_favored_by']:+.1f}")
        if g["notable_total_edge"]:
            flags.append(f"Total: us {g['projected_total']:.1f} vs Vegas {g['total_line']:.1f}")
        if g["notable_win_prob_edge"]:
            flags.append(f"Win%: us {g['home_win_prob']:.0%} vs Vegas {g['vegas_home_win_prob']:.0%}")
        rows += f"""
        <tr>
            <td>{g['away_team']} @ {g['home_team']}</td>
            <td>{"<br>".join(flags)}</td>
        </tr>"""
    return f"""
    <h2>Vs. Vegas - Notable Edges</h2>
    <table style="width:100%; border-collapse:collapse;">
        <tr style="background:#222; color:#fff;">
            <th style="padding:8px; text-align:left;">Matchup</th>
            <th style="padding:8px; text-align:left;">Where we disagree with the market</th>
        </tr>
        {rows}
    </table>
    <p style="color:#999; font-size:12px;">Thresholds: 3+ pt spread edge, 3+ pt total edge, or 8+ pt win probability edge.</p>"""

def build_email_html(games, props, comparison=None):
    week_games = next_week_games(games)
    week_label = f"Week {week_games.iloc[0]['week']}" if not week_games.empty else "Upcoming"

    passing_html = build_props_table(props, {
        "sort": "proj_pass_yards",
        "display": ["proj_pass_attempts", "proj_completions", "proj_pass_yards", "proj_pass_tds"],
        "headers": ["Att", "Comp", "Pass Yds", "Pass TDs"],
    }, "Top Passing Projections")

    rushing_html = build_props_table(props, {
        "sort": "proj_rush_yards",
        "display": ["proj_carries", "proj_rush_yards", "proj_rush_tds"],
        "headers": ["Carries", "Rush Yds", "Rush TDs"],
    }, "Top Rushing Projections")

    receiving_html = build_props_table(props, {
        "sort": "proj_rec_yards",
        "display": ["proj_targets", "proj_receptions", "proj_rec_yards", "proj_rec_tds"],
        "headers": ["Targets", "Rec", "Rec Yds", "Rec TDs"],
    }, "Top Receiving Projections")

    vegas_html = build_vegas_comparison_table(comparison)

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; color:#111;">
        <h1>NFL Edge - {week_label} Picks</h1>
        <p style="color:#666;">Generated {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}</p>

        <h2>Game Predictions</h2>
        {build_games_table(week_games)}

        {vegas_html}

        <h2>Player Props</h2>
        {passing_html}
        {rushing_html}
        {receiving_html}

        <p style="color:#999; font-size:12px; margin-top:30px;">
            Projections are model-based estimates from 2 years of play-by-play data.
            Vegas comparison uses de-vigged lines from DraftKings (via the-odds-api.com) where available.
        </p>
    </body>
    </html>"""

def send_email(html_content, week_label):
    api_key = os.environ["RESEND_API_KEY"]
    recipient = os.environ["RECIPIENT_EMAIL"]

    response = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "from": "onboarding@resend.dev",
            "to": [recipient],
            "subject": f"NFL Edge Picks - {week_label} - {datetime.now().strftime('%b %d, %Y')}",
            "html": html_content,
        },
        timeout=30,
    )
    response.raise_for_status()
    print(f"Email sent to {recipient} (Resend id: {response.json().get('id')})")

def main():
    print("Loading predictions...")
    games, props = load_predictions()

    comparison = None
    comparison_path = os.path.join(PROCESSED_DIR, "vegas_comparison.csv")
    if os.path.exists(comparison_path):
        comparison = pd.read_csv(comparison_path)
        print(f"  Loaded Vegas comparison ({len(comparison)} games)")
    else:
        print("  No Vegas comparison found - email will skip that section")

    print("Building email...")
    week_games = next_week_games(games)
    week_label = f"Week {week_games.iloc[0]['week']}" if not week_games.empty else "Upcoming"
    html = build_email_html(games, props, comparison)

    print("Sending email via Resend...")
    send_email(html, week_label)

if __name__ == "__main__":
    main()
