"""Game schedules and scores from ESPN's public scoreboard API, for the
scoreboard strip at the top of every Edge site and each sport's Schedule tab.

Keep this file identical in mlb-hit-predictor (MLB and NBA) and nfl-edge
(src/games.py, NFL and CFB).

  load(sport)            this sport's current slate, the way ESPN's own
                         scoreboard defines it: the current week for NFL and
                         CFB, the current day for MLB and NBA. Never raises;
                         returns {"label", "games": []} when ESPN can't be
                         reached (it's blocked in some sandboxes).
  write_json(path, ...)  games.json for the scoreboard strip, next to
                         summary.json. The strip refreshes scores from ESPN in
                         the browser and falls back to this file.
  render(...)            the Schedule tab body.

Each site attaches its own pick to a game as game["pick"] = {"text", "result"}
before writing or rendering.
"""
import json
import time
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
ESPN = "https://site.api.espn.com/apis/site/v2/sports/"
SPORTS = {
    "nfl": {"path": "football/nfl", "params": {}},
    # FBS only; the strip and Schedule tab keep games with a Top 25 team.
    "cfb": {"path": "football/college-football", "params": {"groups": "80", "limit": "400"}},
    "mlb": {"path": "baseball/mlb", "params": {}},
    "nba": {"path": "basketball/nba", "params": {}},
}


def espn_url(sport):
    s = SPORTS[sport]
    query = "&".join(f"{k}={v}" for k, v in s["params"].items())
    return f"{ESPN}{s['path']}/scoreboard" + (f"?{query}" if query else "")


def _get(url, params=None):
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=20,
                             headers={"User-Agent": "Mozilla/5.0 (Sports Edge; github.com/ant56-arch)"})
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == 2:
                print(f"  ESPN scoreboard unavailable: {e}")
                return None
            time.sleep(2 ** attempt)


def _team(c):
    t = c.get("team", {})
    rank = (c.get("curatedRank") or {}).get("current")
    record = next((r.get("summary") for r in c.get("records") or [] if r.get("type") in ("total", None)), None)
    probable = next((p.get("athlete", {}).get("shortName") for p in c.get("probables") or []), None)
    try:
        score = int(float(c.get("score"))) if c.get("score") not in (None, "") else None
    except (TypeError, ValueError):
        score = None
    return {
        "abbr": t.get("abbreviation", ""),
        "name": t.get("displayName", ""),
        "short": t.get("shortDisplayName") or t.get("name", ""),
        "location": t.get("location", ""),
        "logo": t.get("logo", ""),
        "rank": rank if isinstance(rank, int) and 1 <= rank <= 25 else None,
        "score": score,
        "record": record,
        "probable": probable,
        "winner": bool(c.get("winner")),
    }


def parse(event):
    comp = (event.get("competitions") or [{}])[0]
    sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
    if set(sides) != {"home", "away"}:
        return None
    status = (comp.get("status") or event.get("status") or {}).get("type", {})
    tv = []
    for b in comp.get("broadcasts") or []:
        for n in b.get("names") or []:
            if n not in tv:
                tv.append(n)
    return {
        "id": str(event.get("id")),
        "start": event.get("date"),
        "state": status.get("state", "pre"),  # pre / in / post
        "detail": status.get("shortDetail") or status.get("detail") or "",
        "tv": ", ".join(tv[:2]),
        "neutral": bool(comp.get("neutralSite")),
        "away": _team(sides["away"]),
        "home": _team(sides["home"]),
    }


def is_top25(g):
    return bool(g["away"]["rank"] or g["home"]["rank"])


def load(sport):
    """The current slate: {"label", "week", "games"}, games sorted by start time."""
    data = _get(espn_url(sport))
    if not data:
        return {"label": "", "week": None, "games": []}
    games = [g for g in (parse(e) for e in data.get("events", [])) if g]
    if sport == "cfb":
        games = [g for g in games if is_top25(g)]
    games.sort(key=lambda g: (g["start"] or "", g["id"]))
    week = (data.get("week") or {}).get("number")
    if sport in ("nfl", "cfb") and week:
        label = f"Week {week}"
    elif games:
        label = start_et(games[0]).strftime("%a, %b %-d")
    else:
        label = ""
    print(f"  {sport.upper()} scoreboard: {len(games)} games ({label or 'none'})")
    return {"label": label, "week": week if sport in ("nfl", "cfb") else None, "games": games}


def start_et(g):
    return datetime.fromisoformat(g["start"].replace("Z", "+00:00")).astimezone(ET)


def write_json(path, sport, slate, updated):
    """games.json for the scoreboard strip."""
    with open(path, "w") as f:
        json.dump({"sport": sport.upper(), "label": slate["label"], "updated": updated,
                   "espn": espn_url(sport), "top25_only": sport == "cfb", "games": slate["games"]}, f, indent=1)


# ── Schedule tab ─────────────────────────────────────────────────────────────
def _status(g):
    if g["state"] == "pre":
        t = start_et(g)
        return t.strftime("%-I:%M %p ET") if t.minute or t.hour else "Time TBA"
    return g["detail"] or ("Final" if g["state"] == "post" else "Live")


def _side(t, show_score):
    rank = f'<span class="sched-rank">{t["rank"]}</span>' if t["rank"] else ""
    logo = f'<img class="sched-logo" src="{escape(t["logo"])}" alt="" loading="lazy">' if t["logo"] else '<span class="sched-logo"></span>'
    score = (f'<span class="sched-score{" win" if t["winner"] else ""}">{t["score"]}</span>'
             if show_score and t["score"] is not None else "")
    sub = t["probable"] or t["record"] or ""
    return (f'<div class="sched-team">{logo}<div class="sched-name">{rank}<span>{escape(t["short"] or t["abbr"])}</span>'
            f'{f"<small>{escape(sub)}</small>" if sub else ""}</div>{score}</div>')


def _game(g, pick_label):
    live = g["state"] == "in"
    show_score = g["state"] != "pre"
    pick = g.get("pick")
    pick_html = ""
    if pick:
        pill = ""
        if pick.get("result") is True:
            pill = '<span class="pill pill-positive">HIT</span>'
        elif pick.get("result") is False:
            pill = '<span class="pill pill-danger">MISS</span>'
        pick_html = (f'<div class="sched-pick"><span class="sched-pick-label">{escape(pick_label)}</span>'
                     f'<span class="sched-pick-text">{escape(pick["text"])}</span>{pill}</div>')
    tv = f'<span class="sched-tv">{escape(g["tv"])}</span>' if g["tv"] else ""
    sep = "vs" if g["neutral"] else "@"
    return f"""<article class="sched-game{' is-live' if live else ''}">
      <div class="sched-top"><span class="sched-status">{escape(_status(g))}</span>{tv}</div>
      {_side(g["away"], show_score)}
      <div class="sched-sep">{sep}</div>
      {_side(g["home"], show_score)}
      {pick_html}
    </article>"""


def render(slate, card, pick_label, empty, note=""):
    """The Schedule tab body: the slate grouped by day, then by start time."""
    games = slate["games"]
    if not games:
        return card("Schedule", "", f'<div class="empty-state">{escape(empty)}</div>')
    days = {}
    for g in games:
        t = start_et(g)
        slot = t.strftime("%-I:%M %p ET") if t.hour or t.minute else "Time TBA"
        days.setdefault(t.strftime("%A, %B %-d"), {}).setdefault(slot, []).append(g)
    parts = []
    for day, slots in days.items():
        body = ""
        for slot, gs in slots.items():
            body += f'<div class="section-label">{escape(slot)}</div>' + '<div class="sched-grid">' + "".join(_game(g, pick_label) for g in gs) + "</div>"
        n = sum(len(gs) for gs in slots.values())
        parts.append(card(escape(day), f"{n} game{'s' if n != 1 else ''}", body))
    foot = f'<div class="table-footnote">{escape(note)}</div>' if note else ""
    return "".join(parts) + foot
