"""
moneyline.py
The moneyline pick, shared by the NFL and CFB pipelines (and defined the same
way on the MLB and NBA sites):

  1. Both sides' American odds become implied probabilities, and the vig is
     removed by normalizing the two to sum to 1.
  2. Our model's win probability for each side (the pure, unblended model -
     the same number the site shows as "Win%") is compared with that no-vig
     book probability. The pick is always the team our model picks to win,
     at its price; its edge is our win % minus the book's.
  3. It's labeled Value when that edge is at least 6 percentage points.
     Every game with a book moneyline still gets a pick either way.
  4. Once final, a pick is graded at 1 unit risked at the book price: a win
     at +135 pays +1.35u, a win at -150 pays +0.667u, a loss is -1u. A tie,
     or a game that never gets a final score, is no decision.
  5. The pick and price lock at kickoff: track_results.py /
     track_cfb_results.py only write a game's moneyline fields while its
     kickoff (the odds feed's commence_time) is still ahead, so a run during
     or after the game can't swap in live in-game odds.

The record only ever counts the current season's live picks - no backtests,
no past seasons.
"""

import numpy as np
import pandas as pd

VALUE_EDGE = 0.06

# Columns each tracking log keeps per game (see pick_columns()).
ML_COLS = ["ml_commence_time", "ml_home_price", "ml_away_price", "ml_pick_side", "ml_pick_team",
           "ml_pick_price", "ml_our_prob", "ml_book_prob", "ml_edge", "ml_value"]
ML_GRADE_COLS = ["ml_won", "ml_push", "ml_units"]


def implied_prob(american):
    """American odds -> implied probability (vig still in)."""
    if american is None or pd.isna(american) or american == 0:
        return np.nan
    american = float(american)
    return -american / (-american + 100) if american < 0 else 100 / (american + 100)


def no_vig(home_price, away_price):
    """Both sides' implied probabilities, normalized to sum to 1."""
    h, a = implied_prob(home_price), implied_prob(away_price)
    if pd.isna(h) or pd.isna(a) or h + a <= 0:
        return np.nan, np.nan
    return h / (h + a), a / (h + a)


def profit_units(price, won):
    """Units won or lost on a 1-unit risk at American `price`."""
    if not won:
        return -1.0
    price = float(price)
    return price / 100 if price > 0 else 100 / -price


def format_price(price):
    return f"+{int(price)}" if price > 0 else f"{int(price)}"


def pick_for_game(home_team, away_team, model_home_prob, home_price, away_price):
    """The moneyline pick for one game, or None when the game has no book
    moneyline on both sides or no model probability."""
    if pd.isna(model_home_prob) or pd.isna(home_price) or pd.isna(away_price):
        return None
    book_home, book_away = no_vig(home_price, away_price)
    if pd.isna(book_home):
        return None
    ours_home = float(model_home_prob)
    home = ours_home >= 0.5
    edge = ours_home - book_home if home else (1 - ours_home) - book_away
    return {
        "ml_home_price": float(home_price),
        "ml_away_price": float(away_price),
        "ml_pick_side": "home" if home else "away",
        "ml_pick_team": home_team if home else away_team,
        "ml_pick_price": float(home_price if home else away_price),
        "ml_our_prob": round(ours_home if home else 1 - ours_home, 4),
        "ml_book_prob": round(book_home if home else book_away, 4),
        "ml_edge": round(edge, 4),
        "ml_value": bool(edge >= VALUE_EDGE - 1e-9),
    }


def add_pick_columns(df, model_prob_col="model_home_win_prob"):
    """Adds the ML_COLS pick fields to a predictions-joined-with-odds frame
    (needs home_team, away_team, moneyline_home, moneyline_away and the
    model's home win probability; commence_time is carried through as the
    kickoff the pick locks at)."""
    df = df.copy()
    rows = []
    for _, r in df.iterrows():
        p = pick_for_game(r["home_team"], r["away_team"], r.get(model_prob_col),
                          r.get("moneyline_home"), r.get("moneyline_away")) or {}
        p["ml_commence_time"] = r.get("commence_time") if p else np.nan
        rows.append(p)
    picks = pd.DataFrame(rows, index=df.index)
    for c in ML_COLS:
        df[c] = picks[c] if c in picks.columns else np.nan
    return df


def lock_and_merge(log, snapshot, key_cols, now=None):
    """Writes each snapshot game's moneyline fields into the log, but only
    for games whose kickoff is still ahead. A game already started keeps
    whatever pick was logged before kickoff (or none), so the pick and price
    are locked at kickoff. Both frames must already have one row per key."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    snap = snapshot[[c for c in key_cols + ML_COLS if c in snapshot.columns]].copy()
    if "ml_commence_time" not in snap.columns:
        return log
    kickoff = pd.to_datetime(snap["ml_commence_time"], utc=True, errors="coerce")
    snap = snap[snap["ml_pick_side"].notna() & kickoff.notna() & (kickoff > now)]

    log = log.copy()
    for c in ML_COLS:
        if c not in log.columns:
            log[c] = pd.Series(np.nan, index=log.index, dtype=object if c in ("ml_commence_time", "ml_pick_side", "ml_pick_team", "ml_value") else float)
    if snap.empty:
        return log
    log = log.set_index(key_cols)
    snap = snap.set_index(key_cols)
    idx = snap.index.intersection(log.index)
    for c in ML_COLS:
        if log[c].dtype != object and snap[c].dtype == object:
            log[c] = log[c].astype(object)
        log.loc[idx, c] = snap.loc[idx, c]
    return log.reset_index()


def grade(log):
    """Fills ml_won / ml_push / ml_units for every logged pick with a final
    score (needs actual_margin). Recomputed from scratch each run."""
    log = log.copy()
    for c in ML_GRADE_COLS:
        log[c] = np.nan
    if "ml_pick_side" not in log.columns or "actual_margin" not in log.columns:
        return log
    has = log["ml_pick_side"].notna() & log["actual_margin"].notna() & log["ml_pick_price"].notna()
    if not has.any():
        return log
    margin = log["actual_margin"].astype(float)
    push = has & (margin == 0)
    decided = has & (margin != 0)
    won = ((log["ml_pick_side"] == "home") & (margin > 0)) | ((log["ml_pick_side"] == "away") & (margin < 0))
    log.loc[has, "ml_push"] = push[has].astype(float)
    log.loc[decided, "ml_won"] = won[decided].astype(float)
    log.loc[decided, "ml_units"] = [profit_units(p, w) for p, w in zip(log.loc[decided, "ml_pick_price"], won[decided])]
    return log


def _record(df):
    decided = df[df["ml_won"].notna()]
    wins = int(decided["ml_won"].sum())
    losses = int(len(decided)) - wins
    pushes = int((df["ml_push"] == 1).sum())
    units = float(decided["ml_units"].sum()) if len(decided) else 0.0
    return {
        "wins": wins, "losses": losses, "pushes": pushes,
        "record": f"{wins}-{losses}" + (f"-{pushes}" if pushes else ""),
        "units": round(units, 2),
        "roi": round(units / len(decided), 4) if len(decided) else None,
        "n_decided": int(len(decided)),
    }


def summarize(log, season):
    """This season's graded moneyline record (all picks, and Value picks
    only), or None when no pick this season has been decided yet."""
    if log is None or log.empty or "ml_won" not in log.columns:
        return None
    df = log[(log["season"] == season) & log["ml_pick_side"].notna() & (log["ml_won"].notna() | (log["ml_push"] == 1))]
    if df.empty or df["ml_won"].notna().sum() == 0:
        return None
    value_flag = df["ml_value"].astype(str).str.lower().isin(["true", "1", "1.0"])
    out = {"season": int(season), **_record(df)}
    out["value"] = _record(df[value_flag])
    return out


def is_value(v):
    return str(v).lower() in ("true", "1", "1.0")


def fmt_units(u):
    return f"{u:+.2f}u"


def fmt_roi(r):
    return f"{r * 100:+.1f}%" if r is not None else "-"
