"""The Model tab: how a sport's model retrains itself and whether it's getting
better, built from its model_history.json (one entry per retrain) and its live
weights file.

Keep this file identical in mlb-hit-predictor (MLB and NBA) and nfl-edge
(src/model_page.py, NFL and CFB). Each site passes a `spec` describing its own
model; this module only renders it with the shared Sports Edge markup.

spec keys:
  intro        one or two sentences on how this model learns
  tiles        [(value, label, sub)] for the headline stat line
  setup        [(label, plain-language value)] for the current setup
  runs         newest first: {date, data_through, tested, decision, tone, reason,
               before, after} - before/after are held-out scores or None
  score_name   what before/after measure, e.g. "Top 10 hit rate"
  score_fmt    number -> display string
  higher_better
  factors      [{label, now, before, fmt, note}] - before may be None
  factors_note footnote under the factors table
  empty        text when there are no retrains yet
"""
from html import escape


def _card(title, subtitle, body):
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    return f"""<section class="card">
    <div class="card-header"><h2>{title}</h2>{sub}</div>
    <div class="card-body">{body}</div>
  </section>"""


def _statline(stats, extra_class=""):
    return f'<div class="statline {extra_class}">' + "".join(
        f'<div class="stat"><div class="stat-value">{escape(str(v))}</div><div class="stat-label">{escape(label)}</div>'
        f'<div class="stat-sub">{escape(sub or "")}</div></div>' for v, label, sub in stats) + "</div>"


def _trend_chart(points, fmt, higher_better):
    """Inline SVG line of the held-out score after each retrain, oldest first."""
    w, h, pad_l, pad_r, pad_t, pad_b = 640, 200, 12, 12, 24, 30
    vals = [v for _, v in points]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or abs(hi) * 0.05 or 1
    lo, hi = lo - span * 0.25, hi + span * 0.25
    n = len(points)
    xs = [pad_l + (w - pad_l - pad_r) * (i / (n - 1) if n > 1 else 0.5) for i in range(n)]
    ys = [pad_t + (h - pad_t - pad_b) * (1 - (v - lo) / (hi - lo)) for v in vals]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" class="trend-dot"/>'
        f'<text x="{x:.1f}" y="{y - 10:.1f}" class="trend-value" text-anchor="middle">{escape(fmt(v))}</text>'
        f'<text x="{x:.1f}" y="{h - 8}" class="trend-label" text-anchor="middle">{escape(d)}</text>'
        for (d, v), x, y in zip(points, xs, ys))
    direction = "Higher is better" if higher_better else "Lower is better"
    return (f'<svg class="trend-chart" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="Score on held-out games after each retrain. {direction}.">'
            f'<polyline points="{line}" class="trend-line"/>{dots}</svg>')


def _change(now, before, fmt):
    if before is None:
        return '<span class="change-flat">new</span>'
    diff = now - before
    if abs(diff) < 1e-4 or fmt(now) == fmt(before):
        return '<span class="change-flat">no change</span>'
    arrow = "&#9650;" if diff > 0 else "&#9660;"
    return f'<span class="change-move">{arrow} {escape(fmt(abs(diff)).lstrip("+"))}</span>'


def render(spec):
    parts = []
    runs = spec["runs"]
    fmt = spec["score_fmt"]

    # How it learns: the headline numbers and the current setup in plain words.
    setup = "".join(f'<li><span class="setup-label">{escape(k)}</span> {escape(v)}</li>' for k, v in spec["setup"])
    parts.append(_card("How It Learns", escape(spec["intro"]),
                       _statline(spec["tiles"], "model-tiles") +
                       '<div class="section-label">Current setup</div>'
                       f'<ul class="setup-list">{setup}</ul>'))

    # Is it getting better: held-out score after each retrain.
    scored = [r for r in reversed(runs) if r.get("after") is not None]
    if scored:
        last = runs[0]
        tiles = []
        if last.get("before") is not None:
            tiles.append((fmt(last["before"]), "Before last retrain", "live model on recent games"))
        tiles.append((fmt(last["after"]), "After last retrain", "same games, never trained on"))
        body = _statline(tiles)
        if len(scored) >= 2:
            body += _trend_chart([(r["date"], r["after"]) for r in scored], fmt, spec["higher_better"])
        else:
            body += ('<div class="table-footnote">A trend line shows up here once there are a few retrains to '
                     'compare.</div>')
        body += (f'<div class="table-footnote">{escape(spec["score_name"])} on the most recent games, which each '
                 f"version is scored on before it's trained on them. "
                 f'{"Higher" if spec["higher_better"] else "Lower"} is better. Live results are on the '
                 f'Accuracy tab.</div>')
        parts.append(_card("Is It Getting Better?", "Score on recent games at each retrain", body))

    # Retrain log.
    if runs:
        rows = "".join(f"""<tr><td class="row-label">{escape(r['date'])}</td>
          <td data-label="Games through">{escape(r['data_through'] or '')}</td>
          <td data-label="Versions tested" class="num">{r['tested']}</td>
          <td data-label="Decision"><span class="pill pill-{r['tone']}">{escape(r['decision'])}</span></td>
          <td data-label="{escape(spec['score_name'])}" class="num">{escape(_before_after(r, fmt))}</td>
          <td data-label="Why" class="why-cell">{escape(r['reason'])}</td></tr>""" for r in runs)
        parts.append(_card("Retrain Log", "Every time the model checked itself against new games, newest first",
                           f"""<table class="data responsive-stack retrain-log">
          <thead><tr><th>Date</th><th>Games through</th><th class="num">Versions tested</th><th>Decision</th>
          <th class="num">{escape(spec['score_name'])}</th><th>Why</th></tr></thead>
          <tbody>{rows}</tbody></table>
          <div class="table-footnote">Each retrain fits several versions of the model on everything except the
          most recent games, scores them on those games, and switches only if a new version is clearly better.
          Then it refits on every game, and a safety check blocks anything that scores worse than the live
          model.</div>"""))
    else:
        parts.append(_card("Retrain Log", "", f'<div class="empty-state">{escape(spec["empty"])}</div>'))

    # What it's weighing.
    if spec["factors"]:
        rows = "".join(f"""<tr><td class="row-label">{escape(f['label'])}{f'<div class="factor-note">{escape(f["note"])}</div>' if f.get('note') else ''}</td>
          <td data-label="Now" class="num accent">{escape(f['fmt'](f['now']))}</td>
          <td data-label="Before last retrain" class="num">{escape(f['fmt'](f['before'])) if f['before'] is not None else '&mdash;'}</td>
          <td data-label="Change" class="num">{_change(f['now'], f['before'], f['fmt'])}</td></tr>""" for f in spec["factors"])
        parts.append(_card("What It's Weighing", "The factors behind every pick, and how the last retrain moved them",
                           f"""<table class="data responsive-stack factor-table">
          <thead><tr><th>Factor</th><th class="num">Now</th><th class="num">Before last retrain</th><th class="num">Change</th></tr></thead>
          <tbody>{rows}</tbody></table>
          <div class="table-footnote">{escape(spec['factors_note'])}</div>"""))
    return "".join(parts)


def _before_after(r, fmt):
    if r.get("after") is None:
        return "—"
    if r.get("before") is None:
        return fmt(r["after"])
    return f"{fmt(r['before'])} → {fmt(r['after'])}"


def decision(run):
    """(label, pill tone) for a history entry."""
    if not run.get("deployed"):
        return "Not updated", "danger"
    if run.get("switched_recipe"):
        return "Switched", "positive"
    return "Kept + refit", "market"


def short_date(iso):
    """'2026-09-23...' -> 'Sep 23, 2026'."""
    from datetime import date
    if not iso:
        return ""
    d = date.fromisoformat(iso[:10])
    return d.strftime("%b %-d, %Y")
