// NFL Edge site behavior: sortable tables, the week-archive picker, and
// trend charts (Chart.js, loaded via CDN in each page's <head>).

function makeSortable(table) {
  const tbody = table.tBodies[0];
  table.querySelectorAll("th[data-sort-key]").forEach((th, colIndex) => {
    th.addEventListener("click", () => {
      const rows = Array.from(tbody.querySelectorAll("tr")).filter(r => !r.classList.contains("day-header"));
      const asc = !th.classList.contains("sorted-asc");
      table.querySelectorAll("th").forEach(h => h.classList.remove("sorted-asc", "sorted-desc"));
      th.classList.add(asc ? "sorted-asc" : "sorted-desc");

      const key = th.dataset.sortKey;
      const isNumeric = th.classList.contains("num");
      rows.sort((a, b) => {
        const av = a.querySelector(`[data-key="${key}"]`)?.dataset.value ?? "";
        const bv = b.querySelector(`[data-key="${key}"]`)?.dataset.value ?? "";
        if (isNumeric) return asc ? parseFloat(av) - parseFloat(bv) : parseFloat(bv) - parseFloat(av);
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      });
      // Sorting collapses any day-group headers - they only make sense in
      // chronological order, not after re-sorting by an arbitrary column.
      tbody.querySelectorAll(".day-header").forEach(r => r.remove());
      rows.forEach(r => tbody.appendChild(r));
    });
  });
}

document.querySelectorAll("table.data[data-sortable]").forEach(makeSortable);

// --- Shared week-picker: populates a <select> from {week_order, weeks} and
// re-renders a content div on change. Used by both history.html (every
// graded week, all seasons - defaults to the most recent) and teams.html
// (the full 2026 season, past + upcoming - defaults to the current week,
// picked via pickDefault since "most recent" there would land on the
// season finale). ---
function buildWeekPicker(data, selectId, contentId, renderWeek, pickDefault) {
  const picker = document.getElementById(selectId);
  const container = document.getElementById(contentId);
  if (!picker || !container || !data) return;

  function render(weekKey) {
    const week = data.weeks[weekKey];
    container.innerHTML = week ? renderWeek(week) : "<p class='muted'>No data for this week.</p>";
  }

  data.week_order.forEach(wk => {
    const opt = document.createElement("option");
    opt.value = wk;
    opt.textContent = data.weeks[wk].label;
    picker.appendChild(opt);
  });
  picker.addEventListener("change", () => render(picker.value));
  if (data.week_order.length) {
    picker.value = pickDefault ? pickDefault(data) : data.week_order[data.week_order.length - 1];
    render(picker.value);
  }
}

// --- Week archive picker (history.html) - all seasons, graded only ---
function initHistoryPicker() {
  if (typeof HISTORY_DATA === "undefined") return;
  buildWeekPicker(HISTORY_DATA, "week-select", "week-content", (week) => {
    let rows = "";
    week.games.forEach(g => {
      rows += `<tr class="${g.model_correct === false ? "row-flag" : ""}">
        <td>${g.away_team} @ ${g.home_team}<div class="faint" style="font-size:11px;">Final: ${g.away_score}-${g.home_score}</div></td>
        <td class="num mono">${g.model_pick}</td>
        <td class="num mono">${g.model_correct ? "<span class='pill pill-positive'>HIT</span>" : "<span class='pill pill-danger'>MISS</span>"}</td>
        <td class="num mono">${g.vegas_pick}</td>
        <td class="num mono">${g.vegas_correct ? "<span class='pill pill-positive'>HIT</span>" : "<span class='pill pill-danger'>MISS</span>"}</td>
      </tr>`;
    });
    return `<table class="data">
        <thead><tr><th>Matchup</th><th class="num">Model Pick</th><th class="num">Result</th><th class="num">Vegas Pick</th><th class="num">Result</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  });
}
initHistoryPicker();

// --- Season week picker (teams.html) - 2026 only, past + upcoming ---
function initTeamsPicker() {
  if (typeof TEAMS_DATA === "undefined") return;
  buildWeekPicker(TEAMS_DATA, "teams-week-select", "teams-week-content", (week) => {
    let rows = "";
    week.games.forEach(g => {
      const border = `border-left:4px solid ${g.color};`;
      const matchup = `${g.away_team} @ ${g.home_team}`;
      const pick = `${g.favored_team} -${g.favored_by.toFixed(1)}`;
      const vegas = g.vegas_favored_team ? `${g.vegas_favored_team} -${g.vegas_favored_by.toFixed(1)}` : "&mdash;";
      let resultCell;
      if (g.graded) {
        const resultPill = g.correct ? "<span class='pill pill-positive'>HIT</span>" : "<span class='pill pill-danger'>MISS</span>";
        resultCell = `${g.away_score}-${g.home_score} ${resultPill}`;
      } else {
        resultCell = g.kickoff || "&mdash;";
      }
      rows += `<tr class="${g.graded && g.correct === false ? "row-flag" : ""}">
        <td style="${border}">${matchup}</td>
        <td class="num mono accent">${pick}</td>
        <td class="num mono market-color">${vegas}</td>
        <td class="num mono">${g.total.toFixed(1)}</td>
        <td class="num mono">${(g.win_pct * 100).toFixed(0)}%</td>
        <td class="num mono">${resultCell}</td>
      </tr>`;
    });
    return `<table class="data">
        <thead><tr><th>Matchup</th><th class="num">Model Pick</th><th class="num">Vegas</th><th class="num">Total</th><th class="num">Win%</th><th class="num">Result</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="table-footnote muted">The colored bar is the team the model favors. Result shows kickoff time for upcoming games, final score once graded.</div>`;
  }, (data) => {
    const currentWeek = data.week_order.find(wk => data.weeks[wk].games.some(g => !g.graded));
    return currentWeek || data.week_order[0];
  });
}
initTeamsPicker();

// --- In-page category tabs (players.html) ---
function initSubtabs() {
  document.querySelectorAll(".subtabs").forEach(bar => {
    bar.querySelectorAll(".subtab").forEach(btn => {
      btn.addEventListener("click", () => {
        bar.querySelectorAll(".subtab").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const targetId = btn.dataset.target;
        document.querySelectorAll(".cat-panel").forEach(panel => {
          panel.hidden = panel.id !== targetId;
        });
      });
    });
  });
}
initSubtabs();

// --- Trend charts (accuracy.html) ---
function initCharts() {
  if (typeof ACCURACY_DATA === "undefined" || typeof Chart === "undefined") return;
  Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  const labels = ACCURACY_DATA.labels;

  function lineChart(canvasId, title, usSeries, vegasSeries, formatFn) {
    const el = document.getElementById(canvasId);
    if (!el) return;
    new Chart(el, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Us", data: usSeries, borderColor: "#c2410c", backgroundColor: "#c2410c", pointRadius: 3, borderWidth: 2, tension: 0.15 },
          { label: "Vegas", data: vegasSeries, borderColor: "#0891b2", backgroundColor: "#0891b2", pointRadius: 3, borderWidth: 2, tension: 0.15 },
        ],
      },
      options: {
        plugins: {
          title: { display: true, text: title, font: { size: 13, weight: "bold" }, color: "#111827" },
          legend: { display: true, position: "top", labels: { color: "#111827", font: { size: 12 } } },
          tooltip: {
            callbacks: { label: (ctx) => `${ctx.dataset.label}: ${formatFn(ctx.parsed.y)}` },
          },
        },
        scales: {
          y: { ticks: { color: "#6b7280" }, grid: { color: "#e2e6ed" } },
          x: { ticks: { color: "#6b7280" }, grid: { display: false } },
        },
      },
    });
  }

  lineChart("chart-accuracy", "Straight-Up Pick Accuracy by Week", ACCURACY_DATA.us_accuracy, ACCURACY_DATA.vegas_accuracy, (v) => (v * 100).toFixed(0) + "%");
  lineChart("chart-spread-mae", "Spread Error by Week (points off actual margin, lower = sharper)", ACCURACY_DATA.us_spread_mae, ACCURACY_DATA.vegas_spread_mae, (v) => "±" + v.toFixed(1));
  lineChart("chart-brier", "Win Probability Calibration by Week (Brier score, lower = sharper)", ACCURACY_DATA.us_brier, ACCURACY_DATA.vegas_brier, (v) => v.toFixed(3));
}
initCharts();
