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

// --- Week archive picker (history.html) ---
function initWeekPicker() {
  const picker = document.getElementById("week-select");
  const container = document.getElementById("week-content");
  if (!picker || !container || typeof HISTORY_DATA === "undefined") return;

  function render(weekKey) {
    const week = HISTORY_DATA.weeks[weekKey];
    if (!week) { container.innerHTML = "<p class='muted'>No data for this week.</p>"; return; }
    let rows = "";
    week.games.forEach(g => {
      const flip = g.sharp_correct === false && g.vegas_correct === true;
      rows += `<tr class="${g.sharp_correct === false ? "row-flag" : ""}">
        <td>${g.away_team} @ ${g.home_team}<div class="faint" style="font-size:11px;">Final: ${g.away_score}-${g.home_score}</div></td>
        <td class="num mono">${g.sharp_pick}</td>
        <td class="num mono">${g.sharp_correct ? "<span class='pill pill-positive'>HIT</span>" : "<span class='pill pill-danger'>MISS</span>"}</td>
        <td class="num mono">${g.vegas_pick}</td>
        <td class="num mono">${g.vegas_correct ? "<span class='pill pill-positive'>HIT</span>" : "<span class='pill pill-danger'>MISS</span>"}</td>
      </tr>`;
    });
    container.innerHTML = `
      <table class="data">
        <thead><tr><th>Matchup</th><th class="num">Our Pick</th><th class="num">Result</th><th class="num">Vegas Pick</th><th class="num">Result</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  HISTORY_DATA.week_order.forEach(wk => {
    const opt = document.createElement("option");
    opt.value = wk;
    opt.textContent = HISTORY_DATA.weeks[wk].label;
    picker.appendChild(opt);
  });
  picker.addEventListener("change", () => render(picker.value));
  if (HISTORY_DATA.week_order.length) {
    picker.value = HISTORY_DATA.week_order[HISTORY_DATA.week_order.length - 1];
    render(picker.value);
  }
}
initWeekPicker();

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
