// NHL PowerScore - runs the trained game model in the browser from data.js (written by src/gamemodel/export.py)
// the prediction math mirrors src/gamemodel/predict.py
const D = window.NHL_DATA;
const M = D.model;
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
const fix = (x, d = 3) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
const signed = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const teams = Object.fromEntries(D.teams.map((t) => [t.team, t]));
const codes = D.teams.map((t) => t.team).sort();
const charts = {};
const seasonLabel = (s) => `${s}-${String(s + 1).slice(2)}`;

// ---------- the model ----------
function blendPower(raw, previous, games, k) {
  if (raw === null || raw === undefined) raw = previous;
  return k > 0 ? (games * raw + k * previous) / (games + k) : raw;
}
// the PowerScore the model uses: this season's once games are played, blended with (or, before any games, equal to) last season's
const currentPower = (t) => blendPower(t.powerRaw, t.powerPrev, t.gamesPlayed, M.k);
function teamInputs(code, goalieId, rest, out) {
  const t = teams[code];
  const goalies = D.goalies[code] || [];
  const goalie = goalies.find((g) => g.goalieId === goalieId) || goalies[0];
  // projected lineup: the top 12 forwards and 6 defensemen on the current roster who are not out (next man up fills in),
  // compared with the team's usual lineup, so offseason additions and losses count too
  const roster = D.players[code] || [];
  const available = roster.filter((p) => !out.has(p.name));
  const dressed = [...available.filter((p) => p.group === "F").slice(0, 12), ...available.filter((p) => p.group === "D").slice(0, 6)];
  const lineupPower = dressed.reduce((s, p) => s + p.rating, 0);
  return {
    t, goalie,
    power: blendPower(t.powerRaw, t.powerPrev, t.gamesPlayed, M.k),
    backToBack: rest === 1 ? 1 : 0,
    goalieRating: goalie ? goalie.goalieRating : 0,
    missing: roster.filter((p) => out.has(p.name) && p.typical).reduce((s, p) => s + Math.max(0, p.rating), 0),
    lineupDelta: lineupPower - t.lineupTypical
  };
}
const FORM_PREFIX = { 5: "form5_", 10: "form_", 20: "form20_" };
function featureRow(a, b, home) {
  // features for team a scoring against team b (same as goalFeatures in train.py)
  const league = a.t.leagueGoals;
  const log = (x) => Math.log(x ?? league);
  const f = FORM_PREFIX[M.form || 10];
  const x = {
    power: a.power, oppPower: b.power, elo: a.t.elo, oppElo: b.t.elo,
    logXGF: log(a.t[f + "xGoalsFor"]), logXGA: log(a.t[f + "xGoalsAgainst"]), logGF: log(a.t[f + "goalsFor"]),
    logOppXGF: log(b.t[f + "xGoalsFor"]), logOppXGA: log(b.t[f + "xGoalsAgainst"]), logOppGA: log(b.t[f + "goalsAgainst"]),
    goalie: a.goalieRating, oppGoalie: b.goalieRating, home,
    backToBack: a.backToBack, oppBackToBack: b.backToBack, logLeague: Math.log(league),
    missing: a.missing, oppMissing: b.missing, lineupDelta: a.lineupDelta, oppLineupDelta: b.lineupDelta
  };
  if (Array.isArray(M.eloFit)) {
    // Elo only keeps the part PowerScore does not explain
    const [intercept, slope] = M.eloFit;
    x.elo -= intercept + slope * x.power;
    x.oppElo -= intercept + slope * x.oppPower;
  }
  const extras = new Set(M.extras || []);
  for (const [side, s] of [["", a], ["opp", b]]) {
    const name = (n) => (side ? side + n[0].toUpperCase() + n.slice(1) : n);
    const t = s.t;
    if (extras.has("adj")) { x[name("logAdjXGF")] = log(t.form_adj_xGF); x[name("logAdjXGA")] = log(t.form_adj_xGA); }
    if (extras.has("ev")) {
      const per60 = (v) => Math.log(Math.max(0.5, t.form_ev_ice > 0 ? v / t.form_ev_ice * 3600 : 2.4));
      x[name("logEvXGF60")] = per60(t.form_ev_xGF); x[name("logEvXGA60")] = per60(t.form_ev_xGA);
    }
    if (extras.has("st")) { x[name("ppXG")] = t.form_pp_xGF ?? 0.5; x[name("pkXG")] = t.form_pk_xGA ?? 0.5; }
    if (extras.has("travel")) { x[name("roadTrip")] = s.roadTrip; x[name("tzShift")] = s.tzShift; x[name("tzFromHome")] = s.tzFromHome; }
  }
  return x;
}
// confidence tiers (same cut points as train.py)
const TIERS = [["Toss-up", 0.5], ["Lean", 0.55], ["Solid", 0.6], ["Strong", 0.65]];
const tierOf = (p) => TIERS.filter(([, cut]) => Math.max(p, 1 - p) >= cut).pop()[0];
const tierStats = (name) => (D.report.tiers || []).find((t) => t.tier === name);
const linear = (part, x) => part.features.reduce((s, f, j) => s + part.coef[j] * (x[f] - part.mean[j]) / part.scale[j], part.intercept);
function poissonPmf(lambda, n) {
  const out = [Math.exp(-lambda)];
  for (let k = 1; k <= n; k++) out.push(out[k - 1] * lambda / k);
  return out;
}
function predict(opts) {
  const h = teamInputs(opts.home, opts.homeGoalie, opts.homeRest, opts.homeOut);
  const a = teamInputs(opts.away, opts.awayGoalie, opts.awayRest, opts.awayOut);
  // travel: the away team is assumed to come straight from home, starting a road trip
  const zones = Math.abs(M.timeZones[opts.home] - M.timeZones[opts.away]);
  Object.assign(h, { roadTrip: 0, tzShift: 0, tzFromHome: 0 });
  Object.assign(a, { roadTrip: 1, tzShift: zones, tzFromHome: zones });
  const hx = featureRow(h, a, 1), ax = featureRow(a, h, 0);
  const homeRate = Math.exp(linear(M.goal, hx)), awayRate = Math.exp(linear(M.goal, ax));
  const pLogistic = 1 / (1 + Math.exp(-linear(M.win, hx)));
  const ph = poissonPmf(homeRate, 12), pa = poissonPmf(awayRate, 12);
  let ahead = 0, tie = 0;
  ph.forEach((p, i) => pa.forEach((q, j) => { if (i > j) ahead += p * q; else if (i === j) tie += p * q; }));
  const pGoals = ahead + 0.5 * tie;
  const homeWin = M.blend * pLogistic + (1 - M.blend) * pGoals;
  // factor contributions vs an average game with both teams rested and nobody out
  const neutral = new Set(["backToBack", "oppBackToBack", "missing", "oppMissing", "lineupDelta", "oppLineupDelta",
    "roadTrip", "tzShift", "tzFromHome", "oppRoadTrip", "oppTzShift", "oppTzFromHome"]);
  const base = Object.fromEntries(M.win.features.map((f, j) => [f, neutral.has(f) ? 0 : M.win.mean[j]]));
  const contrib = Object.fromEntries(M.win.features.map((f, j) => [f, M.win.coef[j] * (hx[f] - base[f]) / M.win.scale[j]]));
  const groups = {
    "PowerScore": ["power", "oppPower"], "Elo (beyond PowerScore)": ["elo", "oppElo"],
    "Recent form": ["logXGF", "logXGA", "logGF", "logOppXGF", "logOppXGA", "logOppGA"],
    "Goaltending": ["goalie", "oppGoalie"], "Rest": ["backToBack", "oppBackToBack"],
    "Missing players": ["missing", "oppMissing", "lineupDelta", "oppLineupDelta"],
    "Shot quality (5v5 / adjusted)": ["logAdjXGF", "logAdjXGA", "oppLogAdjXGF", "oppLogAdjXGA", "logEvXGF60", "logEvXGA60", "oppLogEvXGF60", "oppLogEvXGA60"],
    "Special teams": ["ppXG", "pkXG", "oppPpXG", "oppPkXG"],
    "Travel": ["roadTrip", "tzShift", "tzFromHome", "oppRoadTrip", "oppTzShift", "oppTzFromHome"],
    "Scoring environment": ["logLeague"]
  };
  const factors = Object.entries(groups).filter(([, cols]) => cols.some((c) => c in contrib))
    .map(([factor, cols]) => ({ factor, value: cols.reduce((s, c) => s + (contrib[c] || 0), 0) }));
  factors.push({ factor: "Home ice", value: linear(M.win, base) });
  const grid = ph.slice(0, 9).map((p) => pa.slice(0, 9).map((q) => p * q));
  return { homeWin, awayWin: 1 - homeWin, homeRate, awayRate, shootout: tie, grid, factors, homeGoalie: h.goalie?.name, awayGoalie: a.goalie?.name };
}
window.predictGame = predict;

// ---------- helpers ----------
function table(el, columns, rows) {
  el.innerHTML = "<thead><tr>" + columns.map((c) => `<th class="${c.num ? "num" : ""}">${c.label}</th>`).join("") + "</tr></thead><tbody>" +
    rows.map((r, i) => "<tr>" + columns.map((c) => `<td class="${c.num ? "num" : ""}">${c.value(r, i)}</td>`).join("") + "</tr>").join("") + "</tbody>";
}
const ratingPill = (r, d = 2, cut = 0.05) => `<span class="pill ${r > cut ? "good" : r < -cut ? "bad" : "flat"}">${signed(r, d)}</span>`;
function chartTheme() {
  Chart.defaults.color = css("--ink-2");
  Chart.defaults.borderColor = css("--line");
  Chart.defaults.font.family = css("--body");
}

// ---------- tabs ----------
const tabs = document.querySelectorAll("nav.tabs button");
function showTab(name) {
  if (![...tabs].some((t) => t.dataset.tab === name)) name = "predict";
  tabs.forEach((t) => t.setAttribute("aria-selected", t.dataset.tab === name));
  document.querySelectorAll("section.panel").forEach((p) => (p.hidden = p.id !== name));
  if (name === "teams" && !charts.trend) renderTeam();
  if (name === "model" && !charts.calib) renderModel();
  if (name === "backtest" && !$("btTable").innerHTML) renderBacktest();
  try { history.replaceState(null, "", "#" + name); } catch (e) {}
}
tabs.forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));

// ---------- predict ----------
const state = { homeOut: new Set(), awayOut: new Set() };
function fillSide(side) {
  const code = $(side).value;
  $(side + "Goalie").innerHTML = (D.goalies[code] || []).map((g) =>
    `<option value="${g.goalieId}">${esc(g.name)} · ${g.starts} GS · ${signed(g.goalieRating)}</option>`).join("");
  state[side + "Out"] = new Set();
  const players = (D.players[code] || []).filter((p) => p.typical);  // forwards first, then defense, in depth order
  $(side + "Roster").innerHTML = players.map((p) =>
    `<button type="button" class="chip" aria-pressed="false" data-name="${esc(p.name)}" title="Rating ${p.rating.toFixed(1)} GAR per 82 games">${esc(p.name)}<b>${p.rating.toFixed(1)}</b></button>`).join("");
  $(side + "Roster").querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
    const out = state[side + "Out"], name = chip.dataset.name;
    out.has(name) ? out.delete(name) : out.add(name);
    chip.setAttribute("aria-pressed", out.has(name));
    runPrediction();
  }));
}
function runPrediction() {
  const home = $("home").value, away = $("away").value;
  const r = predict({
    home, away, homeGoalie: Number($("homeGoalie").value), awayGoalie: Number($("awayGoalie").value),
    homeRest: Number($("homeRest").value), awayRest: Number($("awayRest").value), homeOut: state.homeOut, awayOut: state.awayOut
  });
  $("homeName").textContent = home; $("awayName").textContent = away;
  $("homePct").textContent = pct(r.homeWin); $("awayPct").textContent = pct(r.awayWin);
  $("homeBar").style.width = pct(r.homeWin, 2); $("awayBar").style.width = pct(r.awayWin, 2);
  $("probbar").setAttribute("aria-label", `${away} ${pct(r.awayWin)}, ${home} ${pct(r.homeWin)}`);
  $("xScore").textContent = `${away} ${r.awayRate.toFixed(2)} – ${r.homeRate.toFixed(2)} ${home}`;
  let best = { p: 0 };
  r.grid.forEach((row, h) => row.forEach((p, a) => { if (p > best.p) best = { p, h, a }; }));
  $("topScore").textContent = `${home} ${best.h}–${best.a} · ${pct(best.p, 1)}`;
  $("soChance").textContent = pct(r.shootout, 1);
  const tier = tierOf(r.homeWin), stats = tierStats(tier), favorite = r.homeWin >= 0.5 ? home : away;
  const level = TIERS.findIndex(([name]) => name === tier);
  $("tierBadge").innerHTML = `<span class="tag t${level}">${tier}</span> <span>Pick: <b>${favorite}</b>${stats ?
    ` · in the backtest, ${tier.toLowerCase()} picks were right <b>${pct(stats.all.accuracy, 1)}</b> of the time (${stats.all.games.toLocaleString()} games since 2017-18)` : ""}</span>`;
  $("starters").innerHTML = `${away}: ${esc(r.awayGoalie)}<br>${home}: ${esc(r.homeGoalie)}`;
  drawFactors(r, home, away);
  drawGrid(r, home, away);
}
function drawFactors(r, home, away) {
  chartTheme();
  const values = r.factors.map((f) => f.value);
  const colors = values.map((v) => (v >= 0 ? css("--home") : css("--away")));
  if (charts.factors) {
    Object.assign(charts.factors.data.datasets[0], { data: values, backgroundColor: colors });
    charts.factors.options.scales.x.title.text = `← favors ${away}     favors ${home} →`;
    charts.factors.$teams = [home, away];
    return charts.factors.update();
  }
  charts.factors = new Chart($("factorChart"), {
    type: "bar",
    data: { labels: r.factors.map((f) => f.factor), datasets: [{ data: values, backgroundColor: colors, borderRadius: 4, barThickness: 14 }] },
    options: {
      indexAxis: "y", maintainAspectRatio: false, animation: { duration: 250 },
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => {
        const [h, a] = charts.factors.$teams; return `${c.raw >= 0 ? "Favors " + h : "Favors " + a}: ${signed(c.raw, 3)} log-odds`; } } } },
      scales: { x: { title: { display: true, text: `← favors ${away}     favors ${home} →` } }, y: { grid: { display: false } } }
    }
  });
  charts.factors.$teams = [home, away];
}
function drawGrid(r, home, away) {
  const n = 8, max = Math.max(...r.grid.slice(0, n).flatMap((row) => row.slice(0, n))), heat = css("--heat");
  let html = `<thead><tr><th>${home} ↓ ${away} →</th>` + [...Array(n).keys()].map((a) => `<th>${a}</th>`).join("") + "</tr></thead><tbody>";
  for (let h = 0; h < n; h++) {
    html += `<tr><th>${h}</th>`;
    for (let a = 0; a < n; a++) {
      const p = r.grid[h][a], alpha = 0.05 + 0.85 * (p / max);
      html += `<td title="${home} ${h}–${a} ${away}: ${pct(p, 1)}" style="background:rgba(${heat},${alpha.toFixed(3)});${alpha > 0.55 ? "color:#fff" : ""}">${p >= 0.005 ? pct(p, 1) : ""}</td>`;
    }
    html += "</tr>";
  }
  $("scoreGrid").innerHTML = html + "</tbody>";
}

// ---------- rankings ----------
const rankState = { view: "teams", pos: "", limit: 50 };
function sparkline(values, lo, hi) {
  // season PowerScore trend on a shared scale so teams are comparable (first 5 games skipped, too noisy)
  const v = values.slice(5).filter((x) => x !== null);
  if (v.length < 2) return "";
  const w = 96, h = 26, clamp = (x) => Math.min(hi, Math.max(lo, x));
  const pts = v.map((x, i) => [(i / (v.length - 1)) * (w - 4) + 2, h - 3 - ((clamp(x) - lo) / (hi - lo)) * (h - 6)]);
  const [ex, ey] = pts[pts.length - 1];
  return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true"><polyline points="${pts.map((p) => p.map((n) => n.toFixed(1)).join(",")).join(" ")}"/><circle cx="${ex.toFixed(1)}" cy="${ey.toFixed(1)}" r="2.4"/></svg>`;
}
function renderRankings() {
  const rows = [...D.teams].sort((a, b) => currentPower(b) - currentPower(a));
  const max = currentPower(rows[0]), min = currentPower(rows[rows.length - 1]);
  // rank 10 games ago, from each team's PowerScore going into its 10th-to-last game
  const earlier = D.teams.map((t) => { const tr = D.trends[t.team] || []; return [t.team, tr.length > 10 ? tr[tr.length - 10][1] : currentPower(t)]; })
    .sort((a, b) => b[1] - a[1]).map(([code]) => code);
  table($("teamTable"), [
    { label: "Rank", value: (t, i) => { const d = D.preseason ? 0 : earlier.indexOf(t.team) - i;
      return `<span class="rkn">${i + 1}</span><span class="mv ${d > 0 ? "up" : d < 0 ? "down" : "same"}">${d > 0 ? "▲ " + d : d < 0 ? "▼ " + -d : "–"}</span>`; } },
    { label: "Team", value: (t) => `<div class="who"><b>${t.team}</b><span>${t.record.join("-")} ${D.preseason ? seasonLabel(D.dataSeason) : "W-L-SO"}</span></div>` },
    { label: "PowerScore", value: (t) => `<div class="score"><strong>${fix(currentPower(t))}</strong><span class="track"><i style="width:${(8 + 92 * (currentPower(t) - min) / (max - min)).toFixed(1)}%"></i></span></div>` },
    { label: "Season trend", value: (t) => sparkline((D.trends[t.team] || []).map((x) => x[1]), 0.3, 0.9) },
    { label: "Elo", num: true, value: (t) => `<span class="stat2">${fix(t.elo, 0)}</span>` },
    { label: "Recent xG%", num: true, value: (t) => `<span class="stat2">${pct(t.form_xGoalsFor / (t.form_xGoalsFor + t.form_xGoalsAgainst), 1)}</span>` },
    { label: "Likely starter", value: (t) => { const g = (D.goalies[t.team] || [])[0];
      return g ? `<div class="who"><span style="color:var(--ink);font-size:14px">${esc(g.name)}</span><span>${signed(g.goalieRating)} GSAx/60</span></div>` : "–"; } }
  ], rows);
  $("teamTable").querySelectorAll("tbody tr").forEach((tr, i) => {
    tr.classList.add("link");
    tr.title = `Open ${rows[i].team}`;
    tr.addEventListener("click", () => { $("teamSelect").value = rows[i].team; showTab("teams"); renderTeam(); });
  });
  renderPlayers();
  setRankView(rankState.view);
}
function setRankView(view) {
  rankState.view = view;
  $("segTeams").setAttribute("aria-pressed", view === "teams");
  $("segPlayers").setAttribute("aria-pressed", view === "players");
  $("teamRank").hidden = view !== "teams";
  $("playerRank").hidden = view !== "players";
  $("rankTitle").textContent = view === "teams" ? "Team power rankings" : "Player power rankings";
  $("rankNote").textContent = view === "teams"
    ? (D.preseason ? `Preseason: PowerScore is each team's final ${seasonLabel(D.dataSeason)} PowerScore (your team model's predicted points %) until games are played. Click a team for details.`
      : `PowerScore is your team model's predicted points % from ${seasonLabel(D.season)} stats, blended with last season's early on. Arrows show movement over the last 10 games. Click a team for details.`)
    : `Rating is your player model (predicted GAR per 82 games) on each player's recent stats, the value the game model uses. Season PS is your ${seasonLabel(D.rankingsSeason)} PowerScore ranking.`;
}
$("segTeams").addEventListener("click", () => setRankView("teams"));
$("segPlayers").addEventListener("click", () => setRankView("players"));
function renderPlayers() {
  const team = $("playerTeam").value, query = $("playerSearch").value.trim().toLowerCase();
  let players = Object.entries(D.players).flatMap(([code, list]) => list.map((p) => ({ ...p, team: code })))
    .sort((a, b) => b.rating - a.rating).map((p, i) => ({ ...p, overall: i + 1 }));
  if (team) players = players.filter((p) => p.team === team);
  if (rankState.pos === "D") players = players.filter((p) => p.position === "D");
  if (rankState.pos === "F") players = players.filter((p) => p.position !== "D");
  if (query) players = players.filter((p) => p.name.toLowerCase().includes(query));
  const shown = players.slice(0, rankState.limit);
  const top = Math.max(...Object.values(D.players).flat().map((p) => p.rating));
  const positions = { C: "Center", L: "Left wing", R: "Right wing", D: "Defense" };
  table($("playerTable"), [
    { label: "Rank", value: (p) => `<span class="rkn">${p.overall}</span>` },
    { label: "Player", value: (p) => `<div class="who"><span class="player">${esc(p.name)}</span><span>${p.team} · ${positions[p.position] || p.position}</span></div>` },
    { label: "Rating (GAR / 82)", value: (p) => `<div class="score"><strong>${p.rating.toFixed(1)}</strong><span class="track"><i class="${p.rating < 0 ? "neg" : ""}" style="width:${Math.min(100, Math.abs(p.rating) / top * 100).toFixed(1)}%"></i></span></div>` },
    { label: "Season PS", num: true, value: (p) => `<span class="stat2">${fix(p.seasonScore, 1)}</span>` },
    { label: "Last 10", num: true, value: (p) => `<span class="stat2">${p.recentGames} GP</span>` }
  ], shown);
  if (!shown.length) $("playerTable").innerHTML = `<tbody><tr><td class="note">No players match. Try a different name, team or position.</td></tr></tbody>`;
  $("playerMore").hidden = players.length <= rankState.limit;
  $("playerMore").textContent = `Show more (${players.length - shown.length} left)`;
}
$("playerTeam").addEventListener("change", () => { rankState.limit = 50; renderPlayers(); });
$("playerSearch").addEventListener("input", () => { rankState.limit = 50; renderPlayers(); });
$("playerMore").addEventListener("click", () => { rankState.limit += 50; renderPlayers(); });
document.querySelectorAll("#playerRank .seg button").forEach((b) => b.addEventListener("click", () => {
  rankState.pos = b.dataset.pos; rankState.limit = 50;
  document.querySelectorAll("#playerRank .seg button").forEach((x) => x.setAttribute("aria-pressed", x === b));
  renderPlayers();
}));

// ---------- teams ----------
function renderTeam() {
  const code = $("teamSelect").value;
  $("trendTitle").textContent = `${code} through ${seasonLabel(D.dataSeason)}`;
  const trend = D.trends[code] || [];
  chartTheme();
  charts.trend?.destroy();
  charts.trend = new Chart($("trendChart"), {
    type: "line",
    data: { labels: trend.map((t) => t[0]), datasets: [{ label: "PowerScore", data: trend.map((t) => t[1]), borderColor: css("--home"), borderWidth: 2, pointRadius: 0, pointHoverRadius: 5, tension: 0.25, spanGaps: true }] },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false }, tooltip: { callbacks: { title: (c) => `${c[0].label} · game ${c[0].dataIndex + 1}`, label: (c) => `PowerScore ${c.raw === null ? "–" : c.raw.toFixed(3)}` } } },
      scales: { x: { ticks: { maxTicksLimit: 7 }, grid: { display: false } }, y: { title: { display: true, text: "PowerScore (predicted points %)" } } }
    }
  });
  table($("goalieTable"), [
    { label: "Goalie", value: (g) => esc(g.name) },
    { label: "Starts", num: true, value: (g) => g.starts },
    { label: "Last 10", num: true, value: (g) => g.recentStarts },
    { label: "Rating", num: true, value: (g) => ratingPill(g.goalieRating) }
  ], D.goalies[code] || []);
  table($("recentTable"), [
    { label: "Date", value: (g) => g[0] },
    { label: "Opp", value: (g) => (g[2] ? "vs " : "@ ") + g[1] },
    { label: "Result", value: (g) => { const r = g[3] > g[4] ? "W" : g[3] < g[4] ? "L" : "SO"; return `<span class="pill ${r === "W" ? "good" : r === "L" ? "bad" : "flat"}">${r}</span> ${g[3]}-${g[4]}`; } },
    { label: "xG", num: true, value: (g) => `${g[5].toFixed(1)}–${g[6].toFixed(1)}` },
    { label: "Goalie", value: (g) => esc(g[7] ?? "–") }
  ], D.recent[code] || []);
  table($("skaterTable"), [
    { label: "Player", value: (p) => esc(p.name) },
    { label: "Pos", value: (p) => p.position },
    { label: "Lineup", value: (p) => (p.typical ? '<span class="pill good">typical</span>' : '<span class="pill flat">depth</span>') },
    { label: "Games of last 10", num: true, value: (p) => p.recentGames },
    { label: "Current rating", num: true, value: (p) => fix(p.rating, 1) },
    { label: "Season PowerScore", num: true, value: (p) => fix(p.seasonScore, 1) }
  ], D.players[code] || []);
}
$("teamSelect").addEventListener("change", renderTeam);

// ---------- model report ----------
const DEFAULTS = { k: 10, elo: "residual", blend: 0.5, alpha: 1e-4, C: 1.0, goalModel: "glm", players: null, form: 10, extras: [] };
function describe(params) {
  const p = { ...DEFAULTS, ...params }, parts = [];
  if (p.goalModel === "hgb") parts.push("gradient boosting for goals");
  if (p.elo === "raw") parts.push("raw Elo (not residualized)");
  if (p.elo === "none") parts.push("no Elo");
  if (p.k !== 10) parts.push(`prior-season weight ${p.k} games`);
  if (p.blend === 1) parts.push("logistic win model only");
  if (p.blend === 0) parts.push("goal model win chance only");
  if (p.C !== 1.0) parts.push("stronger regularization");
  if (p.players === "missing") parts.push("+ missing players");
  if (p.players === "lineup") parts.push("+ lineup strength");
  if (p.players === "both") parts.push("+ missing players + lineup strength");
  if (p.form !== 10) parts.push(`form half-life ${p.form} games`);
  const names = { adj: "score-adjusted xG", ev: "5-on-5 xG", st: "special teams", travel: "travel" };
  for (const e of p.extras || []) parts.push("+ " + names[e]);
  return parts.length ? parts.join(", ") : "Base model";
}
function renderModel() {
  const r = D.report, bestKey = JSON.stringify(r.best);
  const best = r.candidates.find((c) => JSON.stringify(c.params) === bestKey);
  const elo = r.baselines.find((b) => b.model === "Elo only");
  $("headline").innerHTML = [["Log loss", best.logLoss.toFixed(4)], ["Winner picked", pct(best.accuracy, 1)], ["Brier score", best.brier.toFixed(4)], ["Goals off by (avg)", best.goalMAE.toFixed(2)]]
    .map(([k, v]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  const rows = [...r.baselines.map((b) => ({ ...b, name: b.model, tag: "baseline" })),
    ...r.candidates.map((c) => ({ ...c, name: describe(c.params), tag: JSON.stringify(c.params) === bestKey ? "chosen" : "" }))];
  table($("compareTable"), [
    { label: "Version", value: (m) => `${esc(m.name)} ${m.tag === "chosen" ? '<span class="pill good">chosen</span>' : m.tag ? '<span class="pill flat">baseline</span>' : ""}` },
    { label: "Log loss", num: true, value: (m) => m.logLoss.toFixed(4) },
    { label: "Winner picked", num: true, value: (m) => pct(m.accuracy, 1) },
    { label: "Brier", num: true, value: (m) => m.brier.toFixed(4) },
    { label: "Better than Elo by", num: true, value: (m) => { const d = elo.logLoss - m.logLoss; return `<span class="pill ${d > 0.0001 ? "good" : d < -0.0001 ? "bad" : "flat"}">${signed(d, 4)}</span>`; } }
  ], rows);
  table($("tierTable"), [
    { label: "Tier", value: (t) => `<span class="tag t${TIERS.findIndex(([n]) => n === t.tier)}">${t.tier}</span>` },
    { label: "Favorite's win chance", value: (t) => { const i = TIERS.findIndex(([n]) => n === t.tier), next = TIERS[i + 1]; return next ? `${pct(t.minConfidence)} – ${pct(next[1])}` : `${pct(t.minConfidence)}+`; } },
    { label: "All seasons: accuracy", num: true, value: (t) => `<b>${pct(t.all.accuracy, 1)}</b>` },
    { label: "Share of games", num: true, value: (t) => pct(t.all.share) },
    { label: `${seasonLabel(D.backtestSeason)}: accuracy`, num: true, value: (t) => pct(t.latest.accuracy, 1) },
    { label: "Share", num: true, value: (t) => pct(t.latest.share) }
  ], [...(r.tiers || [])].reverse());
  table($("seasonTable"), [
    { label: "Season", value: (s) => seasonLabel(s.season) },
    { label: "Log loss", num: true, value: (s) => s.logLoss.toFixed(4) },
    { label: "Winner picked", num: true, value: (s) => pct(s.accuracy, 1) },
    { label: "Goals off by", num: true, value: (s) => s.goalMAE.toFixed(2) }
  ], r.bestBySeason);
  const bins = [];
  for (let lo = 0.2; lo < 0.8; lo += 0.1) {
    const g = D.backtest.filter((x) => x[7] >= lo && x[7] < lo + 0.1 && x[3] !== x[4]);
    if (g.length >= 15) bins.push({ x: g.reduce((s, x) => s + x[7], 0) / g.length, y: g.filter((x) => x[3] > x[4]).length / g.length, n: g.length });
  }
  chartTheme();
  charts.calib = new Chart($("calibChart"), {
    type: "scatter",
    data: { datasets: [
      { label: "Perfect calibration", data: [{ x: 0.2, y: 0.2 }, { x: 0.8, y: 0.8 }], type: "line", borderColor: css("--muted"), borderDash: [4, 4], borderWidth: 1, pointRadius: 0 },
      { label: "Model", data: bins, borderColor: css("--home"), backgroundColor: css("--home"), pointRadius: 6, pointHoverRadius: 8, showLine: true, borderWidth: 2 }
    ] },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" }, tooltip: { filter: (c) => c.datasetIndex === 1, callbacks: { label: (c) => `Predicted ${pct(c.raw.x)} → won ${pct(c.raw.y)} (${c.raw.n} games)` } } },
      scales: { x: { min: 0.2, max: 0.8, title: { display: true, text: "Predicted home win chance" }, ticks: { callback: (v) => pct(v) } },
                y: { min: 0.1, max: 0.9, title: { display: true, text: "Home team actually won" }, ticks: { callback: (v) => pct(v) } } }
    }
  });
  const coefs = r.winCoefficients;
  const how = [
    ["Goals", "each team's goals follow a Poisson regression on both teams' pregame features. That gives expected goals and the score grid."],
    ["Win chance", `a logistic regression on the same features, averaged ${Math.round(M.blend * 100)}/${Math.round((1 - M.blend) * 100)} with the win chance implied by the goal model.`],
    ["PowerScore", `your team PowerScore model applied to season-to-date stats. Early in a season it blends with last season's final PowerScore at the weight of ${M.k} games.`],
    ["Elo", "a margin-of-victory Elo rating. It only counts for the part PowerScore doesn't already explain (the two are 93% correlated)."],
    ["Recent form", "exponentially weighted xG and goals for and against (half-life 10 games)."],
    ["Goaltending", "the starter's recency-weighted goals saved above expected per 60, shrunk toward average for goalies with few games."],
    ["Players", "every skater is rated going into every game by your player PowerScore model (predicted GAR per 82 games) applied to his recency-weighted career stats. " +
      (M.players ? `The model uses ${M.players === "both" ? "missing regulars and tonight's lineup vs the team's usual lineup" : M.players === "missing" ? "the rating of regulars who sit out" : "tonight's lineup vs the team's usual lineup"}.` :
        "Missing-player features did not improve the backtest, so the chosen model leaves them out.")],
    ["Schedule", "back-to-back flags for both teams. Home ice is the intercept, and league scoring level adjusts for era."]
  ];
  $("howList").innerHTML = how.map(([k, v]) => `<li><b>${k}:</b> ${v}</li>`).join("");
}

// ---------- backtest ----------
function renderBacktest() {
  const team = $("btTeam").value, tierFilter = $("btTier").value;
  const teamGames = D.backtest.filter((g) => !team || g[1] === team || g[2] === team);
  const games = teamGames.filter((g) => !tierFilter || tierOf(g[7]) === tierFilter).slice().reverse();
  const record = (list) => { let n = 0, c = 0; list.forEach((g) => { if (g[3] !== g[4]) { n++; if ((g[7] >= 0.5) === (g[3] > g[4])) c++; } }); return [c, n]; };
  const [correct, decided] = record(games);
  const byTier = TIERS.slice().reverse().map(([name]) => { const [c, n] = record(teamGames.filter((g) => tierOf(g[7]) === name)); return n ? `${name} ${pct(c / n, 1)} (${n})` : null; }).filter(Boolean).join(" · ");
  $("btSummary").innerHTML = `${games.length} games · winner picked in ${correct} of ${decided} decided before a shootout (<b>${pct(correct / Math.max(decided, 1), 1)}</b>). By tier: ${byTier}. Every game was predicted by a model trained only on seasons before ${seasonLabel(D.backtestSeason)}.`;
  table($("btTable"), [
    { label: "Date", value: (g) => g[0] },
    { label: "Game", value: (g) => `${g[2]} @ ${g[1]}` },
    { label: "Home win %", num: true, value: (g) => pct(g[7]) },
    { label: "Expected", num: true, value: (g) => `${g[6].toFixed(1)}–${g[5].toFixed(1)}` },
    { label: "Final", num: true, value: (g) => `${g[4]}–${g[3]}${g[3] === g[4] ? " SO" : ""}` },
    { label: "Pick", value: (g) => (g[7] >= 0.5 ? g[1] : g[2]) },
    { label: "Tier", value: (g) => { const t = tierOf(g[7]); return `<span class="tag t${TIERS.findIndex(([n]) => n === t)}" style="font-size:11px;padding:3px 6px">${t}</span>`; } },
    { label: "", value: (g) => (g[3] === g[4] ? '<span class="pill flat">SO</span>' : (g[7] >= 0.5) === (g[3] > g[4]) ? '<span class="pill good">✓</span>' : '<span class="pill bad">✗</span>') }
  ], games);
}
$("btTeam").addEventListener("change", renderBacktest);
$("btTier").addEventListener("change", renderBacktest);

// ---------- start ----------
$("asOf").textContent = D.preseason ? `${seasonLabel(D.season)} preseason · ratings through ${D.asOf}` : `${seasonLabel(D.season)} · through ${D.asOf}`;
// labels that depend on which season the backtest and the rankings cover
document.querySelectorAll("[data-backtest-season]").forEach((el) => (el.textContent = el.dataset.backtestSeason.replace("{s}", seasonLabel(D.backtestSeason))));
if (D.preseason) $("preseasonNote").hidden = false;
const teamOptions = (selected) => codes.map((c) => `<option ${c === selected ? "selected" : ""}>${c}</option>`).join("");
const byPower = [...D.teams].sort((a, b) => currentPower(b) - currentPower(a));
$("home").innerHTML = teamOptions(byPower[0].team);
$("away").innerHTML = teamOptions(byPower[1].team);
$("teamSelect").innerHTML = teamOptions(byPower[0].team);
$("playerTeam").innerHTML += teamOptions(null);
$("btTeam").innerHTML += teamOptions(null);
for (const side of ["home", "away"]) {
  fillSide(side);
  $(side).addEventListener("change", () => { fillSide(side); runPrediction(); });
  for (const id of [side + "Goalie", side + "Rest"]) $(id).addEventListener("change", runPrediction);
}
renderRankings();
runPrediction();
showTab((location.hash || "#predict").slice(1));
