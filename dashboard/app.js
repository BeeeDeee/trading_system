// Arena 5m dashboard: čte jen data/*.json (snapshot), nikdy DB (SPEC §15, nástraha 13). Vanilla JS, vlastní SVG grafy.
(function () {
  "use strict";
  var SNAP = "data/snapshot.json", STALE_MS = 60000, POLL_MS = 7000, BAR_MS = 300000;
  var SYMCOL = { BTCUSDT: "var(--s1)", ETHUSDT: "var(--s2)", SOLUSDT: "var(--s3)", XRPUSDT: "var(--s4)", DOGEUSDT: "var(--s5)" };
  var TABS = ["overview", "board", "detail", "costs", "system"];
  var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  var state = { snap: null, tab: "overview", variant: null, vdata: null, sort: { key: "net_mean", dir: -1 }, seenTrades: null, prices: {} };

  // --- pomocné -----------------------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function fin(x) { return x != null && isFinite(x); }
  function num(x, d) { return fin(x) ? Number(x).toFixed(d == null ? 1 : d) : "—"; }
  function sgn(x, d) { if (!fin(x)) return "—"; var s = Number(x).toFixed(d == null ? 1 : d); return (x > 0 ? "+" : "") + s; }
  function cls(x) { return !fin(x) ? "" : (x > 0 ? "pos" : (x < 0 ? "neg" : "")); }
  function arrow(x) { return !fin(x) || x === 0 ? "" : (x > 0 ? "▲ " : "▼ "); }
  function ts(ms) { return ms ? new Date(ms).toISOString().replace("T", " ").slice(0, 19) : "—"; }
  function hm(ms) { return ms ? new Date(ms).toISOString().slice(11, 16) : "—"; }
  function ago(ms) {
    var s = Math.max(0, (Date.now() - ms) / 1000);
    if (s < 60) return Math.round(s) + " s"; if (s < 3600) return Math.round(s / 60) + " min";
    if (s < 86400) return (s / 3600).toFixed(1) + " h"; return (s / 86400).toFixed(1) + " d";
  }
  function money(x, d) { return fin(x) ? Number(x).toLocaleString("cs-CZ", { minimumFractionDigits: d, maximumFractionDigits: d }) : "—"; }
  function price(x) { if (!fin(x)) return "—"; var a = Math.abs(x); return Number(x).toLocaleString("en-US", { minimumFractionDigits: a >= 1000 ? 1 : a >= 10 ? 2 : a >= 1 ? 3 : 5, maximumFractionDigits: a >= 1000 ? 1 : a >= 10 ? 2 : a >= 1 ? 3 : 5 }); }
  function short(sym) { return String(sym).replace("USDT", ""); }
  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }
  function slotParts(slot) { var p = String(slot).split("."); return { variant: p[0] + "." + p[1], symbol: p[2] || "", family: p[0] }; }
  function famChip(v) { return '<span class="fchip">' + esc(v.split(".")[0]) + "</span>"; }
  function table(el, head, rows, empty) {
    var t = $(el);
    if (!rows.length && empty) { t.innerHTML = '<tbody><tr><td class="empty" colspan="' + head.length + '" style="text-align:center">' + esc(empty) + "</td></tr></tbody>"; return; }
    t.innerHTML = "<thead><tr>" + head.map(function (c) { return "<th>" + esc(c) + "</th>"; }).join("") + "</tr></thead><tbody>" + rows.join("") + "</tbody>";
  }
  function td(v, c) { return "<td" + (c ? ' class="' + c + '"' : "") + ">" + v + "</td>"; }
  function emptyBox(title, text) { return '<div class="empty"><b>' + esc(title) + "</b>" + esc(text || "") + "</div>"; }
  function pnlHtml(x, d, unit) { return '<span class="' + cls(x) + '">' + arrow(x) + sgn(x, d) + (unit ? " " + unit : "") + "</span>"; }

  // plynulé počítání čísel
  function tween(el, to, fmt) {
    if (!el) return;
    var from = el._v; el._v = to;
    if (!fin(to)) { el.textContent = "—"; return; }
    if (reduce || !fin(from) || from === to) { el.textContent = fmt(to); return; }
    var t0 = performance.now(), dur = 700;
    cancelAnimationFrame(el._raf);
    (function step(t) {
      var k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = fmt(from + (to - from) * e);
      if (k < 1) el._raf = requestAnimationFrame(step);
    })(t0);
  }

  // --- záložky -----------------------------------------------------------------------------------
  var buttons = document.querySelectorAll("nav [data-tab]");
  function show(name) {
    state.tab = name; store("arena.tab", name);
    buttons.forEach(function (b) { b.setAttribute("aria-selected", String(b.dataset.tab === name)); });
    TABS.forEach(function (k) { $("tab-" + k).hidden = k !== name; });
    render();
  }
  buttons.forEach(function (b) { b.addEventListener("click", function () { show(b.dataset.tab); }); });
  $("clean").checked = store("arena.clean") === "1";
  $("clean").addEventListener("change", function () { store("arena.clean", this.checked ? "1" : "0"); render(); });
  $("variant").addEventListener("change", function () { state.variant = this.value; loadVariant(); });

  // --- načítání ----------------------------------------------------------------------------------
  function load() {
    fetch(SNAP + "?t=" + Date.now(), { cache: "no-store" }).then(function (r) { return r.json(); }).then(function (j) {
      state.snap = j; render();
    }).catch(function () { staleCheck(); });
  }
  function staleCheck() {
    var age = state.snap ? Date.now() - state.snap.generated_at : Infinity;
    $("stale").hidden = age <= STALE_MS;
    $("stale-age").textContent = isFinite(age) ? Math.round(age / 1000) + " s" : "snapshot nenačten";
  }
  function lb() {
    var s = state.snap && state.snap.stats && state.snap.stats.leaderboard;
    if (!s) return {};
    return $("clean").checked ? s.clean : s.all;
  }

  function render() {
    staleCheck();
    var s = state.snap;
    if (!s) return;
    renderTop(s);
    if (state.tab === "overview") renderOverview(s);
    if (state.tab === "board") renderBoard(s);
    if (state.tab === "detail") renderDetailSelect();
    if (state.tab === "costs") loadCosts();
    if (state.tab === "system") renderSystem(s);
  }

  // --- horní pruh --------------------------------------------------------------------------------
  function systemStatus(s) {
    var h = s.health || {}, probs = [], bad = false;
    if (h.kill_switch) { probs.push("kill switch aktivní"); bad = true; }
    Object.keys(h.ws || {}).forEach(function (k) { if (!h.ws[k].connected) { probs.push("WS " + k + " odpojeno"); bad = true; } });
    Object.keys(h.books || {}).forEach(function (k) { if (!h.books[k].synced) probs.push("kniha " + short(k) + " nesync"); });
    if ((h.stale || []).length) { probs.push("stale: " + h.stale.map(short).join(", ")); bad = true; }
    if (h.errors_24h > 0) probs.push(h.errors_24h + " chyb za 24 h");
    if (!probs.length) return { cls: "", text: "Vše běží" };
    return { cls: bad ? "bad" : "warn", text: probs[0] + (probs.length > 1 ? " (+" + (probs.length - 1) + ")" : "") };
  }
  function renderTop(s) {
    $("envline").textContent = s.env + " · data " + s.data_env + " · cost model v" + s.cost_model_version;
    var st = systemStatus(s);
    $("status").innerHTML = '<i class="dot ' + st.cls + '"></i><span>' + esc(st.text) + "</span>";
    document.title = "Arena 5m" + (st.cls ? " ⚠" : "");
  }
  function tickClock() {
    var now = Date.now(), left = BAR_MS - (now % BAR_MS), m = Math.floor(left / 60000), sec = Math.floor(left / 1000) % 60;
    $("countdown").textContent = m + ":" + (sec < 10 ? "0" : "") + sec;
    $("ring").style.strokeDashoffset = String(94.25 * (1 - (BAR_MS - left) / BAR_MS));
  }

  // --- SVG: sparkline ----------------------------------------------------------------------------
  function sparkSvg(vals, color, w, h) {
    var n = vals.length;
    if (n < 2) return "";
    var mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals), pad = (mx - mn) * 0.12 || Math.abs(mx) * 0.0005 || 1;
    mn -= pad; mx += pad;
    var pts = vals.map(function (v, i) { return [i / (n - 1) * w, h - (v - mn) / (mx - mn) * h]; });
    var d = pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join("");
    var last = pts[n - 1];
    return '<path class="spark-area" d="' + d + "L" + w + " " + h + "L0 " + h + 'Z" fill="' + color + '"/>' +
      '<path class="spark-line' + (reduce ? "" : " draw") + '" style="--len:' + (w * 3) + '" pathLength="' + (w * 3) + '" d="' + d + '" stroke="' + color + '"/>' +
      '<circle class="spark-dot" cx="' + last[0].toFixed(1) + '" cy="' + last[1].toFixed(1) + '" r="3.2" fill="' + color + '"/>';
  }

  // --- přehled -----------------------------------------------------------------------------------
  function renderOverview(s) {
    var o = s.overview || {}, eq = o.equity, t = o.today || {}, lp = t.live || {}, pp = t.paper || {};
    var liveOn = s.live_enabled;
    $("env-pill").innerHTML = '<i class="dot ' + (liveOn ? "" : "off") + '"></i>' + (liveOn ? "LIVE · " + esc(s.env) : "jen paper");
    $("eq-lbl").textContent = liveOn ? "Hodnota účtu (" + s.env + ")" : "Účet";
    if (eq) {
      var big = $("eq-big");
      if (!big.firstChild || big.dataset.init !== "1") { big.innerHTML = '<span id="eq-num"></span><small>USDT</small>'; big.dataset.init = "1"; }
      tween($("eq-num"), eq.wallet + (eq.unrealized || 0), function (v) { return money(v, 2); });
      $("eq-sub").innerHTML = "<span>wallet " + money(eq.wallet, 2) + "</span><span>nerealizováno " + pnlHtml(eq.unrealized, 2) +
        "</span><span>margin " + money(eq.margin_used, 2) + "</span>";
    } else {
      $("eq-big").innerHTML = '<span class="muted">—</span>';
      $("eq-sub").innerHTML = '<span class="muted">Účet se zatím nenačetl (live vypnuto nebo čeká na první equity vzorek).</span>';
    }
    renderEquity(s);
    var lpos = o.live_positions || [], ppos = o.paper_positions || [];
    var kp = [
      ["Live P&L dnes", lp.pnl != null ? pnlHtml(lp.pnl, 2) : '<span class="muted">—</span>', lp.n ? lp.n + " obchodů · fee " + num(lp.fee, 2) + " · fund. " + num(lp.funding, 2) : "žádný uzavřený obchod"],
      ["Paper P&L dnes", pp.pnl != null ? pnlHtml(pp.pnl, 2) : '<span class="muted">—</span>', (pp.n || 0) + " obchodů"],
      ["Pozice", String(lpos.length) + '<span class="muted" style="font-size:15px"> live</span> · ' + ppos.length + '<span class="muted" style="font-size:15px"> paper</span>', "otevřených právě teď"],
      ["Obchodů celkem", String((s.stats && s.stats.n_trades) || 0), "paper, všechny varianty"]
    ];
    $("kpis").innerHTML = kp.map(function (k) {
      return '<div class="card kpi"><div class="lbl">' + esc(k[0]) + '</div><div class="val num">' + k[1] + '</div><div class="sub">' + esc(k[2]) + "</div></div>";
    }).join("");
    renderMarket(s);
    renderPositions(s);
    renderFeed(s);
    renderActivity(s);
  }

  function renderEquity(s) {
    var host = $("eq-chart"), c = s.equity_curve;
    if (!c || c.t.length < 2) { host.innerHTML = '<div class="empty" style="padding-top:44px">Křivka účtu se začne kreslit po prvních vzorcích (každou minutu).</div>'; return; }
    lineChart(host, [{ name: "Účet", color: "var(--accent)", area: true, pts: c.t.map(function (t, i) { return [t, c.equity[i]]; }) }],
      { height: 130, yfmt: function (v) { return money(v, 2); }, xfmt: hm, minimal: true, minSpan: 0.0006 });
  }

  function renderMarket(s) {
    var host = $("market"), m = s.market || {}, syms = s.universe || Object.keys(m);
    var live = {};
    (s.overview.paper_positions || []).concat(s.overview.live_positions || []).forEach(function (p) { if (p.px) live[slotParts(p.slot).symbol] = p.px; });
    var ls = s.live_slots || {};
    syms.forEach(function (sym) {
      var d = m[sym], el = host.querySelector('[data-sym="' + sym + '"]');
      if (!d) return;
      var last = live[sym] || d.close[d.close.length - 1], first = d.close[0], chg = (last / first - 1) * 100;
      if (!el) {
        el = document.createElement("div"); el.className = "card mk"; el.dataset.sym = sym;
        el.innerHTML = '<div class="row"><i class="sw" style="background:' + SYMCOL[sym] + '"></i><span class="sym">' + esc(short(sym)) + "</span>" +
          (ls[sym] ? '<span class="tag">LIVE</span>' : "") + '<span class="chg num"></span></div><div class="px num"></div><svg viewBox="0 0 200 54" preserveAspectRatio="none"></svg>';
        host.appendChild(el);
      }
      var pxEl = el.querySelector(".px"), prev = state.prices[sym];
      pxEl.textContent = price(last);
      if (fin(prev) && prev !== last) { pxEl.classList.remove("up", "dn"); void pxEl.offsetWidth; pxEl.classList.add(last > prev ? "up" : "dn"); }
      state.prices[sym] = last;
      var ce = el.querySelector(".chg"); ce.className = "chg num " + cls(chg); ce.textContent = arrow(chg) + Math.abs(chg).toFixed(2) + " %";
      var key = d.close.length + ":" + d.close[d.close.length - 1] + ":" + last, svg = el.querySelector("svg");
      if (svg.dataset.k !== key) {
        var vals = d.close.slice(); if (live[sym]) vals.push(live[sym]);
        var firstDraw = !svg.dataset.k;
        svg.innerHTML = sparkSvg(vals, SYMCOL[sym], 200, 54);
        if (!firstDraw) svg.querySelectorAll(".draw").forEach(function (p) { p.classList.remove("draw"); p.style.strokeDashoffset = 0; });
        svg.dataset.k = key;
      }
    });
  }

  // --- pozice: žebřík -----------------------------------------------------------------------------
  function bpsInFavor(side, entry, p) { return (side === "LONG" ? 1 : -1) * (p - entry) / entry * 1e4; }
  function renderPositions(s) {
    var host = $("positions"), o = s.overview;
    var items = (o.live_positions || []).map(function (p) { return Object.assign({ mode: "live" }, p); })
      .concat((o.paper_positions || []).map(function (p) { return Object.assign({ mode: "paper" }, p); }));
    items.sort(function (a, b) { return (a.mode === "live" ? 0 : 1) - (b.mode === "live" ? 0 : 1) || (a.slot < b.slot ? -1 : 1); });
    $("pos-n").textContent = items.length ? "(" + items.length + ")" : "";
    if (!items.length) { host.innerHTML = '<div class="card" style="grid-column:1/-1">' + emptyBox("Žádná otevřená pozice", "Další vstup přijde se signálem na uzavření 5m baru.") + "</div>"; return; }
    var keep = {};
    items.forEach(function (p) {
      var key = p.mode + ":" + p.slot; keep[key] = 1;
      var sp = slotParts(p.slot), cur = bpsInFavor(p.side, p.entry_px, p.px || p.entry_px);
      var stop = fin(p.hard_stop) ? bpsInFavor(p.side, p.entry_px, p.hard_stop) : null;
      var trail = fin(p.trail) ? bpsInFavor(p.side, p.entry_px, p.trail) : null;
      var pts = [0, cur].concat(stop != null ? [stop] : [], trail != null ? [trail] : []);
      var lo = Math.min.apply(null, pts), hi = Math.max.apply(null, pts), pad = Math.max(4, (hi - lo) * 0.12);
      lo -= pad; hi += pad;
      var pc = function (v) { return ((v - lo) / (hi - lo) * 100).toFixed(2) + "%"; };
      var el = host.querySelector('[data-k="' + key + '"]'), fresh = !el;
      if (fresh) {
        host.querySelectorAll(".empty").forEach(function (e) { e.parentNode.remove(); });
        el = document.createElement("div"); el.className = "card pc" + (p.mode === "live" ? " live" : ""); el.dataset.k = key;
        el.innerHTML = '<div class="hd"><i class="sw" style="background:' + (SYMCOL[sp.symbol] || "var(--muted)") + '"></i><span class="nm">' + esc(short(sp.symbol)) + '</span><span class="fam">' + esc(sp.variant) + "</span>" +
          '<span class="side ' + esc(p.side) + '">' + esc(p.side) + '</span><span class="mode ' + p.mode + '">' + (p.mode === "live" ? "LIVE" : "PAPER") + "</span>" +
          '<span class="pnl num"></span></div>' +
          '<div class="ladder"><div class="rail"><div class="fill"></div></div>' +
          '<div class="mkr stop"><span>stop</span><i></i></div><div class="mkr entry"><span>vstup</span><i></i></div><div class="mkr trail"><span>trailing</span><i></i></div>' +
          '<div class="mkr cur"><b></b></div></div><div class="ft"></div>';
        host.appendChild(el);
      }
      var pnl = el.querySelector(".pnl"); pnl.className = "pnl num " + cls(cur);
      pnl.innerHTML = arrow(cur) + sgn(cur, 1) + "<small>bps</small>";
      var fill = el.querySelector(".fill"), a = Math.min(0, cur), b = Math.max(0, cur);
      fill.style.left = pc(a); fill.style.width = ((b - a) / (hi - lo) * 100).toFixed(2) + "%";
      fill.style.background = cur >= 0 ? "var(--pos)" : "var(--neg)";
      function mk(sel, v, low) {
        var m = el.querySelector(sel); if (!m) return;
        m.style.display = v == null ? "none" : ""; if (v != null) m.style.left = pc(v);
        m.classList.toggle("low", !!low);
      }
      mk(".stop", stop, false); mk(".entry", 0, true); mk(".trail", trail, false);
      var c = el.querySelector(".cur"); c.style.left = pc(cur); c.className = "mkr cur " + (cur >= 0 ? "p" : "n");
      var ft = [];
      ft.push("<span>vstup <b class=num>" + price(p.entry_px) + "</b></span><span>teď <b class=num>" + price(p.px) + "</b></span>");
      if (fin(p.trail_dist_bps)) ft.push("<span>do trailingu <b class=num>" + num(p.trail_dist_bps, 1) + " bps</b></span>");
      if (p.entry_time) ft.push("<span>drží <b>" + ago(p.entry_time) + "</b></span>");
      if (p.state || p.status) ft.push("<span><b>" + esc(p.state || p.status) + "</b></span>");
      el.querySelector(".ft").innerHTML = ft.join("");
    });
    host.querySelectorAll(".pc").forEach(function (e) { if (!keep[e.dataset.k]) e.remove(); });
  }

  // --- feed obchodů ------------------------------------------------------------------------------
  function renderFeed(s) {
    var host = $("feed"), tr = s.recent_trades || [];
    if (!tr.length) { host.innerHTML = emptyBox("Zatím žádný uzavřený obchod", "Obchody se objeví tady hned po uzavření pozice."); state.seenTrades = {}; return; }
    var firstLoad = state.seenTrades == null || !host.querySelector(".fi");
    var seen = state.seenTrades || {};
    host.innerHTML = tr.map(function (t) {
      var sp = slotParts(t.slot_id), win = t.net_bps > 0, isNew = !firstLoad && !seen[t.trade_id];
      return '<div class="fi"' + (isNew ? ' style="background:var(--accent-soft);border-radius:8px"' : "") + '><span class="ic ' + (win ? "p" : "n") + '">' + (win ? "▲" : "▼") + "</span>" +
        '<div><div class="t1">' + esc(short(sp.symbol)) + " " + esc(t.side) + ' <span class="muted" style="font-weight:500">' + esc(sp.variant) + "</span> " +
        '<span class="mode ' + esc(t.mode) + '">' + esc(String(t.mode).toUpperCase()) + "</span>" + (t.tainted ? ' <span class="mode">TAINTED</span>' : "") + "</div>" +
        '<div class="t2">' + esc(t.exit_reason) + " · před " + ago(t.exit_time) + "</div></div>" +
        '<div class="v num ' + cls(t.net_bps) + '">' + sgn(t.net_bps, 1) + ' bps<div class="t2" style="font-weight:500">' + (fin(t.net_pnl) ? sgn(t.net_pnl, 3) + " USDT" : "") + "</div></div></div>";
    }).join("");
    var m = {}; tr.forEach(function (t) { m[t.trade_id] = 1; }); state.seenTrades = m;
  }

  function renderActivity(s) {
    var a = s.activity || {}, hrs = a.entered_by_hour || [], mx = Math.max.apply(null, hrs.concat([1]));
    $("bars24").innerHTML = hrs.map(function (n, i) {
      return '<div title="' + n + ' vstupů" style="height:' + Math.max(2, n / mx * 100) + "%;animation-delay:" + (reduce ? 0 : i * 18) + 'ms"></div>';
    }).join("") || emptyBox("Bez dat", "");
    var by = a.by_action || {}, names = { entered: "vstup", suppressed_position: "potlačeno: otevřená pozice", suppressed_cooldown: "potlačeno: cooldown", blocked_risk: "blokováno riskem", stale: "stale data" };
    var keys = Object.keys(by);
    $("signals").innerHTML = keys.length ? keys.map(function (k) { return '<span class="chip"><b class="num">' + by[k] + "</b> " + esc(names[k] || k) + "</span>"; }).join("") : '<span class="muted">Za posledních 24 h žádné signály.</span>';
  }

  // --- generický čárový graf -----------------------------------------------------------------------
  function niceStep(raw) { var e = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / e; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * e; }
  // series: [{name,color,pts:[[x,y]],dash,area,w}]; o: {height,yfmt,xfmt,minimal,minSpan,zero}
  function lineChart(host, series, o) {
    host._lc = { series: series, o: o };
    draw();
    if (!host._ro && window.ResizeObserver) { host._ro = new ResizeObserver(function () { if (host._w !== host.clientWidth) draw(); }); host._ro.observe(host); }
    function draw() {
      var W = host.clientWidth || 600, H = o.height || 260, L = o.minimal ? 4 : 52, R = o.minimal ? 4 : 12, T = 10, B = o.minimal ? 4 : 24;
      host._w = host.clientWidth;
      var all = [].concat.apply([], series.map(function (s) { return s.pts; }));
      if (all.length < 2) { host.innerHTML = '<div class="empty">Zatím málo dat pro graf.</div>'; return; }
      var xs = all.map(function (p) { return p[0]; }), ys = all.map(function (p) { return p[1]; });
      var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
      if (o.zero) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
      var span = Math.max(y1 - y0, Math.abs(y1) * (o.minSpan || 0), 1e-9), padY = span * 0.1; y0 -= padY; y1 += padY;
      if (x1 === x0) x1 = x0 + 1;
      var X = function (x) { return L + (x - x0) / (x1 - x0) * (W - L - R); }, Y = function (y) { return T + (1 - (y - y0) / (y1 - y0)) * (H - T - B); };
      var g = "";
      if (!o.minimal) {
        var step = niceStep((y1 - y0) / 4), ty = Math.ceil(y0 / step) * step;
        for (; ty <= y1; ty += step) {
          var yy = Y(ty);
          g += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + yy + '" y2="' + yy + '" stroke="var(--grid)"/><text x="' + (L - 8) + '" y="' + (yy + 4) + '" text-anchor="end" fill="var(--muted)" font-size="11">' + esc((o.yfmt || num)(ty)) + "</text>";
        }
        for (var j = 0; j <= 4; j++) {
          var xv = x0 + (x1 - x0) * j / 4;
          g += '<text x="' + X(xv) + '" y="' + (H - 6) + '" text-anchor="' + (j === 0 ? "start" : j === 4 ? "end" : "middle") + '" fill="var(--muted)" font-size="11">' + esc((o.xfmt || String)(xv)) + "</text>";
        }
        if (o.zero) g += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(0) + '" y2="' + Y(0) + '" stroke="var(--axis)"/>';
      }
      series.forEach(function (s, si) {
        var d = s.pts.map(function (p, i) { return (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1); }).join("");
        if (s.area) {
          g += '<defs><linearGradient id="ga' + si + '" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="' + s.color + '" stop-opacity=".28"/><stop offset="1" stop-color="' + s.color + '" stop-opacity="0"/></linearGradient></defs>' +
            '<path d="' + d + "L" + X(s.pts[s.pts.length - 1][0]) + " " + (H - B) + "L" + X(s.pts[0][0]) + " " + (H - B) + 'Z" fill="url(#ga' + si + ')"/>';
        }
        g += '<path d="' + d + '" fill="none" stroke="' + s.color + '" stroke-width="' + (s.w || 2) + '" stroke-linejoin="round" stroke-linecap="round"' + (s.dash ? ' stroke-dasharray="' + s.dash + '"' : "") + "/>";
        var lp = s.pts[s.pts.length - 1];
        if (si === 0) g += '<circle cx="' + X(lp[0]) + '" cy="' + Y(lp[1]) + '" r="3.5" fill="' + s.color + '" stroke="var(--surface)" stroke-width="2"/>';
      });
      g += '<line class="xh" y1="' + T + '" y2="' + (H - B) + '" stroke="var(--axis)" style="display:none"/><rect class="hit" x="' + L + '" y="0" width="' + (W - L - R) + '" height="' + H + '" fill="transparent"/>';
      host.innerHTML = '<svg viewBox="0 0 ' + W + " " + H + '" width="' + W + '" height="' + H + '" role="img">' + g + '</svg><div class="tip"></div>';
      var svg = host.querySelector("svg"), tip = host.querySelector(".tip"), xh = host.querySelector(".xh"), hit = host.querySelector(".hit");
      function near(pts, x) { var lo = 0, hi = pts.length - 1; while (hi - lo > 1) { var m = (lo + hi) >> 1; if (pts[m][0] <= x) lo = m; else hi = m; } return Math.abs(pts[lo][0] - x) <= Math.abs(pts[hi][0] - x) ? pts[lo] : pts[hi]; }
      hit.addEventListener("pointermove", function (e) {
        var r = svg.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * W, xv = x0 + (px - L) / (W - L - R) * (x1 - x0);
        var base = near(series[0].pts, xv), bx = X(base[0]);
        xh.setAttribute("x1", bx); xh.setAttribute("x2", bx); xh.style.display = "";
        var html = '<div class="tt">' + esc((o.xfmtTip || o.xfmt || String)(base[0])) + "</div>";
        series.forEach(function (s) { var p = near(s.pts, base[0]); html += '<div class="tr"><i style="border-color:' + s.color + '"></i>' + esc(s.name) + "<b class=num>" + esc((o.yfmt || num)(p[1])) + "</b></div>"; });
        tip.innerHTML = html; tip.classList.add("on");
        var tw = tip.offsetWidth; tip.style.left = Math.min(Math.max(0, bx / W * r.width - tw / 2), r.width - tw) + "px"; tip.style.top = "0px";
      });
      hit.addEventListener("pointerleave", function () { xh.style.display = "none"; tip.classList.remove("on"); });
    }
  }

  // --- žebříček ----------------------------------------------------------------------------------
  var COLS = [
    ["key", "varianta"], ["n", "n"], ["trades_per_day", "obch./den"], ["net_mean", "net bps ± 95% CI"], ["gross_mean", "gross"],
    ["cost", "náklady"], ["sharpe", "Sharpe"], ["delta", "Δ vs ctl"], ["p_holm", "p_holm"], ["dsr", "DSR"], ["n_req", "n_req"], ["badge", "stav"]
  ];
  function colVal(g, k) {
    if (k === "cost") return (g.cost_bps || {}).total; if (k === "delta") return (g.vs_control || {}).delta;
    if (k === "n_req") return g.n_req && g.n_req.z2; return g[k];
  }
  function renderBoard(s) {
    var L = lb(), keys = Object.keys(L), sk = state.sort;
    var real = keys.filter(function (k) { return k.indexOf("ctl.") !== 0; }), ctl = keys.filter(function (k) { return k.indexOf("ctl.") === 0; }).sort();
    real.sort(function (a, b) {
      var x = colVal(L[a], sk.key), y = colVal(L[b], sk.key);
      if (sk.key === "key" || sk.key === "badge") return sk.dir * (String(x) < String(y) ? -1 : 1) * -1;
      if (!fin(x) && !fin(y)) return a < b ? -1 : 1; if (!fin(x)) return 1; if (!fin(y)) return -1;
      return sk.dir * (x - y);
    });
    var order = real.concat(ctl), mxAbs = 1;
    order.forEach(function (k) { var g = L[k], ci = g.net_ci || []; [g.net_mean, ci[0], ci[1]].forEach(function (v) { if (fin(v)) mxAbs = Math.max(mxAbs, Math.abs(v)); }); });
    function barCell(g) {
      if (!fin(g.net_mean)) return '<span class="muted">—</span>';
      var W = 132, mid = W / 2, sc = function (v) { return mid + Math.max(-mxAbs, Math.min(mxAbs, v)) / mxAbs * (mid - 3); };
      var ci = g.net_ci || [], col = g.net_mean >= 0 ? "var(--div-pos)" : "var(--div-neg)";
      var svg = '<svg class="cellbar" viewBox="0 0 ' + W + ' 20"><line x1="' + mid + '" x2="' + mid + '" y1="2" y2="18" stroke="var(--axis)"/>' +
        '<rect x="' + Math.min(mid, sc(g.net_mean)) + '" y="5" width="' + Math.abs(sc(g.net_mean) - mid) + '" height="10" rx="3" fill="' + col + '"/>' +
        (fin(ci[0]) && fin(ci[1]) ? '<line x1="' + sc(ci[0]) + '" x2="' + sc(ci[1]) + '" y1="10" y2="10" stroke="var(--ink-2)" stroke-width="1.5"/><line x1="' + sc(ci[0]) + '" x2="' + sc(ci[0]) + '" y1="6" y2="14" stroke="var(--ink-2)"/><line x1="' + sc(ci[1]) + '" x2="' + sc(ci[1]) + '" y1="6" y2="14" stroke="var(--ink-2)"/>' : "") + "</svg>";
      return svg + ' <span class="' + cls(g.net_mean) + '" style="display:inline-block;min-width:44px">' + sgn(g.net_mean) + "</span>";
    }
    var head = "<thead><tr>" + COLS.map(function (c) {
      var on = sk.key === c[0];
      return '<th class="s"' + (on ? ' data-dir="' + (sk.dir < 0 ? "↓" : "↑") + '"' : "") + ' data-k="' + c[0] + '">' + esc(c[1]) + "</th>";
    }).join("") + "</tr></thead>";
    var rows = order.map(function (k) {
      var g = L[k], vc = g.vs_control || {}, b = g.badge || "";
      var bc = b === "LIVE" ? "live" : b === "KANDIDÁT" ? "cand" : b === "ZÁPORNÁ" ? "negb" : b === "KONTROLA" ? "ctl" : "";
      return '<tr class="click" data-v="' + esc(k) + '">' + td(esc(k) + famChip(k)) + td(g.n) + td(num(g.trades_per_day, 2)) + td(barCell(g)) +
        td(sgn(g.gross_mean), cls(g.gross_mean)) + td(num((g.cost_bps || {}).total)) + td(num(g.sharpe, 2)) + td(sgn(vc.delta), cls(vc.delta)) +
        td(num(g.p_holm, 3)) + td(num(g.dsr, 2)) + td(g.n_req && g.n_req.z2 != null ? num(g.n_req.z2, 0) : "∞") +
        td('<span class="badge ' + bc + '">' + esc(b) + "</span>") + "</tr>";
    });
    $("board").innerHTML = head + "<tbody>" + rows.join("") + "</tbody>";
    $("board").querySelectorAll("th.s").forEach(function (th) {
      th.addEventListener("click", function () {
        var k = th.dataset.k; state.sort = { key: k, dir: state.sort.key === k ? -state.sort.dir : -1 }; renderBoard(state.snap);
      });
    });
    $("board").querySelectorAll("tr.click").forEach(function (tr) { tr.addEventListener("click", function () { state.variant = tr.dataset.v; show("detail"); }); });

    var Hm = (s.stats && s.stats.heatmap) || {}, syms = s.universe || ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"], mx = 1;
    Object.keys(Hm).forEach(function (v) { Object.keys(Hm[v]).forEach(function (y) { if (Hm[v][y].n >= 30) mx = Math.max(mx, Math.abs(Hm[v][y].net_mean)); }); });
    var hrows = order.map(function (k) {
      return "<tr>" + td(esc(k)) + syms.map(function (sym) {
        var c = (Hm[k] || {})[sym];
        if (!c) return '<td><div class="hc na">—</div></td>';
        if (c.n < 30) return '<td><div class="hc na" title="n < 30">' + sgn(c.net_mean) + "<small>n " + c.n + "</small></div></td>";
        var a = Math.min(1, Math.abs(c.net_mean) / mx), col = c.net_mean >= 0 ? "var(--div-pos)" : "var(--div-neg)";
        return '<td><div class="hc" style="background:color-mix(in srgb,' + col + " " + Math.round(18 + a * 62) + '%,var(--div-mid));color:' + (a > 0.55 ? "#fff" : "var(--ink)") + '">' + sgn(c.net_mean) + "<small>n " + c.n + "</small></div></td>";
      }).join("") + "</tr>";
    });
    table("heat", ["varianta"].concat(syms.map(short)), hrows, "Heatmapa se zaplní po prvních obchodech.");
  }

  // --- detail varianty ---------------------------------------------------------------------------
  function renderDetailSelect() {
    var keys = Object.keys(lb()).sort(), sel = $("variant");
    if (!state.variant) state.variant = store("arena.variant") || keys[0];
    if (sel.options.length !== keys.length) sel.innerHTML = keys.map(function (k) { return "<option>" + esc(k) + "</option>"; }).join("");
    sel.value = state.variant;
    loadVariant();
  }
  function tile(l, v, sub) { return '<div class="card kpi"><div class="lbl">' + esc(l) + '</div><div class="val num">' + v + '</div><div class="sub">' + esc(sub || "") + "</div></div>"; }
  function loadVariant() {
    var v = state.variant;
    if (!v) return;
    store("arena.variant", v);
    var g = lb()[v] || {}, ci = g.net_ci || [];
    $("vinfo").innerHTML = famChip(v) + ' <span class="muted">stav</span> <span class="badge">' + esc(g.badge || "") + "</span>";
    $("vtiles").innerHTML = [
      tile("Obchodů", String(g.n || 0), num(g.trades_per_day, 2) + " / den · " + (g.days || 0) + " dní"),
      tile("Net bps", fin(g.net_mean) ? pnlHtml(g.net_mean, 1) : "—", fin(ci[0]) ? "CI [" + num(ci[0]) + ", " + num(ci[1]) + "]" : ""),
      tile("Gross bps", fin(g.gross_mean) ? pnlHtml(g.gross_mean, 1) : "—", "náklady " + num((g.cost_bps || {}).total) + " bps"),
      tile("Sharpe", num(g.sharpe, 2), "max DD " + num(g.max_dd, 1)),
      tile("Win rate", fin(g.win_rate) ? num(g.win_rate * 100, 0) + " %" : "—", "držení " + num(g.avg_hold_min, 0) + " min"),
      tile("p_holm / DSR", num(g.p_holm, 3) + " / " + num(g.dsr, 2), "n_req (t=2): " + (g.n_req && g.n_req.z2 != null ? num(g.n_req.z2, 0) : "∞"))
    ].join("");
    fetch("data/variants/" + encodeURIComponent(v) + ".json?t=" + Date.now(), { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; }).then(function (d) {
        if (!d || v !== state.variant) return;
        var dayfmt = function (x) { return new Date(x).toISOString().slice(5, 10); };
        var ser = [], pc = d.paper_curve, cc = d.control_curve, lc = d.live_curve;
        function pts(c, key) { return c.t.map(function (t, i) { return [t, c[key][i]]; }); }
        if (pc.t.length > 1) ser.push({ name: "net bps (paper, kumul.)", color: "var(--accent)", area: true, pts: pts(pc, "net") }, { name: "gross bps (paper)", color: "var(--muted)", w: 1.5, pts: pts(pc, "gross") });
        if (cc.t.length > 1) ser.push({ name: "kontrola net", color: "var(--neg)", dash: "5 4", w: 1.5, pts: pts(cc, "net") });
        if (lc.t.length > 1) ser.push({ name: "live net", color: "var(--s2)", pts: pts(lc, "net") });
        $("cum-legend").innerHTML = ser.map(function (s) { return '<span><i style="border-color:' + s.color + (s.dash ? ";border-top-style:dashed" : "") + '"></i>' + esc(s.name) + "</span>"; }).join("") || "";
        if (ser.length) lineChart($("c-cum"), ser, { height: 300, zero: true, yfmt: function (x) { return sgn(x, 0); }, xfmt: dayfmt, xfmtTip: function (x) { return ts(x); } });
        else $("c-cum").innerHTML = emptyBox("Zatím žádné uzavřené obchody", "Kumulativní křivka net vs gross (rozdíl = náklady) se začne kreslit po prvních obchodech.");
        var bs = d.by_symbol || {}, bk = Object.keys(bs);
        $("d-sym").innerHTML = bk.length ? '<table><thead><tr><th>symbol</th><th>n</th><th>net bps</th><th>gross bps</th></tr></thead><tbody>' + bk.map(function (k) {
          var x = bs[k]; return "<tr>" + td('<i class="sw" style="display:inline-block;background:' + (SYMCOL[k] || "var(--muted)") + ';margin-right:7px"></i>' + esc(short(k))) + td(x.n) + td(sgn(x.net_mean), cls(x.net_mean)) + td(sgn(x.gross_mean)) + "</tr>";
        }).join("") + "</tbody></table>" : emptyBox("Bez dat", "");
        var em = d.exit_mix || {}, ek = Object.keys(em), tot = ek.reduce(function (a, k) { return a + em[k]; }, 0);
        var ecol = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];
        $("d-exit").innerHTML = tot ? '<div class="stack">' + ek.map(function (k, i) { return '<div title="' + esc(k) + ": " + em[k] + '" style="flex:' + em[k] + ";background:" + ecol[i % 5] + '"></div>'; }).join("") + "</div>" +
          '<div class="legend" style="margin-top:10px">' + ek.map(function (k, i) { return '<span><i style="border-color:' + ecol[i % 5] + '"></i>' + esc(k) + " <b class=num>" + em[k] + "</b> <span class=muted>" + Math.round(em[k] / tot * 100) + " %</span></span>"; }).join("") + "</div>" : emptyBox("Bez dat", "");
        var h = d.hist || { edges: [], counts: [] };
        renderHist($("c-hist"), h);
        table("d-trades", ["slot", "strana", "vstup", "výstup", "důvod", "net bps", "gross bps", "slip in", "slip out", "barů", "tainted"],
          (d.trades || []).map(function (t) {
            return "<tr>" + td(esc(t.slot_id)) + td(esc(t.side)) + td(ts(t.entry_time)) + td(ts(t.exit_time)) + td(esc(t.exit_reason)) + td(sgn(t.net_bps), cls(t.net_bps)) +
              td(sgn(t.gross_bps)) + td(num(t.entry_slip_bps)) + td(num(t.exit_slip_bps)) + td(t.bars_held) + td(t.tainted ? "ano" : "") + "</tr>";
          }), "Zatím žádné obchody.");
      });
  }
  function renderHist(host, h) {
    var c = h.counts || [];
    if (!c.length || !c.some(function (x) { return x > 0; })) { host.innerHTML = emptyBox("Zatím žádné obchody", "Histogram net bps po 25 bps."); return; }
    var W = host.clientWidth || 600, H = 200, L = 36, B = 24, T = 8, n = c.length, mx = Math.max.apply(null, c), bw = (W - L - 8) / n, g = "";
    c.forEach(function (v, i) {
      var x = L + i * bw, hh = v / mx * (H - T - B), lo = h.edges[i], pos = lo >= 0;
      g += '<rect x="' + (x + 1) + '" y="' + (H - B - hh) + '" width="' + Math.max(1, bw - 2) + '" height="' + hh + '" rx="3" fill="' + (pos ? "var(--div-pos)" : "var(--div-neg)") + '"><title>' + lo + " … " + h.edges[i + 1] + " bps: " + v + "</title></rect>";
      if (i % 4 === 0) g += '<text x="' + x + '" y="' + (H - 6) + '" fill="var(--muted)" font-size="11">' + lo + "</text>";
    });
    g += '<line x1="' + L + '" x2="' + (W - 8) + '" y1="' + (H - B) + '" y2="' + (H - B) + '" stroke="var(--axis)"/><text x="' + (L - 6) + '" y="' + (T + 8) + '" text-anchor="end" fill="var(--muted)" font-size="11">' + mx + "</text>";
    host.innerHTML = '<svg viewBox="0 0 ' + W + " " + H + '" width="' + W + '" height="' + H + '">' + g + "</svg>";
  }

  // --- náklady -----------------------------------------------------------------------------------
  function costRows(o) {
    return Object.keys(o || {}).map(function (k) {
      var c = o[k];
      return "<tr>" + td(esc(short(k))) + td(c.n) + td(num(c.fee_bps, 2)) + td(num(c.entry_slip_med, 2)) + td(num(c.entry_slip_p90, 2)) + td(num(c.exit_slip_med, 2)) + td(num(c.exit_slip_p90, 2)) + td(num(c.funding_bps, 2)) + "</tr>";
    });
  }
  function loadCosts() {
    fetch("data/costs.json?t=" + Date.now(), { cache: "no-store" }).then(function (r) { return r.ok ? r.json() : null; }).then(function (c) {
      if (!c) return;
      $("cmv").textContent = "cost_model_version " + c.cost_model_version + " · " + ts(c.generated_at);
      var head = ["symbol", "n", "fee bps", "entry slip med", "entry p90", "exit slip med", "exit p90", "funding bps"];
      table("costs-live", head, costRows(c.live), "Žádný uzavřený live obchod: náklady se změří po prvních round tripech.");
      table("costs-paper", head, costRows(c.paper), "Zatím žádný uzavřený paper obchod.");
      var m = c.model_error || {};
      table("model-err", ["párů", "net med", "net p90", "vstup med", "výstup med"], m.n ? ["<tr>" + td(m.n) + td(num(m.net_med, 2)) + td(num(m.net_p90, 2)) + td(num(m.entry_med, 2)) + td(num(m.exit_med, 2)) + "</tr>"] : [], "Bez párů live × paper dvojče.");
      table("lat", ["druh", "n", "průměr ms"], Object.keys(c.latencies || {}).map(function (k) { return "<tr>" + td(esc(k)) + td(c.latencies[k].n) + td(num(c.latencies[k].avg_ms, 0)) + "</tr>"; }), "Latence se měří u live obchodů.");
    });
  }

  // --- systém ------------------------------------------------------------------------------------
  function renderSystem(s) {
    var h = s.health || {}, rows = [];
    function row(a, b, c, bad) { rows.push("<tr>" + td(a) + td(b, bad ? "neg" : "") + td(c || "") + "</tr>"); }
    Object.keys(h.ws || {}).forEach(function (k) { var w = h.ws[k]; row("WS " + esc(k), w.connected ? "připojeno" : "ODPOJENO", "spojení " + w.connects + " · poslední zpráva " + num((w.last_msg_age_ms || 0) / 1000, 1) + " s", !w.connected); });
    Object.keys(h.books || {}).forEach(function (k) {
      var b = h.books[k], ok = (h.book_ok_24h || {})[k];
      row("kniha " + esc(short(k)), b.synced ? "sync" : "NESYNC", "resynců " + b.resyncs + " · úrovní " + b.levels + " · book_ok 24 h " + (ok == null ? "—" : num(ok * 100, 0) + " %"), !b.synced);
    });
    Object.keys(h.bar_delay_s || {}).forEach(function (k) { var d = h.bar_delay_s[k], st = (h.stale || []).indexOf(k) >= 0; row("bar " + esc(short(k)), st ? "STALE" : "ok", num(d, 0) + " s od uzavření", st || d > 20); });
    var bv = h.book_validate_result;
    row("book-validate", esc(h.book_validate || "—"), bv ? (bv.passed ? "OK" : "NEPROŠLO") : "", bv && !bv.passed);
    row("rekonciliace", h.last_reconcile ? ts(Number(h.last_reconcile)) : "—", "");
    row("kill switch", h.kill_switch ? "AKTIVNÍ" : "ne", "chyby 24 h: " + h.errors_24h, h.kill_switch);
    row("vzorky knihy", "ok " + (h.samples || {}).ok + " · chybí " + (h.samples || {}).bad, "váha 1m " + (h.weight_1m == null ? "—" : h.weight_1m));
    table("health", ["komponenta", "stav", "detail"], rows);
    table("log", ["čas", "úroveň", "komponenta", "zpráva"], (s.events || []).map(function (e) {
      return "<tr>" + td(ts(e.ts)) + td('<span class="lvl ' + esc(e.level) + '">' + esc(e.level) + "</span>") + td(esc(e.component)) + '<td style="text-align:left;white-space:normal">' + esc(e.msg) + "</td></tr>";
    }), "Žádné události.");
  }

  var saved = store("arena.tab");
  show(saved && TABS.indexOf(saved) >= 0 ? saved : "overview");
  load(); tickClock();
  setInterval(load, POLL_MS);
  setInterval(staleCheck, 5000);
  setInterval(tickClock, 1000);
  setInterval(function () { if (state.tab === "overview" && state.snap) renderFeedAges(); }, 30000);
  function renderFeedAges() { renderFeed(state.snap); }
})();
