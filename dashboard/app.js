// Arena 5m dashboard: čte jen data/*.json (snapshot), nikdy DB (SPEC §15, nástraha 13).
(function () {
  "use strict";
  var SNAP = "data/snapshot.json";
  var STALE_MS = 60000;
  var state = { snap: null, tab: "overview", variant: null, charts: {} };

  function $(id) { return document.getElementById(id); }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function num(x, d) { return x == null || !isFinite(x) ? "—" : Number(x).toFixed(d == null ? 1 : d); }
  function sgn(x, d) { if (x == null || !isFinite(x)) return "—"; var s = Number(x).toFixed(d == null ? 1 : d); return (x > 0 ? "+" : "") + s; }
  function cls(x) { return x == null || !isFinite(x) ? "" : (x > 0 ? "pos" : (x < 0 ? "neg" : "")); }
  function ts(ms) { if (!ms) return "—"; var d = new Date(ms); return d.toISOString().replace("T", " ").slice(0, 19); }
  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }
  function table(el, head, rows) {
    var h = "<thead><tr>" + head.map(function (c) { return "<th>" + esc(c) + "</th>"; }).join("") + "</tr></thead><tbody>";
    h += rows.join("") + "</tbody>";
    $(el).innerHTML = h;
  }
  function td(v, c) { return "<td" + (c ? ' class="' + c + '"' : "") + ">" + v + "</td>"; }

  // --- tabs --------------------------------------------------------------------------------------
  var buttons = document.querySelectorAll("nav [data-tab]");
  function show(name) {
    state.tab = name; store("arena.tab", name);
    buttons.forEach(function (b) { b.setAttribute("aria-selected", String(b.dataset.tab === name)); });
    ["overview", "board", "detail", "costs", "log"].forEach(function (k) { $("tab-" + k).hidden = k !== name; });
    render();
  }
  buttons.forEach(function (b) { b.addEventListener("click", function () { show(b.dataset.tab); }); });
  $("clean").checked = store("arena.clean") === "1";
  $("clean").addEventListener("change", function () { store("arena.clean", this.checked ? "1" : "0"); render(); });
  $("variant").addEventListener("change", function () { state.variant = this.value; loadVariant(); });

  // --- data --------------------------------------------------------------------------------------
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
    $("meta").textContent = "env " + s.env + " · data " + s.data_env + " · live " + (s.live_enabled ? "ZAPNUTO" : "vypnuto") +
      " · config " + s.config_hash + " · cost model v" + s.cost_model_version + " · " + ts(s.generated_at);
    if (state.tab === "overview") renderOverview(s);
    if (state.tab === "board") renderBoard(s);
    if (state.tab === "detail") renderDetailSelect();
    if (state.tab === "costs") loadCosts();
    if (state.tab === "log") renderLog(s);
  }

  // --- přehled -----------------------------------------------------------------------------------
  function renderOverview(s) {
    var o = s.overview, h = s.health, eq = o.equity, t = o.today || {};
    var lp = t.live || {}, pp = t.paper || {};
    var tiles = [
      ["Wallet", eq ? num(eq.wallet, 2) + " USDT" : "—"],
      ["Unrealized", eq ? sgn(eq.unrealized, 2) : "—"],
      ["Live P&L dnes", lp.pnl != null ? sgn(lp.pnl, 2) + " USDT" : "—"],
      ["Live náklady dnes", lp.fee != null ? num(lp.fee, 2) + " fee · " + num(lp.funding, 2) + " fund." : "—"],
      ["Paper obchodů dnes", pp.n || 0],
      ["Paper P&L dnes", pp.pnl != null ? sgn(pp.pnl, 2) + " USDT" : "—"],
      ["Otevřené paper", (o.paper_positions || []).length],
      ["Kill switch", h.kill_switch ? "AKTIVNÍ" : "ne"],
      ["Chyby 24 h", h.errors_24h]
    ];
    $("tiles").innerHTML = tiles.map(function (x) { return '<div class="tile"><span>' + esc(x[0]) + "</span><b>" + esc(x[1]) + "</b></div>"; }).join("");
    var rows = [];
    Object.keys(h.ws || {}).forEach(function (k) {
      var w = h.ws[k];
      rows.push("<tr>" + td("WS " + esc(k)) + td(w.connected ? "připojeno" : "ODPOJENO", w.connected ? "" : "neg") +
        td("spojení " + w.connects) + td("poslední zpráva " + num((w.last_msg_age_ms || 0) / 1000, 1) + " s") + "</tr>");
    });
    Object.keys(h.books || {}).forEach(function (k) {
      var b = h.books[k], ok = (h.book_ok_24h || {})[k];
      rows.push("<tr>" + td("kniha " + esc(k)) + td(b.synced ? "sync" : "NESYNC", b.synced ? "" : "neg") +
        td("resynců " + b.resyncs + " · úrovní " + b.levels) + td("book_ok 24 h " + (ok == null ? "—" : num(ok * 100, 0) + " %")) + "</tr>");
    });
    Object.keys(h.bar_delay_s || {}).forEach(function (k) {
      var d = h.bar_delay_s[k];
      rows.push("<tr>" + td("bar " + esc(k)) + td(num(d, 0) + " s od uzavření", d > 20 ? "neg" : "") +
        td((h.stale || []).indexOf(k) >= 0 ? "STALE" : "") + td("") + "</tr>");
    });
    var bv = h.book_validate_result;
    rows.push("<tr>" + td("book-validate") + td(h.book_validate || "—") + td(bv ? (bv.passed ? "OK" : "NEPROŠLO") : "", bv && !bv.passed ? "neg" : "") + td("") + "</tr>");
    rows.push("<tr>" + td("rekonciliace") + td(h.last_reconcile ? ts(Number(h.last_reconcile)) : "—") + td("") + td("") + "</tr>");
    rows.push("<tr>" + td("vzorky knihy") + td("ok " + (h.samples || {}).ok + " · chybí " + (h.samples || {}).bad) + td("váha 1m " + (h.weight_1m == null ? "—" : h.weight_1m)) + td("") + "</tr>");
    table("health", ["", "stav", "", ""], rows);
    table("live-pos", ["slot", "strana", "vstup", "cena", "trailing", "vzdálenost bps", "algoId"],
      (o.live_positions || []).map(function (p) {
        return "<tr>" + td(esc(p.slot)) + td(p.side) + td(num(p.entry_px, 4)) + td(num(p.px, 4)) + td(num(p.trail, 4)) +
          td(num(p.trail_dist_bps, 1)) + td(esc(p.algo_id || "")) + "</tr>";
      }));
    var pps = (o.paper_positions || []).slice().sort(function (a, b) { return a.slot < b.slot ? -1 : 1; });
    $("paper-n").textContent = "(" + pps.length + ")";
    table("paper-pos", ["slot", "strana", "stav", "vstup", "cena", "trailing", "vzdál. bps", "uPnL bps"],
      pps.map(function (p) {
        return "<tr>" + td(esc(p.slot)) + td(p.side) + td(p.status) + td(num(p.entry_px, 4)) + td(num(p.px, 4)) +
          td(num(p.trail, 4)) + td(num(p.trail_dist_bps, 1)) + td(sgn(p.upnl_bps, 1), cls(p.upnl_bps)) + "</tr>";
      }));
  }

  // --- žebříček ----------------------------------------------------------------------------------
  function renderBoard(s) {
    var L = lb(), keys = Object.keys(L).sort();
    var real = keys.filter(function (k) { return k.indexOf("ctl.") !== 0; });
    real.sort(function (a, b) { return (L[b].net_mean || -1e9) - (L[a].net_mean || -1e9); });
    var order = real.concat(keys.filter(function (k) { return k.indexOf("ctl.") === 0; }));
    var rows = order.map(function (k) {
      var g = L[k], ci = g.net_ci || [null, null], vc = g.vs_control || {};
      return '<tr class="click" data-v="' + esc(k) + '">' + td(esc(k)) + td(g.n) + td(num(g.trades_per_day, 2)) +
        td(sgn(g.net_mean) + " <span class=muted>[" + num(ci[0]) + ", " + num(ci[1]) + "]</span>", cls(g.net_mean)) +
        td(sgn(g.gross_mean), cls(g.gross_mean)) + td(num((g.cost_bps || {}).total)) + td(num(g.sharpe, 2)) +
        td(sgn(vc.delta), cls(vc.delta)) + td(num(g.p_holm, 3)) + td(num(g.dsr, 2)) +
        td(g.n_req && g.n_req.z2 != null ? num(g.n_req.z2, 0) : "∞") +
        td('<span class="badge ' + esc(g.badge) + '">' + esc(g.badge) + "</span>") + "</tr>";
    });
    table("board", ["varianta", "n", "obch./den", "net bps ± 95% CI", "gross bps", "náklady bps", "Sharpe", "Δ vs ctl",
      "p_holm", "DSR", "n_req (t=2)", "stav"], rows);
    document.querySelectorAll("#board tr.click").forEach(function (tr) {
      tr.addEventListener("click", function () { state.variant = tr.dataset.v; show("detail"); });
    });
    var H = (s.stats && s.stats.heatmap) || {}, syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"];
    table("heat", ["varianta"].concat(syms.map(function (x) { return x.replace("USDT", ""); })), order.map(function (k) {
      return "<tr>" + td(esc(k)) + syms.map(function (sym) {
        var c = (H[k] || {})[sym];
        if (!c) return td("", "na");
        var grey = c.n < 30;
        return td(sgn(c.net_mean) + " <span class=muted>(" + c.n + ")</span>", grey ? "na" : cls(c.net_mean));
      }).join("") + "</tr>";
    }));
  }

  // --- detail ------------------------------------------------------------------------------------
  function renderDetailSelect() {
    var keys = Object.keys(lb()).sort(), sel = $("variant");
    if (!state.variant) state.variant = store("arena.variant") || keys[0];
    if (sel.options.length !== keys.length) {
      sel.innerHTML = keys.map(function (k) { return "<option>" + esc(k) + "</option>"; }).join("");
    }
    sel.value = state.variant;
    loadVariant();
  }
  function chart(id, cfg) {
    if (!window.Chart) return;
    if (state.charts[id]) state.charts[id].destroy();
    state.charts[id] = new Chart($(id), cfg);
  }
  function color(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
  function loadVariant() {
    var v = state.variant;
    if (!v) return;
    store("arena.variant", v);
    fetch("data/variants/" + encodeURIComponent(v) + ".json?t=" + Date.now(), { cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : null; }).then(function (d) {
        if (!d) { table("d-trades", ["žádná data"], []); return; }
        function pts(c, key) { return c.t.map(function (t, i) { return { x: t, y: c[key][i] }; }); }
        chart("c-cum", { type: "line", data: { datasets: [
          { label: "net bps (kumul.)", data: pts(d.paper_curve, "net"), borderColor: color("--accent"), pointRadius: 0, borderWidth: 2 },
          { label: "gross bps (kumul.)", data: pts(d.paper_curve, "gross"), borderColor: color("--muted"), pointRadius: 0, borderWidth: 1 },
          { label: "kontrola net", data: pts(d.control_curve, "net"), borderColor: color("--neg"), borderDash: [4, 3], pointRadius: 0, borderWidth: 1 }
        ] }, options: { animation: false, maintainAspectRatio: false, parsing: false,
          scales: { x: { type: "linear", ticks: { callback: function (x) { return new Date(x).toISOString().slice(5, 10); } } } } } });
        var h = d.hist || { edges: [], counts: [] };
        chart("c-hist", { type: "bar", data: { labels: h.edges.slice(0, -1).map(String), datasets: [
          { label: "obchodů", data: h.counts, backgroundColor: color("--accent") }] },
          options: { animation: false, maintainAspectRatio: false, plugins: { legend: { display: false } } } });
        table("d-sym", ["symbol", "n", "net bps", "gross bps"], Object.keys(d.by_symbol || {}).map(function (k) {
          var x = d.by_symbol[k];
          return "<tr>" + td(k) + td(x.n) + td(sgn(x.net_mean), cls(x.net_mean)) + td(sgn(x.gross_mean)) + "</tr>";
        }));
        table("d-exit", ["důvod", "n"], Object.keys(d.exit_mix || {}).map(function (k) { return "<tr>" + td(k) + td(d.exit_mix[k]) + "</tr>"; }));
        table("d-trades", ["slot", "strana", "vstup", "výstup", "důvod", "net bps", "gross bps", "slip in", "slip out", "barů", "tainted"],
          (d.trades || []).map(function (t) {
            return "<tr>" + td(esc(t.slot_id)) + td(t.side) + td(ts(t.entry_time)) + td(ts(t.exit_time)) + td(t.exit_reason) +
              td(sgn(t.net_bps), cls(t.net_bps)) + td(sgn(t.gross_bps)) + td(num(t.entry_slip_bps)) + td(num(t.exit_slip_bps)) +
              td(t.bars_held) + td(t.tainted ? "ano" : "") + "</tr>";
          }));
      });
  }

  // --- náklady -----------------------------------------------------------------------------------
  function costRows(o) {
    return Object.keys(o || {}).map(function (k) {
      var c = o[k];
      return "<tr>" + td(k) + td(c.n) + td(num(c.fee_bps, 2)) + td(num(c.entry_slip_med, 2)) + td(num(c.entry_slip_p90, 2)) +
        td(num(c.exit_slip_med, 2)) + td(num(c.exit_slip_p90, 2)) + td(num(c.funding_bps, 2)) + "</tr>";
    });
  }
  function loadCosts() {
    fetch("data/costs.json?t=" + Date.now(), { cache: "no-store" }).then(function (r) { return r.ok ? r.json() : null; }).then(function (c) {
      if (!c) return;
      $("cmv").textContent = "cost_model_version " + c.cost_model_version + " · " + ts(c.generated_at);
      var head = ["symbol", "n", "fee bps", "entry slip med", "entry p90", "exit slip med", "exit p90", "funding bps"];
      table("costs-live", head, costRows(c.live));
      table("costs-paper", head, costRows(c.paper));
      var m = c.model_error || {};
      table("model-err", ["párů", "net med", "net p90", "vstup med", "výstup med"],
        ["<tr>" + td(m.n || 0) + td(num(m.net_med, 2)) + td(num(m.net_p90, 2)) + td(num(m.entry_med, 2)) + td(num(m.exit_med, 2)) + "</tr>"]);
      table("lat", ["druh", "n", "průměr ms"], Object.keys(c.latencies || {}).map(function (k) {
        return "<tr>" + td(k) + td(c.latencies[k].n) + td(num(c.latencies[k].avg_ms, 0)) + "</tr>";
      }));
    });
  }

  function renderLog(s) {
    table("log", ["čas", "úroveň", "komponenta", "zpráva"], (s.events || []).map(function (e) {
      return "<tr>" + td(ts(e.ts)) + td(e.level, e.level === "ERROR" ? "neg" : "") + td(esc(e.component)) +
        '<td style="text-align:left;white-space:normal">' + esc(e.msg) + "</td></tr>";
    }));
  }

  var saved = store("arena.tab");
  show(saved && $("tab-" + saved) ? saved : "overview");
  load();
  setInterval(load, 15000);
  setInterval(staleCheck, 5000);
})();
