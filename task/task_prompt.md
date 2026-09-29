You are running one daily step of a PAPER-TRADING experiment (no real money, no broker). Each evening you score 20 US stocks/ETFs with three independent lenses; a fixed Python engine then trades 19 strategy variants from those scores (plus a random baseline), each with its own virtual $10,000. Goal: after ~2 months, compare which rules and which lenses have any edge vs SPY buy & hold and vs random picks.

All state lives in the database of this artifact: https://claude.ai/artifact/CgJr4SMobQ7AcnV2y6RUbG . The page renders it automatically; never republish the page.

Write every human-readable text field (notes, messages, regime label/summary, score notes, reasons, invalidation) in CZECH, short and concrete.

RELIABILITY RULES (most important):
- Every run ends by writing runs/<today> with its status, whatever happens (see step 8). A run that fails still writes runs/<today> with status "failed" and a Czech message saying exactly what failed.
- Never invent or estimate prices. If data is missing or fails checks, do not trade on it.
- Never edit engine.py. If it crashes, record the traceback summary in runs/<today> (status "failed") and stop.

## 0. Setup
- Load ArtifactData with ToolSearch ("select:ArtifactData"). Work in a fresh folder, e.g. ./run. Note the start time (UTC ISO).
- Read config/main -> config.json. Read engine/main -> write its "source" field to engine.py and verify sha256 == its "sha256" field (mismatch = failed run).
- Read history/main if it exists ("data" field = {ticker: [[date, open, high, low, close], ...]}) -> history.json.
- Read the latest document in "days" (query, order_by date desc, limit 1) -> prev.json (only the document's data). None on the very first run; then pass "-" as PREV.
- Today = current date in New York. If today is a weekend or NYSE holiday: write runs/<today> = {status "closed", message "Burza zavřená"} and stop. If days/<today> already exists, stop without writing anything.

## 1. Catch up missed trading days (only if needed)
If prev.json's date is older than the previous NYSE trading day, some evenings were missed. For each missed trading day D in order (oldest first):
- Take D's open/high/low/close for the universe from the price-history pages described in step 2 (the same pages list past days) -> today_D.json.
- python3 engine.py check config.json history.json today_D.json check_D.json; if "errors" is non-empty, stop the catch-up and fail the run (write runs/<today> status "failed", message naming the day and errors).
- python3 engine.py history-append history.json today_D.json history.json
- python3 engine.py settle config.json prev.json today_D.json settled_D.json
- echo '{"scores": {}}' > empty.json ; python3 engine.py plan config.json settled_D.json empty.json - out_D.json
- Add to out_D.day: regime {"label": "Doplněno zpětně", "summary": "Večerní běh chyběl; den dopočítán z historických cen bez nových rozhodnutí."}, notes "Doplněno zpětně". Save trades, prices/<D>, days/<D>, and runs/<D> = {status "catchup", message "Doplněno zpětně během běhu <today>", checks: check_D}. Then prev.json = out_D.day.
No scores and no new decisions are made for missed days (that would use hindsight).

## 2. Today's market data (only data up to today's close)
- Primary source: stockanalysis.com price-history pages, one WebFetch per universe ticker:
  ETFs (SPY QQQ IWM TLT GLD XLK XLF XLE XLV XLP XLU): https://stockanalysis.com/etf/<ticker lowercase>/history/
  Stocks (AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA JPM): https://stockanalysis.com/stocks/<ticker lowercase>/history/
  Ask the fetch to "return the most recent 10 rows of the daily price history table verbatim (Date, Open, High, Low, Close, Adj. Close, Volume)". Use the plain Close column, NEVER Adj. Close. The top row's date must equal today; if it does not (page not updated yet), wait ~10 minutes and re-fetch once.
  Do NOT use Stooq (it blocks automated fetching). Only if stockanalysis.com fails for a ticker, use another reputable quote page for that ticker and record it in "source".
- Write today.json = {"date": today, "open": {T: x}, "high": {...}, "low": {...}, "close": {...}}. Leave out tickers without data.
- The same pages give recent history. Backfill only if history.json is missing or a ticker has fewer than 50 rows before today: ask the same page for the most recent 60 rows verbatim and keep rows before today as [date, open, high, low, close].
- Cross-check: find today's closing price of SPY and of one individual stock (rotate daily through AAPL, MSFT, NVDA, AMZN, GOOGL, META, AVGO, TSLA, JPM) from an independent source (e.g. investing.com, finance.yahoo.com, nasdaq.com or a major news report). diff_pct = (primary_close / other_close − 1) × 100, computed in Python. If |diff_pct| > 0.3 for either, re-fetch the primary once; if it still disagrees, fail the run.
- python3 engine.py check config.json history.json today.json check.json
  If "errors" is non-empty: try to fix the data once (re-fetch, other source). If errors remain, fail the run (runs/<today> status "failed", message = the errors) and write nothing else. Warnings are allowed: continue, and carry them into runs/<today>.
- python3 engine.py history-append history.json today.json history.json
- python3 engine.py features history.json features.json   (prints the indicator table)

## 3. Settle today
python3 engine.py settle config.json prev.json today.json settled.json
Executes yesterday's orders at today's open, stops, max-hold exits, marks every variant to market with fees and slippage. settled.json -> day.variants.volny holds the "Claude volně" positions.

## 4. Research
a) Market (3–5 searches): VIX level and move, US 10Y yield move, breadth and sector leadership, main market news today; earnings dates of universe names and macro events (CPI, FOMC, jobs) in the next 5 trading days.
   Summarize regime.label (e.g. "Risk-on, trend", "Neutrální, konsolidace", "Risk-off, zvýšená volatilita") and regime.summary (1–2 sentences).
b) One news search PER universe ticker (20 searches), e.g. "<TICKER> stock news", focused on the last ~48 hours. For ETFs search the theme they track (e.g. XLE → oil/energy sector, TLT → Treasuries/Fed, GLD → gold). For each ticker note briefly in a working file (not saved):
   - what actually happened (new information only; ignore generic commentary, price-target churn and old stories),
   - whether it is better or worse than what the market expected (consensus, guidance, prior narrative),
   - whether it matters for the next 1–10 days or only long term,
   - how the price already reacted today (compare with ret1 from the features table): a strong move in the direction of the news means much of it is already priced in.

## 5. Score EVERY universe ticker -> scores.json
{"date": today, "regime": {...}, "scores": {T: {"trend", "mr", "news", "conviction", "event", "note"}}}
Score each lens INDEPENDENTLY, as if three different analysts wrote them. Do not make them agree. 0 is a normal, common score.
- trend (−2..+2): direction and quality of the 1–4 week trend. +2 = above rising SMA20 and SMA50 with strong relative strength vs SPY; −2 = mirror image.
- mr (−2..+2): expected snap-back over 1–3 days. +2 = sharply oversold short term (RSI14 < 30, big 5-day drop, near the 20-day low) without broken fundamentals; −2 = extremely stretched upward; otherwise 0 or ±1.
- news (−2..+2): the part of fresh news (last ~48 h) that is NOT yet priced in, from step 4b. Judge surprise vs expectations, not whether the news sounds good: good news that was expected, or that the price already fully jumped on, scores 0. +2 = a clear positive surprise with short-term relevance and little price reaction so far; −2 = the mirror image. No relevant new information → 0.
- conviction (1–5): will the ticker outperform SPY over the next 5 trading days? 1 = clearly worse, 3 = no view, 5 = clearly better.
- event (true/false): earnings or another binary event within the next 5 trading days.
- note: one Czech sentence with the main reason.
Use the features table for trend and mr; never compute indicators in your head. SPY must be scored too (one variant uses SPY's trend score as a market filter).

## 6. Decisions of the "Claude volně" variant -> free.json
Using its settled positions and everything above, decide freely for tomorrow's open:
{"decisions": [{"ticker", "action": "BUY"|"SELL"|"HOLD"|"SKIP", "conviction": 1–5, "weight_pct" (BUY, max 20), "stop_price" (optional, BUY), "new_stop" (optional, HOLD; may only raise the stop), "reason", "invalidation"}]}
Every held position gets HOLD or SELL. A BUY needs conviction ≥ 3 and an expected move clearly above the round-trip cost (~0.2 %). Add 1–3 SKIP entries for rejected setups. Cash is fine; don't force trades.

## 7. Plan and save
python3 engine.py plan config.json settled.json scores.json free.json out.json
Add "regime" and "notes" (1 sentence: data problems or anything unusual, else "—") to out.day.
Save with ArtifactData batches (max 50 writes each), days/<today> LAST:
1. trades/<variant>-<ticker>-<entry_date> for each item in out.trades.
2. prices/<today> = today.json; scores/<today> = scores.json; history/main = {"updated": today, "data": <history.json>}.
3. days/<today> = out.day. Re-read it once to confirm it saved.

## 8. Run status -> runs/<today> (ALWAYS, also on failure)
{"date": today, "status": "ok" | "warning" | "failed", "started_at", "finished_at" (UTC ISO), "source": "stockanalysis.com" (plus any per-ticker fallback), "checks": <check.json or {}>, "spy_crosscheck": {"primary": x, "other": y, "other_source": "...", "diff_pct": z}, "stock_crosscheck": {"ticker": T, "primary": x, "other": y, "other_source": "...", "diff_pct": z}, "catchup_days": [...], "message": one Czech sentence}
status "warning" when check.json has warnings, a per-ticker fallback source was used, or days were caught up. Otherwise "ok".

## 9. Final reply (it is sent to the user as a notification)
First line exactly one of: "OK – <date>", "UPOZORNĚNÍ – <date>", "CHYBA – <date>".
Then at most 4 short Czech lines: the 3 best and 2 worst variants by cumulative return, SPY and random; number of fills today; any data problem and what the user should do. On CHYBA, say exactly what failed.
