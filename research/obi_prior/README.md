# obi_prior: explorativní analýza pro rodinu `obi` (SPEC §5.5)

Datum: 2026-10-21. Data: 2026-07-22 až 2026-10-19 (90 dní), BTC/ETH/SOL/XRP/DOGE USDⓈ-M perpy.

- `fetch.py OUT.npz 90`: 5m klines (REST `/fapi/v1/klines`), `bookDepth` a `metrics` z data.binance.vision (~15 min).
- `fetch_oi.py OI.npz`: `metrics` znovu se `sum_open_interest` (kontrakty), `fetch.py` bral jen `_value`.
- `analyze.py OUT.npz OI.npz`: IC, decily, podmíněné signály → `out_analyze.txt`.
  Pozor: permutační p-hodnoty v tomto výstupu jsou příliš optimistické (časy losované nezávisle po symbolech).
- `analyze2.py` (čte `data90.npz`, `oi90.npz` z aktuálního adresáře): SE clusterovaná po dnech UTC, horizont 8 h,
  stabilita po třetinách → `out_analyze2.txt`. Tohle jsou čísla v SPEC §5.5.

Spuštění: `uv run --no-project --with httpx --with numpy python -I fetch.py data90.npz 90` atd.

## Závěr
- Taker flow (delta, CVD) vysvětluje výnos téhož baru, budoucí výnos ≥ 1 h nepredikuje. „Absorpce“ vyšla opačně.
- Blízká kniha (±0,2 %) a široká (±5 %) nic. OI nic.
- Nerovnováha v pásmu ±1 % (z-skóre vůči poslednímu dni) je jediná věc s hrubým výnosem nad náklady:
  +18 bps / 4 h a +26 bps / 8 h při z ≥ 2,5, kladná na 4–5 z 5 symbolů a ve všech třetinách období.
- Je to in-sample výsledek z ~60 testů. Slouží jako prior pro zmrazené parametry, ne jako důkaz hrany.

## Nástrahy dat
- `metrics.create_time` = začátek periody (hodnota platí na konci). `bookDepth.timestamp` = reálný čas snapshotu.
- `bookDepth`: snapshot ~každých 30 s, kumulativní notional pro ±0,2/1/2/3/4/5 %.
