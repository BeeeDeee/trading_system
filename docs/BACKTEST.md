# Backtest (explorativní, mimo experiment)

Spustí všech 140 slotů (brk, obi, ll, ctl; xp ne) nad historií z data.binance.vision stejným kódem jako živý paper:
strategie, `SlotRunner`, `PaperExecutor` a `TrailingSim` nad aggTrades. Výsledek je záložka **Backtest** v kartě Arena.

```
sudo systemd-run --unit=arena-backtest-X --uid=arena --gid=arena --nice=10 -p MemoryMax=3G \
  -p WorkingDirectory=/opt/arena5m -E ARENA_CONFIG_DIR=/opt/arena5m/config \
  /opt/arena5m/.venv/bin/arena -v --log-file /srv/arena/log/backtest.log \
  backtest --from 2026-01-01 --to 2026-07-21 --label oos-2026-01-01_07-21 --workers 3
```
- `--no-fetch` použije stažená data, `--fetch-only` jen stáhne, `--symbols BTCUSDT,ETHUSDT` zúží běh, `--force` jede i s
  chybějícími dny. Data: `/srv/arena/bt/data` (ticky jako npz po dnech), běhy: `/srv/arena/bt/runs/<label>/trades.jsonl.gz`.
- Web je statický: výsledek je v `/srv/arena/site/data/backtest/<label>/`, tlačítko spuštění neexistuje.
- Běh trvá hodiny: ticky se podávají jen symbolům s otevřenou pozicí, ale u BTC je otevřená skoro vždy.
  Změřeno: 10 dní BTC (3 dny + 7 dní rozběhu ctl) ~ 6 min.

## Omezení (jsou i v záložce)
- Fill bez knihy: první obchod po latenci 250 ms + `extra_slippage_bps`, bez dopadu na cenu. Náklady jsou optimistické.
  Sloupce net při 1,5× a 2× nákladech ukazují citlivost.
- `obi` z `bookDepth` (30s snímky), živě z lokální knihy. Filtry (tick, step, minNotional) jsou dnešní.
- Funding z `fundingRate` (měsíční soubory), mark ≈ close baru.
- Prvních 7 dní před `--from` je rozběh (ctl potřebuje týden signálů cílové strategie); jejich obchody se zahodí.
- Obchody s `tainted` (např. mezera v aggTrades) se ze statistik vynechají a spočítají se.
- Holm a DSR jsou jen v rámci backtestu (30 variant). **Do rozhodování na dni 45/90 nevstupují** (SPEC §14.2).
- Priory `obi` vznikly in-sample na 2026-07-22 až 2026-10-19. Starší období je mimo vzorek, ale parametry
  `brk`/`ll` mohly vzniknout z dřívějších výzkumů 15–17 (zkontrolovat před interpretací).
