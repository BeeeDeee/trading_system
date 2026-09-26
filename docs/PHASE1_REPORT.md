# Fáze 1 – data a audit

**Stav 2026-09-26: dokončeno, brána splněna.** Spec §15: audit bez nevysvětlených anomálií, sanity
check SPY v toleranci, parita enginů na reálných datech, výchozí hodnoty zmrazené.

## Pipeline (reprodukovatelně ze snapshotu)

```
scripts/download_sharadar.sh                               # raw zipy (předplatné)
uv run python scripts/raw_to_parquet.py data/raw/sharadar_2026-09-25
uv run python scripts/build_bars.py data/parquet/sharadar_2026-09-25        # ~3 min
uv run python scripts/audit_sharadar.py sharadar_2026-09-25                  # ~2 min
uv run python scripts/build_universe.py sharadar_2026-09-25                  # ~30 s
uv run python scripts/build_panel.py sharadar_2026-09-25                     # ~1 min
uv run python scripts/run_benchmarks.py sharadar_2026-09-25                  # ~20 s
```

Vše běží na stroji s 3 GB RAM (DuckDB s limitem paměti, bars po částech, panel jako memory-map).

## Kroky

| Krok | Výsledek |
|---|---|
| 1.1 Bars | 15 725 titulů (domácí kmenové akcie, primární třída), 37,3 mil. řádků. Delistingy klasifikované podle akcí v poslední obchodní den (shoda 11 355 / 11 355); akvizice s protihodnotou (cash + akcie), 448 likvidací SPAC oddělených od bankrotů. |
| 1.2 Audit | [`docs/audit/sharadar_2026-09-25.md`](audit/sharadar_2026-09-25.md), závěry v [`DATA_FINDINGS.md`](DATA_FINDINGS.md) §6. Rekonstruovaný cap-weighted S&P 500 sleduje SPY v každém roce do 0,95 p.b. |
| 1.3 Univerza | LIQ1000: každý den přesně 1 000 titulů od 1999, celkem 4 580 titulů v historii. SP500_PIT: 492–505 členů/den (chybí sekundární třídy a ADR). Test úniku budoucnosti na syntetice. |
| 1.4 Náklady, panel, benchmarky | Odhad spreadu z OHLC zamítnut (šum), náklady podle pořadí likvidity. Panel 7 228 × 4 580 (1,1 GB, memory-map). Benchmarky za vývojové období níže. |
| 1.5 Zmrazení | [`configs/frozen_defaults.yaml`](../configs/frozen_defaults.yaml) + test, který hlídá shodu s kódem. |

## Benchmarky – vývojové období 2000-01-03 … 2019-12-31

Holdout (od 2020) zůstává zamčený ve vaultu.

| Benchmark | CAGR | Volatilita | Sharpe | Max. propad | Obrat/rok | Náklady |
|---|---|---|---|---|---|---|
| SPY_TR | 6,0 % | 19,0 % | 0,31 | 55,2 % | – | – |
| EW_UNIV (LIQ1000, měsíčně) | 6,4 % | 22,4 % | 0,32 | 61,0 % | 164 % | 28 bps/rok |
| BH_UNIV (od 2000-01-03) | 8,4 % | 19,6 % | 0,43 | 54,8 % | – | – |
| CASH (T-bill) | 1,7 % | 0,1 % | – | 0 % | – | – |

- **Parita enginů na reálných datech:** max. rozdíl denních výnosů vektorový vs. ledger = 4·10⁻¹⁵.
- **Rychlost:** vektorový engine 2,4 s na 20 let × 4 580 titulů (měsíční rebalance), ledger 11 s.
- **Limit propadu 45 %** (spec §9.4) nesplňuje ani jeden investovaný benchmark, jak spec předpokládá.

## Co z toho plyne pro Fázi 2

- Laťka: Sharpe ~0,3 (SPY, EW) a CAGR 6–8 % za 2000–2019 po nákladech.
- Obrat EW benchmarku 164 %/rok stojí ~28 bps/rok. Strategie s týdenní rebalancí a obratem
  1 000 %+ zaplatí stovky bps ročně; nákladový model to zachytí.
