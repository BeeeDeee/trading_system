# strategy_backtester_2026_sep

Lokální výzkumný framework pro poctivé vyhodnocení systematických long-only strategií na denních
datech amerických akcií a ETF (Sharadar, snapshot 2026-09-25).

**Stav: projekt uzavřen (2026-09-27).** Čtyři pre-registrované výzkumy, tři negativní výsledky
a jedna sada praktických závěrů pro pasivní portfolio.

## Závěr v jedné větě

Žádná z testovaných aktivních strategií po nákladech a pod poctivou validací nepřekonala
jednoduché pasivní portfolio; u pasivního portfolia na většině pravidel prakticky nezáleží –
rozhoduje zvolený podíl akcií, nízké náklady a disciplína.

## Výsledky

| Výzkum | Otázka | Výsledek |
|---|---|---|
| **1 – výběr akcií** | Překoná ansámbl strategií nad likvidními US akciemi (momentum, trend, breakout, reversal, low-vol, režim trhu; 3 027 kandidátů, walk-forward výběr) pasivní investici? | **Ne.** Ve vývoji (2005–2019) low-vol ansámbl Sharpe 0,93 vs. SPY 0,49, ale na předem uzamčeném holdoutu **2020–2026 CAGR 2,4 % vs. SPY 15,5 %**, Sharpe 0,01. Hypotéza zamítnuta. |
| **2 – třídy aktiv** | Překoná alokace přes 9 ETF (inverzní volatilita + trendový filtr na každé třídě) portfolio 60/40? | **Ne.** 2010–2026 Sharpe 0,52 vs. 0,85, rozdíl statisticky významně záporný; robustní na náklady, záměnu ETF i délku trendu. |
| **3 – pasivní pravidla** | Záleží na frekvenci rebalance, podílu akcií, zahraničních akciích, typu dluhopisů, zlatu? | Rebalance, typ dluhopisů a zahraničí: **bez významného rozdílu**. Podíl akcií: věc rizikové preference. Zlato 5–10 %: **významně vyšší Sharpe** (+0,04–0,07), s výhradou mimořádného období pro zlato. |
| **4 – aktivace rodin** | Dokáže meta-vrstva (režimy, momentum strategií, LightGBM, ridge) nad 87 strategiemi z 25 rodin poznat, kterou rodinu kdy aktivovat, a překonat trh i držení všech rodin najednou? | **Ne.** 2006–2026 primární LightGBM Sharpe 0,52 vs SPY 0,56 (CAGR 13,5 % vs 11,1 %, ale propad 73 %); veškerý náskok jen v 2020–2026, 2006–2019 výrazně pod SPY. Rank IC 0,034, k překonání trhu by bylo potřeba ~0,05, k „výraznému“ ~0,12. Oracle top-5 má Sharpe 3,4, ale tento potenciál je nepředvídatelný. |

Podrobně: [výzkum 1](docs/FINAL_REPORT.md) · [výzkum 2](docs/research2/REPORT.md) ·
[výzkum 3](docs/research3/REPORT.md) · [výzkum 4](docs/research4/REPORT.md).

## Co jsme se naučili

1. **Vývojové výsledky přeceňují.** Low-vol ansámbl prošel všemi branami (DSR 0,999, CI vůči
   benchmarku > 0) a přesto na holdoutu selhal. Statistika počítá jen formální pokusy, ne volbu
   rodin strategií podle známé literatury ani to, že období 2000–2019 jim přálo.
2. **Robustnostní testy varují dřív.** Slabší výsledek na univerzu S&P 500 byl první signál.
3. **Časování trhu podle zpožděných signálů (volatilita, klouzavé průměry) škodí při rychlých
   odrazech** (březen–červen 2020) a v klidných býčích letech.
4. **Náklady rozhodují:** krátkodobé strategie (reversal, breakout, týdenní rebalance) po nákladech
   ztrácely prakticky vše.
5. **Pasivní 60/40 bylo v letech 2010–2026 mimořádně silné** – laťka, kterou je těžké překonat.
6. **Pre-registrace a trezor fungují:** holdout zůstal čistý do posledního běhu a negativní výsledek
   je proto důvěryhodný.

## Framework

Znovupoužitelný, otestovaný (`uv run pytest`, 107 testů), běží na stroji s 3 GB RAM.

| Oblast | Modul | Co umí |
|---|---|---|
| Data | `qlab.data` | Sharadar → normalizované bars (total return, delistingy s protihodnotou akvizic, SPAC, třídy akcií), syntetický trh se známou pravdou |
| Univerza | `qlab.universe` | LIQ-N podle likvidity k datu, S&P 500 k datu |
| Enginy | `qlab.engine` | vektorový (rychlý screening) a ledger (USD, příkazy, poplatky); shoda 10⁻¹⁵ |
| Strategie | `qlab.strategies` | 6 rodin akciových strategií, multi-asset alokace, mřížky, rozvrhy bez nahlížení do budoucnosti |
| Výběr | `qlab.selection` | tvrdé filtry, robustní skóre podle okolí, korelační deduplikace, `select()` |
| Validace | `qlab.validation` | test úniku budoucnosti, walk-forward, PSR/DSR, PBO, bootstrap, SPA, registr pokusů, trezor na holdout |

Klíčové skripty:

```
scripts/download_sharadar.sh                      # stažení snapshotu (vyžaduje předplatné)
uv run python scripts/raw_to_parquet.py data/raw/sharadar_<datum>
uv run python scripts/build_bars.py data/parquet/sharadar_<datum>
uv run python scripts/audit_sharadar.py sharadar_<datum>
uv run python scripts/build_universe.py sharadar_<datum>
uv run python scripts/build_panel.py sharadar_<datum>
uv run python scripts/run_grid.py sharadar_<datum>          # výzkum 1: mřížka kandidátů
uv run python scripts/run_wfo.py sharadar_<datum> <grid>    # výzkum 1: walk-forward
uv run python scripts/final_evaluation.py sharadar_<datum>  # jednorázový holdout přes trezor
uv run python scripts/r2_build_panel.py sharadar_<datum>    # výzkum 2–3: panel ETF
uv run python scripts/r2_evaluate.py sharadar_<datum>
uv run python scripts/r3_evaluate.py sharadar_<datum>
uv run python scripts/r4_build_panel.py sharadar_<datum>  # výzkum 4: panel akcie + 24 ETF
uv run python scripts/r4_sleeves.py sharadar_<datum>      # výzkum 4: 87 rukávů
uv run python scripts/r4_evaluate.py sharadar_<datum>     # výzkum 4: meta-vrstva, statistika
```

## Kdyby se projekt otevíral znovu

- **Holdout 2020–2026 je spotřebovaný** pro všechny tři výzkumy. Novou hypotézu lze poctivě
  ověřit jen forward testem na datech po 2026-09-25 (framework to umí; data ETF jsou volně
  dostupná, akcie vyžadují obnovu předplatného Sharadaru).
- Nevyužitá data: fundamenty `SF1` (point-in-time), insideři, institucionální držby.
- Začít pre-registrací v `docs/`, commitnout před prvním výpočtem, vést registr pokusů.

## Dokumentace

- Výzkum 1: [specifikace](docs/PROJECT_SPECIFICATION.md), [revize původní spec](docs/SPEC_REVIEW.md),
  [data](docs/DATA_FINDINGS.md), [audit dat](docs/audit/sharadar_2026-09-25.md),
  fáze [0](docs/PHASE0_PLAN.md) · [1](docs/PHASE1_REPORT.md) · [2](docs/PHASE2_REPORT.md) ·
  [3](docs/PHASE3_REPORT.md) · [finální holdout](docs/FINAL_REPORT.md) ·
  [záznam trezoru a výsledky](docs/final/), [zmrazená konfigurace](configs/frozen_defaults.yaml)
- Výzkum 2: [pre-registrace](docs/research2/PREREGISTRATION.md), [audit ETF](docs/research2/DATA_AUDIT.md),
  [report](docs/research2/REPORT.md), [výsledky](docs/research2/results/)
- Výzkum 3: [pre-registrace](docs/research3/PREREGISTRATION.md), [report](docs/research3/REPORT.md),
  [výsledky](docs/research3/results/)
- Výzkum 4: [pre-registrace](docs/research4/PREREGISTRATION.md), [report](docs/research4/REPORT.md),
  [výsledky](docs/research4/results/)
- Archiv: [původní specifikace v1](docs/archive/PROJECT_SPECIFICATION_v1.md)

## Data

Data nejsou v gitu. Snapshot `sharadar_2026-09-25` (raw zipy, Parquet, odvozená data) leží na
serveru v `data/`; API klíč mimo repozitář v `~/.config/sharadar/api_key`.
