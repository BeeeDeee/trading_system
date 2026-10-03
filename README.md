# strategy_backtester_2026_sep

Lokální výzkumný framework pro poctivé vyhodnocení systematických long-only strategií na denních
datech amerických akcií a ETF (Sharadar, snapshot 2026-09-25).

**Stav: projekt uzavřen (2026-09-27), výzkumy 8–11 doplněny 2026-10-03.** Jedenáct pre-registrovaných výzkumů, deset negativních výsledků
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
| **5 – STR-TF** | Vydělává krátkodobý reversal s trendovým filtrem (z-skóre poklesu vůči volatilitě, vstup na open, výstup nad SMA) po nákladech? | **Ne.** Vývoj 1999–2014 Sharpe 1,02, ale DSR 0,52 (N = 993) a výběr je izolovaná špička; validace 2015–2019 **Sharpe 0,32**, 53 % zisku z let 1999–2002. Zamítnuto 4 ze 7 kritérií. |
| **6 – STR-TF při vysokém VIX** | Vydělává reversal, pokud se vstupuje jen při VIX > 25 (Nagel 2012)? | **Ne jako strategie** (DSR 0,0006, CAGR 3 %, investováno 9 % času). **Efekt ale vypadá reálně:** brána obrací čistý výnos na obchod z −2 na +54 bps, stabilně ve všech podúsecích 2003–2026 a rostoucí s prahem. Výzkum 7 ale ukázal, že jde hlavně o odraz celého trhu po vysokém VIX. |
| **7 – SPY + VIX rukáv** | Překoná SPY portfolio, které při VIX > 25 přesune část kapitálu do reversal pozic? | **Ne.** 2003–2026 Sharpe 0,686 vs. SPY 0,685, aktivní výnos +0,25 % p. a. při tracking erroru 6 %, při nákladech ×2 −1,3 p. b. p. a. Zisk rukávu ve výzkumu 6 byl odraz trhu po vysokém VIX, ne výběr akcií. 0 ze 4 kritérií. |
| **8 – funding carry (krypto)** | Vydělává delta-neutrální long spot + short perp na Binance po nákladech víc než T-bill (2023–2026, pre-registrováno, jednou otevřený holdout)? | **Ne.** Ve vývoji 2020–2022 +12 % nad T-bill (Sharpe 8), na holdoutu **+0,7 % (BTC+ETH vždy) a +0,3 % (s filtrem)**, 90% interval obsahuje 0, od 2024-07 záporné. Funding se po nástupu Etheny zmenšil k úrovni T-billu (2/3 kapitálu je nominál). Rotace altcoinů neprošla ani vývojem. |
| **9 – trend a momentum (krypto)** | Zlepší trendový filtr BTC + ETH Sharpe proti držení? Překoná týdenní momentum altcoinů rovné váhy likvidního univerza (2022–2026)? | **Ne.** Vybraný filtr (SMA20) Sharpe 0,36 vs 0,34, CI obsahuje 0. Všech 8 filtrů ale snížilo propad (30–48 % vs 68 %), to je jen popisné. Altcoiny 2022–2026: rovné váhy top 20 −45 % ročně, žádné momentum nemělo kladný CAGR. |
| **10 – doplňující se strategie** | Najdou se dvě strategie, které se doplňují (jedna i short), a vyplatí se je kombinovat nebo přepínat? Akcie (dvojice z 87 rukávů, SPY + trend long/short na ETF, jen ≤ 2019) a krypto (trend se shortem přes perp). | **Ne.** Long-only akciové dvojice v krizi korelují ≥ +0,53, doplňují se jen akcie a dluhopisy (= 60/40). Oracle přepínač Sharpe 1,70 vs statická 0,54, naivní přepínače zachytí ~1–2 %. Trend long/short na ETF vydělal v roce 2008, 2010–2019 ztrácel; verze bez shortu lepší. Krypto: ΔSharpe +0,21, CI [−0,31; +0,75], DSR 0,02; efekt shortu mění znaménko mezi vývojem a holdoutem. |
| **11 – ML z fundamentů a toků** | Vybere LightGBM z fundamentů, insiderů a změn držení 13F měsíčně 50 akcií LIQ1000 lépe než SPY, rovné váhy a stejný model jen z cen? | **Ne (0 ze 6).** 2020-01 → 2026-08 Sharpe 0,57 vs SPY 0,82 (CI rozdílu [−0,67; +0,12]), pod EW i pod modelem jen z cen (0,72). Rank IC ~0,025 (t ≈ 2), obrat 15× ročně. Fundamenty a 13F nepřidaly nic proti cenám. |

Podrobně: [výzkum 1](docs/FINAL_REPORT.md) · [výzkum 2](docs/research2/REPORT.md) ·
[výzkum 3](docs/research3/REPORT.md) · [výzkum 4](docs/research4/REPORT.md) · [výzkum 5](docs/research5/REPORT.md) · [výzkum 6](docs/research6/REPORT.md) · [výzkum 7](docs/research7/REPORT.md) · [výzkum 8](docs/research8/REPORT.md) · [výzkum 9](docs/research9/REPORT.md) · [výzkum 10](docs/research10/REPORT.md) · [výzkum 11](docs/research11/REPORT.md).

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

Znovupoužitelný, otestovaný (`uv run pytest`, 124 testů), běží na stroji s 3 GB RAM.

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
uv run python scripts/r5_build_panel.py sharadar_<datum>  # výzkum 5: panel STR-TF
uv run python scripts/r5_grid.py sharadar_<datum>         # výzkum 5: mřížka 972 konfigurací
uv run python scripts/r5_evaluate.py sharadar_<datum>     # výzkum 5: výběr, robustnost, kritéria
uv run python scripts/r5_final.py sharadar_<datum> validation|late
uv run python scripts/r6_evaluate.py sharadar_<datum>     # výzkum 6: reversal při vysokém VIX
uv run python scripts/r7_evaluate.py sharadar_<datum>     # výzkum 7: jádro SPY + VIX rukáv
uv run python scripts/r8_download.py binance_<datum>      # výzkum 8: Binance perp/spot/funding/mark (data.binance.vision)
uv run python scripts/r8_build_panel.py binance_<datum>   # výzkum 8: panel + T-bill (FRED)
uv run python scripts/r8_audit.py binance_<datum>         # výzkum 8: audit dat
uv run python scripts/r8_dev.py binance_<datum>           # výzkum 8: 17 kandidátů na vývoji
uv run python scripts/r8_final.py binance_<datum> freeze|run   # výzkum 8: jednorázový holdout
uv run python scripts/r9_download.py binance_<datum>      # výzkum 9: všechny spotové páry USDT
uv run python scripts/r9_build_panel.py binance_<datum>   # výzkum 9: panel + audit
uv run python scripts/r9_dev.py binance_<datum>           # výzkum 9: 26 kandidátů na vývoji
uv run python scripts/r9_final.py binance_<datum> freeze|run   # výzkum 9: jednorázový holdout
uv run python scripts/r10_pairs.py sharadar_<datum> [posthoc2004]  # výzkum 10 A: dvojice rukávů (≤ 2019)
uv run python scripts/r10_tsmom.py sharadar_<datum>       # výzkum 10 B: SPY + TSMOM long/short (≤ 2019)
uv run python scripts/r10_crypto_dev.py binance_<datum>   # výzkum 10 C: krypto long/short, vývoj
uv run python scripts/r10_crypto_final.py binance_<datum> freeze|run   # výzkum 10 C: jednorázový holdout
uv run python scripts/r11_dev.py sharadar_<datum>         # výzkum 11: ML z fundamentů/insiderů/13F, vývoj 2010–2019
uv run python scripts/r11_final.py sharadar_<datum> freeze|run   # výzkum 11: jednorázové finále 2020–2026
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
- Výzkum 5: [pre-registrace](docs/research5/PREREGISTRATION.md), [report](docs/research5/REPORT.md),
  [výsledky](docs/research5/results/)
- Výzkum 6: [pre-registrace](docs/research6/PREREGISTRATION.md), [report](docs/research6/REPORT.md),
  [výsledky](docs/research6/results/)
- Výzkum 7: [pre-registrace](docs/research7/PREREGISTRATION.md), [report](docs/research7/REPORT.md),
  [výsledky](docs/research7/results/)
- Archiv: [původní specifikace v1](docs/archive/PROJECT_SPECIFICATION_v1.md)

## Data

Data nejsou v gitu. Snapshot `sharadar_2026-09-25` (raw zipy, Parquet, odvozená data) leží na
serveru v `data/`; API klíč mimo repozitář v `~/.config/sharadar/api_key`.
