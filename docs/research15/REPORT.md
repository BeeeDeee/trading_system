# Výzkum 15 – Multi-timeframe 4h / 1h / 15m na BTC/ETH: výsledek

2026-10-12 · pre-registrace v1.0 · 15m data Binance (snapshot `binance_2026-10-03`) · holdout 2022-01 → 2026-09
otevřen jednou (commit `2cb20923`, otisk metodiky `6c0d3f98…`), bez pádu a bez rerunu.

## Verdikt

**Zamítnuto, 4 ze 6 kritérií.** P = `MTF_BRK_N48_R30_LONG` (4h trend nahoru → 1h close nad maximem 48 h → 15m
potvrzení, stop pod low průrazové svíčky, cíl 3R, časový stop 48 h) vydělal na holdoutu po nákladech a fundingu
**CAGR +5,2 %, Sharpe 0,43, propad 23 %**, 393 obchodů. Nad nulou ho ale statistika neudrží (90% interval
průměrného obchodu zahrnuje 0) a při dvojnásobných nákladech zmizí. Forward test se podle §10 nespouští.

| Holdout 2022-01 → 2026-09 | P | B_RND (500) | B_NO15 (bez 15m) | B_NO4H (bez 4h) | B_4H (long při 4h↑) | B_HOLD 50/50 |
|---|---|---|---|---|---|---|
| Sharpe | **0,43** | medián −0,47 / p95 0,17 | 0,37 | −0,03 | 0,33 | 0,37 |
| CAGR | +5,2 % | – | +4,7 % | −2,3 % | +5,8 % | +4,8 % |
| Max. propad | 23 % | – | 20 % | 43 % | 39 % | 68 % |
| Obchodů / průměr | 393 / +0,152 % | – | 600 / +0,091 % | 650 / −0,005 % | – | – |

| Kritérium (§8) | Hodnota | |
|---|---|---|
| 1. Sharpe > 0 a 90% CI průměru obchodu > 0 | 0,43; +0,152 % [−0,093; +0,396] % | ✗ |
| 2. Sharpe nad p95 B_RND | 0,43 vs 0,17 (P nad 99,2 % náhodných běhů) | ✓ |
| 3. Průměr obchodu nad B_NO15 **a** B_NO4H | +0,152 % vs +0,091 % a −0,005 % | ✓ |
| 4. Náklady ×2 | Sharpe −0,01 | ✗ |
| 5. Oba podúseky | 2022–23: 0,54 / 2024–26: 0,36 | ✓ |
| 6. Průměr obchodu > 0 na BTC i ETH | +0,109 % / +0,193 % | ✓ |

DSR (N = 16) 0,12. Vývoj (2018-02 → 2021-12): P Sharpe 0,79 (B_RND medián 0,30, p95 0,93), na holdoutu 0,43.

## Co z toho plyne

**1. Na rozdíl od výzkumu 14 přidávají vrstvy něco měřitelného.** P porazil 99 % náhodných vstupů ve stejném 4h
režimu se stejnou geometrií výstupu a oba odlehčené benchmarky: bez 15m potvrzení (0,37) i bez 4h filtru (−0,03).
Velikost přínosu je ale malá: Sharpe 0,43 vs 0,33 pro prosté držení long při 4h trendu nahoru (B_4H) a vs 0,37
pro držení. Hlavní hodnotou je nižší propad (23 % vs 39 % a 68 %), ne vyšší výnos.

**2. Žádná statistická jistota.** 393 obchodů s úspěšností 35 % a průměrem +0,15 % dává interval, který zahrnuje
nulu. Průměr v násobcích rizika je −0,01 R: stop (228) i časový stop (92) dohromady převáží cíle (73) a čistý zisk
nese asymetrie, ne přesnost vstupu.

**3. Křehké vůči nákladům a plnění.** Při 15 bps na stranu Sharpe −0,01. Plnění o jednu 15m svíčku později snížilo
Sharpe z 0,43 na 0,22. Pro strategii s 15m rozhodováním je to hlavní praktické omezení: edge by přežil jen s levným
plněním (maker, mikroskluz), které backtest neověřuje.

**4. Short a liquidity sweep nefungují.** Všech 8 `LS` variant je záporných (Sharpe −2,3 až −0,07), všech 8 `SWP`
variant také (−2,3 až −0,5). Kladné jsou jen `BRK` + `LONG` + R = 3 (0,43 a 0,37 pro N = 48 / 12); `BRK_N48_R15_LONG`
0,04. Z tvojí původní myšlenky tedy v datech obstála část „průraz struktury ve směru 4h trendu, jen long“, ne
„vybrání likvidity“ a ne short.

## Robustnost (jen hlášení, nevybírá se podle ní)

- Po letech (P): 2022 +7,5 %, 2023 +5,6 %, 2024 +11,3 %, 2025 +2,8 %, 2026 (do 09) −1,9 %.
- Bez fundingu: Sharpe 0,50 (funding stojí ~0,02 % na obchod). Ceny perpu místo spotu: holdout 0,43; celá historie
  perpu 2020-01 → 2026-09: Sharpe 0,56, CAGR +8,3 %, 599 obchodů.
- Sekundární koš (105 perpů, měsíční top 10 podle objemu, váha 1/10, 2020-05 → 2026-09; jen popisně): Sharpe 0,22,
  CAGR +1,9 %, 2012 obchodů. Po letech: 2020 −0,92, 2021 −1,52, 2022 +0,37, 2023 +0,85, 2024 +0,42, 2025 +0,94,
  2026 +1,78. Na altcoinech tedy pravidla v letech 2020–21 prodělávala a výsledek drží až od 2022.
- Rozpad výstupů P: stop 228, cíl 73, čas 92. Medián obchodu −0,99 %.

## Poznámky k metodice

- Kritéria 1 a 4 nesplnil P o malý kus (dolní mez −0,09 %, Sharpe −0,01 při ×2). Pre-registrace je binární a
  neupravuje se po faktu; úspěch by znamenal všech 6.
- B_RND má medián −0,47: náhodný vstup s touto geometrií (stop na vzdálenosti rizika, cíl 3R, 48 h) v 4h trendu
  prodělává. Přínos P nad náhodou tedy ukazuje, že průraz + potvrzení vybírají lepší chvíle, ne že geometrie
  sama vydělává.
- Pozorování z holdoutu (kladná `BRK_N12_R30_LONG`, koš po 2022) jsou post hoc. Pokud by stálo za to pokračovat,
  jen jako nová pre-registrace s forward testem na datech po 2026-09, ne výběrem z tohoto holdoutu.

Data: `results/dev.json`, `results/final.json`; audit dat `DATA_AUDIT.md`.
