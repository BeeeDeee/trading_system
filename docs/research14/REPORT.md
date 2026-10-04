# Výzkum 14 – Multi-timeframe pullback na BTC/ETH (1h): výsledek

2026-10-04 · pre-registrace v1.0 · hodinová data Binance (snapshot `binance_2026-10-03`) · holdout 2022-01 → 2026-08
otevřen jednou (commit `d208d8d`, otisk metodiky `4bf1c50a…`), bez pádu a bez rerunu.

## Verdikt

**Zamítnuto, 2 ze 6 kritérií.** Nákup BTC/ETH po silném 1h poklesu (z ≤ −1,5 za 6 h), když týdenní i denní trend
ukazují nahoru, držení 48 h (P = `MTF_L6_z15_H48_LONG`), vydělal na holdoutu po nákladech a fundingu
**CAGR +1,6 %, Sharpe 0,18**. Náhodné vstupy ve stejném trendovém režimu se stejným počtem obchodů a stejnou
dobou držení měly medián Sharpe 0,13 a 95. percentil 0,59. **Časování 1h pullbacku nepřidává nic měřitelného nad
samotný vyšší trend.** Forward test se podle §10 nespouští.

| Holdout 2022-01 → 2026-08 | P | B_RND (500) | B_NOHTF (čistá 1h MR) | B_HTF (jen trend W+D, long) | B_HOLD 50/50 |
|---|---|---|---|---|---|
| Sharpe | 0,18 | medián 0,13 / p95 0,59 | −1,04 | **0,74** | 0,34 |
| CAGR | +1,6 % | – | −41,6 % | **+18,4 %** | +3,2 % |
| Max. propad | 32 % | – | 92 % | 31 % | 68 % |
| Obchodů / průměr | 251 / +0,09 % | – | 1029 / −0,41 % | – | – |

| Kritérium (§8) | Hodnota | |
|---|---|---|
| 1. Sharpe > 0 a 90% CI průměru obchodu > 0 | 0,18; +0,09 % [−0,26; +0,45] % | ✗ |
| 2. Sharpe nad p95 B_RND | 0,18 vs 0,59 (P nad 58 % náhodných běhů) | ✗ |
| 3. Průměr obchodu nad B_NOHTF | +0,09 % vs −0,41 % | ✓ |
| 4. Náklady ×2 | Sharpe −0,05 | ✗ |
| 5. Oba podúseky | 2022–23: 0,94 / 2024–26: −0,33 | ✗ |
| 6. Průměr obchodu > 0 na BTC i ETH | +0,06 % / +0,11 % | ✓ |

DSR (N = 16) 0,26. Vývoj (2018-02 → 2021-12): P Sharpe 0,68, ale B_RND medián 0,69 – už ve vývoji na úrovni náhody.

## Co z toho plyne

**1. Edge je ve vyšším trendu, ne v 1h vstupu.** Být long celou dobu, kdy týdenní i denní trend ukazují nahoru
(B_HTF), dalo na holdoutu Sharpe 0,74 a CAGR +18 %, ve vývoji 1,14. Čekání na 1h pokles před vstupem jen
zkrátilo čas v trhu a výsledek zhoršilo; náhodný vstup ve stejném režimu dopadl stejně jako pullback.
(B_HTF nebyl předmětem testu, je to benchmark; trendový filtr na BTC/ETH ale už ve výzkumu 9 neprokázal
lepší Sharpe než držení, jen menší propad.)

**2. Čistá 1h mean reversion v kryptu prodělává.** Sázka proti silnému 1h pohybu bez trendového filtru
(B_NOHTF) měla −0,41 % na obchod a CAGR −42 %. Na hodinových horizontech jsou BTC a ETH spíš momentum.
Filtr vyššího trendu tuto ztrátu odstraní, ale nic nad nulu nepřidá.

**3. Post hoc pozorování (nic nedokazuje):** varianty s 24h oknem (`L24_*`) byly na holdoutu všechny kladné
(Sharpe 0,16–1,27), ve vývoji ale slabší než P, a nejlepší mají jen 22–61 obchodů za 4,7 roku. Pro ně se
náhodná nula nepočítala, takže není jasné, jestli přidávají něco nad B_HTF. Sekundární koš 9 altcoinů
s pravidly P: Sharpe 0,98 vs držení 0,23 (survivorship bias, bez nuly, jen popisně). Pokud by stálo za to
pokračovat, jako nová pre-registrace s forward testem, ne výběrem z tohoto holdoutu.

## Robustnost

- Po letech: 2022 −10 %, 2023 +47 %, 2024 −2 %, 2025 −20 %, 2026 (do 08) +4 %.
- Plnění o hodinu později: Sharpe 0,14. Bez fundingu: 0,28 (funding stojí ~0,06 % na obchod).
- Short strana se v P nepoužívá; LS varianty s L = 6 byly na holdoutu všechny ≤ 0,03.

Data: `results/dev.json`, `results/final.json`; audit dat `DATA_AUDIT.md`.
