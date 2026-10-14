# Výzkum 16 – Dávka 1: osm rodin intradenních strategií na BTC/ETH (15m): výsledek

2026-10-14 · pre-registrace v1.0 · 15m data Binance (snapshot `binance_2026-10-03`) · holdout 2022-01 → 2026-09
otevřen jednou (commit `f7543cea`, otisk metodiky `da8eebdd…`), bez pádu a bez rerunu.

## Verdikt

**Zamítnuto, všech 8 rodin.** Vývojová brána (§7) vyřadila 5 rodin, holdout zamítl zbylé 3. FUND a LEAD splnily
5 ze 6 kritérií a obě padly na stejném: 90% interval průměrného obchodu sahá těsně pod nulu. ORB, nejsilnější
rodina vývoje, na holdoutu nesplnila nic. Forward test se podle §10 nespouští.

### Vývoj 2018-02 → 2021-12 (FUND 2020-04 →), B_RND 1000 běhů

| Rodina | P_f | Sharpe | ×2 náklady | p B_RND | Brána |
|---|---|---|---|---|---|
| ORB | `ORB_H4_LONG` | 1,67 | 0,85 | 0,001 | ✓ |
| FUND | `FUND_Q95_LONG` | 1,88 | 1,80 | 0,001 | ✓ |
| LEAD | `LEAD_K3_LONG` | 1,28 | 0,93 | 0,003 | ✓ |
| CME | `CME_G2` | 0,24 | −0,01 | 0,21 | ✗ |
| SQZ | `SQZ_Q10_LS` | −0,07 | −1,28 | 0,03 | ✗ |
| PULL | `PULL_L40_LONG` | −0,47 | −1,03 | 0,61 | ✗ |
| SHOCK | `SHOCK_K4_LONG` | −1,11 | −1,74 | 0,48 | ✗ |
| RANGE | `RANGE_M25_LONG` | −1,90 | −2,60 | 0,98 | ✗ |

### Holdout 2022-01 → 2026-09 (k = 3)

| Kritérium (§8) | ORB `H4_LONG` | FUND `Q95_LONG` | LEAD `K3_LONG` |
|---|---|---|---|
| 1. Sharpe > 0 a 90% CI obchodu > 0 | −0,88; −0,10 % [−0,17; −0,03] ✗ | 0,47; +0,45 % [−0,11; +0,99] ✗ | 0,72; +0,43 % [−0,006; +0,88] ✗ |
| 2. B_RND, Holm | p = 0,098 (hranice 0,05) ✗ | p = 0,002 (0,017) ✓ | p = 0,014 (0,025) ✓ |
| 3. Náklady ×2 | −2,17 ✗ | 0,32 ✓ | 0,47 ✓ |
| 4. Plnění o svíčku později | −0,99 ✗ | 0,44 ✓ | 0,76 ✓ |
| 5. Podúseky 2022–23 / 2024–26 | −0,51 / −1,17 ✗ | 1,00 / 0,03 ✓ | 1,07 / 0,48 ✓ |
| 6. BTC i ETH | −0,07 % / −0,14 % ✗ | +0,73 % / +0,16 % ✓ | jen ETH +0,43 % ✓ |
| **Splněno** | **0 ze 6** | **5 ze 6** | **5 ze 6** |
| CAGR / max. propad | −24,7 % / 76 % | +9,8 % / 47 % | +3,6 % / 3,3 % |
| Obchodů | 2260 | 267 | 81 |
| DSR (N = 30 / 46) | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 |

Pro srovnání B_HOLD 50/50 Sharpe 0,37 (CAGR +4,8 %, propad 68 %), B_4H LS −0,11.

## Co z toho plyne

**1. ORB byl býčí artefakt vývoje.** Long po průrazu ranního rozpětí vydělával v letech 2019–2021 (Sharpe 1,7,
v roce 2020 +241 %) a porazil náhodné vstupy ve stejném okně dne. Na holdoutu prodělává každý rok kromě 2023 a při
dvojnásobných nákladech má Sharpe −2,2. Všech 8 variant ORB je na holdoutu záporných. Intradenní momentum v kryptu
2019–21 nebylo stabilní vlastností trhu.

**2. FUND a LEAD jsou „téměř“, ale stojí na málo obchodech.** Obě porazily nulu se stejnou stranou i po Holmově
korekci, přežily dvojnásobné náklady, zpoždění plnění a oba podúseky. Neprošly jen intervalem průměrného obchodu:
- **LEAD** (long ETH 4 h po skoku BTC ≥ 3σ, když ETH zaostane): 81 obchodů za 4,75 roku, dolní mez −0,006 %.
  Kladný je každý rok holdoutu (1,47 / 0,62 / 0,36 / 0,48 / 0,68) a všechny 4 varianty LEAD (0,37 až 0,82).
  Propad 3,3 %, ale ETH rukáv je v pozici jen zlomek času, takže CAGR je +3,6 %.
- **FUND** (long 72 h, když je funding pod 5. percentilem 90 dní): 267 obchodů, průměr +0,45 %, ale rok 2025
  Sharpe −0,91 a druhý podúsek jen 0,03. Propad 47 %: je to v podstatě long BTC/ETH ve stresu, s plnou volatilitou.

**3. DSR je u všech nula.** Holdout Sharpe 0,5–0,7 je proti očekávanému maximu z 30 (natož 46) zkoušek malý.
Přesně proto je kritérium 1 tvrdé a výsledek se nezměkčuje.

**4. Krátké držení zabíjí náklady, jak se čekalo.** RANGE, SHOCK, SQZ a PULL (držení 4–48 h, desítky až stovky
obchodů ročně) jsou záporné na vývoji i na holdoutu ve všech variantách. Střední průměr obchodu je −0,05 až −0,6 %,
tedy srovnatelný s náklady 0,15 % nebo horší: chybí i hrubý edge.

**5. Long bias.** Všechna tři P_f jsou jen long. Nula se stejnou stranou drift trhu odfiltruje (FUND i LEAD ji
porazily), ale holdout jsem znal (§2), takže výsledek FUND a LEAD je jen indicie.

## Robustnost (jen hlášení)

| | ORB | FUND | LEAD |
|---|---|---|---|
| Bez fundingu | −0,82 | 0,48 | 0,73 |
| Ceny perpu, holdout | −0,94 | 0,47 | 0,71 |
| Ceny perpu, 2020-01 → 2026-09 | 0,28 | 0,83 | 0,94 |
| Po letech 2022 / 23 / 24 / 25 / 26 | −1,29 / 0,53 / −0,80 / −1,28 / −1,55 | 0,75 / 1,88 / 1,15 / −0,91 / 0,06 | 1,47 / 0,62 / 0,36 / 0,48 / 0,68 |

Holdout všech 30 variant (Sharpe): kladné jsou jen FUND `Q90_LONG` 0,20, `Q95_LS` 0,16, `Q95_LONG` 0,47, CME `G2`
0,04 a všechny čtyři LEAD (0,37–0,82). Sekundární koš altcoinů se nepočítal (§9: jen pro rodiny, které splní §8).

## Co dál

- Podle §10 nic nepokračuje automaticky.
- **LEAD** je jediný kandidát s konzistentním chováním (všechny varianty a roky kladné). Pokračovat lze jen
  **novou pre-registrací** s forward testem na datech od 2026-10-01. Při ~17 obchodech ročně by forward potřeboval
  2–3 roky na smysluplný interval. Rychlejší cesta: rozšířit stejné pravidlo na další dvojice „vůdce → následník“
  (BTC → velké altcoiny) a získat víc obchodů, opět jako nová studie.
- Další dávka strategií by se měla vyhnout držení pod ~1 den, pokud nemá hrubý edge zjevně nad 0,15 % na obchod.
