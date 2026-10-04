# Výzkum 13 – Smart Zones v kryptu: výsledek

2026-10-04 · pre-registrace v1.0 · snapshot `binance_2026-10-03` · holdout 2022-01 → 2026-08 otevřen jednou
(commit `0a885a6`, otisk metodiky `b035f81d…`), bez pádu a bez rerunu.

## Verdikt

**Zamítnuto, 0 ze 6 kritérií.** Nákup likvidních kryptoměn v discount zóně swingového rozpětí (P =
`Z_k10_z50_EQ_touch`: pivot lookback 10, discount = spodní polovina rozpětí, výstup na equilibriu, stop pod
swing low) ztrácel na holdoutu víc než rovné váhy univerza, než BTC HOLD i než náhodné vstupy se stejným
profilem expozice. Forward test se podle §10 nespouští.

| Holdout 2022-01 → 2026-08 | P | B_EW (top 20) | B_BTC | B_RND (500 běhů) |
|---|---|---|---|---|
| CAGR | −66,7 % | −45,6 % | +12,0 % | – |
| Sharpe | −0,92 | −0,42 | 0,48 | medián −0,32, 95. percentil 0,05 |
| Max. propad | 99,5 % | 94,8 % | 66,9 % | – |

| Kritérium (§8) | Hodnota | |
|---|---|---|
| 1. Rozdíl Sharpe vs B_EW (90% CI) | −0,50 [−0,87; −0,17] | ✗ |
| 2. Sharpe nad BTC HOLD | −0,92 vs 0,48 | ✗ |
| 3. Sharpe nad 95. percentilem B_RND | −0,92 vs 0,05 (P je pod **všemi** 500 náhodnými běhy) | ✗ |
| 4. Oba podúseky (2022–23 / 2024–26) | −0,34 / −0,63 | ✗ |
| 5. Zátěžové náklady ×2 | −0,58 | ✗ |
| 6. DSR (N = 24) | 0,000 | ✗ |

Ve vývoji (2018-04 → 2021-12) to bylo stejné: P Sharpe 0,28 proti B_EW 0,77 a BTC 1,06, pod 96 % náhodných
běhů. Na rozdíl od výzkumů 1, 5 a 9 tedy nejde o vývojový výsledek, který se nezopakoval. **Strategie
nefungovala v žádném období.**

## Co z toho plyne

**1. Nákup v discount zóně je v kryptu systematicky horší než náhoda.** Náhodné vstupy do stejného
univerza, se stejnými sloty a dobami držení, měly medián Sharpe −0,32. Zóny výsledek zhoršily na −0,92.
Coin ve spodní části svého posledního rozpětí v kryptu typicky dál padá, což odpovídá krátkodobému
momentu, ne reverzi. 64 % obchodů skončilo stopem, 26 % cílem. Úspěšnost 31 %, medián čistého výnosu
obchodu −6,5 %.

**2. Celá rodina selhala, ne jen vybraný kandidát.** Všech 24 variant mělo na holdoutu záporný CAGR
a Sharpe pod rovnými vahami univerza (nejlepší −0,67). Pivoty jen z close místo high/low, plnění o den
později i univerzum top 10 nebo top 40 dopadly stejně nebo hůř.

**3. Denní stop na close je drahý, ale hlavní problém to není.** Kdyby šel stop provést přesně na jeho
úrovni (horní odhad, intradenní stop příkaz), P by měl CAGR −29,8 % a Sharpe −0,08 místo −66,7 % a −0,92.
Gapy přes stop tedy stojí hodně, ale i horní odhad zůstává záporný a hluboko pod BTC HOLD (0,48).
Intradenní varianta (5 min – 1 h), pro kterou je indikátor primárně určen, zde testována nebyla (§2).

**4. Náklady:** obrat 44× ročně, náklady 10 % p. a. na holdoutu (ve vývoji 15 %).

## Po letech (CAGR / Sharpe)

| Rok | P | B_EW | B_BTC |
|---|---|---|---|
| 2022 | −86 % / −1,53 | −85 % / −1,65 | −64 % / −1,29 |
| 2023 | −18 % / 0,05 | +60 % / 1,06 | +156 % / 2,35 |
| 2024 | −41 % / −0,19 | +5 % / 0,46 | +121 % / 1,76 |
| 2025 | −80 % / −1,45 | −59 % / −0,75 | −6 % / 0,05 |
| 2026 (do 08) | −72 % / −1,57 | −56 % / −1,22 | −15 % / −0,12 |

V medvědím roce 2022 byla strategie stejně špatná jako trh. V růstových letech 2023 a 2024 ztrácela, zatímco
trh rostl.

## Robustnost (P, holdout, jen hlášení)

- Plnění o den později: CAGR −74,0 %, Sharpe −1,24.
- Pivoty jen z close: −65,7 %, −1,00.
- Stop na úrovni stopu (horní odhad): −29,8 %, −0,08.
- Univerzum top 10: −51,8 % (B_EW −26,8 %); top 40: −72,9 % (B_EW −52,5 %).

Všichni kandidáti na holdoutu, statistiky obchodů a roky: `results/final.json`. Vývoj: `results/dev.json`.

## Poznámky k průběhu

- Mřížka měla chybu návrhu: při z = 0,5 je cíl PREM totožný s EQ (6 identických dvojic). N = 24 pro DSR
  zůstalo, na verdikt to nemá vliv.
- Filtr výzkumu 9 propustil fiat páry `AUDUSDT` a `BKRWUSDT`; ve výzkumu 13 vyřazeny. V top 20 nebyly nikdy.
- Simulátor výzkumu 5 dostal cenový stop a cíl, zákaz zpětného nákupu na stejném openu a volitelné plnění
  stopu na jeho úrovni. Výchozí chování je beze změny.

## Kontext

Všech jedenáct uzavřených výzkumů aktivních strategií (1, 2, 4–11, 13) skončilo zamítnutím. Výzkum 13 je nejjasnější
z nich: populární indikátor z TradingView po nákladech v denním kryptu nejen nepřekonal trh, ale byl
horší než náhodný výběr okamžiku vstupu.
