# Výzkum 9 – trend a momentum v kryptu: výsledek

2026-10-03 · pre-registrace v1.0 · snapshot `binance_2026-10-03` · holdout 2022-01 → 2026-08 otevřen jednou
(commit `1522234`, otisk metodiky `c885d5d9…`). Běh spadl po otevření až při zápisu JSON a byl zopakován
beze změny metodiky (`scripts/r9_final_rerun.py`, řádek `rerun` v `runs/vault_r9/vault.log`).

## Verdikt

**Zamítnuto.** Ani trendový filtr na BTC + ETH (P_T = `T_sma20`), ani průřezové momentum altcoinů
s tržním filtrem (P_X = `X_K10_L30_mf`) nesplnily všechna kritéria z §6. Forward test se podle §9 nespouští.

| | P_T `T_sma20` | 50/50 BTC/ETH | P_X `X_K10_L30_mf` | rovné váhy top 20 |
|---|---|---|---|---|
| CAGR | +6,7 % | +3,3 % | −19,0 % | −45,1 % |
| Sharpe | 0,36 | 0,34 | −0,11 | −0,41 |
| Max. propad | 38,5 % | 68,2 % | 71,7 % | 94,8 % |
| Rozdíl Sharpe (90% CI) | +0,02 [−0,52; +0,56] | | +0,30 [−0,25; +0,87] | |
| Podúseky 2022–23 / 2024–26 | −0,13 / +0,12 | | +0,47 / +0,18 | |
| Zátěžové náklady (rozdíl Sharpe) | −0,12 | | +0,27 | |
| DSR (N = 26) | 0,09 | | 0,48 | |
| Kritéria 1–5 | ✗ ✗ ✗ ✓ ✗ | | ✗ ✓ ✓ ✓ ✗ | |

BTC HOLD na holdoutu: CAGR +12,0 %, Sharpe 0,48, propad 66,9 %.

## Co z toho plyne

**1. Trendový filtr spolehlivě snižuje propad, ale výnos nezaručuje.** Všech 8 filtrů mělo na holdoutu
propad 30–48 % proti 68 % u držení. Sedm z osmi mělo i vyšší Sharpe (0,47–0,81 proti 0,34). Vybraný
`T_sma20`, ve vývoji nejlepší, byl ale na holdoutu druhý nejhorší (Sharpe 0,36): v letech 2023 a 2024 ho
časté signály vyhazovaly z rostoucího trhu (2023: +28 % proti +122 %). **Výběr podle vývoje vybral špatně.**
Tohle je stejná lekce jako ve výzkumu 1: celá rodina vypadá rozumně, konkrétní vítěz z vývoje se neopakuje.
Že „pomalejší trendový filtr na BTC/ETH snižuje propad při podobném výnosu“, je **post-hoc pozorování**.
Ověřit se dá jen novou pre-registrací a forward testem, ne tímto holdoutem.

**2. Altcoiny v letech 2022–2026 ničily hodnotu.** Rovné váhy 20 nejlikvidnějších coinů: −45 % ročně,
propad 95 %. Žádná z 18 variant momenta neměla kladný CAGR. Tržní filtr ztráty zhruba polovičí,
ale vydělávat nepomůže. Pro krypto paper bota to znamená, že benchmark „rovné váhy univerza“ je velmi
slabá laťka. Porazit ho nic neznamená, rozhoduje srovnání s BTC HOLD.

**3. Náklady zabíjejí rychlé varianty.** Ve vývoji stály týdenní rotace top 3 až 20 % p. a.

## Robustnost (P_T / P_X, holdout)

- Hotovost nesoucí T-bill: P_T CAGR +8,8 %, P_X −17,3 %.
- Plnění o den později: P_T +6,0 % (propad 48 %), P_X −23,0 %.
- P_T jen BTC / jen ETH: +5,6 % / +5,1 %.
- P_X univerzum top 10: +1,7 % (benchmark −26 %); top 40: −45 % (benchmark −52 %).

Výsledky po letech a všichni kandidáti na holdoutu: `results/final.json`.

## Kontext výzkumů 1–9

Osm z devíti pre-registrovaných výzkumů skončilo zamítnutím. Opakuje se stejný vzor: vývojové období
vypadá dobře (výzkum 9: Sharpe 1,3–1,5), holdout ne. Jediný robustní poznatek napříč výzkumy se týká
**rizika, ne výnosu**: diverzifikace (výzkum 3) a trendový filtr (výzkum 9, jen popisně) snižují propady.
