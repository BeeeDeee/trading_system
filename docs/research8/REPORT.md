# Výzkum 8 – funding carry na Binance: výsledek

2026-10-03 · pre-registrace v1.1 · snapshot `binance_2026-10-03` · holdout otevřen jednou
(commit `7644423`, otisk metodiky `0ff8e850…`, `runs/vault_r8/vault.log`).

## Verdikt

**Zamítnuto.** Delta-neutrální funding carry (long spot + short perp) na Binance po nákladech
v letech 2023–2026 spolehlivě nepřekonal americký T-bill. P1 (`S0`, BTC + ETH vždy zajištěno) i P2
(`S1_L30_th0`, zajištěno jen při kladném 30denním fundingu) nesplnily kritéria 1 a 2 z §6.
Forward test se nespouští.

## Holdout 2023-01 → 2026-08

Pre-registrovaný holdout končil 2026-09-30. Měsíční soubory **spotu** za září 2026 ale v době stahování
na Binance ještě nebyly (perp a funding ano). Simulátor proto 2026-09-01 vyhodnotil BTC i ETH jako
delistované a zavřel je se 2% slippage. Je to chyba dat, ne strategie, a audit ji měl chytit
(kontroloval konec dat jen u perpů; opraveno). Proto jsou uvedené obě verze:

| | P1 `S0` (pre-reg., do 09-30) | P1 `S0` (do 08-31, post hoc) | P2 `S1_L30_th0` (pre-reg.) | P2 (do 08-31, post hoc) |
|---|---|---|---|---|
| Nadvýnos nad T-bill p. a. | −0,14 % | **+0,74 %** | −0,59 % | **+0,28 %** |
| 90% interval (bootstrap) | [−1,88; +1,59] | [−0,37; +2,09] | [−2,24; +1,07] | [−0,74; +1,58] |
| Funding p. a. (na kapitál) | 5,30 % | 5,42 % | 4,91 % | 5,02 % |
| T-bill p. a. | 4,39 % | 4,41 % | 4,39 % | 4,41 % |
| Náklady p. a. | 0,98 % | 0,20 % | 1,05 % | 0,28 % |
| Nadvýnos 2023-01 → 2024-06 | +3,00 % | +3,00 % | +2,26 % | +2,26 % |
| Nadvýnos 2024-07 → konec | −2,24 % | **−0,82 %** | −2,49 % | **−1,08 %** |
| Zátěžové náklady | −0,34 % | +0,58 % | −0,89 % | +0,02 % |
| PSR / DSR | 0,42 | 1,00 | 0,00 | 0,00 |
| Kritéria 1/2/3/4 | ✗ ✗ ✗ ✗ | ✗ ✗ ✓ ✓ | ✗ ✗ ✗ ✗ | ✗ ✗ ✓ ✗ |

Verdikt je stejný v obou verzích: kritérium 2 (kladný nadvýnos v obou podúsecích) a kritérium 1
(dolní mez intervalu > 0) neprošly ani bez artefaktu.

Po letech (P1, pre-reg. běh): 2023 +1,1 %, 2024 +4,2 %, 2025 −1,0 %, 2026 −6,5 % nad T-bill (2026 je
ovlivněný artefaktem; leden–srpen 2026 bez něj −2,4 % p. a.). Likvidace v holdoutu: 1 (ETH 2025-05-08, podle mark price).
Robustnost (pre-reg. běh, P1): s = 1/2 −1,44 %, s = 3/4 +0,11 %, plnění o den později −0,15 %.

## Proč to nefunguje

**Prémie se zmenšila pod úroveň T-billu.** Průměrný funding BTC (roční sazba na nominál) po čtvrtletích:

| | 23Q1 | 23Q4 | 24Q1 | 24Q2 | 24Q3 | 24Q4 | 25Q1 | 25Q2 | 25Q3 | 25Q4 | 26Q1 | 26Q2 | 26Q3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTC | 8,2 % | 12,2 % | 22,3 % | 9,3 % | 3,5 % | 12,6 % | 5,2 % | 3,5 % | 7,0 % | 4,7 % | 1,2 % | 1,0 % | 6,5 % |
| ETH | 9,1 % | 14,1 % | 22,9 % | 10,3 % | 4,8 % | 14,0 % | 5,1 % | 4,3 % | 6,0 % | 4,3 % | 0,0 % | 0,8 % | 4,9 % |

Na kapitál se z toho dostane jen 2/3, protože třetina leží jako kolaterál. Když funding na nominál
klesne pod ~6–7 % p. a., carry prohrává s T-billem, který v holdoutu nesl ~4,4 %. Od poloviny 2024
(nástup Etheny a dalších syntetických dolarů, které tento obchod dělají ve velkém) je funding většinu
času právě tam. Ve vývoji (2020–2022) carry vypadal výborně (S1: +12 % nad T-bill, Sharpe 8), ale
v té době byl T-bill u nuly a funding v býčím trhu 2021 extrémní. Opakuje se lekce z výzkumu 1:
**vývojové období přeceňuje.**

Rotace mezi altcoiny (S2) neprošla ani vývojem: vyšší funding sežraly náklady (5–15 % p. a.) a likvidace.

## Co z toho plyne

- Strukturální prémie existuje (funding byl v průměru kladný), ale pro malého taker obchodníka **po
  nákladech a proti bezrizikové sazbě není**. Výjimkou je euforie (24Q1), kterou ale nejde předem
  poznat jinak než trailing fundingem, a ten (S1) v holdoutu nepomohl.
- K tomu riziko burzy, které ve výsledku není vůbec.
- Kdo chce tento obchod dělat, potřeboval by maker poplatky, výnos z kolaterálu (USDT na perp účtu
  nenese nic, T-bill ano) nebo křížové burzy. To jsou nové hypotézy, testovatelné jen forward,
  protože historie je teď spotřebovaná.

## Průběh (log v `PREREGISTRATION.md` §10)

1. Pre-registrace v1.0 commitnuta před stažením dat.
2. Audit: neplatné páry perp/spot (FTT, RAY, STRAX…), doplněno pravidlo platnosti páru před výsledky.
3. Vývoj v1.0 odhalil, že likvidace spouštěly wicky poslední ceny. Binance likviduje podle mark price →
   v1.1, N = 34.
4. Vývoj v1.1, výběr P2, zamčení metodiky, jedno otevření holdoutu.
5. Post-hoc: artefakt chybějícího zářijového spotu, přepočet do 2026-08-31
   (`scripts/r8_posthoc_truncated.py`, záměrně mimo trezor, zapsáno v logu). Verdikt se nemění.

Výsledky: `results/dev_v1.0.json`, `results/dev.json`, `results/final.json`,
`results/final_posthoc_to_2026-08-31.json`.
