# Výzkum 4 – aktivace rodin strategií podle režimu (report)

**Datum:** 2026-09-27 · **Snapshot:** `sharadar_2026-09-25` · **Pre-registrace:**
[PREREGISTRATION.md](PREREGISTRATION.md) (commit `07d346e`, kód `89888d6` commitnut před prvním během)
· **Výsledky:** [results/](results/)

## Verdikt

**Hypotéza se zamítá. Žádné z 4 pre-registrovaných kritérií primární metoda nesplnila.**
Meta-vrstva nad 87 rukávy z 25 rodin neumí spolehlivě poznat, kterou rodinu aktivovat.
Průměrná rank IC nejlepší metody (LightGBM) je 0,034 (t = 1,6, statisticky neprůkazná).
K překonání SPY v poměru výnosu k riziku by bylo potřeba IC zhruba 0,05–0,06, k „výraznému"
překonání (Sharpe ~1) IC ~0,12, tedy 3–4× víc, než jakékoli testované metodě vyšlo.

Potenciál přepínání je obrovský (oracle top-5: Sharpe 3,4), jenže je téměř celý
nepředvídatelný. To je hlavní poznatek výzkumu.

## Výsledky (OOS 2006-01 … 2026-09, po nákladech, simulace na úrovni pozic)

| | CAGR | Sharpe | Max. propad | Obrat/rok | ΔSharpe vs SPY, 90% CI | DSR |
|---|---|---|---|---|---|---|
| **M3 ML_GBM (primární)** | **13,5 %** | **0,52** | **72,6 %** | 16,5× | −0,03 [−0,29; +0,19] | 0,88 |
| M5 BLEND | 11,7 % | 0,51 | 65,6 % | 13,9× | −0,05 [−0,28; +0,16] | 0,87 |
| M4 ML_RIDGE | 9,0 % | 0,40 | 70,4 % | 15,5× | −0,16 [−0,40; +0,07] | 0,73 |
| M1 FACTOR_MOM | 8,7 % | 0,38 | 37,3 % | 10,7× | −0,17 [−0,48; +0,14] | 0,71 |
| M2 REGIME | 7,7 % | 0,34 | 72,0 % | 11,2× | −0,22 [−0,44; −0,01] | 0,64 |
| SPY | 11,1 % | 0,56 | 55,2 % | – | | |
| 60/40 | 8,3 % | 0,63 | 32,3 % | – | | |
| EW_ALL (všechny rukávy, bez přepínání) | 8,9 % | 0,43 | 58,9 % | 4,5× | | |

M3 vs EW_ALL: ΔSharpe +0,10 [−0,09; +0,27]. SPA p vs SPY: 0,16.

| Kritérium (M3) | Výsledek |
|---|---|
| 1. CI ΔSharpe vs EW_ALL > 0 | ne (dolní mez −0,09) |
| 2. CI ΔSharpe vs SPY > 0 | ne (dolní mez −0,29) |
| 3. DSR ≥ 0,90 | ne (0,88) |
| 4. Sharpe > SPY v obou podúsecích | ne (2006–2019: 0,28 vs 0,49) |

### Podúseky – výsledek stojí na 2020–2026

| M3 vs SPY | CAGR | Sharpe |
|---|---|---|
| 2006–2019 | 5,3 % vs 9,1 % | 0,28 vs 0,49 |
| 2020–2026 | **32,7 % vs 15,5 %** | **0,97 vs 0,68** |

Celkový CAGR nad SPY vzniká **jen** v letech 2020–2026 (2020: +68 %, 2026 do září: +71 %).
Toto období je spotřebované a obecně známé (pre-registrace §2). V „čistší" části 2006–2019
metoda za trhem výrazně zaostala a v roce 2008 prošla propadem 73 %. Rozdíl mezi podúseky je
typický pro model, který zachytil jeden příznivý režim, ne stabilní vztah.

## Diagnostika

**Strop a potřebná přesnost** (lineární měsíční výnosy rukávů, 2006–2026):

| Selektor | Rank IC | CAGR | Sharpe |
|---|---|---|---|
| Oracle top-1 (zná budoucnost) | 1,00 | 292 % | 3,85 |
| Oracle top-5 | 1,00 | 194 % | 3,44 |
| Oracle + šum | 0,19 | 32 % | 1,28 |
| Oracle + šum | 0,12 | 23 % | 0,95 |
| Oracle + šum | 0,08 | 18 % | 0,77 |
| Oracle + šum | 0,05 | 14 % | 0,62 (≈ SPY) |
| **M3 ML_GBM (skutečnost)** | **0,034** | 15 % | 0,55 |
| M5 BLEND | 0,026 | 12 % | 0,53 |
| M4 ML_RIDGE | 0,022 | 10 % | 0,42 |
| M1 FACTOR_MOM | 0,014 | 9 % | 0,37 |
| M2 REGIME | 0,005 | 8 % | 0,34 |

Křivka odpovídá skutečným metodám: s IC ~0,03 se dostaneme přesně tam, kde M3 skončila.
**Chybí prediktivní signál, ne portfoliová mechanika.**

**Co meta-vrstva aktivovala** (`results/activation.csv`): nejčastěji momentum (46 % měsíců),
value+momentum (42 %), value (38 %). Rotace mezi obdobími odpovídá tomu, co v předchozích
letech fungovalo. Je to v podstatě momentum strategií se zpožděním, a proto přichází pozdě
při otočkách (2008: GLD a low-vol až po propadu; 2009 naopak +59 %).

**Robustnost** (`results/robustness.csv`): ve všech variantách (náklady ×2, top-3, top-10,
bez limitu na rodinu, jen akcie, jen ETF) je ΔSharpe M3 vs SPY záporný (−0,02 až −0,18).
Vs. EW_ALL je bodově kladný (+0,05 až +0,11) kromě ETF-only. Slabá a neprůkazná stopa
hodnoty přepínání mezi akciovými rukávy, v žádném případě „výrazné překonání trhu".

## Proč to nevyšlo

1. **Doplňkovost rodin je fakt, ale neví se o ní předem.** Rodiny se v čase střídají, jenže
   pořadí příštího měsíce je z dostupných informací (minulé výnosy, volatilita, VIX, trend,
   sazby, šíře trhu, rozptyl) predikovatelné jen nepatrně. Faktorové momentum (M1) mělo
   IC 0,014, režimová tabulka (M2) 0,005.
2. **Obrovský oracle je statistický artefakt výběru maxima.** Mezi 87 volatilními rukávy je
   každý měsíc někdo +15 %. Nejlepší z mnoha náhodných veličin vypadá vždy skvěle, i když
   neexistuje žádný vzorec, jak ho najít.
3. **Přepínání stojí peníze a riziko.** Obrat 11–17× ročně a koncentrace do 5 rukávů
   zvyšují propady (M3 73 % vs SPY 55 %), aniž by za to byl spolehlivě vyšší výnos.
4. **Diverzifikace sama trh nepřekonává.** EW_ALL (Sharpe 0,43) je pod SPY: většina rodin
   akciových faktorů po nákladech v letech 2006–2026 za trhem zaostávala (`results/sleeves_oos.csv`).

## Doporučení

- **Neobchodovat.** Forward test se podle §9 nespouští (kritéria 1–3 nesplněna).
- Pokud chce někdo myšlenku dál rozvíjet, jediná nadějná stopa je slabé zlepšení vs. EW_ALL
  u akciových rukávů. Ověřit ji lze **jen forward testem** a očekávaný přínos je v řádu
  desetin Sharpe, ne násobků trhu.
- Praktický závěr výzkumů 1–4 se nemění: nízkonákladové pasivní portfolio s vhodným podílem
  akcií (výzkum 3).
