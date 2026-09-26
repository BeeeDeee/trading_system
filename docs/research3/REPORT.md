# Výzkum 3 – pravidla pasivního portfolia (výsledky)

**Datum:** 2026-09-26 · Pre-registrace `PREREGISTRATION.md` v1.0 i skript commitnuty před během.
Náklady 5 bps na stranu, bez daní. Reference: 60 % SPY / 40 % IEF, měsíční rebalance.
Významnost: bootstrap rozdílu Sharpe, Holm-Bonferroni v rámci otázky, hladina 10 %.

## Shrnutí jednou větou na otázku

| Otázka | Závěr |
|---|---|
| Q1 Jak často rebalancovat | **Na frekvenci prakticky nezáleží** (Sharpe 0,66–0,68); roční rebalance nebo pásmo 10 p.b. dají stejný výsledek s 3–30× méně obchody. |
| Q2 Kolik akcií | Věc rizikové preference – více akcií = vyšší výnos i propad; žádný „nejlepší" poměr, poměry se v čase mění. |
| Q3 Akcie mimo USA | V tomto období **uškodily** (Sharpe −0,07 až −0,10), ale ne statisticky významně; jde o jedno období amerického náskoku. |
| Q4 Typ dluhopisů | Rozdíly malé a nevýznamné; IEF (7–10 let) je rozumná volba. |
| Q5 Zlato | Jediný **statisticky významný** rozdíl: 5–10 % zlata zvýšilo Sharpe o 0,04–0,07 v obou polovinách období – **s velkou výhradou** (viz níže). |

## Q1 – Rebalance 60/40 (2002-08 … 2026-09)

| Varianta | CAGR | Volatilita | Sharpe | Max. propad | Rebalancí | Obrat/rok | ΔSharpe [90% CI] | Holm |
|---|---|---|---|---|---|---|---|---|
| měsíčně (ref.) | 8,5 % | 10,6 % | 0,66 | 32,3 % | 290 | 27 % | – | – |
| čtvrtletně | 8,6 % | 10,4 % | 0,67 | 31,8 % | 97 | 19 % | +0,01 [0,00; +0,03] | ne |
| ročně | 8,5 % | 10,2 % | 0,68 | 30,2 % | 25 | 11 % | +0,02 [0,00; +0,05] | ne |
| nikdy | 9,5 % | 12,2 % | 0,66 | 32,7 % | 1 | 4 % | 0,00 [−0,06; +0,07] | ne |
| pásmo 5 p.b. | 8,5 % | 10,7 % | 0,66 | 31,8 % | 19 | 12 % | −0,01 [−0,02; +0,01] | ne |
| pásmo 10 p.b. | 8,8 % | 11,0 % | 0,67 | 30,5 % | 9 | 11 % | +0,01 [−0,02; +0,04] | ne |

„Nikdy" má vyšší výnos jen proto, že podíl akcií driftoval nahoru (vyšší riziko); na Sharpe nic nezíská.

## Q2 – Podíl akcií (SPY/IEF, měsíčně)

| Akcie/dluhopisy | CAGR | Volatilita | Sharpe | Max. propad | Sharpe 2002–14 | Sharpe 2014–26 |
|---|---|---|---|---|---|---|
| 0/100 | 3,2 % | 6,8 % | 0,24 | 23,9 % | 0,58 | −0,13 |
| 20/80 | 5,1 % | 5,6 % | 0,60 | 19,1 % | 0,93 | 0,30 |
| 40/60 | 6,8 % | 7,2 % | **0,71** | 19,8 % | 0,83 | 0,59 |
| **60/40** | 8,5 % | 10,6 % | 0,66 | 32,3 % | 0,64 | 0,68 |
| 80/20 | 10,0 % | 14,5 % | 0,61 | 44,7 % | 0,53 | 0,70 |
| 100/0 | 11,4 % | 18,8 % | 0,58 | 55,2 % | 0,47 | 0,71 |

Pořadí podle Sharpe se mezi polovinami období **obrátilo** (v první byly lepší dluhopisy, ve druhé
akcie; dluhopisy poškodil rok 2022). Volbu poměru proto nelze „optimalizovat" z historie.

## Q3 – Akcie mimo USA (2003-05 …, 40 % IEF)

| Akciová část | CAGR | Sharpe | Max. propad | ΔSharpe [90% CI] | Holm |
|---|---|---|---|---|---|
| jen SPY (ref.) | 8,5 % | 0,67 | 32,3 % | – | – |
| 60 % SPY / 30 % EFA / 10 % EEM | 7,9 % | 0,59 | 34,4 % | −0,07 [−0,15; 0,00] | ne |
| 50 % SPY / 40 % EFA / 10 % EEM | 7,7 % | 0,57 | 35,0 % | −0,10 [−0,19; 0,00] | ne |

## Q4 – Typ dluhopisů (2004-01 …, 60 % SPY)

| Dluhopisy | CAGR | Sharpe | Max. propad | ΔSharpe [90% CI] |
|---|---|---|---|---|
| IEF 7–10 let (ref.) | 8,2 % | 0,63 | 32,3 % | – |
| SHY 1–3 roky | 7,6 % | 0,56 | 35,0 % | −0,07 [−0,14; +0,01] |
| TLT 20+ let | 8,4 % | 0,64 | 31,2 % | +0,01 [−0,10; +0,11] |
| AGG agregát | 8,0 % | 0,58 | 35,5 % | −0,05 [−0,09; 0,00] |
| TIP inflační | 8,2 % | 0,60 | 36,7 % | −0,03 [−0,08; +0,03] |
| 50 % IEF + 50 % TIP | 8,2 % | 0,62 | 34,4 % | −0,01 [−0,04; +0,02] |

Žádný rozdíl po Holmově korekci významný není.

## Q5 – Zlato (2004-12 …)

| Varianta | CAGR | Volatilita | Sharpe | Max. propad | ΔSharpe [90% CI] | Holm |
|---|---|---|---|---|---|---|
| 60/40 (ref.) | 8,2 % | 10,6 % | 0,63 | 32,3 % | – | – |
| 5 % zlata | 8,4 % | 10,2 % | 0,66 | 30,3 % | +0,04 [+0,01; +0,07] | **ano** |
| 10 % zlata | 8,6 % | 9,9 % | 0,70 | 28,5 % | +0,07 [+0,01; +0,13] | **ano** |

**Výhrada:** zlato mělo v letech 2005–2026 mimořádné období (GLD +9,5 % ročně, srovnatelně
s akciemi). V letech 1980–2000 zlato ztratilo přes polovinu reálné hodnoty – taková fáze v našich
datech není. Statistická významnost uvnitř vzorku tedy neznamená, že se efekt zopakuje. Robustní
část zjištění: zlato má nízkou korelaci s akciemi i dluhopisy (0,06–0,22), takže malá příměs
snižuje propady (28,5 % vs. 32,3 %) – to je vlastnost, ne náhoda jednoho období.

## Praktické závěry (pro čistý výzkum; bez daní)

1. **Rebalance jednou ročně nebo při odchylce 10 p.b. stačí** – stejný výsledek, výrazně méně
   obchodů. Pro reálného investora to znamená i méně daňových událostí.
2. **Poměr akcií a dluhopisů volit podle snesitelného propadu**, ne podle backtestu – historie
   „nejlepší" poměr neurčí (obrácené pořadí mezi polovinami období).
3. **Mezinárodní diverzifikace a typ dluhopisů** v tomto vzorku nic prokazatelně nezměnily;
   argumenty pro ně jsou mimo backtest (diverzifikace rizika jedné země, inflační ochrana).
4. **Malá příměs zlata (5–10 %)** je jediná prokazatelná úprava v tomto vzorku, s výhradou
   výjimečného období pro zlato.

Reprodukce: `uv run python scripts/r3_evaluate.py sharadar_2026-09-25` (~3 min).
