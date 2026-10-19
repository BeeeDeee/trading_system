# BTC/H2 brzdy BTC expozice — výsledky

Snapshot `binance_2026-10-03`, období 2020-05-01 až 2026-09-30. Průzkumná post hoc analýza na již zkoumané historii, nikoli nezávislé potvrzení. Kompletní roční výsledky jsou v ignorovaném souboru `runs/r18_btc_h2_risk_overlays_history.json`.

Každé pravidlo se vyhodnocuje po denním close a nová BTC váha platí od následujícího dne. Náklady overlaye jsou 10 bps za každý dolar obratu při převážení mezi BTC/H2/hotovostí; vnitřní H2 náklady a funding jsou zachované. Denní cílové váhy znamenají předpoklad pravidelného rebalancování.

## Jak dlouho trvaly propady BTC

- Nejhlubší epizoda: vrchol 2021-11-08, dno 2022-11-21, drawdown **76,6 %**. Pokles od vrcholu do dna trval **378 dní** a návrat k předchozímu vrcholu dalších zhruba **469 dní** (847 dní celkem od vrcholu).
- Další hluboká epizoda: vrchol 2021-04-13, dno 2021-07-20, drawdown **53,1 %** za 98 dní; obnova vrcholu zabrala dalších 91 dní.
- Nejhorší klouzavý 14denní výnos: **−40,5 %**, 2021-05-10 až 2021-05-23.
- Nejhorší klouzavý 7denní výnos: **−33,3 %**, 2022-06-12 až 2022-06-18.

Takže ano, samotný prudký pád může nastat během dvou týdnů. Největší drawdown ale nebyl dvoutýdenní událost: ztráta se skládala mnoho měsíců a návrat byl ještě delší.

## Mix 50/50 BTC/H2

| Overlay | CAGR | Max. DD | Sharpe | Průměrný násobek BTC váhy |
|---|---:|---:|---:|---:|
| Statický mix | 29,86 % | 49,30 % | 1,029 | 1,00 |
| Volatilita 7 dní, cíl 60 % | 28,23 % | 43,84 % | 1,063 | 0,93 |
| Volatilita 14 dní, cíl 60 % | 28,91 % | 44,80 % | 1,088 | 0,93 |
| Volatilita 30 dní, cíl 60 % | 27,39 % | 47,88 % | 1,038 | 0,94 |
| Volatilita 60 dní, cíl 60 % | 27,43 % | 48,04 % | 1,030 | 0,94 |
| 30denní maximum, práh 10 % | **24,08 %** | **35,91 %** | 1,048 | 0,75 |
| 30denní maximum, práh 15 % | 19,92 % | 47,02 % | 0,855 | 0,84 |
| 30denní maximum, práh 20 % | 22,18 % | 52,85 % | 0,886 | 0,91 |

## Mix 25/75 BTC/H2

| Overlay | CAGR | Max. DD | Sharpe | Průměrný násobek BTC váhy |
|---|---:|---:|---:|---:|
| Statický mix | 20,74 % | 28,28 % | 1,213 | 1,00 |
| Volatilita 7 dní, cíl 60 % | 19,76 % | 24,72 % | 1,250 | 0,93 |
| Volatilita 14 dní, cíl 60 % | 20,07 % | 25,38 % | **1,274** | 0,93 |
| Volatilita 30 dní, cíl 60 % | 19,37 % | 27,45 % | 1,232 | 0,94 |
| Volatilita 60 dní, cíl 60 % | 19,41 % | 27,53 % | 1,224 | 0,94 |
| 30denní maximum, práh 10 % | **17,57 %** | **20,33 %** | 1,244 | 0,75 |
| 30denní maximum, práh 15 % | 15,69 % | 25,53 % | 1,077 | 0,84 |
| 30denní maximum, práh 20 % | 16,89 % | 30,54 % | 1,097 | 0,91 |

Drawdown pravidlo s 10% prahem omezilo v tomto vzorku drawdown nejvíc, ale drželo BTC expozici v průměru jen na 75 % cílové váhy a stálo část CAGR. Volatility targeting s oknem 14 dní měl lepší Sharpe než statický mix, avšak drawdown snížil jen mírně. Prahy 15 % a 20 % byly horší než statická kontrola v některých metrikách; vyšší práh tedy automaticky neznamenal lepší ochranu.

## Interpretace a omezení

- Overlay neodstranil prudký propad; reaguje až po denním close a další den. V rychlém pádu kus ztráty nevyhnutelně zůstane.
- 10% drawdown brzda uspěla historicky v tomto testu, ale je jednou z několika zkoušených variant. Její výběr z těchto dat je zatížený multiple testingem.
- Volatilita není totéž co směr: BTC může být velmi volatilní i při prudkém růstu, takže volatility brake může snížit také růstovou expozici.
- Nejrozumnější kandidáti pro další paper tracking jsou statický 25/75, jeho 14denní volatility targeting a 10% drawdown brake. Výsledky je třeba dál sledovat na nových datech; z této tabulky nevyplývá, který z nich bude nejlepší příště.
