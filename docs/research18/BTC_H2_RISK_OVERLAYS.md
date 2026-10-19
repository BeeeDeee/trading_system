# BTC/H2 mixy — brzdy BTC expozice

Verze 0.1 · 2026-10-19 · Popisný post hoc experiment; nemění primární H2 ani jeho forward rozhodnutí.

## Otázky

1. Jak dlouho trvaly nejhlubší historické drawdowny BTC a jak velké ztráty vznikly během libovolných 7 a 14 po sobě jdoucích dní?
2. Zlepší jednoduché, zpožděné řízení BTC složky drawdowny mixů BTC/H2?

## Data a výchozí mixy

Stejný snapshot a období 2020-05-01 až 2026-09-30 jako předchozí srovnání. H2 zůstává základní `H2_btc200`, včetně původních 10% vah na obchod, 10 bps nákladů na stranu a fundingu. Výchozí poměry BTC/H2 jsou 50/50 a 25/75. Neinvestovaná část je hotovost s nulovým výnosem. Nezvyšuje se páka.

Všechny signály se počítají po denním close a platí až následující den. Při změně BTC váhy účtuje model 10 bps za každý dolar obratu. Začáteční nákup se ignoruje stejně jako v původním BTC B&H benchmarku. Výsledky jsou post hoc, výběrově zkreslené a citlivé na náklady.

## Předem zafixovaná sada

- Kontroly: statické mixy 50/50 a 25/75.
- Volatility brake: BTC složka se následující den omezí faktorem `min(1, 60% / RV)`, kde RV je anualizovaná směrodatná odchylka denních BTC výnosů. Testovaná zpětná okna: 7, 14, 30 a 60 kalendářních dní. Expozice se nikdy nezvýší nad statickou váhu; rozdíl je hotovost.
- Drawdown brake: BTC složka se následující den sníží na 25 % své cílové váhy, když BTC zavře aspoň 10 %, 15 % nebo 20 % pod nejvyšším close posledních 30 dní. Snížená expozice se obnoví následující den po close BTC nad jeho 20denním jednoduchým průměrem.
- Každý ze 7 overlayů se aplikuje zvlášť na oba výchozí mixy: 14 variant plus 2 kontroly.
- Diagnostika BTC: nejhlubší drawdown epizody, délka od vrcholu do dna, délka do obnovení vrcholu (pokud nastalo před koncem vzorku) a nejhorší klouzavý 7denní a 14denní výnos.
- Metriky mixů: CAGR, celkový výnos, max drawdown, Sharpe, nejhorší kalendářní rok, obrat a zaplacené dodatečné náklady.

## Rozhodovací log

| Datum | Změna |
|---|---|
| 2026-10-19 | Na žádost vlastníka předem určeny čtyři krátké RV pohledy a tři drawdown prahy pro dva výchozí BTC/H2 mixy. Jde pouze o popisnou historickou analýzu. |
