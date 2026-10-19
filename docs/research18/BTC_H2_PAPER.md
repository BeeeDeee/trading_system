# Paper tracking mixů BTC/H2

Verze 0.1 · 2026-10-19 · Průběžné popisné sledování, bez vlivu na rozhodnutí H1/H2.

## Začátek a účel

Tři varianty portfolia začínají ve stejný den jako forward test research 18: **2026-10-20**. Každá začíná s vlastní základnou 1,00; sleduje se BTC spot, H2_btc200 a hotovost. Jde o paper simulaci bez odesílání příkazů.

## Varianty

Všechny varianty cílí na alokaci 25 % BTC / 75 % H2. H2 je původní H2_btc200: 10 % kapitálu na každý altcoinový obchod, výstup po 4 hodinách, náklady 10 bps na stranu a původní funding model. Neinvestovaná část je hotovost s nulovým výnosem. Páka se nepoužívá.

1. **Mix 25/75** — stálý cíl BTC/H2.
2. **RV14** — cílová BTC váha se následující den násobí `min(1, 0,60 / RV14)`, kde RV14 je anualizovaná směrodatná odchylka posledních 14 denních BTC výnosů. Váha BTC nikdy nepřekročí 25 %.
3. **DD10** — pokud BTC zavře alespoň 10 % pod nejvyšším close posledních 30 dní, BTC váha se od následujícího dne sníží na čtvrtinu cílové váhy (6,25 %). Obnoví se na 25 % den po close BTC nad jeho 20denním jednoduchým průměrem.

Signály se vyhodnocují po UTC denním close a použijí se nejdříve následující den. Každý dolar obratu při změně vah stojí 10 bps. Zahrnuta je historie před začátkem paper testu pouze pro zahřátí volatility, 30denního maxima a 20denního průměru; výnosová řada paper testu začíná 2026-10-20. Průběžný stav se přepočítává od začátku období ze stejných uložených dat.

Dashboard zobrazuje CAGR (od začátku testu anualizované), celkový výnos, max drawdown, Sharpe, aktuální den dat a denní výnosové řady. Vzorek bude zpočátku příliš krátký pro spolehlivý závěr. Tyto varianty netvoří další hypotézy v Holmově korekci H1/H2 a nemění jejich prahy ani rozhodnutí.

## Rozhodovací log

| Datum | Změna |
|---|---|
| 2026-10-19 | Na žádost vlastníka předem zafixovány tři paper varianty: mix 25/75, RV14 cíl 60 % a DD10. Nová samostatná řada začíná 2026-10-20. |
