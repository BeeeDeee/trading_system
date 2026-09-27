# Výzkum 7 – pasivní jádro SPY + reversal rukáv při vysokém VIX (pre-registrace)

Verze 1.0 · 2026-09-27 · Stav: **pre-registrace, žádný výsledek zatím spočten**.
Navazuje na výzkumy 5 a 6 (panel, signál, brána VIX, simulátor, náklady).
Změny po commitu tohoto dokumentu = nový pokus v registru a řádek v §8.

## 1. Otázka

Výzkum 6: reversal s bránou VIX > 25 vydělával +54 bps čistě na obchod, ale kapitál byl 91 % času
v hotovosti. **Překoná portfolio, které drží SPY a jen při signálu přesune část kapitálu do reversal
pozic, samotný SPY po nákladech?**

## 2. Poctivé omezení předem

- Hypotéza je post hoc (vznikla z výsledků výzkumů 5–6) a celá historie 1999–2026 je viděná.
  Historický test ji může jen zamítnout, nebo kvalifikovat na forward test (§7).
- Rukáv nahrazuje SPY právě ve stresových dnech. Jeho přínos je tedy výnos pozic **minus výnos SPY**
  po dobu držení, ne celých +54 bps.

## 3. Pravidlo

- **Jádro:** veškerý volný kapitál je v SPY (total return). V simulátoru „hotovost“ nese přes noc
  `ret_co` SPY a během dne `ret_oc` SPY. Prodej se tedy účtuje na open a peníze nesou jen tu část dne,
  kdy byly skutečně v SPY.
- **Rukáv:** primární pravidlo výzkumu 6 beze změny (default STR-TF, nové vstupy jen po close
  s VIX > 25). Pozice = 1/10 NAV celého portfolia, max. 10 pozic, strop 1 % ADV20, kapitál 1 mil. USD.
- **Náklady:** každý nákup pozice = prodej SPY a každý prodej = nákup SPY. K nákladům pozice
  (model výzkumu 5) se připočte **2 bps za stranu** za obchod v SPY.
- **Benchmark:** SPY total return (buy-and-hold, bez nákladů).

## 4. Období

Jeden běh 1999-01-04 → 2026-09-25. Hodnocení **2003–2026** a podúseky 2003–2014, 2015–2019,
2020–2026. 1999–2002 se jen hlásí.

## 5. Kritéria (2003–2026); nutné splnit **všechna**

1. Rozdíl Sharpe (portfolio − SPY) > 0 a dolní mez 90% intervalu (stationary bootstrap, blok 21 dní,
   1000 vzorků, seed 0) > 0,
2. rozdíl CAGR (portfolio − SPY) > 0 v každém podúseku,
3. rozdíl CAGR > 0 i při nákladech ×2 (rukáv i SPY),
4. Deflated Sharpe aktivního výnosu (portfolio − SPY) ≥ 0,95. N = pokusy výzkumů 5 + 6 + 7,
   rozptyl Sharpe z mřížky výzkumu 5.

Sharpe = průměr / sm. odchylka denních výnosů × √252, bez odečtu bezrizikové sazby (jako výzkumy 5–6).

## 6. Robustnost (jen hlášení)

Práh VIX 20 a 30, rukáv bez brány VIX, vstup open *t*+2, max. propad a beta k SPY, výsledky po letech.

## 7. Co znamená výsledek

- Nesplní-li se cokoli z §5: zamítnuto.
- Splní-li se vše: kandidát na forward test (minimálně 12 měsíců, protože VIX > 25 je jen ~20 % dní).
  Nutná obnova dat.

## 8. Log rozhodnutí

| Datum | Rozhodnutí | Důvod |
|---|---|---|
| 2026-09-27 | Pre-registrace v1.0 | – |
