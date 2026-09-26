# Výzkum 2 – audit dat ETF (R1)

Období 2007-03-01 … 2026-09-25, total return z `closeadj` (Sharadar funds), kalendář = obchodní dny SPY.

| ETF | TR CAGR | Cenový CAGR | Výnos z distribucí | Volatilita | Max. denní pohyb | Dny bez open |
|---|---|---|---|---|---|---|
| VEU | 5.5% | 2.6% | 2.9% | 21.8% | 13.6% | 0 |
| VWO | 5.2% | 2.5% | 2.7% | 26.4% | 20.3% | 0 |
| VNQ | 4.9% | 0.6% | 4.2% | 29.5% | 19.5% | 0 |
| TLT | 2.5% | -0.6% | 3.2% | 15.0% | 7.5% | 0 |
| TIP | 3.2% | 0.2% | 3.0% | 6.3% | 4.5% | 0 |
| IYR | 4.2% | 0.5% | 3.7% | 28.5% | 20.6% | 5 |
| IEF | 3.0% | 0.4% | 2.6% | 6.9% | 3.4% | 0 |
| EFA | 4.9% | 1.9% | 3.0% | 21.7% | 15.9% | 0 |
| EEM | 5.4% | 3.3% | 2.1% | 28.0% | 22.8% | 0 |
| AGG | 2.8% | -0.3% | 3.1% | 5.4% | 6.8% | 0 |
| SPY | 11.1% | 9.1% | 2.0% | 19.6% | 14.5% | 0 |
| GSG | -0.6% | -0.6% | 0.0% | 23.5% | 12.1% | 0 |
| DBC | 2.4% | 1.4% | 1.0% | 19.3% | 7.9% | 0 |
| IAU | 9.7% | 9.7% | -0.1% | 18.2% | 11.7% | 0 |
| GLD | 9.5% | 9.6% | -0.1% | 18.2% | 11.3% | 0 |

Závěry:
- Výnos z distribucí odpovídá typu aktiva: akcie ~2–3 %, dluhopisy ~3 %, REIT ~4 %, zlato 0 % (−0,1 % = poplatek fondu).
- Extrémní denní pohyby odpovídají známým událostem (2008, březen 2020); žádné skoky z neupravených splitů.
- Korelace denních výnosů (hlavní ETF): akcie mezi sebou 0,66–0,88, akcie vs. státní dluhopisy −0,26 až −0,30, zlato vs. akcie 0,06–0,18.
- Pět dní bez použitelného open (IYR, jen alternativa) – neobchodovatelné, engine je přeskočí.
