# Výzkum 6 – STR-TF jen při vysokém VIX: report

2026-09-27 · Pre-registrace: [PREREGISTRATION.md](PREREGISTRATION.md) ·
Výsledky: [summary.json](results/summary.json)

## Verdikt

**Zamítnuto podle pre-registrace.** Splněno 5 ze 6 kritérií, ale Deflated Sharpe je 0,0006
(hranice 0,95). Jako samostatná strategie to nefunguje. **Samotný efekt podmínění ale vypadá
reálně** (viz níže) a je to nejsilnější nález výzkumů 5–6.

| # | Kritérium (default STR-TF, VIX > 25, 2003–2026) | Hodnota | Výsledek |
|---|---|---|---|
| C1 | Hrubý výnos / náklady na obchod ≥ 2 | 4,09 | prošlo |
| C2 | CAGR při nákladech ×2 > 0 | 1,8 % | prošlo |
| C3 | Čistý výnos na obchod > 0 ve všech podúsecích | 70 / 102 / 28 bps | prošlo |
| C4 | Brána zlepšuje čistý výnos na obchod | +54 vs. −2 bps | prošlo |
| C5 | Sharpe > 95. percentil náhodného benchmarku | 0,33 vs. 0,27 | prošlo (těsně) |
| C6 | Deflated Sharpe ≥ 0,95 (N = 1008) | **0,0006** | **zamítá** |

## Co to znamená

> **Výhrada z výzkumu 7:** Tento výzkum měřil výnos proti hotovosti. Když idle kapitál drží SPY,
> přínos rukávu zmizí (Sharpe 0,686 vs. 0,685, viz [výzkum 7](../research7/REPORT.md)).
> Velká část +54 bps na obchod je odraz celého trhu po vysokém VIX, ne výběr akcií.

**Efekt:** Stejné pravidlo bez brány prodělává (−2 bps čistě na obchod, Sharpe −0,04). S bránou
VIX > 25 vydělává +54 bps na obchod, a to ve všech třech podúsecích, i při dvojnásobných nákladech
a i se vstupem o den později. Výnos na obchod roste s prahem monotónně:

| Práh | Obchodů | Čistě / obchod | Investovanost | Sharpe | CAGR |
|---|---|---|---|---|---|
| bez brány | 11 210 | −2 bps | 69 % | −0,04 | −2,3 % |
| VIX > 20 | 3 416 | 22 bps | 20 % | 0,26 | 2,5 % |
| **VIX > 25** | 1 499 | **54 bps** | 9 % | 0,33 | 3,0 % |
| VIX > 30 | 653 | 107 bps | 4 % | 0,37 | 2,7 % |
| relativní (80. percentil 252 d) | 2 335 | 14 bps | 14 % | 0,13 | 0,8 % |

To odpovídá Nagelovi (2012): reversal je odměna za likviditu a platí se hlavně ve stresu.
Absolutní úroveň VIX funguje lépe než relativní (relativní práh 2015–2019 prodělával).

**Proč to přesto není strategie:** Kapitál je investovaný 9 % času. CAGR 3 % a Sharpe 0,33
nestačí na test, který počítá s ~1000 předchozími pokusy v této rodině (DSR). Výnos je navíc
soustředěný do několika let (2007, 2009, 2010, 2021) a rok 2020 byl −15 %: v březnu 2020 brána
nakupovala propady, které ještě pokračovaly. Období 1999–2002 (jen hlášení): Sharpe 0,70, 79 bps.

**Upozornění:** Hypotéza je post hoc (§2 pre-registrace) a všechna data byla viděna. Monotónní
závislost na prahu a stabilita napříč podúseky jsou dobrá znamení, ale ne důkaz.

## Varianty (2003–2026)

| Varianta | Sharpe | CAGR | Čistě / obchod |
|---|---|---|---|
| Náklady ×3 | 0,12 | 0,7 % | 19 bps (2020–2026 záporné) |
| Vstup open *t*+2 | 0,29 | 2,4 % | 38 bps |
| Konfigurace vybraná ve výzkumu 5 + VIX > 25 | 0,44 | 2,9 % | 75 bps |
| Konfigurace vybraná ve výzkumu 5 bez brány | 0,49 | 5,1 % | 20 bps |

## Možné pokračování (nová hypotéza, jen forward)

Efekt je zajímavý jako **doplněk**, ne jako samostatná strategie: kapitál je většinu času
v hotovosti. Logický další krok je portfolio „pasivní jádro + reversal rukáv aktivovaný VIX > 25“
(rukáv si peníze půjčí z jádra jen ve stresu). Šlo by o novou hypotézu. Poctivě ověřit ji lze jen
forward testem na nových datech a VIX > 25 nastává jen zhruba 20 % dní, takže test by potřeboval
roky, ne měsíce.

## Reprodukce

```
uv run python scripts/r6_evaluate.py sharadar_2026-09-25 1000
```

Registr pokusů: `runs/registry/research6.sqlite` (N pro DSR = výzkum 5 + 6).
