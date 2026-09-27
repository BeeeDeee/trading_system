# Výzkum 7 – jádro SPY + reversal rukáv při vysokém VIX: report

2026-09-27 · Pre-registrace: [PREREGISTRATION.md](PREREGISTRATION.md) ·
Výsledky: [summary.json](results/summary.json)

## Verdikt

**Zamítnuto, 0 ze 4 kritérií.** Rukáv proti samotnému SPY nepřidává nic měřitelného. Zisk
+54 bps na obchod z výzkumu 6 byl z velké části jen výnos trhu, který se ve stresových obdobích
odrazil. SPY ho vydělá také.

| # | Kritérium (2003–2026) | Hodnota | Výsledek |
|---|---|---|---|
| C1 | Rozdíl Sharpe > 0 a dolní mez 90% CI > 0 | +0,001, CI [−0,08; +0,08] | zamítá |
| C2 | Rozdíl CAGR > 0 ve všech podúsecích | +0,56 / −1,01 / +0,46 p. b. | zamítá |
| C3 | Rozdíl CAGR > 0 při nákladech ×2 | −1,27 p. b. | zamítá |
| C4 | DSR aktivního výnosu ≥ 0,95 (N = 1014) | 0,000001 | zamítá |

## Čísla (2003–2026)

| Varianta | Sharpe | CAGR | Max. propad | Průměrně v rukávu |
|---|---|---|---|---|
| **SPY** | 0,685 | 11,60 % | 55,2 % | – |
| Jádro + rukáv, VIX > 25 | 0,686 | 11,80 % | 54,7 % | 7,9 % |
| – náklady ×2 | 0,616 | 10,33 % | 56,6 % | 8,2 % |
| – VIX > 20 | 0,506 | 8,30 % | 61,2 % | 19,6 % |
| – VIX > 30 | 0,689 | 11,60 % | 56,4 % | 3,4 % |
| – vstup open *t*+2 | 0,695 | 11,84 % | 51,3 % | 7,0 % |
| – bez brány VIX | 0,074 | −0,77 % | 60,4 % | 68,7 % |

- Aktivní výnos (portfolio − SPY): +0,25 % ročně při tracking erroru 6,0 %, tedy informační
  poměr zhruba 0,04. Takový výsledek nejde odlišit od nuly.
- Beta k SPY je 0,97. Rukáv jen vyměňuje SPY za jednotlivé akcie, které ve stresu kolísají
  s trhem, a za tu výměnu platí náklady na obě strany.
- Rozdíl po letech se střídá: +7,7 p. b. (2007), +9,7 p. b. (2021), ale −4,4 p. b. (2011)
  a −4,0 p. b. (2016). Stabilní přínos v tom není.
- 1999–2002 (jen hlášení): +4,2 p. b. ročně oproti SPY. Opět výsledek éry před decimalizací.

## Co to říká o výzkumu 6

Výzkum 6 porovnával podmíněný reversal s **hotovostí**. Když místo hotovosti leží peníze v trhu,
výhoda zmizí. Nákup propadů při vysokém VIX vydělával hlavně proto, že po vysokém VIX se odráží
celý trh (známý efekt vysoké očekávané prémie za riziko ve stresu), ne proto, že by vybrané akcie
porážely trh. Test proti hotovosti tak přisoudil strategii tržní betu. Výsledek výzkumu 6 je proto
potřeba číst s touto výhradou (doplněno do jeho reportu).

## Závěr pro řadu výzkumů 5–7

Krátkodobý reversal v likvidních US akciích nepřináší po nákladech nic navíc oproti držení SPY:
- nepodmíněný (výzkum 5): zamítnut na validaci,
- podmíněný VIX jako samostatná strategie (výzkum 6): jen tržní beta ve stresu,
- jako rukáv k SPY (výzkum 7): rozdíl nulový.

Laťka je stejná jako v předchozích výzkumech: pasivní jádro je těžké porazit.

## Reprodukce

```
uv run python scripts/r7_evaluate.py sharadar_2026-09-25
```
