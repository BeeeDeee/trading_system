# H2_btc200 – přerozdělení nevyužité expozice

Verze 0.1 · 2026-10-19 · Popisný sizing experiment; pravidlo a rozhodnutí hlavního forward testu H2 se nemění.

## Zmrazené pravidlo

Na otevřených H2 obchodech se každých 15 minut nastaví stejná váha podle počtu právě otevřených mincí. Cílová
hrubá expozice portfolia je 80 %, maximální váha jedné mince je 40 %:

`váha na minci = min(40 %, 80 % / počet otevřených mincí)`.

Příklady: 1 otevřená mince = 40 % celkem; 2 = 40 % každá (80 % celkem); 3 = 26,67 % každá; 10 = 8 % každá.
Když nejsou otevřené žádné obchody, expozice je nula. Tím se nepřidává páka nad vlastní kapitál a současně se
nepřelévá celých 80 % do jediné mince.

Signály, výběr mincí, filtr BTC nad MA200, TP/SL/trailing, délka držení, funding a náklady jsou beze změny vůči
každé odpovídající variantě v `EXIT_VARIANTS.md`. Poplatky 10 bps se účtují za změnu nominálu: vstup, výstup i
převážení již otevřených pozic při změně počtu souběžných obchodů. Výnos pozice a funding se počítají podle stejného
modelu jako v hlavním backtestu.

## Vyhodnocení

Všech 30 výstupních variant se spustí znovu s tímto jediným sizing pravidlem. Hlášené metriky: Sharpe, CAGR,
max drawdown, celkový výnos, průměrná/maximální hrubá expozice, obrat a zaplacené náklady. Srovnání s předchozími
30 běhy s fixní váhou 10 % na coin je popisné. Nejlepší výsledek z mřížky není nezávisle potvrzená strategie;
výpočet na historii 2020–2026 je post hoc a nemění H2 forward rozhodnutí.

## Log rozhodnutí

| Datum | Změna |
|---|---|
| 2026-10-19 | Na žádost vlastníka stanoveno jedno pravidlo alokace: 80 % cílová celková expozice, 40 % limit na coin, rovnoměrné váhy mezi otevřené pozice. Před výpočtem zmrazeno; žádné parametry nebyly vybrány podle výsledků. |
