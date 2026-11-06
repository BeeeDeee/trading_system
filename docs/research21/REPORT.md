# Výzkum 21 – MR na akciích a proxy filtr šum/informace: výsledek

2026-11-06 · pre-registrace v1.1 · Sharadar `sharadar_2026-09-25` · jeden plný běh (131 s), zámek `3676214f1`, commit kódu `f97776dee`.
**Žádný čistý holdout:** veškerá historie je už viděná, jde o post hoc odhad velikosti efektu. Report je napsaný rychle (konec limitu), obsahuje čísla z `results/full.json`, ne rozbor.

## Verdikt

**(a) Žádný filtr neprošel všemi kritérii.** V, G, E: „nelze posoudit“ (blokovaný podíl pod 15 %, pravidlo ponecháno po rozhodnutí vlastníka); I a C: „ne“ (2 z 5). Podle pre-registrace se
**LLM klasifikátor (úroveň 2) nad tímto signálem nestaví**; news logger běží dál.

| Filtr | Verdikt | Kritéria | Blokováno | Průměr povolených | Průměr blokovaných | Spread [90% CI] | Povolené, náklady ×2 |
|---|---|---|---|---|---|---|---|
| E | nelze posoudit | 1/5 | 14.1 % | +0.146 % | +0.046 % | +0.100 % [-0.131; +0.344] | -0.039 % |
| V | nelze posoudit | 3/5 | 7.3 % | +0.198 % | -0.092 % | +0.290 % [+0.011; +0.565] | -0.077 % |
| G | nelze posoudit | 1/5 | 10.4 % | +0.196 % | +0.009 % | +0.187 % [-0.041; +0.392] | -0.078 % |
| I | ne | 2/5 | 25.0 % | +0.201 % | +0.103 % | +0.098 % [-0.070; +0.275] | -0.070 % |
| C | ne | 2/5 | 34.5 % | +0.149 % | +0.098 % | +0.051 % [-0.104; +0.197] | -0.035 % |

## Základ S0 a kontroly
S0: 16141 obchodů, průměrný čistý obchod +0.176 % (CI +0.078; +0.292), Sharpe 0.38, CAGR 8.5 %, max. propad -61 %.
SPY: Sharpe 0.53, CAGR 8.7 %, propad -55 %. Kontrola C0 (akcie v režimu, náhodné pořadí): Sharpe -0.00, takže reversal nese informaci, ale S0 nepřekoná SPY (0,38 vs 0,53).
DSR (N = 6): S0 0.93, V 0.92, G 0.94, I 0.92.

## Co z toho plyne a co je neověřené
- Jediný filtr, který prošel 3 kritérii, je V (objem > 3×): spread +0.29 % s CI nad nulou, povolené nad p95 náhodných vstupů, kladný ve všech třetinách. Selhal na nákladech ×2 a na podílu blokovaných (7,3 %). Je to **jediný náznak**, ne kandidát na forward (pre-registrace ho nepřipouští).
- **Kritérium 4 (náklady ×2) selhalo u všech filtrů.** Nekontroloval jsem, zda je za tím reálný efekt nákladů v letech 1999–2001 (45 bps na stranu při ×1,5 u openu), nebo chyba v `net_at`/`trade_costs`. Před dalším použitím ověřit proti `robustness.r7` (náklady ×3 v simulátoru).
- Robustnost (r1–r7) je v `full.json`, v reportu nerozebraná. README řádek a rozbor třetin chybí.
