# Výzkum 3 – pravidla pasivního portfolia (pre-registrace)

Verze 1.0 · 2026-09-26 · Stav: **pre-registrace, žádný výsledek zatím spočten**. Čistý výzkum.
Znovu používá framework (panel ETF z výzkumu 2 rozšířený o SHY, engine, bootstrap).

## 1. Účel a rámec

Nehledáme strategii, která porazí trh. Porovnáváme **pravidla pasivního portfolia**, která musí
zvolit každý investor: jak často rebalancovat, kolik akcií, zda diverzifikovat mimo USA, jaké
dluhopisy, zda přidat zlato. Otázka je vždy: **liší se varianty prokazatelně, a kolik stojí?**

Předem očekávaný a legitimní výsledek: většina rozdílů bude statisticky nerozlišitelná.
I to je užitečné – pak rozhoduje jednoduchost, náklady a riziková preference, ne backtest.

## 2. Společná pravidla

- Data: Sharadar `funds`, total return z `closeadj`, kalendář SPY; snapshot 2026-09-25.
- Náklady 5 bps na stranu, plně investováno (bez hotovosti), bez pák a bez daní.
- **Referenční portfolio: 60 % SPY + 40 % IEF, měsíční rebalance** (rozhodnutí po close prvního
  obchodního dne měsíce, jako výzkum 1–2).
- Každá otázka má vlastní období = nejdelší společná historie jejích ETF; reference se počítá
  na stejném období.
- Metriky: CAGR, volatilita, Sharpe (nad T-bill), max. propad, délka propadu, obrat, náklady.
- Rozdíl vůči referenci: 90% CI (stationary bootstrap, blok 21 dní, 2 000 opakování) pro rozdíl
  Sharpe a CAGR. **Významnost:** Holm-Bonferroni přes varianty dané otázky na hladině 10 %
  (bootstrap p-hodnota oboustranného testu rozdílu Sharpe).
- Stabilita: každá otázka zvlášť pro první a druhou polovinu jejího období.

## 3. Otázky a varianty

**Q1 – Rebalance 60/40** (SPY/IEF, 2002-08 → 2026-09)
- měsíčně (reference), čtvrtletně, ročně (první obchodní den roku), nikdy (buy-and-hold s driftem),
- pásma: rebalance na cíl, když se váha akcií odchýlí o víc než 5 p.b. / 10 p.b. (kontrola
  po každém close).

**Q2 – Podíl akcií** (SPY/IEF měsíčně, 2002-08 → 2026-09)
- 0 / 20 / 40 / 60 (reference) / 80 / 100 % akcií. Popisná křivka riziko/výnos, žádný „vítěz" –
  volba je věc rizikové preference.

**Q3 – Akcie mimo USA** (2003-05 → 2026-09, akciová část 60 %, dluhopisy 40 % IEF)
- jen USA (reference: 60 % SPY),
- globální A: akciová část 60 % SPY / 30 % EFA / 10 % EEM,
- globální B: akciová část 50 % SPY / 40 % EFA / 10 % EEM.

**Q4 – Typ dluhopisů** (2004-01 → 2026-09, akcie 60 % SPY, dluhopisová část 40 %)
- IEF 7–10 let (reference), SHY 1–3 roky, TLT 20+ let, AGG agregát, TIP inflační, 50 % IEF + 50 % TIP.

**Q5 – Zlato** (2004-12 → 2026-09)
- 60/40 (reference), 57/38/5 GLD, 54/36/10 GLD (zlato na úkor obou částí proporčně).

Celkem 5 + 5 + 2 + 5 + 2 = 19 variant proti referencím.

## 4. Výstup

`REPORT.md`: tabulka na otázku, rozdíly s CI a Holm-významností, stabilita v polovinách období,
praktické shrnutí.

## 5. Rozhodovací log

| Datum | Rozhodnutí | Zdůvodnění |
|---|---|---|
| 2026-09-26 | Pre-registrace v1.0 | Volba uživatele (možnost 2 po výzkumu 2) |
