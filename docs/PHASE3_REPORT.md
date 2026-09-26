# Fáze 3 – walk-forward výběr a ansámbl

**Stav 2026-09-26: brána formálně splněna** (DSR ≥ 0,90 a dolní hranice 90% CI rozdílu Sharpe vůči
EW_UNIV > 0). Výsledek má ale podstatná omezení – viz „Interpretace".

## Jak to proběhlo

- Metodika (`select()`, WFO, ansámbl, statistika brány) byla zapsaná a commitnutá **před** prvním
  během (commity „Spec: pre-register Phase 3 gate…" a „Phase 3 code…"). Nic se po běhu neměnilo.
- 15 foldů: test = kalendářní rok 2005 … 2019, trénink = 2000-01-03 … konec předchozího roku
  minus 21denní embargo (rozšiřující se okno).
- V každém foldu: tvrdé filtry → robustní skóre (min(Sharpe, medián okolí)) → korelační shluky
  (0,85) → K nejlepších reprezentantů. Primární K = 2, sekundární K = 3.
- Portfolio: pozice členů sečtené v jednom enginu, změna členů na přelomu roku platí náklady na
  čistý rozdíl pozic. Ledger engine s 10 000 USD: parita 1·10⁻¹⁵, 9 973 příkazů za 15 let.
- Registr pokusů: `N_meth = 2` (obě předem registrované varianty).
- Reprodukce: `uv run python scripts/run_wfo.py sharadar_2026-09-25 a03e443786b83e08`
  (výstup `data/derived/…/wfo/e166501421d4bd05/`).

## Výsledky poskládaného out-of-sample 2005–2019 (po nákladech)

| | CAGR | Volatilita | Sharpe | Max. propad | Obrat/rok | Náklady |
|---|---|---|---|---|---|---|
| **Primární (K = 2)** | **9,2 %** | **8,5 %** | **0,93** | **9,6 %** | 300 % | 26 bps |
| Sekundární (K = 3) | 8,5 % | 8,0 % | 0,90 | 8,7 % | 508 % | 47 bps |
| EW_UNIV | 8,4 % | 21,4 % | 0,42 | 60,6 % | – | – |
| SPY_TR | 8,9 % | 18,3 % | 0,49 | 55,2 % | – | – |

| Statistika (primární) | Hodnota | Brána |
|---|---|---|
| DSR (N_meth = 2) | 0,999 | ≥ 0,90 ✔ |
| Rozdíl Sharpe vs. EW_UNIV (90% CI) | +0,50 [+0,07; +0,91] | dolní mez > 0 ✔ |
| Rozdíl Sharpe vs. SPY (90% CI) | +0,44 [+0,03; +0,81] | – |
| Rozdíl CAGR vs. EW_UNIV (90% CI) | +0,8 p.b. [−7,1; +8,9] | – |
| Rozdíl CAGR vs. SPY (90% CI) | +0,3 p.b. [−5,9; +6,5] | – |
| SPA p-hodnota (rozdíl denních výnosů) vs. EW / SPY | 1,0 / 1,0 | – |
| Sharpe při nákladech ×2 / ×3 | 0,90 / 0,87 | – |

Roky (primární vs. SPY): 2008 **−2,3 %** vs. −36,8 %; 2011 +10,0 % vs. +1,9 %; 2018 +2,0 % vs.
−4,6 %; naopak 2009 +18,2 % vs. +26,4 %, 2013 +20,1 % vs. +32,3 %, 2019 +13,6 % vs. +31,2 %.

## Interpretace – bez přikrášlení

1. **Metodika nepřináší vyšší výnos, jen výrazně nižší riziko.** CAGR je na úrovni SPY (9,2 % vs.
   8,9 %) a CI rozdílu CAGR je široké kolem nuly; SPA test na surovém výnosu nic nenachází
   (p = 1,0, protože průměrný denní rozdíl je dokonce mírně záporný). Statisticky významný je
   jen rozdíl **Sharpe** – při poloviční volatilitě a desetinovém propadu.
2. **„Pár" je fakticky jedna strategie.** Ve všech 15 letech vybral výběr dvě varianty low_vol
   (liší se hlavně overlayem vol_target vs. trend_filter). Jejich korelace v testovacích letech
   má medián 0,88. Původní představa „dvou doplňujících se strategií" se na datech nepotvrdila –
   nic jiného než low_vol v likvidním univerzu po nákladech neobstálo.
3. **Low-vol byl známou anomálií** (publikace 2006–2011). Výběr ho našel sám a konzistentně, ale
   rodina v mřížce byla zvolena s vědomím literatury. DSR s N_meth = 2 je proto optimistický vůči
   „lidským" stupňům volnosti před projektem.
4. **Sektorové a úrokové riziko.** 10–20 nejméně volatilních akcií = převážně utility, REIT a
   spotřební zboží. Období 2005–2019 (klesající sazby) jim přálo. Holdout 2020–2026 obsahuje
   prudký růst sazeb v roce 2022 – nejdůležitější test.
5. **Velký tracking error** vůči trhu: v silných býčích letech zaostává o 8–18 p.b.
6. Limit propadu 45 % a overlaye tvarují výsledek směrem k defenzivě – to odpovídá zadání.

## Co zbývá před finálním holdoutem (spec §12)

- Robustnost na univerzu `SP500_PIT` a citlivost na politiku delistingu (optimistická /
  pesimistická) – spustit na zmrazené metodice ve vývojovém období, ne na holdoutu.
- Rozhodnutí o Fázi 4 (ML overlay) – musí padnout **před** otevřením holdoutu.
