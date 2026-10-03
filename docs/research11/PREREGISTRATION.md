# Výzkum 11 – ML výběr akcií z fundamentů, insiderů a toků 13F (pre-registrace)

Verze 1.0 · 2026-10-03 · Stav: **pre-registrace, nic zatím spočteno**.
Navazuje na výzkum 4 (panel `panel_r4`, LIQ1000, point-in-time skóre, LightGBM, nákladový model) a výzkum 10
(závěr: přepínání z minulých výnosů nefunguje, ML potřebuje nové informace a velký počet sázek).
Změny po commitu = nový pokus v registru a řádek v §10.

## 1. Otázka

**Vybere LightGBM z bodových fundamentů, transakcí insiderů a změn institucionálního držení (13F) měsíčně
50 akcií z LIQ1000 tak, aby portfolio po nákladech překonalo SPY, rovné váhy LIQ1000 a stejný model
jen s cenovými příznaky?**

Proč průřez a ne přepínání: měsíčně ~1 000 akcií = ~12 000 pozorování ročně místo ~300 u přepínání rodin
(fundamentální zákon, výzkum 10). Literatura (Gu, Kelly, Xiu 2020) hlásí přínos ML právě na průřezu.

## 2. Poctivé omezení předem

- **Období 2020–2026 není neviděné.** Cenové strategie na něm selhaly (výzkum 1) a výzkum 4 na něm vyhodnotil
  jednofaktorové rukávy value, quality, growth a insider. Kombinace přes ML a toky 13F tam testovány nebyly.
  Výsledek může hypotézu **zamítnout, nebo kvalifikovat na forward test**, nic víc.
- **13F začíná 2013-06-30** (Sharadar SF3 v tomto snapshotu), insideři 2008-01. Model s toky se proto učí
  vztah 13F jen z let 2014+. Snapshot obsahuje dodatečné opravy 13F (amendments), může tak mírně předbíhat.
- Hyperparametry jsou pevné předem (výzkum 4), žádné ladění. Modely jsou tři, výběr mezi nimi se nedělá.

## 3. Data, univerzum, rozvrh

- Panel `panel_r4` (akcie LIQ1000 k datu + ETF), total return, delistingy s protihodnotou.
- Rozhodnutí po close **prvního obchodního dne měsíce** (dny výzkumu 4), plnění na openu dalšího dne.
- Univerzum: členové LIQ1000 v den rozhodnutí.
- **Cíl (label):** výnos od plnění v měsíci *m* do plnění v měsíci *m*+1 (open → open, včetně delistingu),
  převedený na průřezový percentil v rámci univerza. Label je použitelný pro trénink, až když je celý
  realizovaný před prvním rozhodnutím testovacího roku.

## 4. Příznaky (všechny point-in-time k close dne rozhodnutí)

Do modelu vstupují jako průřezové percentily v rámci univerza (NaN zůstává NaN).

**P – cena (12, z výzkumu 4):** mom_12_1, mom_6_1, mom_3, lowvol_63, lowvol_252, lowbeta_252, strev_21,
strev_5, ltrev_36_12, high52, trend_200, trend_50.

**F – fundamenty (15):** z výzkumu 4 value_ep, value_bp, value_ebitda_ev, value_sp, size (tržní
kapitalizace), quality_gpa, quality_roe, quality_lowlev, invest_lowag, growth_rev, growth_eps, dividend.
Nové (ART/ARQ, použitelné den po podání, max. stáří 400 dní):
- `accruals` = −(netinc − ncfo) / assets (ART),
- `net_issuance` = −(sharesbas / sharesbas před rokem − 1) (ART),
- `eps_surprise` = (čistý zisk poslední kvartál − čistý zisk stejný kvartál před rokem) / tržní kapitalizace k datu
  rozhodnutí (ARQ; viz §10).

**I – insideři (5), okno 91 dní, podle data podání < den rozhodnutí:** `ins_buy` (nákupy P / tržní
kapitalizace, z výzkumu 4), `ins_sell` (prodeje S / tržní kapitalizace), `ins_n_buyers`, `ins_n_sellers`
(počet různých osob), `ins_net_n` = kupující − prodávající.

**H – 13F (5), z `holdings_ticker`, kvartál ke dni D použitelný až od D + 46 dní (lhůta 45 dní):**
- `io` = hodnota akcií v držení 13F / tržní kapitalizace ke dni D,
- `d_io` = změna `io` proti předchozímu kvartálu,
- `d_holders` = log(počet držitelů / počet držitelů v předchozím kvartálu),
- `d_breadth` = změna podílu všech 13F filerů, kteří akcii drží (Chen, Hong, Stein 2002),
- `putcall` = (držitelé putů − držitelé callů) / držitelé akcií.

## 5. Modely a portfolio

| Model | Příznaky |
|---|---|
| `M_P` | P (kontrola: stejné ML jen z cen) |
| `M_PFI` | P + F + I |
| **`M_ALL`** (primární) | P + F + I + H |

- LightGBM regresor, parametry výzkumu 4 (`GBM_PARAMS`: 200 stromů, learning rate 0,03, 15 listů,
  min. 200 vzorků v listu, …), seed 0.
- Walk-forward: přetrénování každý leden, rozšiřující se okno, trénink od rozhodnutí 2003-01.
- Portfolio: top 50 podle predikce, rovné váhy 2 %, měsíčně, vektorový engine, nákladový model výzkumu 4
  (stupně podle likvidity), zátěž ×2.
- Benchmarky: SPY (buy and hold, 5 bps), **EW LIQ1000** (rovné váhy všech členů univerza, měsíčně,
  stejné náklady).

## 6. Fáze a trezor

| Fáze | Testovací roky | Data |
|---|---|---|
| Vývoj | 2010–2019 (M_ALL hlavně 2014–2019) | jen ≤ 2019-12-31; poslední rozhodnutí, jehož label končí v roce 2020, se vynechá |
| Finále | 2020-01 → 2026-08 | jednou přes nový trezor `runs/vault_r11` (hranice 2019-12-31) |

Finále se spustí bez ohledu na výsledek vývoje (nic se nevybírá). Ve finále se přetrénovává každý leden
2020–2026 na všech datech, jejichž label je realizovaný.

## 7. Hlášení

Pro každý model a fázi: CAGR, Sharpe, volatilita, max. propad, obrat, náklady, výsledky po letech; **rank IC**
predikce s realizovaným výnosem po měsících (průměr, t-statistika, podíl kladných); důležitost příznaků
(gain) po skupinách P/F/I/H.

## 8. Kritéria (finále 2020-01 → 2026-08, nutné splnit **všechna**)

1. ΔSharpe M_ALL − SPY > 0 a dolní mez 90% intervalu > 0 (stationary bootstrap párově, blok 21, 1000, seed 0),
2. ΔSharpe M_ALL − EW LIQ1000 > 0,
3. ΔSharpe M_ALL − M_P > 0 **a** průměrný rank IC M_ALL > M_P (fundamenty a toky přidávají),
4. ΔSharpe M_ALL − SPY > 0 v obou podúsecích 2020-01 → 2022-12 a 2023-01 → 2026-08,
5. ΔSharpe M_ALL − SPY > 0 při zátěžových nákladech ×2,
6. Deflated Sharpe aktivního výnosu (M_ALL − SPY) ≥ 0,95, N = 3 (rozptyl Sharpe z vývoje); popisně i
   s N = 3 027 (mřížka výzkumu 1).

Splnění = kandidát na forward test, ne důkaz (§2).

## 9. Postup

1. Commit této pre-registrace.
2. Příznaky F/I/H + testy (point-in-time: zkrácení a perturbace, lhůta 13F, podání insiderů), label, WFO.
3. Vývoj `scripts/r11_dev.py`, commit výsledků.
4. Finále `scripts/r11_final.py freeze|run`, jednou.
5. `REPORT.md`, řádek v README.

## 10. Log změn

| Datum | Změna | Důvod |
|---|---|---|
| 2026-10-03 | Verze 1.0 | – |
| 2026-10-03 | Před výpočtem: `eps_surprise` z čistého zisku / tržní kapitalizace místo EPS / ceny. Sharadar zpětně upravuje EPS na splity (AAPL 2019-Q4 eps 1,26 místo vykázaných 4,99), nepřepočtená cena k datu by tak míchala budoucí splity. | únik budoucnosti |
