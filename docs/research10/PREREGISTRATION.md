# Výzkum 10 – doplňující se strategie a short strana (pre-registrace)

Verze 1.0 · 2026-10-03 · Stav: **uzavřeno, zamítnuto** (`REPORT.md`).
Navazuje na výzkum 4 (87 rukávů, meta-vrstva), výzkum 2 (ETF panel) a výzkum 9 (krypto trend, trezor).
Změny po commitu = nový pokus v registru a řádek v §9.

## 1. Otázky

Nápad: najít dvě strategie, které se doplňují (když jedna ztrácí, druhá vydělává), případně jednu long a jednu
short, a přepínat mezi nimi modelem. Výzkum 4 přepínání přes 87 long-only rukávů zamítl. Zde se oddělí tři věci:

- **A (dvojice, popisně + výběr):** Existují mezi rukávy výzkumu 4 dvojice, kde každý rukáv sám vydělává a
  vzájemně se v krizích doplňují? Kolik dá **statická** kombinace 50/50, kolik by dal **dokonalý** měsíční
  přepínač (oracle) a kolik z toho zachytí naivní přepínače? Tím se změří, jak velký prostor pro ML vůbec je.
- **B (krok 1, akcie + trend long/short):** Zlepší statická kombinace 50 % SPY + 50 % time-series momentum
  long/short na 9 ETF (managed-futures styl) Sharpe proti SPY a proti 60/40, a přidá short strana něco proti
  stejné strategii bez shortu?
- **C (krok 2, krypto long/short):** Zlepší BTC + ETH s trendovým pravidlem, které pod klouzavým průměrem jde
  **short přes perp** (místo do USDT), Sharpe proti držení a proti filtru do hotovosti z výzkumu 9?

## 2. Poctivé omezení předem

- **Akcie a ETF nemají čistý holdout.** Data po 2019-12-31 chrání trezor výzkumu 1 a 2020–2026 je spotřebované.
  A i B proto používají jen roky ≤ 2019: výběr 2001/2004–2012, kontrola 2013–2019. Období 2013–2019 autor zná
  z výzkumů 1–4. Výsledek A a B může hypotézu **zamítnout, nebo kvalifikovat na forward test**, nic víc.
- Time-series momentum na futures je v literatuře známé (Moskowitz, Ooi, Pedersen 2012) a víme, že v roce
  2008 vydělalo. Na 9 ETF s denními daty ale nebylo v tomto projektu testováno se short stranou.
- **Krypto:** holdout 2022-01 → 2026-08 je stejný jako ve výzkumu 9 a jeho průběh (medvědí 2022) je známý.
  Short strana v medvědím roce pomůže skoro jistě; otázka je, zda to vydrží i 2023–2026.
- **Ekvivalence předem:** „držení 50 % + short rukáv 50 % jen pod SMA“ (varianta HS v §5) má čistou expozici
  0,5 nad SMA a 0 pod ní, tj. je to trendový filtr výzkumu 9 v poloviční velikosti. Očekáváme, že se tak bude
  chovat. Skutečně nová je jen varianta LS (plný long nad SMA, plný short pod ní).
- ML přepínač se zde **netrénuje**. Smysl má jen tehdy, když A ukáže, že naivní přepínání zachytí podstatnou část
  oracle a statická kombinace nechává velký prostor.

## 3. Společná pravidla

- Rozhodnutí po close dne *t*, plnění na openu *t*+1 (kontrakt projektu).
- Nový simulátor `qlab.research10.sim.simulate_signed`: podepsané cílové váhy (short < 0), součet |w| ≤ 1,
  NaN = pozici neměnit. Náklady za stranu z obchodované hodnoty. Short: výnos −r, výnosy z prodeje short pozice
  nenesou úrok (T-bill se platí jen na hotovost bez nich), denní short carry (poplatek za půjčku nebo funding).
  Pro long-only váhy musí dávat stejný výsledek jako `qlab.engine.vector.simulate` (test parity).
- Kombinace dvou rukávů: váhy 50/50 obnovované první obchodní den v měsíci, mezi tím driftují. Náklady na obnovu
  mezi rukávy se zanedbávají (pod 1 bp měsíčně u ETF), náklady uvnitř rukávů jsou zahrnuté.
- Sharpe: denní, anualizace √252 (akcie, ETF), √365 (krypto).

## 4. Část A – dvojice rukávů výzkumu 4

- Data: denní čisté výnosy 87 rukávů (`research4/sleeves/returns.npy`), bez CASH, jen do 2019-12-31.
- **Okno výběru 2001-01-02 → 2012-12-31**, způsobilé jsou rukávy s daty po celé okno.
  (GLD, DBC a další mladší ETF tím vypadnou; zlato a komodity pokrývá B.)
- Pro každou dvojici z různých rodin: Sharpe obou, Sharpe kombinace 50/50, korelace denních výnosů celkem
  a v krizi (dny, kdy je SPY > 15 % pod svým maximem), CAGR, max. propad.
- Přepínače na dvojici, měsíčně, 100 % v jednom rukávu: **oracle** (vybere rukáv s vyšším výnosem příští měsíc),
  **mom1** (vítěz minulého měsíce), **mom12** (vítěz za 12 měsíců). Zachycení = (SR přepínače − SR statické) /
  (SR oracle − SR statické).
- Výběr: dvojice, kde oba rukávy mají Sharpe ≥ 0,3 v okně výběru, seřazené podle Sharpe kombinace;
  **P_A** = první. Hlásí se top 10.
- **Kontrola 2013-01-02 → 2019-12-31** na P_A a top 10 (popisně i kritéria §7).
- Srovnání: SPY (rukáv `etf_SPY`), 60/40 (`etf_SPY` + `etf_IEF` 60/40 měsíčně), EW všech způsobilých rukávů.

## 5. Kandidáti

### 5.1 Část B – TSMOM na ETF (12 kandidátů)

- Univerzum: SPY, EFA, EEM, IEF, TLT, TIP, GLD, DBC, VNQ (panel výzkumu 2); ETF je způsobilý po 252 dnech
  historie.
- Měsíčně (první obchodní den, po close): signál = znaménko výnosu za L měsíců (21·L dní) **minus** T-bill za
  stejné období, L ∈ {3, 6, 12}.
- Váhy: `eq` = ±1/N způsobilých; `iv` = ±(1/σ63) normalizované na Σ|w| = 1.
- Typ: `LS` (záporný signál = short), `LO` (záporný signál = hotovost; kontrola, zda short přidává).
- Náklady 5 bps/stranu (výzkum 4), zátěž 10 bps. Short: poplatek 0,5 % p. a. z hodnoty shortu, prodej bez úroku.
  Hotovost nese T-bill (`extra_cash_ret`).
- Portfolio: **50 % SPY + 50 % TSMOM**, měsíčně obnovované.
- Výběr na **2004-01-02 → 2012-12-31**: **P_B** = kandidát `LS` s nejvyšším Sharpe kombinace. Kontrola P_LO =
  `LO` se stejným L a váhami.
- Kontrola 2013-01-02 → 2019-12-31.

### 5.2 Část C – krypto long/short (8 kandidátů)

- BTC a ETH, každý 50 % NAV, každý se svým signálem close > SMA(n), n ∈ {20, 50, 100, 200} (výzkum 9).
- `LS_n`: nad SMA long spot 0,5 NAV; pod SMA short perp 0,5 NAV. Obchoduje se jen při změně signálu
  (velikost = 0,5 aktuálního NAV), bez průběžného rebalance. Před dostatkem historie: hotovost.
- `HS_n`: 50 % rukáv držení (BTC/ETH 50/50 měsíčně) + 50 % short rukáv (coin short 0,5 NAV rukávu pod SMA,
  jinak USDT), měsíčně obnovované.
- Short se počítá na spotových cenách (rozdíl perp–spot se zanedbává) a dostává skutečný funding Binance
  (`fund_hold` z panelu výzkumu 8, kladný funding platí long → short dostává). Před 2019-12-31 funding chybí
  a bere se 0. USDT nenese nic.
- Náklady: model výzkumu 9 (poplatek 10 bps + slippage podle objemu) pro obě nohy; zátěž ×2.
  **Zátěž fundingu:** short po celé období platí 0,03 % denně (≈ 11 % p. a.) místo skutečného fundingu.
- Srovnání: `HOLD` = BTC/ETH 50/50 měsíčně (benchmark T výzkumu 9), `LC_n` = filtr do hotovosti výzkumu 9 se
  stejným n (`T_sma{n}`).
- **Vývoj 2018-04-01 → 2021-12-31**: **P_C** = kandidát s nejvyšším Sharpe.
- **Holdout 2022-01-01 → 2026-08-31** jednou přes nový trezor `runs/vault_r10`.

Celkem pokusů: B 12, C 8 (registr `runs/registry/research10.sqlite`). A je popisná diagnostika; výběr P_A je
jeden pokus nad 1 dvojicí z deterministického pravidla, ale prohledává se ~3 000 dvojic, proto se hlásí i DSR
s N = počet způsobilých dvojic.

## 6. Hlášení

Pro každou část: CAGR, Sharpe, volatilita, max. propad, obrat, náklady, výsledky po letech, korelace
s SPY/HOLD v krizi. A navíc: rozdělení Sharpe statických kombinací, oracle vs. naivní přepínače, zachycení.

## 7. Kritéria (nutné splnit **všechna** dané části)

**A (P_A na kontrole 2013–2019):** ΔSharpe vs SPY > 0 a vs 60/40 > 0; max. propad menší než SPY;
ΔSharpe vs 60/40 > 0 v obou podúsecích 2013–2016 a 2017–2019. Splnění = kandidát na forward test, ne důkaz.

**B (P_B, kombinace 50/50, kontrola 2013–2019):** ΔSharpe vs SPY > 0; ΔSharpe vs 60/40 > 0;
ΔSharpe vs kombinace s P_LO > 0 (short přidává); ΔSharpe vs 60/40 > 0 při zátěžových nákladech;
max. propad menší než SPY. Splnění = kandidát na forward test.

**C (P_C, holdout):**
1. ΔSharpe vs HOLD > 0 a dolní mez 90% intervalu > 0 (stationary bootstrap párově, blok 21, 1000, seed 0),
2. ΔSharpe vs `LC_n` se stejným n > 0 (short strana přidává proti hotovosti),
3. ΔSharpe vs HOLD > 0 v obou podúsecích 2022-01 → 2023-12 a 2024-01 → 2026-08,
4. ΔSharpe vs HOLD > 0 při zátěži nákladů ×2 i při zátěži fundingu,
5. Deflated Sharpe aktivního výnosu (P_C − HOLD) ≥ 0,95 (N = 8 + 26 pokusů výzkumu 9 = 34).

Splnění C = kandidát na forward test (paper bot), protože holdout není neviděný (§2).

## 8. Postup

1. Commit této pre-registrace (nic spočteno).
2. Simulátor + testy (parita s vektorovým enginem, short P&L, carry, NaN = držet, point-in-time signálů).
3. `scripts/r10_pairs.py` (A), `scripts/r10_tsmom.py` (B), `scripts/r10_crypto_dev.py` (C vývoj).
4. `scripts/r10_crypto_final.py freeze|run` (C holdout, jednou).
5. `REPORT.md`.

## 9. Log změn

| Datum | Změna | Důvod |
|---|---|---|
| 2026-10-03 | Verze 1.0 | – |
| 2026-10-03 | Upřesnění před výpočtem: váhy `iv` se normalizují přes **všechny** způsobilé ETF (Σ 1/σ), takže `LS` má Σ\|w\| = 1 a `LO` je přesně `LS` se shorty nahrazenými hotovostí (stejně jako `eq`). | Pre-registrace u `LO` nebyla jednoznačná |
| 2026-10-03 | A: kritérium 3 mělo obrácené znaménko (`max_drawdown` je kladné číslo); opraveno před interpretací, verdikt beze změny (kritéria 1, 2, 4 neprošla). Okno 2001–2012 vyřadilo i dluhopisové ETF (rukáv IEF/TLT od 2003), proto navíc **post hoc** popisný běh s oknem 2004–2012 (`pairs_posthoc2004.json`), zapsaný v registru jako `other`. | chyba v kódu; neočekávaný dopad pravidla způsobilosti |
| 2026-10-03 | A: P_A zamítnuto (1 ze 4). B: P_B = LS_L3_eq zamítnuto (1 z 5). C vývoj: P_C = HS_20, všech 8 variant pod filtrem do hotovosti. | `results/pairs.json`, `tsmom.json`, `crypto_dev.json` |
| 2026-10-03 | C: metodika zamčena (`c0c4d945…`), holdout otevřen jednou. P_C zamítnuto (3 z 5: CI obsahuje 0, DSR 0,02). | `results/crypto_final.json`, `REPORT.md` |
