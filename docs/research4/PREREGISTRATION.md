# Výzkum 4 – aktivace rodin strategií podle režimu (pre-registrace)

Verze 1.0 · 2026-09-27 · Stav: **pre-registrace, žádný výsledek meta-vrstvy zatím spočten**.
Znovu používá framework výzkumů 1–3 (data, enginy, test úniku, statistika, registr).
Změny po commitu tohoto dokumentu = nový pokus v registru a řádek v §11.

## 1. Otázka

Různé rodiny strategií se vzájemně doplňují: momentum vydělává v jiných obdobích než value,
low-vol v jiných než high-beta, dluhopisy a zlato v jiných než akcie. **Dokáže meta-vrstva
z bodových (point-in-time) informací rozpoznat, kterou rodinu v daném měsíci „aktivovat", a tím
po nákladech překonat (a) trh a (b) prosté držení všech rodin najednou?**

(b) je klíčové srovnání: izoluje hodnotu *přepínání* od hodnoty *diverzifikace*.

## 2. Poctivé omezení předem

- **Historie 2020–2026 není čistý holdout** (spotřebována výzkumy 1–3). Seznam rodin níže je
  zvolen podle literatury, ale autor ví, že 2020–2026 přálo růstu, momentu a velkým firmám.
  Walk-forward přes 2006–2026 je proto **výzkumná evidence**, ne důkaz. Oba podúseky se hlásí zvlášť.
- **Jediný čistý test je forward test** (§9).
- ETF jsou dnešní „přeživší" vlajkové fondy; akcie jsou bez survivorship biasu (LIQ1000 k datu).
- Strop (oracle) z §6 není investovatelný; slouží jen ke změření, kolik z potenciálu přepínání
  meta-vrstva zachytí.

## 3. Knihovna rukávů (sleeves)

Rukáv = plně specifikovaná long-only strategie s vlastními denními čistými výnosy.
Všechny: rozhodnutí po close prvního obchodního dne měsíce, fill na open dalšího dne, měsíční
rebalance, bez overlayů (časování je úkol meta-vrstvy), první rozhodnutí 2000-01-03.

**Akciové rukávy** – univerzum LIQ1000 k datu, rovné váhy top-N podle skóre, max. váha 10 %,
N ∈ {20, 50}, náklady podle modelu výzkumu 1. Rodiny a varianty skóre (vyšší = lepší):

| Rodina | Varianty skóre |
|---|---|
| momentum | výnos 12-1 m; 6-1 m; 3 m |
| low_vol | −volatilita 63 d; −volatilita 252 d |
| low_beta | −beta k SPY 252 d |
| high_beta | +beta k SPY 252 d |
| st_reversal | −výnos 21 d; −výnos 5 d |
| lt_reversal | −výnos 36-12 m |
| high52 | cena / max 252 d |
| trend | cena / SMA 200 − 1; cena / SMA 50 − 1 |
| value | E/P; B/P; EBITDA/EV; S/P (z `daily`); kompozit (průměr pořadí 4 poměrů) |
| quality | hrubý zisk / aktiva; ROE; −dluh/vlastní kapitál; kompozit |
| investment | −meziroční růst aktiv |
| growth | meziroční růst tržeb; meziroční růst EPS |
| dividend | dividendový výnos (DPS za 12 m / cena) |
| size | −tržní kapitalizace (malé v LIQ1000); +tržní kapitalizace (největší) |
| insider | hodnota nákupů insiderů (kód P) za 91 dní / tržní kapitalizace |
| value_momentum | průměr pořadí E/P a momenta 12-1 |
| quality_value | průměr pořadí kompozitu quality a kompozitu value |

Celkem 31 variant skóre × 2 hodnoty N = **62 akciových rukávů**.

Fundamenty: `fundamentals` dimenze `ARQ`/`ART` (as-reported, `date` = datum podání u SEC),
hodnota použitelná od **následujícího** obchodního dne po podání; poměry z `daily` se berou
k close dne rozhodnutí. Meziroční růst = poslední podání vs. podání o 4 kvartály dřív.
Akcie bez hodnoty skóre se nevybírají.

**ETF rukávy** – buy-and-hold jednoho ETF (Sharadar `funds`, total return), náklady 5 bps/stranu,
dostupné od prvního dne s 252 dny historie:
SPY, QQQ, IWM, MDY, EFA, EEM, EWJ, IEF, TLT, SHY, LQD, TIP, GLD, DBC, VNQ,
XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU (24 rukávů) + **CASH** (T-bill).

Celkem **87 rukávů v 25 rodinách** (17 akciových, 8 ETF) (ETF: rodina = třída aktiv: us_equity, intl_equity, bonds,
gold, commodities, real_estate, sector, cash).

## 4. Meta-vrstva (aktivace)

Rozhodnutí první obchodní den měsíce po close (stejný den jako rukávy). Vstupem jsou jen
denní čisté výnosy rukávů do dne t a tržní data do dne t. Výstup: váhy přes rukávy
(long-only, součet 1). Portfolio = vážený součet cílových vah akcií/ETF rukávů → **simulace na
úrovni pozic** (netování obchodů mezi rukávy, skutečné náklady přepínání).

Vzorek pro učení: (měsíc m, rukáv s) s ≥ 12 m historie rukávu. Cíl: výnos rukávu za měsíc m+1
minus průřezový průměr přes rukávy v měsíci m+1.

**Příznaky rukávu:** výnos 1, 3, 6, 12 m; volatilita 1 m, 12 m; propad od 12m maxima;
beta k SPY 12 m; Sharpe 12 m; rodina (kategorie); typ (akcie/ETF).
**Tržní příznaky (stejné pro všechny rukávy):** SPY výnos 1, 3, 12 m; SPY nad SMA 200;
realizovaná volatilita SPY 1 m a 3 m; VIX a VIX / medián VIX 252 d; úroveň T-billu a jeho změna
za 12 m; průřezový rozptyl měsíčních výnosů akcií LIQ1000; šíře trhu (podíl LIQ1000 nad SMA 200);
výnos IEF − SPY za 3 m.

**Metody (N_meth = 5, všechny předem, bez ladění hyperparametrů):**

| Kód | Metoda | Skóre rukávu |
|---|---|---|
| **M3 (primární)** | ML_GBM | LightGBM regrese, 200 stromů, learning rate 0,03, 15 listů, min. 200 vzorků v listu, subsample 0,7, colsample 0,7, seed 0 |
| M1 | FACTOR_MOM | výnos rukávu za 12 m („momentum strategií") |
| M2 | REGIME | režim = (SPY nad SMA 200) × (VIX nad mediánem 252 d) → 4 stavy; skóre = (n·průměr v režimu + 12·celkový průměr)/(n + 12) budoucích měsíčních relativních výnosů v tréninku |
| M4 | ML_RIDGE | ridge (α = 10) na standardizovaných příznacích + interakce rodina × tržní příznaky |
| M5 | BLEND | průměr průřezových pořadí M1, M2, M3 |

Pravidlo výběru pro všechny: **top-5 rukávů podle skóre, rovné váhy, nejvýše 2 rukávy z jedné
rodiny**. Walk-forward: roční refit, rozšiřující se okno od 2001-01, embargo 1 měsíc (cíl
posledního tréninkového vzorku končí před prvním testovacím dnem), testovací roky 2006 … 2026.

## 5. Benchmarky

- **SPY** (total return) – trh.
- **EW_ALL** – rovné váhy všech dostupných rukávů, měsíčně (diverzifikace bez přepínání).
- **EW_STOCK** – rovné váhy akciových rukávů.
- **60/40** – 60 % SPY + 40 % IEF, měsíčně.

## 6. Diagnostika (bez vlivu na závěr)

- **Oracle top-1 a top-5**: se znalostí budoucího měsíce – strop potenciálu přepínání
  (lineární výnosy rukávů, bez nákladů přepínání).
- **Rank IC** každé metody: Spearmanova korelace skóre a skutečného výnosu příštího měsíce, průměr
  a t-statistika přes měsíce.
- **Křivka potřebné přesnosti**: selektor „oracle + šum" s cílovými IC 0,02 … 0,30 → jaká IC
  je potřeba k překonání SPY a EW_ALL.
- Četnost aktivace rodin v čase.

## 7. Kritéria úspěchu (primární M3, OOS 2006-01 … 2026-09, čistě po nákladech)

1. dolní mez 90% CI rozdílu Sharpe M3 − EW_ALL > 0 (přepínání přidává hodnotu),
2. dolní mez 90% CI rozdílu Sharpe M3 − SPY > 0,
3. DSR (N_meth = 5) ≥ 0,90,
4. bodový rozdíl Sharpe M3 − SPY > 0 v obou podúsecích 2006–2019 i 2020–2026.

Statistika: stationary bootstrap (blok 21 dní, 1 000 replikací), SPA vůči SPY a EW_ALL.
Splní-li kritéria jiná metoda než M3, je to evidence pro další výzkum, ne úspěch hypotézy.

## 8. Robustnost (předem daná)

1. Náklady ×2.
2. Top-3 a top-10 místo top-5.
3. Bez limitu 2 rukávů na rodinu.
4. Jen akciové rukávy; jen ETF rukávy (+ CASH).

## 9. Forward test

Jen pokud M3 splní kritéria 1–3: zmrazení (hash, commit), 12 měsíců 2026-10 … 2027-09, měsíční
cílové váhy zapsané do append-only logu před dalším obchodním dnem. Akciové rukávy vyžadují
obnovené předplatné Sharadaru.

## 10. Postup prací

| Krok | Obsah |
|---|---|
| R4.1 | Panel: LIQ1000 + 24 ETF; PIT matice fundamentů na dny rozhodnutí; test úniku |
| R4.2 | Knihovna rukávů: denní výnosy + měsíční cílové váhy (řídké) 2000–2026 |
| R4.3 | Meta příznaky, walk-forward metod, simulace na úrovni pozic, statistika, diagnostika, robustnost, report |

## 11. Rozhodovací log

| Datum | Rozhodnutí | Zdůvodnění |
|---|---|---|
| 2026-09-27 | Pre-registrace v1.0 | Uživatel chce otestovat původní záměr: více rodin + rozpoznání, kdy kterou aktivovat. Registr výzkumu 4: `runs/registry/research4.sqlite` |
