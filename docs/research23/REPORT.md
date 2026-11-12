# Výzkum 23 – VWAP Wave (Drysdale) na BTC/ETH 5m: výsledek

2026-11-12 · pre-registrace v1.0.2 · Binance perpy BTCUSDT + ETHUSDT 5m (snapshot `binance_2026-10-03`) ·
vývoj 2020-01 → 2022-12 · N = 20 (registr `runs/registry/research23.sqlite`) · výsledky `results/dev.json`

## Verdikt

**Zamítnuto na vývojových datech, všechny 4 setupy.** Všech 20 variant má po nákladech (7,5 bps na stranu)
záporný Sharpe i průměrný obchod. Vývojovou bránou (§7) neprošel nic, **holdout 2023–2026 se neotevřel** a zůstává
nedotčený pro tato pravidla. Nejde o hraniční případ: i nejlepší varianta ztrácí 7 bp na obchod a S2–S4
prodělávají už **před náklady**.

| Setup | P_f (nejlepší z 4) | Obchodů | Obchodů/den | Čistý Sharpe | Čistý obchod | Hrubý obchod | B_RND medián | p B_RND |
|---|---|---|---|---|---|---|---|---|
| S1 Price Discovery Continuation | `PB_EOD` | 7 368 | 6,7 | −1,60 | −8,5 bp | **+6,5 bp** | −3,6 bp | 0,998 |
| S2 Fade Value Area Extremes | `REJ_OPP` | 4 980 | 4,5 | −8,40 | −18,8 bp | −3,8 bp | −21,8 bp | 0,004 |
| S3 Return to Value | `PB_OPP` | 2 914 | 2,7 | −5,97 | −19,2 bp | −4,2 bp | −24,2 bp | 0,002 |
| S4 VWAP Bounce | `STR_2R` | 7 539 | 6,9 | −5,60 | −13,2 bp | +1,8 bp | −10,5 bp | 1,000 |

Obchody za oba rukávy dohromady. Hrubý = bez poplatků a skluzu, s fundingem. Medián B_RND je čistý průměrný obchod
náhodných vstupů téhož dne se stejnou stranou a stejnou vzdáleností stopu a cíle.

Všech 16 variant při k = 1 a 4 při k = 2: čistý průměrný obchod −7 až −21 bp, Sharpe −1,6 až −22,8 (`dev.json`).
Primární varianty podle textu letáku: S1 `PB_EOD` −8,5 bp, S2 `REJ_VWAP` −17,6 bp, S3 `PB_VWAP` −17,6 bp,
S4 `STR_BAND` −16,4 bp. Pásma 2σ místo 1σ nepomohla ani jednomu setupu.

## Co z toho plyne

**1. Mean-reversion setupy (S2, S3, S4) nemají edge ani bez nákladů.** Fade okrajů value area a návrat k VWAP
mají hrubě −1 až −5 bp na obchod. Načasování S2 a S3 je proti náhodnému vstupu téhož dne o 3–5 bp lepší (p ≤ 0,004),
ale to jen znamená „méně špatné než náhoda se stejnou geometrií stopu a cíle“. Stop ~0,4 % a cíl u VWAP nebo
protějšího pásma dávají hit rate ~32 %, a to výplatu nepokryje.

**2. S1 (pokračování po průrazu) je jediný s kladným hrubým výsledkem, ale nepokryje náklady.** Hrubě +3 až +8 bp
na obchod podle varianty (Sharpe 1,2–2,3 bez nákladů). Zisk ale:
- je skoro celý v **longu** (`PB_EOD` long +11,6 bp, short +1,2 bp) a na **ETH** (BTC +2,6 bp),
- slábne v čase: Sharpe bez nákladů 2020 1,6 → 2021 1,4 → 2022 0,65. Při 2,5 bp na stranu je rok 2022 záporný.
To je stejný vzor jako ORB ve výzkumu 16: intradenní momentum v býčím trhu 2020–21, který se na holdoutu nevrátil.
B_RND u S1 neříká nic o načasování. Náhodný vstup téhož dne dostane stranu z pozdějšího průrazu, tedy vidí
směr dne dopředu (vstup v 6:00 long v den, kdy cena odpoledne prorazí nahoru). Proto je nula „lepší“ než setup
(p = 0,998).

**3. Náklady na kryptu jsou pro 5m scalp smrtelné, jak se čekalo.** 15 bp na kolo proti mediánovému stopu
~0,4 % znamená, že náklady jsou ~0,4 R na každý obchod. S 3–20 obchody denně to žádný z popsaných mechanismů
neunese.

**4. Konvence plnění výsledek nezkreslují.** Stop i cíl ve stejné svíčce (konzervativně se počítá stop) nastane
u 0,0–0,4 % obchodů. Kdyby se počítal cíl, hrubý průměr se zlepší o < 0,3 bp.

## Co to neříká

- **Netestuje to ES**, pro které je leták napsaný. ES stojí ~1–2 bp na kolo (1 tick + poplatek), ne 15 bp. Jediný
  setup, který by tam mohl přežít, je **S1**: na kryptu má hrubý edge 3–8 bp. Průzkumně (vývoj, žádná nová
  varianta) při 1 bp na stranu je S1 `BRK_EOD` +5,9 bp a Sharpe 1,6, ale 2022 jen 0,18. S2–S4 mají hrubý výsledek
  záporný, takže je nezachrání ani nulové náklady. Test na ES/SPX je dodatek v1.1 a potřebuje data (Dukascopy
  blokuje, placený zdroj čeká na rozhodnutí vlastníka).
- Operacionalizace je moje (§4 pre-registrace). Diskrétní trader s kontextem (profil objemu, Initial Balance,
  „čtení“ auction market theory z placené knihy) může obchodovat jinak. Leták ale žádnou pravidlovou výhodu
  nedokládá: neobsahuje jediné číslo, jen čtyři vybrané grafy z jednoho dne (18. 3. 2025).

## Průběh a odchylky

1. Pre-registrace v1.0 commitnuta před výpočtem (`d046e3fa`).
2. v1.0.1 (před jakýmkoli výpočtem výnosu): S1 backtest svíčka musí zavřít u pásma (§12 pre-registrace).
3. První vývojový běh spadl na zápisu do registru (špatný druh pokusu), nic se nezaregistrovalo ani nespočítalo.
4. Druhý běh doběhl. Pak se ukázalo, že B_RND byl zkreslený dolů (pravidlo nepřekrývání záviselo na výsledku
   obchodu). Oprava v1.0.2 s regresním testem, vývojový běh zopakován, rerun je v registru jako `other` bez
   nových konfigurací (N = 20). Opravou se změnila jen čísla B_RND. Verdikt je stejný, protože bránou neprošla
   žádná varianta ani v Sharpe.
5. **Vedlejší nález pro výzkum 15:** `research15.strategy.random_book` má stejný vzor (`if occ[e:x+1].any():
   continue` po simulaci výstupu stopem/cílem). Jeho nula B_RND byla proto vychýlená dolů, tedy **příznivě pro
   strategii**. Tvrzení výzkumu 15, že `BRK N48 long R3` „porazil 99 % náhodných vstupů“, je nadhodnocené. Verdikt
   výzkumu 15 (zamítnuto) se tím nemění. `research16.engine.random_book` postižený není: náhodné obchody tam končí
   časem (délka držení se losuje předem), takže zamítnutí nezávisí na výsledku. Výzkumy 16–18 tedy platí.
