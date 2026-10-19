# Výzkum 18 – LEAD: dvě hypotézy po post hoc backtestu variant (pre-registrace forward testu)

Verze 1.0 · 2026-10-19 · Stav: **schváleno vlastníkem, čeká na implementaci a zmrazení kódu**.
Navazuje na výzkum 17 a na post hoc backtest 30 variant (`/lead/backtest.html`, `FORWARD.md` §9 z 2026-10-19).
Po commitu verze 1.0 platí: změny = řádek v §12 s důvodem, **před** dalším vyhodnocením.

## 1. Otázky

Dvě hypotézy, každá se **fixním pravidlem** a **fixním rozhodnutím**. Obě vznikly po pohledu na historii (§2), proto
jediným důkazem je forward na nových datech.

- **H1 – `LEAD_LAG0`.** Vydělá long altcoinu na 4 h po skoku BTC ≥ 3σ, když altcoin za stejnou hodinu **klesl**
  (varianta `V05_lag0` z `qlab.research17.variants`), po nákladech a fundingu?
- **H2 – `LEAD_K3_BTC200`.** Vydělá primární pravidlo výzkumu 17 (`LEAD_K3_LONG` = `V01_base`), když se obchoduje **jen
  tehdy, je-li BTC spot nad svým 200denním průměrem**?

## 2. Poctivé omezení předem

- **Obě hypotézy jsem vybral po pohledu na stejnou historii 2020-05 → 2026-09.** H1 je nejlepší z 30 variant
  (Sharpe 1,25; 0,88 při 20 bps; DSR 0,69 i s mírnější laťkou, tedy ne nad 0,95). H2 je jedna ze čtyř vrstev, které jsem
  zkusil (BTC 30denní trend, BTC nad MA200, vol targeting, snížení podle propadu; Sharpe V01 1,23 oproti 0,80).
  **Historická čísla nejsou důkaz.** Kumulativně je na těchto datech aspoň N = 47 + 30 + 12 = 89 pokusů.
- **Neviděná data jsou jen ta po commitu verze 1.0.** Data od 2026-10-01 do commitu už viděna byla (dashboard LEAD
  forward ukazuje varianty včetně V05 denně), proto test **začíná prvním UTC dnem po commitu** (§6).
- **H1 je podmnožina signálů H2 před filtrem** (`rA < 0` implikuje `rA/rB < 0,5`), korelace denních výnosů 0,74.
  Hypotézy nejsou nezávislé; Holmova korekce (§7) je konzervativní i při závislosti.
- **Kandidáta na MA200 jsem nenavrhl z teorie, ale z toho, že BTC za posledních 30 dní −20 % je v datech špatný
  režim** (27 z 30 variant tam ztrácí). MA200 je učebnicová volba bez ladění prahu, ale mechanismus je pořád post hoc.
- **Počet pozorování je počet událostí BTC, ne obchodů** (jeden skok spustí obchody ve více altcoinech). Interval a
  p-hodnoty se počítají po událostech (§7).
- **Síla testu je malá** (§8). I když je hypotéza pravdivá a edge má historickou velikost, pravděpodobnost úspěchu za
  12 měsíců je zhruba 20–30 % u H1 a 10–20 % u H2. Nejčastější výsledek bude proto „neprůkazné“, ne „potvrzeno“.

## 3. Data

Totéž co `FORWARD.md` §5, sdílené (jeden stahovač, jedna sada souborů `data/raw/binance_forward17/`):
- Svíčky 15m perpů a BTC spotu z denních ZIPů `data.binance.vision` (sha256), ceny a funding perpu, členství podle
  měsíčních top 10 (`qlab.research15.universe`), členství pevné po výpočtu.
- Historie před začátkem testu z `binance_2026-10-03`, indikátory (σ za 672 svíček, MA200) se počítají z dat před
  začátkem testu, na začátku se nedrží žádná pozice.

## 4. Pravidla (beze změny po commitu)

Společné: vůdce BTC spot; σ₁ₕ = 2σ_B (σ z 672 svíček 15m); `rB`, `rA` = log změna za 4 svíčky; svíčky *k*−4 … *k*
reálné u BTC i altcoinu; vstup za close svíčky *k*, výstup za close *k* + 16 (4 h), bez stopu a cíle; signály během
otevřené pozice na stejném altcoinu se ignorují; vstup jen do člena měsíce; otevřená pozice doběhne.

| | Signál (long altcoinu) | Navíc |
|---|---|---|
| **H1** | `rB ≥ 3 · σ₁ₕ` **a** `rA < 0` (v kódu `ratio = 0,0`, tj. `rA / rB < 0`; pozn.: `rA = 0` signál nedává) | nic |
| **H2** | `rB ≥ 3 · σ₁ₕ` **a** `rA / rB < 0,5` (primární pravidlo výzkumu 17) | **jen když close BTC spotu ve svíčce *k* > průměr closů BTC spotu za posledních 19 200 svíček (200 dní) včetně *k***; bez platného průměru žádný vstup |

Filtr H2 se vyhodnocuje při **vstupu**; otevřená pozice se kvůli poklesu BTC pod MA200 nezavírá.

## 5. Obchod a portfolio

Beze změny proti výzkumu 17 (§5 `research17/PREREGISTRATION.md`): 10 rukávů po 1/10 kapitálu pro člena měsíce, hotovost
bez úroku, **10 bps na stranu**, funding perpu, denní výnos = součet 15m výnosů dne (UTC). Každá hypotéza má
**vlastní portfolio** (žádné sdílení kapitálu mezi H1 a H2).

## 6. Období

| Fáze | Období |
|---|---|
| Začátek forwardu | **první UTC den po commitu verze 1.0** (`T0`), 00:00 UTC |
| 1. rozhodnutí (§7) | konec dne `T0` + 12 měsíců − 1 den (23:45 UTC) |
| 2. rozhodnutí | konec dne `T0` + 24 měsíců − 1 den, jen pokud 1. rozhodnutí u dané hypotézy vyšlo „neprůkazné“ |

Před `T0` se pro tento test nic nehodnotí. Průběžné reporty (popisné) se smějí generovat, ale nikdy nic nezastavují,
neprodlužují ani nemění a nevybírá se podle nich.

## 7. Kritéria a rozhodnutí

Sharpe = průměr / směrodatná odchylka denních výnosů × √365, denní mřížka od `T0`, dny bez expozice = 0.

**Událost** = svíčka *k*, na které vznikl vstup aspoň v jednom altcoinu příslušné hypotézy.
**p-hodnota hypotézy** = event bootstrap (převzorkují se události i se všemi svými obchody, 10 000 vzorků, seed 0),
statistika = průměrný čistý výnos obchodu, jednostranně: `p = (1 + #{vzorky s průměrem ≤ 0}) / 10 001`.

**Hypotéza uspěla**, právě když platí **všechno**:
1. **Holmova korekce přes H1 a H2** při α = 0,025 na jedno rozhodnutí: menší z dvou p-hodnot < 0,0125, větší < 0,025.
   (Úroveň 0,025 je Bonferroniho půlka z 0,05 pro dvě rozhodnutí §6; dohromady tak celé studium neudělá falešný
   úspěch s pravděpodobností větší než ~5 %.)
2. Sharpe portfolia > 0 při nákladech ×1 (10 bps).
3. Sharpe portfolia > 0 při nákladech 15 bps na stranu.
4. Počet událostí ≥ minimum: **H1 ≥ 30, H2 ≥ 60**.

**Hypotéza zamítnuta**, právě když při událostech ≥ minimum platí, že Sharpe při ×1 ≤ 0 **nebo** průměrný čistý výnos
obchodu ≤ 0. Zamítnutá hypotéza se uzavírá a v dalším rozhodnutí se nepočítá do Holma (druhá se testuje s α = 0,025
a prahem 0,025).

**Jinak neprůkazné:** (a) málo událostí, nebo (b) kladný odhad, ale kritérium 1 nebo 3 nesplněno. Po 1. rozhodnutí
se neprůkazná hypotéza prodlužuje do 2. rozhodnutí (kumulativně 24 měsíců, nová data se přidávají k dosavadním);
neprůkazné i potom = **nepotvrzeno**, bez dalšího prodlužování (pak nová pre-registrace s novými daty).

Výsledek při ×1 je rozhodující; citlivost na náklady, zpoždění a ostatní scénáře se jen hlásí.

## 8. Síla testu (hrubý odhad, předem)

Za předpokladu, že skutečný průměrný obchod = historický (optimistické, protože výběr zvyšuje historii), a že
směrodatná chyba roste s odmocninou počtu událostí:

| | hist. události (6,4 roku) | hist. průměrný obchod, SE | události za 12 / 24 měsíců | síla 12 měsíců | síla 24 měsíců |
|---|---|---|---|---|---|
| H1 | 345 | +0,61 %, 0,17 % | ~54 / ~108 | ~20–30 % | ~40–55 % |
| H2 | ~690 (odhad) | ≥ +0,21 %, 0,09 % (hodnota V01 bez filtru) | ~108 / ~215 | ~10–15 % | ~15–25 % |

Důsledek: nejpravděpodobnější výsledek je **neprůkazné**. Pre-registrace slouží hlavně k tomu, aby falešný úspěch
nebyl možný, ne k tomu, aby rychle potvrdila edge. Dá se tím také odhadnout, zda vůbec stojí za to čekat 24 měsíců
(odpověď: u H2 pravděpodobně jen kvůli odmítnutí, ne potvrzení).

## 9. Jen hlášení (nerozhoduje)

- **Kombinace V05 + filtr MA200** (nepatří do žádné hypotézy: je to spojení dvou post hoc výběrů).
- **H2 bez filtru (= `V01_base`) a H1 bez ničeho navíc** vedle sebe, rozdíl H2 − V01 s event bootstrapem (zda filtr
  skutečně pomáhá, ne jen vypadá).
- Všech 30 variant `FORWARD.md` §10 dál jako popisné; nové varianty ne.
- Náklady 7,5 / 15 / 20 bps, plnění o 1 a 2 svíčky později, bez fundingu, po měsících, po altcoinech, počet událostí,
  kolik dní bylo BTC nad MA200, kolik signálů filtr zahodil.
- Skutečný skluz po skoku BTC se **neměří tady**; jde o samostatný dokument (`FORWARD.md` §7, §8).

## 10. Co znamená výsledek

- **Uspěla H1 nebo H2 (nebo obě):** paper edge potvrzený na 12–24 měsících neviděných dat při 10 a 15 bps. Teprve pak
  má smysl samostatně změřit reálný skluz a rozhodnout o případném malém kapitálu (nový dokument). Žádná živá
  alokace před tím.
- **Zamítnuta:** hypotéza se uzavírá. Zamítnutí H2 neovlivňuje rozhodnutí §4 `FORWARD.md` (primární pravidlo výzkumu 17
  běží dál podle vlastního protokolu).
- **Nepotvrzeno:** nic se nemění v primárním forwardu; případné další pokračování je nová pre-registrace.
- Rozhodnutí §4 `FORWARD.md` (primární pravidlo, do 2027-03-31) a toto studium jsou **oddělená**; sdílejí data, ale
  každé má své kritérium. Rozhodnutí výzkumu 17 se do Holma tohoto studia nepočítá.

## 11. Implementace a postup

1. Schválení vlastníkem, verze 1.0, **commit před prvním stažením dat po `T0`** (cron 6:30 stahuje data bez výpočtu
   tohoto studia; samotný výpočet se spustí až po zmrazení kódu).
2. `qlab.research18`: H1 přes `qlab.research17.variants` (`Spec`, `candidates`, `run_spec`), H2 jako `V01_base`
   s maskou `allowed &= btc_close > MA200`. Event bootstrap a Holm. Testy: H1 shoduje obchody s `V05_lag0`, H2 bez
   filtru shoduje obchody s `V01_base`, filtr zahazuje signály pod MA200 a jen při vstupu, MA200 bez budoucích dat
   (kontrola bodu v čase), Holm na ručním příkladu, event bootstrap zachovává současnost obchodů.
3. `scripts/r18_forward.py freeze` zapíše otisk `RULE_FILES` (viz `FORWARD.md` §6 + `research18/*.py`) do
   `docs/research18/forward_lock.json`; `status` počítá a ukládá `results/forward_<datum>.json`; `final` je povoleno
   jen při termínech §6 a ≥ minimu událostí a zapíše `results/decision_<číslo>.json`.
4. Záložka v dashboardu LEAD forward se doplní popisně (bez rozhodování před termínem).
5. Registr `runs/registry/research18.sqlite`: zapsat obě hypotézy jako N = 2 před prvním výpočtem. Kumulativně
   N ≥ 91 na stejných datech.

## 12. Log rozhodnutí

| Datum | Změna |
|---|---|
| 2026-10-19 | Verze 0.1, návrh. |
| 2026-10-19 | Verze 1.0 schválena vlastníkem beze změn. Před commitem se na datech po 2026-10-19 nic nepočítalo ani nezkoumalo. |
| 2026-10-19 | Upřesnění implementace (rozhodnuto při psaní kódu, před jakýmkoli výpočtem na datech od `T0`; pravidla §4–7 se nemění): (a) Zamítnutá hypotéza opouští Holmovu rodinu už v **témže** rozhodnutí; zbylá se pak testuje s prahem α = 0,025 (§7, poslední věta čtu takto). (b) Hypotéza, která uspěla v 1. rozhodnutí, je uzavřená a ve 2. se nepočítá; ve 2. rozhodnutí je rodinou jen to, co bylo „neprůkazné“. (c) Neprůkazná hypotéza ve 2. rozhodnutí dostane stav `not_confirmed`. (d) Podmínka MA200 se u zpožděného plnění (jen popisné scénáře §9) hodnotí na svíčce plnění. (e) Událost = signální svíčka *k* (u zpoždění se od vstupu odečte zpoždění). (f) Test sdílí stahovač a členství s výzkumem 17; kvůli 24 měsícím se členství počítá do 2028-10 (`forward.MEMBERS_END`), což nemění nic z výzkumu 17. (g) Zámek pokrývá `RULE_FILES` výzkumu 17, `research17/variants.py` a `research18/{setup,rules,evaluate}.py`. (h) Záložka v dashboardu zatím není; výsledky jsou v `results/status_<datum>.json`. |
