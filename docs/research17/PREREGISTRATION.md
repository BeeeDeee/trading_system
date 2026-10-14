# Výzkum 17 – LEAD: BTC vede altcoiny (pevné pravidlo z výzkumu 16, nová aktiva) (pre-registrace)

Verze 1.0 · 2026-10-14 · Stav: **schváleno, implementace**.
Navazuje na výzkum 16, kde `LEAD_K3_LONG` (long ETH 4 h po skoku BTC, když ETH zaostane) splnil na holdoutu
5 ze 6 kritérií a neprošel jen intervalem průměrného obchodu (81 obchodů). Po commitu verze 1.0 platí: změny =
nový pokus v registru a řádek v §12.

## 1. Otázka

Funguje **beze změny** pravidlo `LEAD_K3_LONG` z výzkumu 16 na altcoinech, na kterých nikdy neběželo? Konkrétně:
vydělá long altcoinu na 4 h po skoku BTC ≥ 3σ, když altcoin zaostane, po nákladech a fundingu víc než stejné
obchody v náhodném čase?

## 2. Poctivé omezení předem

- **Pravidlo jsem vybral podle holdoutu výzkumu 16** (ETH 2022–2026). ETH proto v testu **není**, je jen
  referenční řádek. Testují se altcoiny, na kterých LEAD nikdy nepočítal.
- **Altcoiny nejsou úplně neviděné.** Z výzkumu 15 znám chování jiné strategie (long průrazy) na stejném koši
  2020–2026: ztrátové 2020–21, ziskové od 2022. Vím také, že trh krypta v holdoutu rostl. O LEAD na altcoinech
  nevím nic.
- **Nic se neladí.** Parametry, strana, držení a definice zaostávání jsou převzaté beze změny (N = 1). Není tedy
  vývojová fáze: celé období je test. Kumulativně jde na stejných datech o N = 47 (16 + 30 + 1).
- **Obchody nejsou nezávislé.** Jeden skok BTC spustí obchody ve více altcoinech současně. Efektivní počet
  pozorování je počet **událostí BTC**, ne obchodů. Interval i nula se proto počítají po událostech (§6, §8).
- **Historický test může jen zamítnout, nebo kvalifikovat na forward test** (§10).

## 3. Data

Panel z výzkumu 15 beze změny (`data/derived/binance_2026-10-03/r15_15m/`, `docs/research15/DATA_AUDIT.md`):
- **Vůdce:** BTCUSDT spot 15m (stejně jako ve výzkumu 16).
- **Následníci:** USDT-M perpy z koše výzkumu 15, ceny perpu, vlastní funding (bez dat paušál 0,01 % / 8 h, týká
  se HYPE). Jen kryptoměny s protějškem na spotu, bez stablecoinů a tokenizovaného zlata.
- **Univerzum v čase:** v každém měsíci členové **top 10 podle objemu za 30 dní před začátkem měsíce** (bez BTC
  a ETH; `qlab.research15.universe`, manifest `r15_manifest.json`). Výběr je podle data, bez survivorship biasu.
  Delistované kontrakty zůstávají v datech do posledního obchodu.
- Svíčky BTC a altcoinu se párují podle času (15m index), ne podle pozice v poli.

## 4. Pravidlo (převzato z výzkumu 16, `families.lead`, beze změny)

Vše jen z dat uzavřených do konce 15m svíčky *k*, vstup za close *k*:
- σ_B = sm. odchylka 15m log výnosů BTC za 672 svíček (≥ 75 % platných), σ₁ₕ = 2σ_B.
- `rB = log(cB[k] / cB[k−4])`, `rA = log(cA[k] / cA[k−4])`; svíčky *k*−4 … *k* reálné u BTC i altcoinu.
- **Signál:** `rB ≥ 3 · σ₁ₕ` **a** `rA / rB < 0,5` → **long** altcoin. (Short ani K = 2 se netestují, jsou v §9.)
- **Výstup:** close svíčky *k* + 16 (4 h), bez stopu a cíle. Během pozice se signály na stejném altcoinu ignorují.
- **Vstup jen do členů** v daném měsíci; otevřená pozice doběhne i po vypadnutí.

## 5. Obchod a portfolio

- 10 rukávů, každý **1/10 kapitálu** pro člena (nečinný rukáv drží hotovost, bez úroku). Pozice +1 nominálu
  rukávu, bez páky. Při souběhu všech 10 je portfolio na 100 % long.
- **Náklady 10 bps na stranu** (taker 5 bps + skluz 5 bps; vyšší než u BTC/ETH, protože vstup přichází v rychlém
  trhu a altcoiny mají horší knihu). Zátěž **×2 = 20 bps**.
- **Funding:** v každém vypořádání během pozice platí long `+rate` daného perpu.
- Denní výnos portfolia = součet 15m výnosů dne (UTC).

## 6. Nula a benchmarky

- **B_RND_EVT (hlavní nula, 1000 běhů, seed 0..999):** každá **událost** (svíčka *k*, kdy nastal signál aspoň na
  jednom altcoinu) se přesune na náhodnou reálnou svíčku téhož kalendářního měsíce; **všechny obchody z jedné
  události dostanou stejnou náhodnou svíčku** (zachová se současnost obchodů napříč altcoiny). Stejné altcoiny,
  long, 4 h, stejné náklady a funding. Obchod, který by se překrýval s jiným na stejném altcoinu, se vynechá.
  Měří, zda načasování po skoku BTC přidává něco nad náhodný long ve stejném měsíci.
  p-hodnota = (1 + #{běhy se Sharpe ≥ P}) / 1001.
- **B_NOLAG:** stejné pravidlo **bez podmínky zaostávání** (long každý člen po skoku BTC). Odliší lead-lag od
  pouhého pokračování momenta. Jen hlášení.
- **B_HOLD:** rovné váhy členů, denní rebalance, bez nákladů.

## 7. Období a trezor

| Fáze | Období |
|---|---|
| Test | **2020-05-01 → 2026-09-30** (první měsíc s 10 členy), jednou, přes `Vault` (`runs/vault_r17/`) |
| Podúseky | 2020-05 → 2022-12 a 2023-01 → 2026-09 |
| Forward | od 2026-10-01 (§10) |

Bez vývojové fáze (pravidlo je pevné). Indikátory se mohou počítat z dat před 2020-05, na začátku se nic nedrží.
Obchod otevřený na konci se uzavře za close poslední svíčky.

## 8. Kritéria (nutné splnit **všechna**)

Sharpe = průměr / sm. odchylka denních výnosů × √365.

1. Sharpe > 0 a průměrný čistý výnos obchodu > 0 s dolní mezí 90% intervalu > 0. **Bootstrap po událostech**
   (převzorkují se události BTC i se všemi svými obchody), 2000 vzorků, seed 0.
2. p-hodnota B_RND_EVT < 0,05.
3. Sharpe > 0 při nákladech ×2 (20 bps na stranu).
4. Sharpe > 0 při plnění o svíčku později (close *k*+1).
5. Sharpe > 0 v obou podúsecích.
6. **Šíře:** Sharpe > 0 i po vyřazení altcoinu s největším celkovým příspěvkem **a** kladný průměr obchodu
   u ≥ 50 % altcoinů s ≥ 10 obchody.

DSR (N = 1 a kumulativně N = 47) se hlásí, není kritériem. Minimální rozsah: má-li test < 30 událostí, je
výsledek **neprůkazný** (ne zamítnutí ani úspěch).

## 9. Robustnost (jen hlášení)

Varianty K = 2, strana LS (short, když BTC spadne a altcoin zaostane), B_NOLAG; výsledky po letech; bez fundingu;
náklady 7,5 bps; ETH 2020-05 → 2026-09 jako referenční řádek (známý); statistiky po altcoinech (počet obchodů,
průměr, příspěvek); rozložení počtu obchodů na událost; počet událostí; BTC perp jako vůdce místo spotu.

## 10. Co znamená výsledek

- Nesplní-li cokoli z §8: **zamítnuto**, LEAD se uzavírá.
- Splní-li vše: **forward paper test** od 2026-10-01 (altcoiny top 10 + ETH, stejná pravidla). Vyhodnocení na
  datech do **2027-03-31**: Sharpe > 0 a průměrný obchod > 0 při nákladech ×1. Kritérium je pevné už teď.

## 11. Implementace a postup

1. Schválení vlastníkem, verze 1.0, commit (před jakýmkoli výpočtem signálu na altcoinech).
2. `qlab.research17`: párování BTC a altcoinu podle času, signál přes `qlab.research16.families` (tentýž kód
   jako ve výzkumu 16), portfolio 10 rukávů s členstvím, B_RND_EVT, bootstrap po událostech. Testy: bod v čase,
   shoda signálu s výzkumem 16 na páru BTC/ETH, párování při rozdílném začátku dat, členství, nula zachovává
   současnost obchodů a měsíce, ruční přepočet obchodu (náklady 10 bps, funding).
3. Commit kódu. Registr `runs/registry/research17.sqlite`: zapsat P před výpočtem.
4. `scripts/r17_final.py freeze|run` (jediný běh).
5. `REPORT.md`, řádek v README, verdikt do §12.

## 12. Log rozhodnutí

| Datum | Změna |
|---|---|
| 2026-10-14 | Verze 0.1, návrh. |
| 2026-10-14 | Verze 1.0 schválena vlastníkem beze změn. |
| 2026-10-14 | Upřesnění implementace (rozhodnuto při psaní kódu; pravidla §4–8 se nemění): (a) Sharpe a podúseky se počítají na pevné denní mřížce 2020-05-01 → 2026-09-30 včetně dnů bez expozice (výnos 0), stejná mřížka u nuly. (b) Událost = svíčka *k* vůdce, na které vznikl signál aspoň u jednoho člena; při plnění o svíčku později se událost určuje podle signálu (vstup − 1). (c) Signál vyžaduje reálné svíčky BTC *k*−4 … *k* a platné σ (≥ 75 % z 672 svíček); se zaostáváním i reálné svíčky následníka *k*−4 … *k*, bez zaostávání (B_NOLAG) jen *k*. (d) Nula: náhodná svíčka se losuje z reálných svíček BTC spotu téhož měsíce v testovaném období; obchod se vynechá, nemá-li následník v té svíčce reálnou svíčku nebo už drží pozici (události v původním pořadí); počet obchodů nuly proto může být o málo nižší než skutečný. (e) Zaostávání: `ra/rb < 0,5` platí i pro short (robustnost LS) stejným vzorcem jako ve výzkumu 16. (f) Kritérium 6: nejvýznamnější altcoin = nejvyšší součet vážených denních příspěvků; podíl kladných = altcoiny s ≥ 10 obchody a kladným průměrem obchodu. (g) B_HOLD = průměr denních výnosů členů daného měsíce (bez nákladů). (h) ETH referenčně: perp ETH, jediný rukáv s váhou 1, stejné pravidlo. (i) DSR s N = 47 používá jako proxy rozptyl denních Sharpe z mřížky výzkumu 16 (u N = 1 je očekávané maximum 0). (j) Hash metodiky pokrývá pre-registraci, `scripts/r17_final.py`, `research15/strategy.py`, `research15/universe.py` a celé `research16/`, `research17/`. **Zveřejnění:** testy proběhly výhradně na syntetických trzích, na skutečných cenách altcoinů jsem před otevřením trezoru nic nepočítal (trezor je do otevření neprostupný i pro data). |
