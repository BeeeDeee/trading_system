# Pre-registrace vyhodnocení – akciový paper-trading bot

**Zapečetěno:** 2026-10-02 (commit a tag `prereg-v1-akcie`), před tím, než kdokoli viděl výsledek, ze kterého by
šlo cokoli usoudit. K tomuto dni existují nejvýš 4 večerní běhy (od 2026-09-28) a žádné 5denní okno není uzavřené.
Vyhodnocovací skript byl napsaný a vyzkoušený jen na syntetických datech (`python tools/prereg_eval.py --demo`).

Tento dokument **nahrazuje** sekci „Pravidla vyhodnocení“ v README pro všechno, co se má tvrdit jako výsledek.
Sesterský dokument pro krypto bota je na větvi `crypto-paper-bot` v `docs/PREREGISTRATION.md`. Oba používají stejné
metody, aby šly výsledky porovnat.

## 1. Otázka

Přidává večerní rešerše LLM informaci o relativních výnosech 20 amerických titulů (11 ETF, 9 akcií) na příští
obchodní den(y) **nad rámec obyčejného momenta**, které jde spočítat bez LLM?

## 2. Okno a data

| | |
|---|---|
| Primární okno | plány (večerní skóre) s datem **2026-09-29 až 2026-11-25** včetně (42 obchodních dní) |
| Vyhodnocení | nejdříve **2026-12-04** (uzavřená 1 i 5denní okna), `python tools/prereg_eval.py export.json` |
| Segment | `code.tag` v `v1.1.0`–`v1.1.2` (v1.1.0 = pravidla platná od plánů 29. 9.) |
| Platnost | aspoň **32 dní** se skóre, jinak *neprůkazné kvůli datům* |
| Výnos | rank IC = Spearman přes univerzum mezi skóre a výnosem close(d) → close(d+h) minus průměr univerza |
| Export | kolekce `days`, `scores`, `prices`, `history` z databáze deníku do jednoho JSON (formát v hlavičce skriptu) |

- **Žádné průběžné rozhodování.** Pokus se kvůli výsledkům neukončuje ani neprodlužuje. Změna okna je možná jen
  před 2026-11-25, z neobsahových důvodů, jako nová verze dokumentu a dřív, než se kdo podívá na výsledky.
- **Release se změnou pravidel nebo promptu** během okna ukončí primární okno dnem před release.
- Chybějící skóre se nedoplňují.

## 3. Primární hypotézy

Každá: **dvoustranný Studentův t-test, α = 0,05**, a navíc musí sedět znaménko průměru v první i druhé polovině okna.

| | Měřeno | Potvrzeno když |
|---|---|---|
| **H1** | denní rank IC `composite` (= trend + mr + news + 0,5·(conviction − 3), jako engine) na 1 den | průměr > 0, p < 0,05, poloviny > 0 |
| **H2** | denní rozdíl IC(`composite`) − IC(`mom20`) na 1 den. `mom20` = výnos za 20 obchodních dní ke stejnému close, bez LLM. | průměr > 0, p < 0,05, poloviny > 0 |
| **H3** | denní rozdíl výnosu `zaklad` − rovné váhy univerza (`universe_equity`), po nákladech | průměr > 0, p < 0,05, poloviny > 0 |

**„LLM přidává informaci“ platí, jen když platí H1 i H2.** Bot nemá variantu bez LLM, a proto se mechanická kontrola
`mom20` počítá ve vyhodnocení z cen. Je definovaná tady, předem, a dál se nemění.

## 4. Sekundární testy

| Rodina | Obsah | Korekce |
|---|---|---|
| A | IC `composite` a `composite` − `mom20` na 5 dnech, t-test nad nepřekrývajícími se **10denními bloky** | Holm, 2 |
| B | IC pohledů `trend`, `mr`, `news`, `conviction` na 1 den | Holm, 4 |
| C | ostatní varianty (včetně `nahoda`) vs rovné váhy univerza | Holm přes všechny |

Explorativní a nikdy důkaz: `tools/random_null.py`, rozptyl skóre (spread), hit rate, srovnání s SPY,
pořadí variant a cokoli, co tu není vyjmenované.

## 5. Co se stane podle výsledku (rozhodnuto předem)

| Výsledek | Další krok |
|---|---|
| H1 ✓, H2 ✓ | replikace: nový pre-registrovaný 3měsíční běh beze změny pravidel. Teprve potom se smí tvrdit „funguje“. |
| H1 ✓, H2 ✗ | skóre je převážně momentum; LLM nepřidává nic, co by nešlo spočítat levněji |
| H1 ✗ | nulový výsledek, zveřejní se celý výstup |
| opačné znaménko | reportovat, znaménko neotáčet bez nové registrace |
| < 32 dní | neprůkazné kvůli datům |

## 6. Statistická síla

`python tools/power.py --days 43 --coins 20` (na větvi `crypto-paper-bot`; stejný model trhu):

| skutečné IC 1 d | síla H1 (t-test, 42–43 dní) |
|---|---|
| 0,00 | 4,0 % (falešně pozitivní, jednostranně, nominálně 2,5 %) |
| 0,04 | 20 % |
| 0,065 | 44 % |
| 0,10 | 81 % |
| 0,15 | 98 % |

Sekundární rodina A (5 dní) má jen 4 nezávislé 10denní bloky. Její síla je nízká (pod 30 % i pro IC 5 d ≈ 0,16),
slouží proto jen jako kontrola směru.

Spolehlivě (≥ 80 %) jde odhalit jen IC 1 d kolem **0,10**. U akcií je to ještě méně pravděpodobné než u krypta,
protože dobré akciové signály mívají IC 0,02–0,05. Pozor taky na to, že univerzum je z poloviny sektorová a
třídová ETF (TLT, GLD, XLE…). Část IC tak může pocházet z rotace mezi třídami aktiv, ne z výběru akcií.
Nepotvrzení **neznamená**, že edge neexistuje.

## 7. Proč se mění pravidla z README

1. `tools/analyze.py` převzorkovává jednotlivé dny (ne bloky) i na 5denním horizontu. Překrývající se okna jsou
   ale silně autokorelovaná, takže interval vychází příliš úzký a falešné „signály“ jsou častější, než říká 5 %.
   Nově: primárně 1 den, 5 dní jen přes nepřekrývající se bloky.
2. 4 pohledy × 2 horizonty + 19 variant, každá na 5 %, bez korekce. Nově 3 primární hypotézy a Holmova korekce.
3. Chyběla mechanická kontrola (bez ní nejde odlišit „LLM ví něco navíc“ od „LLM přečte graf“). Nově `mom20`.
4. Konec „po ~2 měsících“ nebyl pevný. Nově pevné okno a datum vyhodnocení.

## 8. Pečeť

```
7772e7bb1509e8762506616cb0c25c21fdb39ea9b3b73f74b7b150be3ea8988d  tools/prereg_eval.py
c2be59cd1435eb7a4a40e8fe7fe77c0ca519558f7f070570479092d18d02fa9b  config/config.json
6206c93db082eab21fb0a737d4555122ec8d9943acfeb860945d1f53f3997c5a  engine/engine.py
```

Ověření: `sha256sum -c` nad výpisem výše, `git show prereg-v1-akcie`. Pozdější změna = nová verze s novým tagem
a záznamem v sekci „Změny“.

## Změny

(žádné)
