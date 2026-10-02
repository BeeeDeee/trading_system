# Pre-registrace vyhodnocení – crypto paper-trading bot

**Zapečetěno:** 2026-10-02 (commit a tag `prereg-v1`), před tím, než kdokoli viděl výsledek, ze kterého by šlo
cokoli usoudit. K tomuto dni existují 3 běhy (2026-09-30 bez skóre LLM, 2026-10-01 a 2026-10-02 se skóre).
Autor viděl dashboard z prvních dvou dnů, kde se nedá nic rozpoznat, protože žádné 7denní okno ještě není uzavřené.
Vyhodnocovací skript byl napsaný a vyzkoušený jen na syntetických datech (`tools/simulate.py`).

Tento dokument **nahrazuje** sekci „Pravidla vyhodnocení“ v README pro všechno, co se má tvrdit jako výsledek.
README pravidla zůstávají jako orientační pohled v dashboardu. Důvody jsou v sekci „Proč se mění pravidla z README“.

## 1. Otázka

Přidává denní online rešerše LLM (`claude-opus-5-5`) informaci o krátkodobých relativních výnosech
kryptoměn **nad rámec obyčejného momenta**, kterou lze spočítat bez LLM?

## 2. Okno a data

| | |
|---|---|
| Primární okno | rozhodnutí s datem **2026-10-01 až 2026-11-30** včetně (61 dní) |
| Vyhodnocení | nejdříve **2026-12-15** (uzavřená 1, 7 i 14denní okna), jedním příkazem `python tools/prereg_eval.py` |
| Segment | verze `v1.0.0`–`v1.0.4` (jen provozní opravy bez změny pravidel), model `claude-opus-5-5` |
| Platnost | aspoň **45 dní** s validním skóre LLM, jinak je výsledek *neprůkazný kvůli datům* |
| Výnos | rank IC = Spearman přes univerzum (bez BTC) mezi skóre z dat do close D−1 a forward výnosem vs BTC od téhož close |

- **Žádné průběžné rozhodování.** Pokus se kvůli výsledkům neukončuje ani neprodlužuje. Prodloužení se může
  rozhodnout jen před 2026-11-30 a jen z neobsahových důvodů (např. výpadek). Potom je to nová verze tohoto
  dokumentu s novým tagem, a to dřív, než se kdo podívá na výsledky.
- **Release se změnou pravidel, promptu nebo modelu** během okna ukončí primární okno dnem před release.
  Data po release se do potvrzující analýzy nezapočítají.
- Dny bez skóre (selhání LLM) se nevyplňují. V H1 a H2 chybí, H3 je obsahuje, protože LLM varianty v takový
  den drží pozice, což je součást strategie.

## 3. Primární hypotézy

Každá: **dvoustranný Studentův t-test, α = 0,05**, a navíc musí sedět znaménko průměru v první i druhé polovině okna.

| | Měřeno | Potvrzeno když |
|---|---|---|
| **H1** | denní rank IC `composite` na horizontu 1 den | průměr > 0, p < 0,05, obě poloviny > 0 |
| **H2** | denní rozdíl IC(`composite`) − IC(`rel30`) na 1 den (párově podle dne) | průměr > 0, p < 0,05, obě poloviny > 0 |
| **H3** | denní rozdíl výnosu `zaklad` − `mech_momentum` (čisté náklady, ocenění k 00:00 UTC) | průměr > 0, p < 0,05, obě poloviny > 0, **a** průměr > 0 i ve stresovém scénáři |

**Hlavní závěr „LLM přidává informaci“ platí, jen když platí H1 i H2.** H3 říká, jestli se to projeví i v penězích
po nákladech.

Primární horizont je 1 den (ne 7), protože jen u něj jsou denní pozorování nezávislá, takže test drží
deklarovanou chybovost, a protože všechny denní varianty rebalancují denně.

## 4. Sekundární testy

| Rodina | Obsah | Korekce |
|---|---|---|
| A | IC `composite` a IC `composite` − `rel30` na 7 dnech; t-test nad průměry nepřekrývajících se **14denních bloků** | Holm, 2 testy |
| B | IC jednotlivých pohledů na 1 den: `trend`, `news`, `mr`, `conviction`, `p_outperform_btc_7d`, `expected_move_7d_pct` | Holm, 6 testů |
| C | 16 denních LLM variant: denní rozdíl výnosu vs `b_ew20` (rovné váhy univerza; odděluje výběr od alt-bety) | Holm, 16 testů |
| – | kalibrace `p_outperform_btc_7d` (Brier vs konstantní základní míra) | jen popisně |

**Explorativní, nikdy důkaz:** hodinové signály a hodinové varianty (`tools/analyze.py --hourly`, Benjamini–Hochberg
FDR 10 %), percentily vůči nulovému rozdělení (`tools/random_null.py`), všechny post-hoc přehrávky
(`tools/replay.py`), a cokoli, co tu není vyjmenované. Pořadí variant podle výnosu se nikdy neprezentuje jako výsledek.

## 5. Co se stane podle výsledku (rozhodnuto předem)

| Výsledek | Interpretace | Další krok |
|---|---|---|
| H1 ✓, H2 ✓ | LLM nese informaci nad momentem | **replikace**: nový pre-registrovaný 3měsíční běh, beze změny pravidel. Teprve potvrzená replikace se smí prezentovat jako „funguje“. |
| H1 ✓, H2 ✗ | skóre je převážně přebalené momentum | LLM je drahý způsob, jak spočítat `rel30`; další výzkum na mechanických signálech |
| H1 ✗ | žádná prokazatelná informace v 1denním horizontu | zveřejnit jako nulový výsledek; LLM skóre krátkodobých výnosů se dál nerozvíjí |
| opačné znaménko (p < 0,05) | skóre je systematicky špatně | reportovat. Znaménko se neotáčí bez nové pre-registrace a nového běhu. |
| < 45 validních dní | neprůkazné kvůli datům | reportovat; nové okno jen s novou pre-registrací |

Ve všech případech se zveřejní celý výstup `tools/prereg_eval.py`, včetně nepotvrzených testů.

## 6. Statistická síla (co vůbec jde odhalit)

`python tools/power.py` simuluje syntetický trh s 19 coiny a 61 dny: skóre s denní autokorelací 0,9, výnosy
s tlustými konci, 500 simulací. Síla = podíl simulací, kde test najde kladný edge. Řádek s IC 0 je míra falešně
pozitivních výsledků (jednostranně, nominálně 2,5 %).

| skutečné IC 1 d | H1 (t-test, 1 d) | IC 7 d | README bootstrap 7 d | sekundární A (bloky 14 d) |
|---|---|---|---|---|
| 0,00 | 2,2 % | 0,00 | **8,8 %** | 2,6 % |
| 0,04 | 25 % | 0,06 | 31 % | 12 % |
| 0,06 | 52 % | 0,10 | 45 % | 22 % |
| 0,10 | 91 % | 0,16 | 82 % | 44 % |
| 0,15 | 100 % | 0,23 | 96 % | 67 % |

Spolehlivě (≥ 80 %) jde odhalit jen IC 1 d kolem **0,08–0,10**. Dobré profesionální signály mívají IC 0,02–0,05.
Nejpravděpodobnější poctivý výsledek tohoto pokusu je proto „nepotvrzeno“, i kdyby malý edge existoval.
Nepotvrzení **neznamená** důkaz, že edge neexistuje. Znamená, že není dost velký na to, aby ho šlo vidět za
2 měsíce.

U peněžních výsledků je to ještě horší: s tracking errorem alt-portfolia vůči BTC kolem 40 % p. a. je směrodatná
chyba nadvýnosu za 2 měsíce zhruba ±16 procentních bodů. Výsledek „varianta X porazila BTC o 10 %“ je tak
v pásmu šumu.

## 7. Proč se mění pravidla z README

1. **Blokový bootstrap na 7 a 14 dnech je příliš benevolentní.** Na datech bez edge hlásí falešný pozitivní
   výsledek v 8,8 % (7 d) a 13 % (14 d) případů místo 2,5 %. Při 61 dnech je nezávislých 7denních bloků jen ~8
   a percentilový interval je pak příliš úzký. Nově: primárně 1 den (t-test drží chybovost, simulace 2,2 % jednostranně),
   7 dní jen sekundárně přes nepřekrývající se 14denní bloky (simulace 2,6 %; za cenu nízké síly). 14 dní se
   nepotvrzuje vůbec (61 dní = 2 nezávislé bloky).
2. **Mnohonásobné testování.** README testuje 7 pohledů × 4 horizonty, 18 variant a 30 hodinových kombinací,
   každou na 5 %. Náhodou by tak vyšlo ~4 „edge“. Nově jsou 3 primární hypotézy a korigované rodiny.
3. **„Soustavně porážejí“ (README bod 3) nebylo definované.** Nově H2 a H3 s konkrétním testem.
4. **Konec pokusu „8–12 týdnů“ nechával prostor vybrat si konec podle výsledku.** Nově pevné okno a datum.

## 8. Známé slabiny

- 61 dní je jeden tržní režim. I potvrzený výsledek platí pro tento režim, proto se před jakýmkoli tvrzením
  vyžaduje replikace (bod 5).
- IC se počítá z close 00:00 UTC, plnění je ~04:00 UTC. IC je tedy mírně optimistický vůči skutečně obchodovatelnému
  výnosu; H3 to zachycuje přesně.
- `rel30` je jedna konkrétní definice momenta. H2 neříká, že LLM porazí *každé* momentum.
- Skóre LLM v sobě nese i informaci, kterou má model z rešerše o pohybu z posledních hodin před během.
  Je to legitimní informace dostupná v čase rozhodnutí (běh je po close), ale znamená to, že část IC na 1 den může
  pocházet z rychlé reakce na zprávy, ne z „úsudku“.

## 9. Pečeť

Soubory, které tato pre-registrace připíná (sha256):

```
0a4463eccfbb028b170c4b6e75bf4dfc2fc2227f72234ca92a4c92540b7d9346  tools/prereg_eval.py
5bb78e675c5bbf554c061b281076263fadf026e336b9e29aa3607d69be107f53  tools/power.py
939caf44c9ca105ac0cf71c3740845acaa47438873914839c2a00b3457d6e1e4  tests/test_prereg.py
9c180bcc5d5cba09f5d61b71be078700eddd7fed6ac4ef00754b42fec8a5d8fd  config/config.json
ea21f53b1f825276c9bd4d9dae726353ba6526d0da428f1e9bf3b5753498d5c7  engine/cpb/analytics.py
8223c6f37ef2e76b3ed9e876826db6127a3ebd31a862362ef7667168ded088f5  engine/cpb/portfolio.py
```

Ověření: `sha256sum -c` nad výpisem výše, `git show prereg-v1`. Commit je veřejně pushnutý na GitHub
(časové razítko mimo kontrolu autora). Jakákoli pozdější změna těchto souborů je nová verze s novým tagem,
s vysvětlením v sekci „Změny“ níže. Starší verze zůstává platná pro data, která už v té době existovala.

## Změny

(žádné)
