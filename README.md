# Paper trading bot

Experiment: každý obchodní večer Claude ohodnotí 20 amerických titulů třemi pohledy
(trend, mean-reversion, zprávy) a celkovým přesvědčením. Pevný Python engine z jednoho
skóre obchoduje 19 variant pravidel (každá s vlastními $10 000) a náhodný výběr.
Po ~2 měsících se ukáže, jestli má nějaké pravidlo nebo pohled edge proti SPY a náhodě.
Bez skutečných peněz a bez brokera.

## Struktura

| Cesta | Obsah |
|---|---|
| `engine/engine.py` | Veškerá deterministická logika (bez závislostí): `history-append`, `features`, `check`, `validate`, `settle`, `plan`, `apply-split` |
| `config/config.json` | Univerzum, náklady, definice všech variant a náhodné baseline |
| `task/task_prompt.md` | Zadání naplánované úlohy (orchestrace večerního běhu), včetně připnutého release |
| `dashboard/dashboard.src.html` | Zdroj stránky deníku (čte databázi artefaktu) |
| `tools/simulate.py` | Syntetický trh přes skutečný engine, generuje ukázková data |
| `tools/build_dashboard.py` | Vloží ukázková data do dashboardu, výsledek je `dashboard/dashboard.html` |
| `tools/pin_prompt.py`, `tools/verify_manifest.py` | Připnutí release a kontrola konzistence |
| `tools/random_null.py` | Vlastní nulové rozdělení pro každou variantu: stejná aktivita, náhodné tituly (skill vs štěstí) |
| `tools/analyze.py` | Má některý pohled informaci? Rank IC a rozdíly skóre s bootstrap intervaly přes dny |
| `tests/test_engine.py` | Testy invariantů (hotovost, ocenění, náklady, podmnožiny titulů, determinismus, kontrola dat) |
| `MANIFEST.sha256`, `VERSION`, `CHANGELOG.md` | Otisky a historie verzí |

Práce s LLM je jen v zadání úlohy: sběr dat, rešerše zpráv, skóre titulů a rozhodnutí
varianty „Claude volně“. Všechno ostatní je engine.

## Denní běh (22:15 Praha, po–pá)

1. Klon větve `paper-trading-bot`, ověření SHA-256 otisků enginu a konfigurace proti připnutí v zadání (při nesouladu běh selže).
2. Ceny z stockanalysis.com, kontrola `engine.py check`, nezávislé ověření SPY a jedné akcie.
3. `settle`: vypořádání včerejších pokynů na openu, stopy, doba držení, ocenění.
4. Rešerše a skóre všech 20 titulů (Claude), `validate` skóre.
5. `plan`: pokyny na zítřek pro všechny varianty.
6. Zápis do databáze deníku (`days`, `trades`, `scores`, `prices`, `history`, `runs`).

## Vývoj

```bash
python tests/test_engine.py          # invarianty
python tools/simulate.py             # ukázková data pro dashboard
python tools/build_dashboard.py      # dashboard/dashboard.html
python tools/analyze.py export.json      # jsou pohledy lepší než šum? (--demo pro ukázku)
python tools/random_null.py export.json  # je varianta lepší než náhodný výběr? (--demo)
```

## Release (změna pravidel nebo enginu)

Pro férové srovnání variant se během pokusu pravidla neplánují měnit. Když je změna
nutná, je to vždy nový release, ne úprava za běhu:

1. Změna v `engine/` nebo `config/`, `python tests/test_engine.py`.
2. Zápis do `CHANGELOG.md`.
3. `python tools/pin_prompt.py vX.Y.Z` (aktualizuje `VERSION`, `MANIFEST.sha256` a připnutí v zadání).
4. `python tools/verify_manifest.py`, commit, `git tag vX.Y.Z`, `git push` (tag je jen označení; kód se připíná otisky, ne tagem).
5. **Záloha v databázi:** `pin_prompt.py` vytvoří `build/db_mirror_engine.json` a `build/db_mirror_config.json` (surový text souborů). Nahraj je do databáze deníku: `engine/main` a `engine/config`. Repo je soukromé a úloha z něj nemusí umět klonovat (GitHub vrací 403), takže běžně poběží ze zálohy. Bez aktuální kopie by běh selhal na kontrole otisku. `config/main` je jen parsovaná kopie pro dashboard a jako zdroj se nesmí použít.
6. Obsah `task/task_prompt.md` vložit do naplánované úlohy.
7. V dashboardu je u každého dne vidět, který tag ho spočítal, a v analýze se výsledky před a po změně nesmí míchat.

## Poznámky k datům

- Ceny jsou neupravené (`Close`, nikdy `Adj. Close`), engine drží posledních 80 řádků historie.
- Backtest se záměrně nedělá: LLM zná historii, takže by jeho skóre před datem znalostí
  byla zatížená lookahead biasem. Jediný platný test je dopředný (tento pokus).

## Pravidla vyhodnocení (zapsáno předem, před prvními výsledky)

Aby po dvou měsících nešlo vybrat vítěze zpětně (z 19 variant vždy nějaká vyhraje):

1. **Pohled (trend / mr / news / přesvědčení) má edge**, jen když 95% interval spolehlivosti
   rank IC na horizontu 5 dní (`tools/analyze.py`) neobsahuje 0 **a** znaménko je stejné v první
   i druhé polovině pokusu.
2. **Varianta má edge**, jen když její výnos leží nad 95. percentilem vlastního nulového
   rozdělení (`tools/random_null.py`) **a** je nad SPY. Při 19 variantách čekej jednu
   „výhru“ náhodou; jedna izolovaná varianta nad 95 % nic nedokazuje, důležitější je vzor
   (např. souhlasí pořadí `plne` vs `kontrarian` s IC složeného skóre?).
3. Sloupec „vs náhoda“ v dashboardu je jedna náhodná cesta a slouží jen orientačně.
4. Výsledky před a po případném release se nemíchají.

## Známá omezení

- Ceny jsou bez dividend (price return), stejně jako SPY a univerzum. Porovnání je konzistentní, ale TLT a dividendové ETF jsou mírně znevýhodněné.
- 19 vzájemně korelovaných variant za ~40 dní má malou sílu; na výsledky variant se dívej jen jako na hrubý test.
- Chyba ceny jednoho tickeru zastaví celý den (záměr: nikdy neobchodovat na špatných datech). Chybějící den se doplní zpětně.
- Náhodná baseline v enginu je jedna cesta; skutečné srovnání dělá `tools/random_null.py`.
