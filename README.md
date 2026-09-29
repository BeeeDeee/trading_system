# Paper trading bot

Experiment: každý obchodní večer Claude ohodnotí 20 amerických titulů třemi pohledy
(trend, mean-reversion, zprávy) a celkovým přesvědčením. Pevný Python engine z jednoho
skóre obchoduje 19 variant pravidel (každá s vlastními $10 000) a náhodný výběr.
Po ~2 měsících se ukáže, jestli má nějaké pravidlo nebo pohled edge proti SPY a náhodě.
Bez skutečných peněz a bez brokera.

## Struktura

| Cesta | Obsah |
|---|---|
| `engine/engine.py` | Veškerá deterministická logika (bez závislostí): `history-append`, `features`, `check`, `settle`, `plan` |
| `config/config.json` | Univerzum, náklady, definice všech variant a náhodné baseline |
| `task/task_prompt.md` | Zadání naplánované úlohy (orchestrace večerního běhu), včetně připnutého release |
| `dashboard/dashboard.src.html` | Zdroj stránky deníku (čte databázi artefaktu) |
| `tools/simulate.py` | Syntetický trh přes skutečný engine, generuje ukázková data |
| `tools/build_dashboard.py` | Vloží ukázková data do dashboardu, výsledek je `dashboard/dashboard.html` |
| `tools/pin_prompt.py`, `tools/verify_manifest.py` | Připnutí release a kontrola konzistence |
| `tests/test_engine.py` | Testy invariantů (hotovost, ocenění, náklady, podmnožiny titulů, determinismus, kontrola dat) |
| `MANIFEST.sha256`, `VERSION`, `CHANGELOG.md` | Otisky a historie verzí |

Práce s LLM je jen v zadání úlohy: sběr dat, rešerše zpráv, skóre titulů a rozhodnutí
varianty „Claude volně“. Všechno ostatní je engine.

## Denní běh (22:15 Praha, po–pá)

1. Klon větve `paper-trading-bot`, ověření SHA-256 otisků enginu a konfigurace proti připnutí v zadání (při nesouladu běh selže).
2. Ceny z stockanalysis.com, kontrola `engine.py check`, nezávislé ověření SPY a jedné akcie.
3. `settle`: vypořádání včerejších pokynů na openu, stopy, doba držení, ocenění.
4. Rešerše a skóre všech 20 titulů (Claude).
5. `plan`: pokyny na zítřek pro všechny varianty.
6. Zápis do databáze deníku (`days`, `trades`, `scores`, `prices`, `history`, `runs`).

## Vývoj

```bash
python tests/test_engine.py          # invarianty
python tools/simulate.py             # ukázková data pro dashboard
python tools/build_dashboard.py      # dashboard/dashboard.html
```

## Release (změna pravidel nebo enginu)

Pro férové srovnání variant se během pokusu pravidla neplánují měnit. Když je změna
nutná, je to vždy nový release, ne úprava za běhu:

1. Změna v `engine/` nebo `config/`, `python tests/test_engine.py`.
2. Zápis do `CHANGELOG.md`.
3. `python tools/pin_prompt.py vX.Y.Z` (aktualizuje `VERSION`, `MANIFEST.sha256` a připnutí v zadání).
4. `python tools/verify_manifest.py`, commit, `git tag vX.Y.Z`, `git push` (tag je jen označení; kód se připíná otisky, ne tagem).
5. Obsah `task/task_prompt.md` vložit do naplánované úlohy.
6. V dashboardu je u každého dne vidět, který tag ho spočítal, a v analýze se výsledky před a po změně nesmí míchat.

## Poznámky k datům

- Ceny jsou neupravené (`Close`, nikdy `Adj. Close`), historie 50 dní zpět.
- Backtest se záměrně nedělá: LLM zná historii, takže by jeho skóre před datem znalostí
  byla zatížená lookahead biasem. Jediný platný test je dopředný (tento pokus).
