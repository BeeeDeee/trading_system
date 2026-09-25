# strategy_backtester_2026_sep

Lokální výzkumný framework pro vývoj a poctivé vyhodnocení long-only strategií na denních datech
US akcií (Sharadar).

Stav: specifikace, bez implementace. Další krok: Fáze 0 (základ a enginy na syntetických datech). Sharadar data jsou potřeba až od Fáze 1.

## Dokumenty

- [`docs/PROJECT_SPECIFICATION.md`](docs/PROJECT_SPECIFICATION.md) – aktuální specifikace (v2)
- [`docs/SPEC_REVIEW.md`](docs/SPEC_REVIEW.md) – revize v1 a zdůvodnění změn
- [`docs/archive/PROJECT_SPECIFICATION_v1.md`](docs/archive/PROJECT_SPECIFICATION_v1.md) – původní specifikace

## Data

Sharadar dump patří do `data/raw/<snapshot_id>/` a nikdy do gitu (viz `.gitignore` a spec §4.1).
