# strategy_backtester_2026_sep

Lokální výzkumný framework pro vývoj a poctivé vyhodnocení long-only strategií na denních datech
US akcií (Sharadar).

Stav: Fáze 0 hotová (viz `docs/PHASE0_PLAN.md`). Další krok: Fáze 1 – datový audit a normalizace reálných dat.

Testy: `uv run pytest`

## Dokumenty

- [`docs/PROJECT_SPECIFICATION.md`](docs/PROJECT_SPECIFICATION.md) – aktuální specifikace (v2)
- [`docs/SPEC_REVIEW.md`](docs/SPEC_REVIEW.md) – revize v1 a zdůvodnění změn
- [`docs/PHASE0_PLAN.md`](docs/PHASE0_PLAN.md) – plán a výsledky Fáze 0
- [`docs/DATA_FINDINGS.md`](docs/DATA_FINDINGS.md) – předběžná zjištění o datech Sharadar
- [`docs/archive/PROJECT_SPECIFICATION_v1.md`](docs/archive/PROJECT_SPECIFICATION_v1.md) – původní specifikace

## Data

Sharadar dump patří do `data/raw/<snapshot_id>/` a nikdy do gitu (viz `.gitignore` a spec §4.1).

Aktuální snapshot `sharadar_2026-09-25` je na serveru. Znovu stáhnout (vyžaduje aktivní předplatné,
klíč v `~/.config/sharadar/api_key` nebo `$SHARADAR_API_KEY`):

```
scripts/download_sharadar.sh                 # všechny tabulky do data/raw/sharadar_<dnes>/
uv run python scripts/raw_to_parquet.py data/raw/sharadar_<datum>   # → data/parquet/<snapshot>/
```
