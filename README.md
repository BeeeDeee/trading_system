# strategy_backtester_2026_sep

Lokální výzkumný framework pro vývoj a poctivé vyhodnocení long-only strategií na denních datech
US akcií (Sharadar).

Stav: Fáze 0–3 hotové (`docs/PHASE0_PLAN.md`, `docs/PHASE1_REPORT.md`, `docs/PHASE2_REPORT.md`, `docs/PHASE3_REPORT.md`). Další krok: rozhodnutí o Fázi 4 (ML) a robustnostní běhy před holdoutem.

Testy: `uv run pytest`

## Dokumenty

- [`docs/PROJECT_SPECIFICATION.md`](docs/PROJECT_SPECIFICATION.md) – aktuální specifikace (v2)
- [`docs/SPEC_REVIEW.md`](docs/SPEC_REVIEW.md) – revize v1 a zdůvodnění změn
- [`docs/PHASE0_PLAN.md`](docs/PHASE0_PLAN.md) – plán a výsledky Fáze 0
- [`docs/PHASE1_REPORT.md`](docs/PHASE1_REPORT.md) – Fáze 1: pipeline dat, audit, benchmarky
- [`docs/PHASE2_REPORT.md`](docs/PHASE2_REPORT.md) – Fáze 2: 3 027 kandidátů, výsledky rodin, PBO/DSR
- [`docs/PHASE3_REPORT.md`](docs/PHASE3_REPORT.md) – Fáze 3: walk-forward 2005–2019, brána, interpretace
- [`configs/frozen_defaults.yaml`](configs/frozen_defaults.yaml) – zmrazené výchozí hodnoty metodiky
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
