# Backend

Python trading application (modular monolith).

## Requirements

- Python 3.10+

## Setup

```bash
cd backend
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

Full Windows path instructions: [`docs/LOCAL_SETUP.md`](../docs/LOCAL_SETUP.md).

## Download data + run backtest

```bash
download-data          # BTC/USDT 1h -> ../data/processed/BTCUSDT_1h.parquet
run-backtest           # writes ../results/<run-id>/
pytest
```

Equivalent:

```bash
python -m app.cli.download_data
python -m app.cli.run_backtest
```

## Control plane (stub)

```bash
uvicorn app.main:app --reload
```
