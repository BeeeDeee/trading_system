# Local Development Setup

> Develop and backtest on your machine. Use the VPS for paper/live later.

## Target path (Windows)

```text
C:\git\others\trading_system
```

This Cursor session runs on the **VPS**. Your Windows folder is separate —
clone (or pull) the git repo there.

## 1. Prerequisites (Windows)

- Git
- Python **3.10+** ([python.org](https://www.python.org/downloads/) — tick “Add to PATH”)
- Optional: VS Code / Cursor opened on `C:\git\others\trading_system`

Check:

```powershell
python --version
git --version
```

## 2. Get the code

```powershell
mkdir C:\git\others -Force
cd C:\git\others
git clone git@github.com:BeeeDeee/trading_system.git trading_system
cd trading_system
```

If SSH is not set up, use HTTPS:

```powershell
git clone https://github.com/BeeeDeee/trading_system.git trading_system
```

Then open that folder in Cursor.

## 3. Python environment

```powershell
cd C:\git\others\trading_system\backend
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 4. Env file

```powershell
cd C:\git\others\trading_system
copy .env.example .env
```

Milestone 1 backtests need **no exchange API keys**.

## 5. Download BTC/USDT 1h history

From `backend` with venv active (needs network once):

```powershell
cd C:\git\others\trading_system\backend
download-data
```

Or:

```powershell
python -m app.cli.download_data
```

Writes `data\processed\BTCUSDT_1h.parquet`.

## 6. Run the reference backtest

```powershell
run-backtest
```

Or:

```powershell
python -m app.cli.run_backtest
```

Outputs go to `results\<run-id>\`:

- `equity.csv`
- `trades.csv`
- `metrics.json`
- `config.yaml`

Config: `config\backtest.yaml`.

## 7. Tests

```powershell
cd C:\git\others\trading_system\backend
pytest
```

## Linux / macOS / VPS equivalent

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
download-data
run-backtest
pytest
```

## What you have after this

| Piece | Role |
|---|---|
| `backend/app/` | Trading platform code |
| `config/backtest.yaml` | Backtest parameters |
| `data/` | Parquet market data (gitignored) |
| `results/` | Backtest outputs (gitignored) |
| `docs/` | Architecture |
| EMA cross strategy | Reference pipeline through risk → simulated fills |

Next coding work: improve strategies/features, then Milestone 1b (news/sentiment), then paper trading on the VPS.
