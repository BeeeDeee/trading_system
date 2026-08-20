# Backend Skeleton

This directory contains the minimal Python application skeleton and type-only
domain ports. It intentionally has no exchange, database, strategy, or trading
pipeline implementation yet.

## Requirements

- Python 3.10 or newer

## Setup

```bash
cd backend
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Verification

```bash
python -m compileall -q app
PYTHONPATH=. pytest ../tests
```

## Run the Control Plane

```bash
uvicorn app.main:app --reload
```

The application currently exposes only the FastAPI application factory. Routes
and runtime wiring are deliberately deferred until the simulated-execution
vertical slice is implemented.
