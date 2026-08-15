# API

> External control and observation API of the trading platform.

## 1. Purpose

FastAPI provides the external API of the platform.

The API is a control-plane interface.

It is not the implementation of the trading engine.

---

## 2. API Responsibilities

The API provides access to:

- system status,
- health,
- configuration,
- strategies,
- portfolio state,
- orders,
- trades,
- risk state,
- operational commands,
- reports.

---

## 3. Initial API Structure

```text
/api/v1/

health
status

strategies
portfolio
positions

orders
trades

risk
configuration

reports

runtime
````

Exact endpoints will be defined during API design.

---

## 4. Runtime Control

The API may expose commands such as:

```text
START
STOP
PAUSE
RESUME
HALT
RECONCILE
```

The API sends commands to the Trading Runtime.

FastAPI must not directly manipulate internal runtime state.

Commands are represented by `RuntimeCommand` objects containing `command_id`,
`type`, `timestamp`, `requested_by`, and `payload`. The runtime reports
`accepted`, `rejected`, `completed`, or `failed`.

Command semantics are defined in `RUNTIME.md`. In particular, `STOP` does not
automatically close positions and `HALT` prohibits new orders.

---

## 5. Request Flow

```text
HTTP Request
      ↓
FastAPI
      ↓
Application Service
      ↓
Domain / Runtime
      ↓
Result
      ↓
API Response
```

---

## 6. API Rules

* API contracts must be versioned.
* API models should not expose exchange-specific models.
* API requests must not bypass risk controls.
* Long-running operations should not block request handling.
* Administrative operations must be authenticated.
* Sensitive information must not be returned by default.

---

## 7. Future API Areas

Potential future areas include:

* backtesting,
* model management,
* market-data inspection,
* system diagnostics,
* dashboards.
