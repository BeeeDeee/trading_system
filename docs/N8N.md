# n8n Integration

> Role and boundaries of n8n in the trading platform.

## 1. Purpose

n8n is the external orchestration and automation layer.

It is not part of the core trading engine.

---

## 2. Responsibilities

n8n may handle:

- scheduling,
- workflows,
- notifications,
- reporting,
- external SaaS integrations,
- operational automation,
- periodic jobs.

---

## 3. Typical Workflow

```text
n8n
  ↓
FastAPI
  ↓
Application Service
  ↓
Trading Platform
  ↓
Result
  ↓
n8n
  ↓
Telegram / Email / External System
````

---

## 4. Examples

### Daily report

```text
Schedule
  ↓
n8n
  ↓
GET /reports/daily
  ↓
Telegram
```

### Model retraining

```text
Schedule
  ↓
n8n
  ↓
POST /models/retrain
  ↓
Python
  ↓
Result
  ↓
Telegram
```

### Operational alert

```text
Health check
  ↓
FastAPI
  ↓
Failure detected
  ↓
n8n
  ↓
Telegram
```

---

## 5. Realtime Trading Boundary

n8n must not participate in the realtime trading loop.

Forbidden:

```text
Market Event
   ↓
n8n
   ↓
Strategy
   ↓
Risk
   ↓
Order
```

Core trading must continue even if n8n is unavailable.

---

## 6. Business Logic Boundary

Trading business logic belongs in Python.

Do not duplicate:

* strategies,
* indicators,
* risk rules,
* execution rules,
* portfolio logic

inside n8n workflows.

---

## 7. Data Collection

Python is the primary data acquisition layer for data required by trading and analysis.

n8n may collect data for unrelated automation workflows, but such data should not become a hidden part of the trading decision pipeline.

---

## 8. Integration Rule

n8n interacts with the platform through documented APIs.

It should behave like an external operator of the platform, not like an internal trading module.
