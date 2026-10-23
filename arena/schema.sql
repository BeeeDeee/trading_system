CREATE TABLE IF NOT EXISTS bars(symbol TEXT, open_time INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL, qv REAL, n INTEGER,
                  tbv REAL, tbqv REAL, PRIMARY KEY(symbol, open_time));      -- tbqv = kline pole `Q`
CREATE TABLE IF NOT EXISTS book_bands(symbol TEXT, ts INTEGER, pct REAL, bid_notional REAL, ask_notional REAL, mid REAL,
                  source TEXT CHECK(source IN ('live','bookDepth')), PRIMARY KEY(symbol, ts, pct, source));
CREATE TABLE IF NOT EXISTS bar_book(symbol TEXT, open_time INTEGER, pct REAL, obi REAL, n_samples INTEGER, book_ok INTEGER,
                  source TEXT, PRIMARY KEY(symbol, open_time, pct));
CREATE TABLE IF NOT EXISTS slots(slot_id TEXT PRIMARY KEY, family TEXT, variant_id TEXT, symbol TEXT, params_json TEXT,
                   params_hash TEXT, is_control INTEGER, created_at INTEGER);
CREATE TABLE IF NOT EXISTS slot_state(slot_id TEXT, mode TEXT CHECK(mode IN ('paper','live')), status TEXT, state_json TEXT,
                   updated_at INTEGER, PRIMARY KEY(slot_id, mode));
CREATE TABLE IF NOT EXISTS signals(id INTEGER PRIMARY KEY, slot_id TEXT, bar_time INTEGER, side TEXT, intent_json TEXT,
                   action TEXT,   -- entered | suppressed_position | suppressed_cooldown | blocked_risk | stale
                   decided_at INTEGER, config_hash TEXT);
CREATE TABLE IF NOT EXISTS trades(trade_id TEXT PRIMARY KEY, slot_id TEXT, mode TEXT, twin_trade_id TEXT, signal_id INTEGER,
                   side TEXT, qty REAL, notional REAL,
                   decision_px REAL, arrival_mid REAL, entry_px REAL, entry_time INTEGER, entry_fee REAL,
                   exit_trigger_px REAL, exit_px REAL, exit_time INTEGER, exit_fee REAL, exit_reason TEXT,
                   funding REAL, gross_pnl REAL, net_pnl REAL, gross_bps REAL, net_bps REAL,
                   entry_slip_bps REAL, exit_slip_bps REAL, mfe_bps REAL, mae_bps REAL, bars_held INTEGER,
                   tainted INTEGER DEFAULT 0, cost_model_version INTEGER, config_hash TEXT);
CREATE TABLE IF NOT EXISTS orders(client_id TEXT PRIMARY KEY, trade_id TEXT, kind TEXT, is_algo INTEGER, exchange_id TEXT,
                   status TEXT, request_json TEXT, last_response_json TEXT, sent_at INTEGER, ack_at INTEGER);
CREATE TABLE IF NOT EXISTS fills(exchange_trade_id TEXT PRIMARY KEY, client_id TEXT, trade_id TEXT, px REAL, qty REAL,
                   fee REAL, fee_asset TEXT, is_maker INTEGER, ts INTEGER, realized_pnl REAL);
CREATE TABLE IF NOT EXISTS income(tran_id TEXT PRIMARY KEY, symbol TEXT, income_type TEXT, amount REAL, asset TEXT, ts INTEGER,
                   trade_id TEXT);
CREATE TABLE IF NOT EXISTS equity(ts INTEGER PRIMARY KEY, wallet REAL, unrealized REAL, margin_used REAL);
CREATE TABLE IF NOT EXISTS latencies(trade_id TEXT, kind TEXT, ms INTEGER);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts INTEGER, level TEXT, component TEXT, msg TEXT, data_json TEXT);
CREATE TABLE IF NOT EXISTS config_versions(config_hash TEXT PRIMARY KEY, yaml TEXT, activated_at INTEGER);
CREATE INDEX IF NOT EXISTS idx_book_bands_ts ON book_bands(ts);
CREATE INDEX IF NOT EXISTS idx_signals_slot ON signals(slot_id, bar_time);
CREATE INDEX IF NOT EXISTS idx_trades_slot ON trades(slot_id, mode, entry_time);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
