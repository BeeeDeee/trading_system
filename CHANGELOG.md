# Changelog

Každá změna pravidel, enginu, promptu, schématu, dashboardu nebo modelu = nový release (VERSION, MANIFEST.sha256,
tag, `tools/pin.py --install`). Každý záznam běhu nese `version.tag`, `manifest_sha256` a připnutý model, takže je
vždy dohledatelné, co ho spočítalo. Výsledky různých verzí se v analýze nemíchají.

## v1.0.0 (2026-09-30)
První verze. Orphan větev `crypto-paper-bot` (vzory převzaté kopií z akciového bota `paper-trading-bot` v1.1.2:
deterministický engine, varianty v configu, připnutí otisky, validace skóre bez doplňování, catch-up bez rozhodnutí,
náhodná baseline + nulové rozdělení, rank IC s bootstrapem, pravidla vyhodnocení předem).
- Engine (stdlib): fetch z Binance s fallbackem OKX/Coinbase/Kraken a doslovným uložením odpovědí, kontroly dat,
  křížové kontroly (close vs OKX, snímek vs CoinGecko/CoinPaprika), indikátory, týdenní univerzum 20 coinů.
- Krok LLM: `claude -p` s `claude-opus-5-5`, nástroje jen WebSearch/WebFetch/Read/Write, `--restricted`, bez MCP;
  validace, jedna oprava přes `--resume`, kontrola modelu.
- 18 variant + 5 benchmarků, 3 nákladové scénáře (čistý/stres/hrubý), stopy na 5min svíčkách, delisting, redenominace.
- Zámek rozhodnutí před snímkem, hash chain přes běhy i opravy, replay celého pokusu bajt po bajtu.
- Dashboard (statický HTML + JSON, CSP), systemd timer, Caddy + basic auth + fail2ban, install.sh po krocích.
