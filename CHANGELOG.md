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
- Před nasazením doplněno: karanténa coinu s chybnými daty (den se zastaví jen kvůli drženému coinu, BTC nebo ETH),
  plnění za bid/ask ze snímku (bookTicker) místo poslední ceny, snímek všech obchodovatelných coinů univerza,
  nulové rozdělení počítané skutečným enginem (cpb/null.py) místo aproximace.
- Hodinový běh (`engine.py hourly`, timer HH:02): hodinové svíčky, kniha, futures sentiment; 7 hodinových variant
  (`h_momentum`, `h_reverze`, `h_breakout`, `h_kniha`, `funding_kontra`, `llm_nacasovani`, `seance_usa`), hodinové
  záznamy v hash chainu a replay, hodinový dataset a IC analýza, derivátová tabulka v denních podkladech pro LLM.

## v1.0.1 (2026-09-30)
Oprava provozu, žádná změna pravidel ani enginu (na stejných vstupech totožné výsledky).
- `run_daily.sh`: `git add` jen pro existující složky. Dřív chybějící `corrections/` shodila celý `git add` (chyba byla
  potlačená), takže první ostrý běh 2026-09-30 se necommitnul ani nepushnul; commitne se při dalším běhu.
- Výpis notifikace se v journalu neopakuje dvakrát; úspěšný push se zaloguje.

## v1.0.2 (2026-10-01)
Oprava provozu, žádná změna pravidel ani enginu.
- `run_daily.sh`: když GitHub push odmítne (větev mezitím posunul release commit), bot přeskládá svoje datové commity
  na novější stav (`git pull --rebase --autostash`) a pushne znovu; vše pod hodinovým zámkem. Nově stažený kód se
  spustí až po připnutí (`tools/pin.py --install`), jinak běh odmítne pracovat.
- 2026-10-01: denní běh (OK, první s Claude) se kvůli tomu nepushnul; data byla commitnutá na VPS a pushnou se ručně.

## Provozní změny (bez release, mimo připnuté soubory)
- 2026-10-01: denní běh přesunut z 00:20 UTC na 06:00 Europe/Prague (04:00/05:00 UTC), opakování 10:00 a 14:00.
  Rozhodnutí stále vychází z denní svíčky uzavřené v 00:00 UTC; plnění proběhne o ~4–5 h později než v prvních dvou
  dnech (30. 9. a 1. 10. v ~00:22 UTC). Hodinový běh beze změny (HH:02).

