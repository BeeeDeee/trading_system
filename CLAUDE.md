# arena5m: pokyny pro implementaci

Závazná specifikace je `SPEC.md`. Před každou fází si přečti příslušnou sekci a sekci 18 (nástrahy).

- Postupuj po fázích P0 až P7 (SPEC §17). Fázi neoznačuj za hotovou, dokud nesplňuje akceptační kritéria a testy
  neprojdou. Výsledek zapiš do `CHANGELOG.md`.
- Rozhodnutí D1–D9 (SPEC §1) neměň. Když narazíš na rozpor s realitou API, zastav se, popiš ho a zeptej se vlastníka.
- Body označené **OVĚŘIT** ověř proti https://developers.binance.com/docs/derivatives/usds-margined-futures nebo proti
  živému API (read-only) a výsledek zapiš do `docs/VERIFIED.md` (datum, endpoint, co platí).
- Nepoužívej trading knihovny (python-binance, ccxt). Klient je vlastní a tenký (SPEC §6).
- Výchozí `env` je `testnet` a `live.enabled: false`. Na mainnet s reálnými příkazy nikdy bez výslovného souhlasu
  vlastníka v konverzaci. Read-only volání na mainnet (exchangeInfo, klines, account) jsou v pořádku.
- Klíče jen z `/etc/arena/env` nebo env proměnných. Nikdy je nevypisuj ani nelogni.
- Styl: Python 3.12, `uv`, typy, dataclasses, žádné globální proměnné kromě konfigurace. Strategie jsou čisté funkce.
- Testy: `uv run pytest`. Síťové testy jsou označené `@pytest.mark.net` a v defaultním běhu se přeskakují.
- Vlastník komunikuje česky, chce stručné, přímé odpovědi bez přikrášlování výsledků.
