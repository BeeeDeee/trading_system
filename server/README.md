# Večerní bot na serveru

Přesun naplánované úlohy z Claude aplikace na tento server. Pravidla se **nemění**: `engine/engine.py` a
`config/config.json` jsou byte-identické s release v1.1.2 (SHA-256 pin v `task/task_prompt.md`, runner ho kontroluje
při každém běhu a při nesouladu běh selže). Mění se jen orchestrace a zdroj cen.

| Dřív (Claude cloud) | Teď (server) |
|---|---|
| naplánovaná úloha, `ArtifactData` databáze | `python3 -m evening run`, soubory v `~/evening-data` (git repo, záloha) |
| ceny: WebFetch stockanalysis.com | ceny: Yahoo chart API (OHLC), křížová kontrola SPY + 1 akcie přes api.nasdaq.com (obě vrací JSON; stockanalysis je za Cloudflare) |
| kroky 4–6 (rešerše, skóre, rozhodnutí „Claude volně“) | `claude -p` (WebSearch, WebFetch, Read, Write), text kroků 4–6 **doslovně** vyříznutý z `task/task_prompt.md` |
| ostatní kroky: LLM volá `engine.py` | Python orchestrace volá stejné příkazy `engine.py` |
| artefakt čte DB | `report.py` vloží data do `dashboard/dashboard.src.html` (CSP s hashem skriptu, bez Google Fonts) → `/srv/evening/current` → `https://<ip>/#evening` |

Orchestrace: `evening/runner.py` (kroky 0–9 promptu: pin, NYSE kalendář, doplnění zmeškaných dní, kontrola dat,
split přes Yahoo `events=split`, settle, LLM, validate + 1 oprava, plan, uložení `days` až jako poslední, `runs/<den>` vždy).

## Ověřeno
- `tools/replay_check.py DATA`: všech 10 uložených dnů (2026-09-28 … 10-09) se přes `settle` + `plan` s uloženými skóre a
  rozhodnutími reprodukuje **identicky** (včetně obchodů). Seed + engine + config na serveru jsou ty, co dávaly cloudové výsledky.
- `python3 -m unittest discover -s tests`: kalendář, pin, výřez promptu, běžný den, doplnění, zavřená burza, výpadek dat,
  nesoulad křížové kontroly, dry-run, notifikace a dashboard (bez sítě a bez LLM).
- Dry-run dne 2026-10-09 na kopii dat se skutečným Yahoo/Nasdaq a injektovanými cloudovými skóre: stejné plnění a pozice,
  rozdíly jen v `last`/equity z centů u 7 titulů (viz níže).

## Zdroj cen: známý rozdíl proti cloudovým datům
Yahoo a Nasdaq se shodují na všech 796 porovnaných hodnotách (OHLC, 20 titulů, od 28. 9.). Uložená data ze stockanalysis.com
se od obou liší u 54 z nich (6,8 %, nejvíc 0,23 %, hlavně otevírací ceny ETF a občas close o pár centů). Server tedy používá
data, která potvrzují dva nezávislé zdroje; historická cloudová data se nepřepisují. Dopad na equity ≤ $1,3 z $10 000.

## Provoz
```
cd server && ./run_daily.sh --dry-run --force [--date YYYY-MM-DD]   # na kopii dat
python3 -m evening --data ~/evening-data report --publish /srv/evening
python3 tools/replay_check.py ~/evening-data
python3 tools/import_export.py EXPORT_DIR ~/evening-data           # při přechodu: dnešní export DB artefaktu
journalctl -u evening-daily -n 100
```
Timer: po–pá 16:30 New York + 17:30 a 19:00 (no-op, když den existuje). Čas v New Yorku, takže se nehýbe vůči
závěru burzy v týdnech, kdy se USA a Evropa přepínají na letní čas v jiný den.

## Přechod z cloudu (cutover)
Dvě instance nesmí běžet naostro najednou. Pořadí: (1) `install.sh` kroky user…web + dry-runy, (2) v den přechodu po
večerním běhu cloudu nový export DB artefaktu → `import_export.py`, `replay_check.py`, (3) vypnout naplánovanou úlohu
v Claude aplikaci, (4) `install.sh timer`. Pre-registrace (`PREREGISTRATION.md`): primární okno plánů je do 2026-11-25
a release se změnou promptu ho ukončuje; rozhodnutí 2026-10-10: cutover hned, změna je zapsaná v `INFRA_CHANGE.md`
(`code.source = "server"` ve všech nových záznamech). Model opus-5-5, effort high.
