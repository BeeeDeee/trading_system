# Změna infrastruktury: běh večerního bota z Claude cloudu na vlastní server

Rozhodnuto: 2026-10-10 (uživatel: cutover hned, ne až po 2026-11-25). Poslední cloudový den: 2026-10-13 (naimportován z DB artefaktu, replay 12/12 OK). První serverový běh: 2026-10-14 (timer zapnut 2026-10-13).

## Co se NEMĚNÍ (pin v1.1.2)
- `engine/engine.py` (sha256 6206c93d…), `config/config.json` (sha256 c2be59cd…) – byte-identické, hash se kontroluje při každém běhu.
- Pravidla, universe, 19 variant + náhodná baseline, scoring/rozhodovací text promptu (kroky 4–6 doslovně z `task/task_prompt.md`).
- Model: claude-opus-5-5, effort high (`EVENING_MODEL`, `EVENING_EFFORT`); každý běh zapisuje použitý model do `runs/<D>.json`.

## Co se MĚNÍ
- Harness: Claude cloud scheduled task -> `claude -p` na serveru (nástroje WebSearch, WebFetch, Read, Write; `--restricted`).
- Zdroj cen: stockanalysis.com (z IP serveru blokováno, 403) -> Yahoo chart API + křížová kontrola Nasdaq API (SPY + rotující akcie, tolerance 0,3 %).
  Dopad změřen na 796 hodnotách: 54 se liší (6,8 %), max 0,23 %, hlavně otevírací ceny ETF; vliv na equity ≤ 1,3 USD z 10 000.
- Kontrola proti Nasdaq: Nasdaq doplňuje historii se zpožděním (týž večer ani po 17:00 ET řádek není). Den se proto uloží hned s cenami z Yahoo (`pending`) a Nasdaq je potvrdí zpětně při dalším běhu; rozdíl > 0,3 % nebo nepotvrzeno po 4 obchodních dnech = upozornění v run záznamu a ntfy. Rozhodnuto 2026-10-14.
- Úložiště: DB artefaktu -> soubory (`~/evening-data`), dashboard `/evening/` na serveru.
- Plánování: systemd timer po–pá 16:30 / 17:30 / 19:00 America/New_York.
- Záznamy od cutoveru nesou `code.source = "server"`.

## Dopad na pre-registraci
Pravidla ani prompt se nemění, mění se jen runtime a zdroj cen. Posoudit při vyhodnocení (od 2026-12-04), zda to
ukončuje primární segment; do té doby je hranice segmentu = datum prvního serverového běhu (dny před ním = cloud, po něm = server),
takže jde vyhodnotit obě části i společně.
