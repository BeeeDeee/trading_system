# Changelog

Každá změna pravidel nebo enginu = nový commit, nový tag a nové připnutí v zadání
naplánované úlohy. Výsledky v deníku nesou u každého dne `code.tag` a `code.commit`,
takže je vždy dohledatelné, jaká verze je spočítala.

## v1.0.0 (2026-09-29)
První verzovaná verze. Kód beze změny oproti prvnímu večernímu běhu (28. 9. 2026).
- Engine: indikátory, kontrola dat, vypořádání na open dalšího dne, stopy, doba držení,
  poplatky 5 bps a slippage 5 bps (u stopu +20 bps), plán pro všechny varianty.
- 19 variant + náhodný výběr, univerzum 20 titulů (11 ETF, 9 akcií), $10 000 na variantu.
- Zadání úlohy připíná verzi otisky enginu a konfigurace (SHA-256), ověřují se při každém běhu.
  Kód se stahuje z větve `paper-trading-bot`, záloha je kopie v databázi deníku.
