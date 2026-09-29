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

## v1.0.1 (2026-09-29)
Jen robustnost, žádná změna pravidel: na platných vstupech dává výsledky totožné s v1.0.0
(ověřeno 30 simulovanými dny, všech 20 portfolií i obchody).
- `plan` už nepadá na špatném vstupu od LLM (float/None/text ve skóre, neznámý ticker, špatný `conviction`,
  `weight_pct`, `stop_price`); hodnoty ořízne nebo ignoruje a skóre nikdy nedopočítává.
- Nový příkaz `validate`: před `plan` ohlásí chybějící tituly a hodnoty mimo rozsah.
- Split: `check` rozpozná čistý poměr (`split_suspects`), nový `apply-split` přepočte historii, pozice, stopy
  a základny benchmarků. Bez toho by split zablokoval `check` (i doplnění zpětně) natrvalo.
- `stop_price` nad nákupní cenou se ignoruje (dřív okamžitý stop-out); `new_stop` nad cenou také.
- Zápis JSON je atomický, CLI kontroluje počet argumentů.
- Nové testy: gap-stop, trailing stop, max. doba držení, denní limit, tržní filtr, doplnění zpětně,
  špatný vstup, split, atomický zápis. Nový nástroj `tools/analyze.py`.

## v1.1.0 (2026-09-29)
Změna pravidel, provedená po prvním (startovním) dni, kdy ještě neexistovaly žádné obchody. Pokyny vytvořené
v1.0.x večer 28. 9. se vykonají 29. 9. na openu (settle se nemění); pravidla v1.1.0 platí pro všechny plány od 29. 9.
- `consensus`: vstup vyžaduje aspoň 2 kladné pohledy a **žádný záporný** (`max_neg_lenses`, výchozí 0).
  Dřív prošlo i (+1, +1, −2). Týká se 10 variant.
- Denní limit ztráty platí i pro `top_n` varianty (`plne`, `top3`, `rotace`, `kontrarian`) a náhodnou baseline.
- `top_n`: držený titul bez skóre 2 plány po sobě se prodá (dřív se držel navždy); prodaný titul
  se týž den nekupuje zpět.
- Zadání: věta o zkrácených obchodních dnech.
- Nástroj `tools/random_null.py`, pravidla vyhodnocení předem v README.
