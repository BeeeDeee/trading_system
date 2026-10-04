# Výzkum 13 – audit dat (`binance_2026-10-03`)

`panel_r13` = `panel_r9` + `extra_high`, `extra_low` ze stejných raw svíček a se stejnými filtry řádků.

- panel 2017-08-01 → 2026-08-31, 3318 dní, 663 párů
- všechny matice `panel_r9` (vč. `open`, `qv`) bitově shodné: **ano**
- close přepočtený z raw svíček bitově shodný s `panel_r9`: **ano**; high/low existují přesně na řádcích svíček
- svíček: 779068; high < max(open, close): 1; low > min(open, close): 0; high < low: 0

