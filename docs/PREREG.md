# PREREG (SPEC §14.1): NÁVRH, vlastník projde, upraví a commitne PŘED prvním paper barem

- Verze konfigurace: `arena config-show` → `config_hash` doplnit při startu. `cost_model_version: 1`.
- Varianty: 30 strategických (brk, obi, ll; 140 slotů přes symboly) + kontrolní rodina ctl (nepočítá se do N). Hashe v `config/variants.lock.json`.
- Kumulativní N trialů: 30 (zapsat do vlastníkova registru).
- Rozhodovací pravidla: SPEC §14.2 beze změny (n ≥ 500 bez tainted, net > 0 při 1,5× nákladech po kalibraci ≥ 30 live obchody na symbol, Holm p < 0,05 vs kontrola, kladný průměr na ≥ 4 z 5 symbolů (ll 3 ze 4), žádný měsíc pod −3 sd).
- Termíny kontrol: den 45 a den 90 od startu. Start (UTC): ______  → den 45: ______  → den 90: ______
- Nikdy: ladit parametry pod stejným ID, hlásit nejlepší variantu bez DSR, více live slotů na symbol, mazat obchody.
- Live (sondy) běží zatím jen na demo/testnetu; mainnet odložen. Live je jen měření nákladů; live výsledky nejsou důkaz edge.
