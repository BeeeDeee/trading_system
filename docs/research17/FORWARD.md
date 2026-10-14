# Výzkum 17 – forward paper test LEAD na altcoinech: protokol

Verze 1.0 · 2026-10-14 · Stav: **schváleno, běží**.
Provádí §10 pre-registrace výzkumu 17 (`PREREGISTRATION.md`). Pravidlo, náklady a univerzum se **nemění**. Tento
dokument jen určuje, jak se data po 2026-09-30 sbírají, jak se z nich počítá a kdy se rozhoduje. Po commitu
verze 1.0 platí: změny = řádek v §9 s důvodem, **před** dalším vyhodnocením.

## 1. Co se testuje

Pravidlo `LEAD_K3_LONG` beze změny (`qlab.research17`, commit kódu `c54c782f`): BTC spot 1h pohyb ≥ 3σ nahoru
(σ₁ₕ = 2σ, σ z 672 svíček 15m) a altcoin zaostává (`rA/rB < 0,5`) → long altcoinu na 4 h (16 svíček), vstup za
close signální svíčky. 10 rukávů po 1/10 kapitálu, nový vstup jen do členů měsíčního top 10 (bez BTC a ETH),
náklady 10 bps na stranu, funding perpu. Ceny a funding USDT-M perpu, vůdce BTCUSDT spot.

## 2. Období

- **Forward:** 2026-10-01 00:00 UTC → **2027-03-31 23:45 UTC**. Všechna data po 2026-09-30 jsou pro tento test
  neviděná: do schválení tohoto protokolu se nestahují ani nezkoumají.
- Indikátory (σ za 672 svíček) se počítají z dat před 2026-10-01, na začátku se nedrží žádná pozice.

## 3. Primární a popisné výsledky

- **Primární portfolio = 10 altcoinových rukávů jako v testu** (váha 1/10, cash při nečinnosti). ETH se v testu
  nepočítal do portfolia, jen jako referenční řádek, a tak zůstává i ve forwardu. *(Upřesnění §10: „altcoiny top 10
  + ETH“ čtu jako „sbírat i ETH“, ne „ETH v primárním portfoliu“.)*
- Jen popisně: ETH samostatně (váha 1), náklady 7,5 a 20 bps, plnění o svíčku později, bez fundingu, BTC perp
  jako vůdce, bez podmínky zaostávání, p-hodnota náhodné nuly po událostech (1000 běhů) a event bootstrap
  intervalu. K = 2 a short se nepočítají.

## 4. Rozhodnutí (jednou)

**Forward uspěl**, právě když na datech do 2027-03-31 platí obojí při nákladech 10 bps (×1):
1. Sharpe primárního portfolia (denní mřížka od 2026-10-01, dny bez expozice = 0, × √365) > 0,
2. průměrný čistý výnos obchodu primárního portfolia > 0.

Jinak **zamítnuto**. Další pravidla (stanovena předem, aby nešlo zastavit podle výsledku):
- **Minimální rozsah:** < 30 událostí BTC → výsledek je neprůkazný a test se prodlužuje o 3 měsíce (do
  2027-06-30, případně 2027-09-30, …), dokud nemá ≥ 30 událostí na konci čtvrtletí. Rozhoduje se až tehdy.
- **Průběžné reporty jsou popisné.** Smějí se spouštět kdykoli, ale nikdy nezastavují, neprodlužují ani nemění
  test, a nevybírá se podle nich nic (žádné ladění, žádný výběr altcoinů).
- Výsledek při ×1 je rozhodující. Citlivost na náklady (7,5 / 20 bps) se jen hlásí.

## 5. Data

| Co | Zdroj | Poznámka |
|---|---|---|
| Svíčky 15m perp (BTC, ETH, členové), BTC spot | `data.binance.vision` **denní** ZIPy `…/daily/klines/{sym}/15m/` | sha256 podle `.CHECKSUM`, stahují se dny do včerejška (UTC); den, který ještě není zveřejněn, ukončí stahování |
| Historie před 2026-10-01 | existující snapshot `binance_2026-10-03` | beze změny |
| Nový člen bez historie ve snapshotu | měsíční 15m ZIPy do 2026-09 | stejná ověření |
| Funding | Binance REST `fapi/v1/fundingRate` od 2026-10-01 | ukládá se surový JSON; chybí-li záznam, paušál 0,01 % / 8 h (jako dosud) |
| Členství | měsíční 1d ZIPy perpů | člen měsíce *m* = top 10 podle objemu za 30 dní před *m*, pravidla `qlab.research15.universe` beze změny; vypočte se, až je měsíc *m*−1 zveřejněn |

- **Členství je pevné:** jakmile se měsíc vypočte, další běh jej musí vrátit totéž, jinak se zastaví (revize dat).
- Měsíc bez vypočteného členství (měsíční ZIP *m*−1 ještě není) se do vyhodnocení nezapočítá. Vyhodnocení
  běží k poslednímu dni, který mají všichni potřební aktéři, a pro který je členství známé (`asof`).
- Panel se staví týmž způsobem jako v `scripts/r15_build_panel.py` (přenesený close, příznak `real`, funding na
  svíčku podle času záznamu). Shoda staveče s panelem snapshotu je v testech.

## 6. Skripty a záznam

- `scripts/r17_forward.py freeze` zapíše otisk souborů pravidla do `forward_lock.json`. Každé další spuštění ho
  ověří, jinak končí.
- `… download` stáhne data (idempotentní, vše po souborech s ověřeným checksum).
- `… status` postaví panel, spočítá primární a popisné výsledky k `asof` a uloží
  `results/forward_<asof>.json` (včetně otisku použitých ZIPů a commitu). Soubor se **commituje**: čas commitu
  dokládá, že výsledek vznikl před pozdějšími daty.
- `… final` je povoleno až při `asof ≥ 2027-03-31` (resp. konci prodloužení) a ≥ 30 událostech, jednou, zapíše
  `results/forward_final.json` s rozhodnutím podle §4.
- Průběžný režim: 1× týdně (cron, `scripts/r17_forward_weekly.sh`) `download` + `status`, kontrola zdraví
  (`qlab.research17.ops`), obnova dashboardu (`qlab.research17.dashboard`, záložka „LEAD forward“ v portalu) a upozornění
  přes ntfy při chybě stahování, selhání statusu, zaostávajících datech, zastaralých symbolech nebo chybějícím členství.
  Nové `forward_<datum>.json` se commituje ručně. Při chybě stahování se nic neinterpoluje.

## 7. Co forward neověřuje

- **Skutečný skluz.** Paper test používá 10 bps na stranu a z 15m svíček nelze změřit, kolik by stálo skutečné
  plnění po skoku BTC (rychlý trh, tenčí kniha). Zbytková nejistota je u kritéria ×1 výrazná: při 20 bps byl
  v testu Sharpe 0,06, při 7,5 bps 0,98.
- Nezávislost událostí a vliv jedné minci. Hlásí se, nerozhodují.

## 8. Co se stane při úspěchu

Forward 6 měsíců s 10 bps jako ×1 je důkaz jen pro paper. Před reálným kapitálem je potřeba samostatně změřit
skluz a rozhodnout o velikosti pozic (nový dokument).

## 9. Log rozhodnutí

| Datum | Změna |
|---|---|
| 2026-10-14 | Verze 0.1, návrh. |
| 2026-10-14 | Verze 1.0 schválena vlastníkem beze změn (primární portfolio = 10 altcoinů, prodloužení po 3 měsících při < 30 událostech). Před zmrazením a prvním stažením se nestahovala ani nezkoumala žádná data po 2026-09-30. |
| 2026-10-14 | Provozní doplněk (nemění pravidlo ani rozhodnutí): týdenní běh, upozornění a dashboard; `forward_<datum>.json` obsahuje navíc denní výnosy a seznam obchodů. |
