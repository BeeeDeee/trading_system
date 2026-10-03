# Crypto paper-trading bot

Denní pokus na 20 kryptoměnách: Claude (`claude -p`) jednou denně udělá online rešerši a ohodnotí univerzum,
deterministický Python engine z toho obchoduje **18 denních variant**, **7 hodinových variant** a **5 benchmarků**,
každou s virtuálními $10 000 (kotace USDT). Vedle denního běhu sbírá engine každou hodinu hodinové svíčky, knihu
objednávek a sentiment z futures (funding, open interest, long/short); LLM se volá dál jen jednou denně. Žádná burza, žádné API klíče k obchodování, žádné skutečné peníze. Plánovaná délka 8–12 týdnů.

Hlavní produkt je **point-in-time dataset** (ceny, indikátory, skóre LLM, rozhodnutí, plnění) a férové srovnání
proti kontrolám: `mech_momentum` (stejná pravidla bez LLM), náhodný výběr, kontrarián, BTC HOLD a rovné váhy univerza.
Altcoiny jsou z velké části beta k BTC a schopnost LLM předpovídat krátkodobé výnosy je neprokázaná; nulový výsledek
je stejně cenný jako pozitivní.

> **Repo je veřejné a do větve se pushuje všechno** (skóre, rozhodnutí, ledger, raw odpovědi API). Heslo dashboardu
> chrání jen pohodlný pohled, ne data.

## Principy

| Princip | Jak je vynucen |
|---|---|
| LLM nesahá na stav | `claude -p` běží s nástroji jen `WebSearch, WebFetch, Read, Write`, `--restricted` (soubory jen v `runs/D/llm/`), bez Bash, bez MCP. Zapíše jen `scores.json`. Vše ostatní dělá engine. |
| Ceny jen z kódu | Veřejná REST API burz; každá odpověď doslova v `runs/D/raw/` + `raw_manifest.json` (URL, čas, HTTP status, sha256). Nic se neodhaduje ani nedopočítává. Chybná data = žádný obchod. |
| Žádný lookahead | Cílové váhy všech variant se zamknou (`decisions.json.locked_at`) **před** stažením snímku pro plnění; `snapshot.fetched_at` i čas serveru Binance musí být pozdější, jinak běh selže. |
| Append-only, tamper-evident | Každý záznam (`run.json`, `corrections/*.json`) nese `prev_hash`, sha256 všech souborů běhu a `this_hash`; `data/chain.jsonl` je jen přidávaný. Oprava = nový záznam `correction`. |
| Připnutá verze | `MANIFEST.sha256` (engine, config, prompt, schéma, dashboard, `run_daily.sh`) + připnutí mimo repo v `~/.config/cpb/pin.json`. Nesoulad = běh `failed`, žádné obchody. |
| Determinismus | Seedovaná náhoda, stabilní řazení (tie-break hashem data a symbolu), kanonický JSON na každé hranici fází. `tools/verify_chain.py` přehraje celý pokus z uložených vstupů bajt po bajtu. |
| Hodinová data | Hodinový běh nevolá LLM, nemění denní rozhodnutí a jeho selhání neovlivní denní varianty. |
| Web je nedůvěryhodný | Prompt to říká explicitně; engine validuje schéma, ořízne rozsahy, zahodí nevalidní coin (nikdy nedoplní), odstraní řídicí znaky, pustí jen http(s) URL. Dashboard vkládá text z LLM jen přes `textContent` a má CSP s hashem jediného skriptu. |
| Připnutý model | `config.llm.model = claude-opus-5-5`; ověřuje se proti `init` a `modelUsage` v transcriptu. Jiný model = skóre se nepoužije (LLM varianty drží). Změna modelu = nový release = nový segment. |

## Denní běh (vše UTC)

systemd timer v **06:00 pražského času** (04:00 UTC v létě, 05:00 UTC v zimě), `Persistent=true`, opakování v 10:00
a 14:00 pražského času jen když den chybí. Rozhoduje se z denní svíčky uzavřené v 00:00 UTC; datum běhu je datum UTC.

```
A  fetch      univerzum (pondělí: přestavba), denní svíčky (Binance → OKX → Coinbase → Kraken), stav párů
   check      chybějící/duplicitní svíčky, nekladné ceny, high<low, zastaralá data, nulový objem, pohyb > 60 %
              bez potvrzení druhou burzou (±3 %), revize uložené historie, podezření na redenominaci;
              křížová kontrola close BTC + rotujícího coinu proti OKX (0,5 %, jeden nový fetch, pak selhání)
   settle     stopy na 5min svíčkách od posledního plnění, delisting (nucený prodej), ocenění k 00:00
   features   indikátory → features.json + features.md (tabulka pro LLM)
B  claude -p  rešerše + skóre → llm/scores.json + transcript; validace; při chybách 1 oprava (--resume)
C  lock       cílové váhy všech 23 portfolií → decisions.json (locked_at)
   snapshot   bid/ask (bookTicker) všech obchodovatelných coinů + čas serveru; křížová kontrola středu BTC + rotujícího
              coinu vs CoinGecko/CoinPaprika (0,5 %)
   fill       stopy do času snímku, výstupy, rebalanc ve 3 nákladových scénářích
D  report     run.json + hash chain, dashboard, commit + push, notifikace
```

## Hodinový běh (vše UTC)

systemd timer `cpb-hourly.timer` v HH:02 každou hodinu (zmeškané hodiny se **nedoplňují**, nikdy se nerozhoduje zpětně):

```
fetch     uzavřené hodinové svíčky (1. běh: 35 dní), top 100 úrovní knihy (spread, hloubka, nerovnováha),
          perpetual futures Binance (funding, změna OI za 1 h, poměr long/short účtů); paralelně, raw gzip + manifest
check     jako denně po hodinách; pohyb > 25 % za hodinu potvrzuje OKX → Coinbase → Kraken; karanténa coinu
phase_h   trailing stopy na nových hodinových svíčkách, delisting, ocenění k HH:00, hodinové indikátory
plan      cílové váhy 7 hodinových variant → decisions.json (zamčeno před snímkem)
snapshot  bid/ask + čas serveru (po zámku), BTC vs OKX
execute   stejné účetnictví a náklady jako denní varianty, 3 scénáře
```

Stav hodinových variant je oddělený (`data/hourly/state.json`, historie `data/hourly/history/<COIN>/<den>.json`),
záznamy `runs/D/hourly/HH/` jsou v hash chainu (`type: hourly`) a replay je přehrává bajt po bajtu. Denní a hodinový
běh mají vlastní zámky, zápis do řetězce hlídá společný krátký zámek. Commit hodinových záznamů dělá denní běh
(jeden commit „hourly D“), rychlý dashboard se obnovuje každou hodinu. Denní tabulka pro LLM obsahuje navíc
derivátový sentiment z posledního hodinového běhu.

Když krok B selže (timeout, limit, nevalidní výstup, jiný model), varianty bez LLM a benchmarky běží normálně,
LLM varianty drží pozice (stopy a max. doba držení platí dál) a běh má status `warning`. Skóre se nikdy nedoplňuje.

**Karanténa coinu:** chybná data coinu, který nedrží žádné portfolio (a není to BTC ani ETH), celý den nezastaví.
Coin jde na ten den do karantény: jeho svíčky se neuloží, nemá indikátory, nejde koupit a další běh ho stáhne znovu.
Chybná data drženého coinu, BTC nebo ETH běh zastaví (držený coin nejde ocenit a cena se nikdy neodhaduje).
Coin bez platné ceny ve snímku se ten den neobchoduje (jeho cílová váha zůstane v hotovosti); je-li držený, běh selže.

Zmeškané dny se při dalším běhu doplní jako `catchup`: jen vypořádání (stopy, delisting) a ocenění, žádná rozhodnutí.
Selhaný pokus se uloží do `runs/D/failed-HHMMSS/` (s raw daty a přesnou chybou), zařadí se do hash chainu a stav se
nemění; další pokus týž den (ruční nebo timer v 06:20) může proběhnout. Existuje-li `runs/D/run.json`, běh nic nedělá.

## Varianty (config/config.json)

`composite = trend + news + 0,5·mr + (conviction − 3)`; **brána** = close > SMA20 ∧ RSI14 < 80 ∧ bez `risk_flag`.
Váhy < 2 % se zahodí, max 25 % na coin (zbytek hotovost), obchod jen při odchylce > 2 % kapitálu od cíle
(úplný výstup vždy), min. obchod $10.

| id | Výběr | Alokace | Otázka |
|---|---|---|---|
| `zaklad` | LLM top 10, composite > 0, brána | ∝ composite | referenční varianta |
| `rovne` | jako zaklad | rovné | pomáhá vážení skóre? |
| `inv_vol` | jako zaklad | ∝ 1/ATR% | rizikové vážení vs skóre? |
| `skore_vol` | jako zaklad | ∝ composite/ATR% | kombinace? |
| `bez_brany` | LLM top 10 | ∝ composite − min + 1 (vždy investováno) | přidávají indikátory něco? |
| `btc_filtr` | jako zaklad | ∝ composite; BTC < SMA50 → 100 % cash | pomáhá tržní režim? |
| `vol_target` | jako zaklad | ∝ composite × min(1, 50 % / vol portfolia z 30d kovariance) | řízení expozice? |
| `top3` | prvních 3 z LLM top 10, composite > 0, brána | ∝ composite, max 40 % | koncentrace? |
| `tydenni` | jako zaklad | rebalanc jen v pondělí | kolik stojí obrat? |
| `stop` | jako zaklad | stop 10 % od nákupu, trailing 15 % od maxima (5min svíčky); vystopovaný coin se týž den nekupuje | pomáhají stopy? |
| `jen_trend` | celé univerzum, trend ≥ 1, top 10 | rovné | edge trendu |
| `jen_zpravy` | news ≥ 1, top 10 | rovné, max držení 5 d | edge zpráv |
| `jen_mr` | mr ≥ 1, top 10 | rovné, max držení 3 d | krátkodobá reverze |
| `pravdepodobnost` | p_outperform > 0,55, top 10 | ∝ (p − 0,5) | jsou pravděpodobnosti užitečné? |
| `claude_volne` | volné rozhodnutí LLM (≤ 25 %/coin, ≤ 100 %) | volné | umí úsudek víc než pravidla? |
| `mech_momentum` | **bez LLM**: rel30 vs BTC > 0, brána (bez risk_flag), top 10 | ∝ rel30 | **přidává LLM něco nad momentem?** |
| `kontrarian` | nejhorších 10 podle composite | rovné, vždy investováno | když vede, skóre je šum |
| `nahodny` | 10 náhodných (seed = sha256(datum)) | rovné | baseline štěstí |

### Hodinové varianty (bez LLM, kromě `llm_nacasovani`)

Rozhodují v hodinovém běhu z dat uzavřených do HH:00 a plní se za bid/ask snímku o pár sekund později.

| id | Pravidlo | Otázka |
|---|---|---|
| `h_momentum` | top 3 podle výnosu vs BTC za 4 h (> 0), po 1/3, pozice drží min. 4 h | přežije intradenní momentum náklady? |
| `h_reverze` | hodinový výnos < −2,5 σ vlastní hodinové volatility (30 d) a BTC ne pod −1 %; max 3 × 15 %; prodej po 3 h nebo nad open propadové hodiny | vrací se přehnané hodinové propady? |
| `h_breakout` | close nad maximem předchozích 24 h a objem > 3× průměr; 20 %, max 5 pozic, trailing stop 3 %, max 24 h | fungují průrazy s objemem? |
| `h_kniha` | nerovnováha top 100 úrovní knihy > 0,25 a podíl agresivních nákupů v hodině > 55 %; top 3 × 20 %; drží min. 2 h, dokud signál trvá | předpovídá mikrostruktura příští hodiny? |
| `funding_kontra` | v 00/08/16 UTC: 5 coinů s nejnižším fundingem pod 0,01 % (bez horních 20 % fundingu), po 20 % | je přeplněnost derivátů kontrariánský signál? |
| `llm_nacasovani` | váhy jako `zaklad` z posledního denního rozhodnutí; nákup až při hodinovém RSI < 40 nebo ≥ 1 % pod open dne, jinak ve 23 h; prodeje hned | zlepší načasování vstup do výběru LLM? |
| `seance_usa` | top 5 univerza podle kapitalizace jen 13–21 UTC, jinak hotovost | vzniká výnos v amerických hodinách? |

`h_reverze`, `h_breakout` a `seance_usa` záměrně hodně obchodují; poplatky je nejspíš sežerou, hrubý scénář ukáže,
jestli je v signálu něco před náklady. Nerovnováha knihy se počítá z top 100 úrovní (u BTC to je jen pár dolarů od
středu; ±1 % hloubky by vyžadovalo 5000 úrovní na coin a hodinu).

Benchmarky (stejné náklady): BTC HOLD, 50/50 BTC/ETH (měsíční rebalanc), top 10 univerza vážené kapitalizací
(týdně), **rovné váhy celého univerza** (týdně; odděluje výběr od alt-bety), hotovost.

Po maximální době držení (`jen_zpravy`, `jen_mr`) se coin v tom běhu nekupuje zpět. Coin, který vypadne z univerza,
se dál oceňuje, dokud ho varianta neprodá. Pár, který přestane být `TRADING`, se prodá za poslední platnou close
se slippage +200 bps (varování).

## Náklady

Poplatek 10 bps/strana; slippage podle 24h objemu páru (poslední denní svíčka): ≥ $500M 5 bps, $100–500M 10 bps,
$20–100M 25 bps, pod $20M 50 bps; stop +30 bps; delisting +200 bps. Nákup za `ask × (1 + slip)`, prodej
za `bid × (1 − slip)` (nejlepší nabídka a poptávka ve snímku; slippage podle objemu modeluje dopad nad špičku knihy).
Ocenění je ve středu (bid + ask) / 2, resp. za denní close. Spread nad 2 % dává varování. Každé portfolio má **tři paralelní ledgery** se stejnými rozhodnutími:
čistý (výše), **stresový** (40 bps, 2× slippage, spread jednou) a **hrubý** (0, plnění ve středu). Úrovně stopů se počítají z ceny snímku
(před náklady), takže jsou ve všech scénářích stejné.

## Struktura

```
engine/engine.py         CLI: run | notify | report | verify | validate | apply-redenomination | correction
engine/cpb/              canon (kanonický JSON), http (raw záznam), fetch (burzy), universe, check, features,
                         llm (validace + claude -p), portfolio (cílové váhy), ledger (účetnictví), pipeline (fáze),
                         runner (život běhu, pin), chain (hash chain), replay, corrections, analytics, report,
                         hourly (hodinový běh a 7 hodinových strategií), null (nulová rozdělení přes engine)
config/config.json       univerzum, data, náklady, pravidla, varianty, benchmarky, model
task/daily_prompt.md     denní prompt pro claude -p;  task/scores.schema.json  formát výstupu
dashboard/template.html  zdroj dashboardu (inline CSS/JS, grafy v SVG, bez CDN)
run_daily.sh             orchestrace pro systemd (flock, timeout 75 min, report, commit, push, ntfy)
run_hourly.sh            hodinový běh (flock, timeout 10 min, rychlý dashboard; necommituje)
deploy/                  install.sh (po krocích s potvrzením), systemd unit + timer, Caddyfile, fail2ban
tools/                   simulate, verify_chain, replay (post-hoc), export_dataset, analyze, random_null, pin, verify_manifest
tests/test_engine.py     invarianty (python -m unittest discover -s tests)
universe/D.json          zmrazené týdenní snímky univerza (včetně vyřazených a důvodu)
runs/D/                  raw/, raw_manifest.json, universe.json, check.json, features.json|md, llm/(prompt, scores,
                         transcript.jsonl, meta), validate.json, scores.json (validované), decisions.json,
                         snapshot.json, inputs.json (vše pro replay), fills.json, ledger.json, config.json, run.json
data/                    history/<COIN>.json (denní svíčky), state.json (portfolia), chain.jsonl,
                         hourly/state.json + hourly/history/<COIN>/<den>.json, null_cache.json
runs/D/hourly/HH/        hodinové záznamy (raw, inputs, features, decisions, snapshot, fills, ledger, run.json)
corrections/             opravy (redenominace, poznámky) – také v hash chainu
public/                  index.html + data.json (dashboard; nic jiného, hlídá test)
```

## Provoz

### Instalace (VPS, jednou)

```bash
bash deploy/install.sh            # všechny kroky, každý se ptá; nebo jednotlivě: audit user repo env claude venv pin web tls fail2ban ufw timer
```

Ručně je potřeba: **CoinGecko Demo API klíč** (zdarma, coingecko.com → Developer; bez něj se použije CoinPaprika),
**token Claude**: jako svůj uživatel `claude setup-token` (přihlášení předplatným, bez API klíče) a výsledek vložit
v kroku `env` jako `CLAUDE_CODE_OAUTH_TOKEN` (krok `claude` ho ověří testovacím voláním),
**deploy key** (krok `repo` vypíše veřejný klíč → GitHub → Settings → Deploy keys → *Allow write access*),
**heslo dashboardu** (krok `web`, bcrypt), **branch protection** pro `crypto-paper-bot` (Settings → Branches:
zakázat force-push a mazání), volitelně **ntfy** téma (dlouhé náhodné jméno – témata na ntfy.sh jsou veřejná).

HTTPS bez domény: krok `tls` nabídne (1) certifikát Let's Encrypt přímo na IP (krátkodobý profil, Caddy obnovuje),
(2) `38-109-11-51.sslip.io` s běžným LE certifikátem, (3) samopodepsaný jen nouzově (varování prohlížeče se na
mobilu snadno odklikne i při skutečném útoku). Caddy: jen statické soubory `index.html` a `data.json`, basic auth
(bcrypt), `Cache-Control: no-store`, `X-Content-Type-Options`, `Referrer-Policy: no-referrer`, `X-Robots-Tag: noindex`,
HSTS, CSP; fail2ban banuje po 5 neúspěšných přihlášeních. ufw: 22, 80, 443.
**Pozor:** porty publikované Dockerem (na tomto VPS n8n `0.0.0.0:5678`) ufw neblokuje.

### Ruční ovládání

```bash
./run_daily.sh                       # dnešní běh (idempotentní), report, commit, push, notifikace
./run_daily.sh --dry-run             # vše v dočasné kopii, včetně claude -p; nic se necommituje ani nepublikuje
./run_hourly.sh                      # hodinový běh pro právě uzavřenou hodinu + rychlý dashboard (bez commitu)
./run_hourly.sh --dry-run            # v dočasné kopii
.venv/bin/python engine/engine.py run --dry-run --allow-unpinned --no-llm   # rychlý test dat a plnění bez LLM
.venv/bin/python engine/engine.py verify           # MANIFEST vs připnutí + hash chain
.venv/bin/python tools/verify_chain.py             # + replay celého pokusu bajt po bajtu
systemctl list-timers 'cpb-*'; journalctl -u cpb-daily -n 200; journalctl -u cpb-hourly -n 50
```

Rozhodovací běh jde spustit jen pro dnešek (snímek pro plnění musí být živý). Zmeškané dny doplní další běh sám.

### Řešení chyb

| Příznak | Co dělat |
|---|---|
| `CHYBA … kontrola dat selhala` | Detail v `runs/D/failed-*/run.json` a `raw/`. Přechodný výpadek: spusť `./run_daily.sh` znovu (nebo počkej na 06:20). Nikdy neupravuj data ručně. |
| `… vypadá jako redenominace 1:N` | Ověř rešerší (oznámení burzy/projektu). Pak `engine.py apply-redenomination COIN RATIO --effective-date D --evidence URL [--new-symbol X]` (RATIO = nové jednotky za starou; hodnota portfolií se nemění, záznam jde do hash chainu) a spusť běh znovu. |
| `… pohyb > 60 % bez potvrzení` | Druhá burza pohyb nepotvrdila nebo coin nemá. Počkej/zopakuj; pokud je pohyb skutečný a žádná druhá burza coin nemá, řeš to novým release (výjimka v configu), ne ručně. |
| `připnutá verze nesedí` | Někdo změnil kód/config bez release, nebo release není připnutý: `tools/verify_manifest.py`, případně `tools/pin.py --install` po řádném release. |
| `krok LLM selhal` (UPOZORNĚNÍ) | `runs/D/llm/meta.json` + `transcript.jsonl`. Vyčerpaný limit/odhlášení: `claude setup-token` a aktualizovat env. LLM varianty ten den jen držely. |
| `lookahead` | Hodiny serveru: `timedatectl` (NTP). Běh se nezapsal jako úspěšný, nic se neobchodovalo. |
| push selhal | Deploy key / síť; další běh pushne i starší commity. Nikdy nepoužívej force-push. |
| Oprava dříve zapsaného běhu | `engine.py correction --date D --message "…" --refers-to runs/D/run.json` (nový záznam, nic se nepřepisuje). |

## Release (změna pravidel, enginu, promptu nebo modelu)

Během pokusu se pravidla neplánují měnit. Když je změna nutná, je to vždy nový release, nikdy úprava za běhu:

1. změna → `python -m unittest discover -s tests` (zelené) → `python tools/simulate.py` (dashboard v pořádku)
2. zápis do `CHANGELOG.md`
3. `python tools/pin.py vX.Y.Z` (VERSION + MANIFEST.sha256) → `python tools/verify_manifest.py --no-pin`
4. commit, `git tag vX.Y.Z`, push (tag i větev)
5. na VPS: `git pull --ff-only` jako `cpb`, pak `tools/pin.py --install`
6. dashboard i analýza oddělují segmenty podle verze a modelu (svislá čára v grafu; `tools/analyze.py --version`)

## Pravidla vyhodnocení (zapsáno předem, před prvními výsledky)

> **Potvrzující vyhodnocení se řídí [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md)** (zapečetěno 2026-10-02, tag
> `prereg-v1`): 3 primární hypotézy, pevné okno 2026-10-01 – 2026-11-30, vyhodnocení od 2026-12-15 skriptem
> `tools/prereg_eval.py`. Pravidla níže zůstávají jako orientační pohled dashboardu; bootstrap na 7/14 dnech je
> příliš benevolentní (viz pre-registrace, bod 7).
>
> **Forward test pomalého trendového filtru** (`trend_btc200`, BTC nad SMA200, jinak USDT) je pre-registrovaný
> v [`docs/PREREGISTRATION_TREND.md`](docs/PREREGISTRATION_TREND.md) (tag `prereg-trend-v1`); do bota přibude
> releasem po 2026-11-30, okno do 2028-11-30.

1. **Pohled/skóre má edge** jen tehdy, když 95% interval spolehlivosti rank IC (7denní výnos vs BTC,
   `tools/analyze.py`, blokový bootstrap přes dny) neobsahuje 0 **a** znaménko sedí v první i druhé polovině pokusu.
2. **Varianta má edge** jen tehdy, když je nad 95. percentilem vlastního nulového rozdělení (`tools/random_null.py`)
   **a** nad BTC HOLD **a** přežije stresové náklady (stresový výnos > 0 a stále nad BTC HOLD).
3. **LLM přidává hodnotu** jen tehdy, když LLM varianty soustavně porážejí `mech_momentum` se stejnými pravidly
   alokace (zejména `zaklad` vs `mech_momentum` a IC composite vs IC rel30).
4. Při 25 variantách čekej 1–2 „výhry“ náhodou; rozhoduje vzor (souhlasí pořadí `zaklad` vs `kontrarian` s IC
   composite? vede `mech_momentum`?), ne jednotlivý vítěz. Výsledky různých verzí a modelů se nemíchají.
5. **Hodinové signály** (nerovnováha knihy, agresivní nákupy, funding, z-skóre, 4h momentum, ΔOI, L/S) se hodnotí hlavně
   přes rank IC vs forward výnos vs BTC na 1/4/24 h (`tools/analyze.py --hourly`, průměr IC za den, bootstrap přes dny):
   edge jen když 95% CI neobsahuje 0, znaménko sedí v obou polovinách a je aspoň 20 dní dat. Hodinová varianta má
   edge podle stejných pravidel jako denní (bod 2), s vlastním nulovým rozdělením.
6. Post-hoc přehrání jiných pravidel (`tools/replay.py --config alt.json`) je in-sample a nikdy není důkaz.

## Nástroje pro vytěžení dat

```bash
python tools/export_dataset.py [--parquet]   # build/dataset/panel.csv: datum × coin × indikátory, skóre, forward výnosy 1/3/7/14 d abs i vs BTC
python tools/export_dataset.py --hourly      # panel_hourly.csv: hodina × coin × výnosy, z-skóre, objem, agresivní nákupy, kniha, funding, ΔOI, L/S, bid/ask + forward 1/4/24 h
python tools/analyze.py --hourly             # rank IC hodinových signálů
python tools/analyze.py [--version vX] [--model M]   # rank IC každého pohledu, CI, poloviny, kalibrace (Brier, reliability)
python tools/random_null.py [--paths 2000]   # percentil každé varianty ve vlastním nulovém rozdělení
python tools/replay.py --config alt.json     # POST-HOC, in-sample
python tools/simulate.py --days 60           # syntetický trh přes skutečný engine → build/sim/repo/public/index.html
python tools/prereg_eval.py                  # PRE-REGISTROVANÉ vyhodnocení (od 2026-12-15); --from/--to = jen test nástroje
python tools/power.py                        # statistická síla pravidel na syntetických datech
```

## Známá omezení

- **Plnění** je za nejlepší bid/ask ± slippage podle objemu, bez skutečné hloubky knihy (na špičce bývá u menších coinů jen pár set dolarů); proto stresový scénář.
- **Stopy** se vyhodnocují na 5min svíčkách (ne tick po ticku); v rámci svíčky se předpokládá nejhorší pořadí (nejdřív stop, pak nové maximum). Pětiminutovka, ve které proběhl nákup, se nepočítá.
- **Nulové rozdělení** (`engine/cpb/null.py`) běží skutečným účetním enginem: stejné dny, počet coinů, profil vah, počet výměn, pásmo rebalancu, max. držení, blokace a náklady (bid/ask snímku) jako varianta, jen náhodné coiny. Se skutečnými volbami varianty reprodukuje její výnos na cent (test). Jediná aproximace: stopy náhodných pozic se kontrolují na denních svíčkách (5min svíčky se ukládají jen pro coiny skutečné varianty `stop`). Dashboard ho přepočítává v pondělí (a denně první 4 týdny) s 1000 cestami, jinak ukazuje uložený výsledek s datem (`data/null_cache.json`).
- **Rank IC** používá close předchozího dne (informace, kterou mělo rozhodnutí), ne cenu plnění (~1 h později).
- Den se zastaví jen při chybě dat drženého coinu, BTC nebo ETH; ostatní coiny jdou do karantény.
- 18 korelovaných variant za 8–12 týdnů má malou statistickou sílu; hlavní síla je v IC (20 coinů × dny).
- Data v gitu rostou cca 2–4 MB/den (denní raw odpovědi a inputs ~0,3–1,5 MB, hodinové běhy ~1–3 MB; první hodinový běh ~3 MB kvůli 35 dnům historie).
- Hodinové varianty se na denní ose grafu oceňují prvním úspěšným hodinovým během dne (normálně 00:02 = stav k 00:00); chybí-li celý den hodinových běhů, v křivce je mezera.
- Nulové rozdělení hodinových variant kopíruje počet držených coinů, profil vah a počet výměn po každém hodinovém běhu (ne pravidla výstupu, která závisí na signálu); stopy na hodinových svíčkách.
- Futures data jsou jen informativní vstup (sentiment); obchoduje se výhradně spot.
- Kalibrace a IC mají smysl až po ~20 dnech s uzavřeným 7denním oknem.
