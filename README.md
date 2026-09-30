# Crypto paper-trading bot

Denní pokus na 20 kryptoměnách: Claude (`claude -p`) jednou denně udělá online rešerši a ohodnotí univerzum,
deterministický Python engine z toho obchoduje **18 variant pravidel** a **5 benchmarků**, každou s virtuálními
$10 000 (kotace USDT). Žádná burza, žádné API klíče k obchodování, žádné skutečné peníze. Plánovaná délka 8–12 týdnů.

Hlavní produkt je **point-in-time dataset** (ceny, indikátory, skóre LLM, rozhodnutí, plnění) a férové srovnání
proti kontrolám: `mech_momentum` (stejná pravidla bez LLM), náhodný výběr, kontrarián, BTC HODL a rovné váhy univerza.
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
| Web je nedůvěryhodný | Prompt to říká explicitně; engine validuje schéma, ořízne rozsahy, zahodí nevalidní coin (nikdy nedoplní), odstraní řídicí znaky, pustí jen http(s) URL. Dashboard vkládá text z LLM jen přes `textContent` a má CSP s hashem jediného skriptu. |
| Připnutý model | `config.llm.model = claude-opus-5-5`; ověřuje se proti `init` a `modelUsage` v transcriptu. Jiný model = skóre se nepoužije (LLM varianty drží). Změna modelu = nový release = nový segment. |

## Denní běh (vše UTC)

systemd timer `00:20` (po uzavření denní svíčky), `Persistent=true`, opakování `06:20` a `12:20` jen když den chybí.

```
A  fetch      univerzum (pondělí: přestavba), denní svíčky (Binance → OKX → Coinbase → Kraken), stav párů
   check      chybějící/duplicitní svíčky, nekladné ceny, high<low, zastaralá data, nulový objem, pohyb > 60 %
              bez potvrzení druhou burzou (±3 %), revize uložené historie, podezření na redenominaci;
              křížová kontrola close BTC + rotujícího coinu proti OKX (0,5 %, jeden nový fetch, pak selhání)
   settle     stopy na 5min svíčkách od posledního plnění, delisting (nucený prodej), ocenění k 00:00
   features   indikátory → features.json + features.md (tabulka pro LLM)
B  claude -p  rešerše + skóre → llm/scores.json + transcript; validace; při chybách 1 oprava (--resume)
C  lock       cílové váhy všech 23 portfolií → decisions.json (locked_at)
   snapshot   ticker Binance + čas serveru; křížová kontrola BTC + rotující coin vs CoinGecko/CoinPaprika (0,5 %)
   fill       stopy do času snímku, výstupy, rebalanc ve 3 nákladových scénářích
D  report     run.json + hash chain, dashboard, commit + push, notifikace
```

Když krok B selže (timeout, limit, nevalidní výstup, jiný model), varianty bez LLM a benchmarky běží normálně,
LLM varianty drží pozice (stopy a max. doba držení platí dál) a běh má status `warning`. Skóre se nikdy nedoplňuje.

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

Benchmarky (stejné náklady): BTC HODL, 50/50 BTC/ETH (měsíční rebalanc), top 10 univerza vážené kapitalizací
(týdně), **rovné váhy celého univerza** (týdně; odděluje výběr od alt-bety), hotovost.

Po maximální době držení (`jen_zpravy`, `jen_mr`) se coin v tom běhu nekupuje zpět. Coin, který vypadne z univerza,
se dál oceňuje, dokud ho varianta neprodá. Pár, který přestane být `TRADING`, se prodá za poslední platnou close
se slippage +200 bps (varování).

## Náklady

Poplatek 10 bps/strana; slippage podle 24h objemu páru (poslední denní svíčka): ≥ $500M 5 bps, $100–500M 10 bps,
$20–100M 25 bps, pod $20M 50 bps; stop +30 bps; delisting +200 bps. Nákup za `snapshot × (1 + slip)`, prodej
za `snapshot × (1 − slip)`. Každé portfolio má **tři paralelní ledgery** se stejnými rozhodnutími:
čistý (výše), **stresový** (40 bps, 2× slippage) a **hrubý** (0). Úrovně stopů se počítají z ceny snímku
(před náklady), takže jsou ve všech scénářích stejné.

## Struktura

```
engine/engine.py         CLI: run | notify | report | verify | validate | apply-redenomination | correction
engine/cpb/              canon (kanonický JSON), http (raw záznam), fetch (burzy), universe, check, features,
                         llm (validace + claude -p), portfolio (cílové váhy), ledger (účetnictví), pipeline (fáze),
                         runner (život běhu, pin), chain (hash chain), replay, corrections, analytics, report
config/config.json       univerzum, data, náklady, pravidla, varianty, benchmarky, model
task/daily_prompt.md     denní prompt pro claude -p;  task/scores.schema.json  formát výstupu
dashboard/template.html  zdroj dashboardu (inline CSS/JS, grafy v SVG, bez CDN)
run_daily.sh             orchestrace pro systemd (flock, timeout 75 min, report, commit, push, ntfy)
deploy/                  install.sh (po krocích s potvrzením), systemd unit + timer, Caddyfile, fail2ban
tools/                   simulate, verify_chain, replay (post-hoc), export_dataset, analyze, random_null, pin, verify_manifest
tests/test_engine.py     invarianty (python -m unittest discover -s tests)
universe/D.json          zmrazené týdenní snímky univerza (včetně vyřazených a důvodu)
runs/D/                  raw/, raw_manifest.json, universe.json, check.json, features.json|md, llm/(prompt, scores,
                         transcript.jsonl, meta), validate.json, scores.json (validované), decisions.json,
                         snapshot.json, inputs.json (vše pro replay), fills.json, ledger.json, config.json, run.json
data/                    history/<COIN>.json (denní svíčky), state.json (portfolia), chain.jsonl
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
.venv/bin/python engine/engine.py run --dry-run --allow-unpinned --no-llm   # rychlý test dat a plnění bez LLM
.venv/bin/python engine/engine.py verify           # MANIFEST vs připnutí + hash chain
.venv/bin/python tools/verify_chain.py             # + replay celého pokusu bajt po bajtu
systemctl list-timers cpb-daily.timer; journalctl -u cpb-daily -n 200
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

1. **Pohled/skóre má edge** jen tehdy, když 95% interval spolehlivosti rank IC (7denní výnos vs BTC,
   `tools/analyze.py`, blokový bootstrap přes dny) neobsahuje 0 **a** znaménko sedí v první i druhé polovině pokusu.
2. **Varianta má edge** jen tehdy, když je nad 95. percentilem vlastního nulového rozdělení (`tools/random_null.py`)
   **a** nad BTC HODL **a** přežije stresové náklady (stresový výnos > 0 a stále nad BTC HODL).
3. **LLM přidává hodnotu** jen tehdy, když LLM varianty soustavně porážejí `mech_momentum` se stejnými pravidly
   alokace (zejména `zaklad` vs `mech_momentum` a IC composite vs IC rel30).
4. Při ~18 variantách čekej 1 „výhru“ náhodou; rozhoduje vzor (souhlasí pořadí `zaklad` vs `kontrarian` s IC
   composite? vede `mech_momentum`?), ne jednotlivý vítěz. Výsledky různých verzí a modelů se nemíchají.
5. Post-hoc přehrání jiných pravidel (`tools/replay.py --config alt.json`) je in-sample a nikdy není důkaz.

## Nástroje pro vytěžení dat

```bash
python tools/export_dataset.py [--parquet]   # build/dataset/panel.csv: datum × coin × indikátory, skóre, forward výnosy 1/3/7/14 d abs i vs BTC
python tools/analyze.py [--version vX] [--model M]   # rank IC každého pohledu, CI, poloviny, kalibrace (Brier, reliability)
python tools/random_null.py [--paths 2000]   # percentil každé varianty ve vlastním nulovém rozdělení
python tools/replay.py --config alt.json     # POST-HOC, in-sample
python tools/simulate.py --days 60           # syntetický trh přes skutečný engine → build/sim/repo/public/index.html
```

## Známá omezení

- **Plnění** je za snímek posledního obchodu (±slippage), bez modelu hloubky knihy; u malých coinů to může být optimistické (proto stresový scénář).
- **Stopy** se vyhodnocují na 5min svíčkách (ne tick po ticku); v rámci svíčky se předpokládá nejhorší pořadí (nejdřív stop, pak nové maximum). Pětiminutovka, ve které proběhl nákup, se nepočítá.
- **Nulové rozdělení** (dashboard i `random_null.py`) je aproximace: stejný počet coinů, expozice a počet výměn, rovné váhy, výnosy close-to-close mezi běhy, náklady paušálně. Není to plná simulace pásma rebalancu.
- **Rank IC** používá close předchozího dne (informace, kterou mělo rozhodnutí), ne cenu plnění (~1 h později).
- Při chybě dat jednoho coinu se zastaví celý den (záměr: nikdy neobchodovat na špatných datech).
- 18 korelovaných variant za 8–12 týdnů má malou statistickou sílu; hlavní síla je v IC (20 coinů × dny).
- Data v gitu rostou cca 0,3–1,5 MB/den (raw odpovědi, inputs.json).
- Kalibrace a IC mají smysl až po ~20 dnech s uzavřeným 7denním oknem.
