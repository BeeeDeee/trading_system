# Pre-registrace forward testu – pomalý trendový filtr na BTC

**Zapečetěno:** 2026-10-03 (commit a tag `prereg-trend-v1`), dřív než existuje kód varianty a dřív než začne
jediný den testovacího okna. Varianta se do bota přidá releasem **až po 2026-11-30**, kdy končí
pre-registrované okno LLM pokusu (`docs/PREREGISTRATION.md`), aby se okna nemíchala.

## 1. Původ hypotézy (post hoc, otevřeně)

Hypotéza vznikla **po** otevření holdoutu výzkumu 9 (`strategy_backtester_2026_sep`, `docs/research9/REPORT.md`).
Tam pre-registrovaný kandidát `T_sma20` neprošel. Popisně ale všech 8 trendových filtrů na BTC + ETH snížilo
propad (30–48 % proti 68 % u držení 50/50) a pomalé filtry měly vyšší Sharpe (SMA50 0,81, SMA200 0,76,
30denní momentum 0,71 proti 0,34). Historie 2022–2026 je teď spotřebovaná. Proto jediný poctivý další krok
je **forward test na datech, která ještě neexistují**.

**Volba SMA200** je konvence (nejběžnější publikovaný trendový filtr), ne vítěz holdoutu. Na holdoutu výzkumu 9
měl nejvyšší Sharpe SMA50 (0,81), SMA200 měl nejmenší propad (29,7 %). Délka 200 se během testu nemění.

## 2. Otázka

**Sníží jednoduchý filtr „BTC jen nad 200denním průměrem, jinak USDT“ v reálném čase výrazně propad proti
BTC HOLD, aniž by zhoršil výnos na jednotku rizika?**

Nejde o hledání edge ve výnosu, ale o ověření **řízení rizika**: tvrzení, které jde po skončení testu
obhájit před dalšími lidmi.

## 3. Varianty

| id | Role | Pravidlo |
|---|---|---|
| `trend_btc200` | **primární** | 100 % BTC, když close BTC denní svíčky uzavřené v 00:00 UTC > SMA200 closů (včetně té svíčky), jinak 100 % USDT |
| `trend_5050_200` | sekundární, popisná | BTC 50 % + ETH 50 %, každý coin filtrován vlastní SMA200 stejně; vstupující coin dostane min(50 % NAV, volná hotovost). Navazuje přímo na rodinu T z výzkumu 9. |

- **Obchod jen při změně signálu.** Žádný denní rebalanc; držená pozice se jinak nemění.
- Rozhodnutí v denním běhu (06:00 Praha) ze svíčky uzavřené v 00:00 UTC, zamknutí vah před snímkem a plnění za
  bid/ask snímku: **stejný mechanismus, náklady a tři nákladové scénáře jako ostatní varianty bota**.
- Hotovost (USDT) nenese nic. Varianty nepoužívají LLM; selhání kroku LLM je neovlivní.
- SMA200 vyžaduje 200 platných denních closů za sebou. Chybí-li, signál je „hotovost“ (nikdy se nedopočítává).
- Benchmarky: `b_btc` (BTC HOLD) pro primární, `b_btc_eth` (50/50, měsíční rebalanc) pro sekundární.
  Oba už v botovi běží.

## 4. Okno a vyhodnocení

| | |
|---|---|
| Start | první úspěšný denní běh releasu s těmito variantami, **nejdříve 2026-12-01, nejpozději 2026-12-31**. Pozdější start = nová verze tohoto dokumentu. |
| Konec | **2028-11-30** (24 měsíců) |
| Vyhodnocení | od 2028-12-01, jedním skriptem, který bude napsán a commitnut **před** startem okna (§7) |
| Průběžně | dashboard ukazuje variantu živě, ale **nic se podle průběžného výsledku nerozhoduje**: žádné ukončení, prodloužení ani změna parametrů. |
| Platnost | maximální propad BTC HOLD v okně (denní closy) **≥ 25 %**. Jinak trh nenabídl situaci, kde by filtr mohl pomoct: okno se **jednou automaticky prodlouží o 12 měsíců** (do 2029-11-30) a pak se vyhodnotí bez ohledu na propad. |

## 5. Kritéria (primární varianta; nutné splnit všechna)

Denní výnosy z ocenění ledgeru v 00:00 UTC, Sharpe = průměr / sm. odchylka × √365, bez bezrizikové sazby.

1. **Propad:** max. propad `trend_btc200` ≤ **0,6 ×** max. propad BTC HOLD (čistý scénář),
2. **Výnos na riziko:** Sharpe `trend_btc200` ≥ Sharpe BTC HOLD (čistý scénář),
3. **Náklady:** kritéria 1 a 2 platí i ve **stresovém** scénáři,
4. **Provedení:** každá změna signálu se provedla v nejbližším úspěšném denním běhu (kontrola z `decisions.json`
   a `fills.json`), bez ručního zásahu do pozic.

**Statistická síla:** filtr mění signál jen několikrát ročně. Žádný test významnosti za 24 měsíců nemá smysl a
dokument ho nepředstírá. Splnění kritérií znamená „filtr v reálném čase udělal, co měl“ (případová studie
s předem daným pravidlem), ne statistický důkaz. Nesplnění v platném okně znamená zamítnutí.

Popisně se hlásí: CAGR, Calmar, počet změn signálu, náklady, čas v hotovosti, výsledky ve všech třech
scénářích, sekundární varianta proti 50/50, nejhorší měsíc, a jak dlouho trvalo zotavení z propadu.

## 6. Co znamená výsledek

| Výsledek | Další krok |
|---|---|
| Všechna kritéria splněna | Pravidlo lze prezentovat jako ověřený nástroj řízení rizika (s počtem změn signálu a délkou okna). Teprve potom má smysl uvažovat o skutečných penězích, malým kapitálem a s novou pre-registrací. |
| Kritérium 1 nebo 2 nesplněno | Zamítnuto. Popisný výsledek výzkumu 9 byl náhoda daná obdobím. |
| Jen kritérium 3 nesplněno | Filtr funguje jen bez realistických nákladů; zamítnuto. |
| Okno neplatné i po prodloužení | Vyhodnotí se podle §5 a výsledek se označí jako „bez medvědího trhu“. |

## 7. Implementace (release po 2026-11-30)

1. Nový typ varianty `trend` v `engine/cpb/portfolio.py` (SMA z denní historie, obchod jen při změně signálu)
   a dvě varianty v `config/config.json`.
2. `config.data.history_days` 150 → **230** (SMA200 potřebuje 200 closů plus rezervu).
3. Testy: SMA200 přepočtená ručně, obchod jen při změně signálu, chybějící close → hotovost, perturbace
   budoucích svíček nezmění rozhodnutí, **replay všech dosavadních běhů zůstane bajt po bajtu stejný**
   (existující varianty se nesmí změnit).
4. `tools/prereg_trend_eval.py` podle §4–5, vyzkoušený jen na `tools/simulate.py`, commitnutý a otisknutý
   (sha256 se doplní do sekce Změny) **před** startem okna.
5. Release `v1.1.0`: CHANGELOG, `tools/pin.py`, tag, push, na VPS `git pull` a `tools/pin.py --install`.

Do té doby se v botovi nic nemění: tento dokument není mezi připnutými soubory.

## Změny

(žádné)
