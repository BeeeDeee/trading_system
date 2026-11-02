# Výzkum 19 – trend ensemble, RSI(2) a overlaye jako portfolio streamy: výsledek

2026-11-02 · pre-registrace v1.1 · Sharadar `sharadar_2026-09-25` (ETF) a Binance `binance_2026-10-03` (krypto)
Jeden plný běh (commit kódu `546f30652`, zámek `d65d61dc3`, otisk pravidel `51ff7d8c…`), 973 s, bez pádu a bez rerunu.
**Žádný čistý holdout:** veškerá historie už byla použita ve výzkumech 1–18 (viz pre-registrace §2). Je to
replikace literárních pravidel s pevnými parametry, vyhodnocená jednou na celé historii. Čísla jsou tedy
v-sample; rozhodující je forward test.

## Verdikt

**2 ze 7 primárních hypotéz splnily všechna kritéria: H1b (trend na koši ETF) a H2 (RSI(2) na ETF) → kandidáti na
forward test (§7.6).** H1a (krypto), H3a/b/c (overlaye) a H4 (kombinace) neprošly.

| Hypotéza | Kritérií | Verdikt | Co rozhodlo |
|---|---|---|---|
| H1a krypto Donchian ensemble, top 20 | 6 z 8 řádků | **ne** | Sharpe 0,90 < BTC hold 0,96; alfa vůči BTC t = 1,04 (práh 2) |
| H1b ETF Donchian ensemble | 6 z 6 | **ano** | – |
| H2 RSI(2) na ETF | 7 z 7 | **ano** | – |
| H3a turn-of-month | 0 ze 3 | ne | rozdíl TOM − mimo TOM není významný; overlay zhoršil propad |
| H3b vol targeting H4 | 1 z 2 | ne | téměř nikdy nezasáhl (průměr k = 0,999), Sharpe beze změny |
| H3c filtr momentum crash | 0 z 2 | ne | Sharpe +0,007, CI zahrnuje 0; propad stejný |
| H4 kombinace tří sleevů | 2 ze 4 | ne | rozdíl Sharpe vs 60/40 CI [−0,43; +0,63]; při nákladech ×2 0,69 < 0,73 |

**Co to neznamená.** H1b a H2 prošly mechanickými kritérii na datech, která jsou celá viděná. Prošly ale
rozumně: žádné kritérium nebylo těsné kromě DSR u H1a (které beztak neprošla). Očekávaný live Sharpe podle
haircutu 0,5× je H1b ≈ 0,38 a H2 ≈ 0,25. H1b je navíc nízkovýnosový proud (CAGR 3,6 %), ne náhrada 60/40.

## H1a – krypto Donchian ensemble, rotace top 20 (2019-01 → 2026-08)

| Kritérium (§7.1) | Hodnota | |
|---|---|---|
| 1. Dolní mez 90% CI Sharpe > 0 | Sharpe 0,90 [0,21; 1,54] | ✓ |
| 2a. Sharpe > BTC hold | 0,90 vs **0,96** | ✘ |
| 2b. Alfa vůči BTC > 0, t > 2 | alfa +2,4 % p. a., **t = 1,04**, beta 0,065 | ✘ |
| 3. Sharpe 2022-01 → 2026-08 > 0 | 0,43 | ✓ |
| 4a. Náklady 25 bps na stranu > 0 | 0,90 | ✓ |
| 4b. Náklady ×2 > 0 | 0,77 | ✓ |
| 5. Sharpe > p95 náhodného načasování | 0,90 vs 0,64 (medián nuly 0,42) | ✓ |
| 6. DSR (N = 7) > 0,95 | 0,952 (N = 34: 0,892; kumulativní krypto N = 106: 0,840) | ✓ (těsně) |

Pravidlo má statisticky kladný Sharpe a časování mu přidává proti náhodnému posunu, ale **není lepší než držet
BTC**. Portfolio je investované v průměru jen 7,5 % (vol cíl 25 % na aktivum, dělení 20), takže absolutní
výnos je malý: CAGR ≈ 5,9 % (odhad z ročních výnosů) při max. propadu 12,3 % (BTC hold: ≈ 46 %, propad 77 %). Sharpe je invariantní vůči
investovanosti, proto to samo o sobě kritéria neposouvá.

| Popisně | |
|---|---|
| Sharpe BTC-only replikace (H1a-BTC, sekundární) | **1,15** (BTC hold 0,96, ensemble top 20: 0,90) |
| BTC/ETH 50/50 měsíčně / rovné váhy top 20 | 0,96 / 0,52 |
| Roky (výnos) | 2019 +1,9 %, 2020 +15,3 %, 2021 +19,9 %, **2022 −6,3 %**, 2023 +10,9 %, 2024 +12,4 %, **2025 −2,6 %**, 2026 −1,2 % |
| Post-publikační úsek 2025-04 → 2026-08 | Sharpe **−0,13** |
| Obrat / náklady | 3,5× ročně / 0,9 % ročně |
| Podíl kladných výnosů z 5 % nejlepších dní | 41 % (obchody se vektorový engine nevede; dny jako náhrada) |

Robustnost (jen hlášení, z ní se nevybírá): bez pásma 0,91 · T-bill 0,86 · top 10 **1,07** · top 40 0,88 · bez
podmínky stáří 0,82 · plnění o den později 0,84. H1a-BTC: bez pásma 1,17, náklady ×2 1,10.

## H1b – ETF Donchian ensemble (2007-07 → 2026-09)

| Kritérium (§7.2) | Hodnota | |
|---|---|---|
| 1. Dolní mez CI Sharpe > 0 | Sharpe 0,77 [0,42; 1,10] | ✓ |
| 2. Sharpe 2010 → 2026 > 0 | 0,72 | ✓ |
| 3. Výnos 2008 a 2022 > 60/40 | 2008: **+2,9 %** vs −17,7 %; 2022: **+2,0 %** vs −16,6 % | ✓ |
| 4. Náklady ×3, Sharpe > 0 | 0,72 | ✓ |
| 5. Korelace se SPY < 0,5 | 0,31 | ✓ |

Popisně: CAGR 3,6 % (60/40: 7,9 %), max. propad 4,5 % (60/40: 32,2 %), investovanost 45 %. Sharpe vůči 60/40
(0,64), SPY (0,55) a rovným vahám 16 ETF (0,50) vyšel vyšší, ale kritérium to **není** (plán záměrně nesoudí H1b
proti 60/40 na Sharpe, výzkum 2 to už udělal). **Proti náhodnému cyklickému posunu signálu je náskok malý:**
Sharpe 0,77 vs p95 nuly 0,74 (medián 0,54). To není kritérium; je to varování, že část Sharpe dává vol sizing
a koš, ne načasování. Roky se záporným výnosem: jen 2015 (−1,4 %) a 2018 (−0,5 %).
DSR: N = 7: 0,988, N = 34: 0,955, kumulativní krypto N není relevantní.

Robustnost: bez pásma 0,78 · náklady ×2 0,74 · ×3 0,72 · vol cíl 15 % 0,76 · jen 9 ETF z výzkumu 2: 0,68.
Poznámka k datům: UUP má jeden neobchodovatelný den 2007-03-15; podle pravidla okna to vynuluje její signály
do cca března 2008 (L = 250).

## H2 – RSI(2) mean reversion na ETF (1999-11 → 2026-09)

| Kritérium (§7.3) | Hodnota | |
|---|---|---|
| 1. Průměrný čistý obchod, dolní mez CI > 0 | **+0,265 %** [+0,173; +0,355] %, 1764 obchodů | ✓ |
| 2. Po 2009: průměrný obchod a Sharpe > 0 | +0,259 %, Sharpe 0,51 | ✓ |
| 3. Průměrný obchod > p95 náhodných vstupů (c2) | 0,265 % vs 0,148 % (medián c2 0,067 %) | ✓ |
| 4. Sharpe > c1 (vždy long v režimu) nebo propad < 0,5× | Sharpe 0,49 vs 0,26; propad 16,5 % vs 49,4 % | ✓ |
| 5. Náklady ×2, průměrný obchod > 0 | +0,200 % | ✓ |
| 6. Plnění na open (r1), průměrný obchod > 0 | +0,258 % | ✓ |

Popisně: úspěšnost 69,5 %, průměrné držení 3,7 dne, čas v trhu 24 % (4 sloty), CAGR 5,5 %, propad 16,5 %.
Zisk z let 1999–2002 je jen **3,4 %** (varování z výzkumu 5 se neprojevilo). Roky záporné: 2002, 2008, 2011,
2018, 2020 (−2,5 %). H2-SPY (sekundární) vedle čísel z rešerše: 202 obchodů, úspěšnost **74,8 %** (rešerše 82 %),
držení 3,4 dne (3,7), čas v trhu 10,2 % (10 %), Sharpe **0,55** (0,62). Slabší než literatura, ale stejný řád.
DSR (nepatří mezi kritéria): N = 7: 0,898, N = 34: 0,731.

Robustnost (průměrný obchod / Sharpe): r1 plnění na open 0,26 % / 0,48 · r2 výstup RSI > 70: 0,35 % / 0,56 ·
r3 součet RSI2 za 2 dny: 0,28 % / 0,55 · r4 náklady ×2: 0,20 % / 0,36, ×3: 0,14 % / 0,23 · r5 bez SMA200: 0,33 % /
0,52, ale propad 29,8 % · r6 jen 4 indexová ETF: 0,39 % / 0,49 · r7 2 pozice: 0,23 % / 0,44. Edge přežívá
všechny varianty, citlivost na náklady je největší (×3 spolkne polovinu).

## H3 – overlaye

- **H3a (TOM):** SPY má v oknu TOM (19 % dní) o **+1,7 bps/den** vyšší výnos než mimo, 90% CI [−4,0; +7,2] bps →
  nevýznamné (2002–2026). Overlay na H2 (nevyužitá hotovost do SPY): Sharpe 0,504 vs 0,508, propad **23,3 %** vs 16,5 %.
- **H3b (vol targeting H4):** H4 má málokdy σ20 nad 10 %, průměr škálování k = 0,999, Sharpe 0,8520 vs 0,8522. Test
  je v praxi degenerovaný: cíl 10 % je nad skutečnou volatilitou H4 (propad celého H4 je 6 %).
- **H3c (crash filtr):** crash stav je aktivní 4,9 % dní u krypta a 8,8 % u ETF; Sharpe H4 0,859 vs 0,852,
  rozdíl CI [−0,016; +0,032], propad stejný (6,06 %).

## H4 – kombinované portfolio (2019-01 → 2026-08, váhy 1/3 reset měsíčně)

| Kritérium (§7.5) | Hodnota | |
|---|---|---|
| 1. Sharpe H4 > každý sleeve | H4 **0,85** vs A 0,50, B 0,72, C 0,64 | ✓ |
| 2. Dolní mez CI Sharpe(H4) − Sharpe(60/40) > 0 | rozdíl +0,13, CI [**−0,43**; +0,63] | ✘ (očekáváno) |
| 3. Propad H4 < 60/40 | 6,1 % vs 21,2 % | ✓ |
| 4. Náklady ×2: Sharpe H4 > 60/40 | 0,69 vs 0,73 | ✘ |

Sharpe 60/40 je v tomto období 0,73, SPY 0,79; CAGR H4 6,2 % vs 60/40 10,5 %. Diverzifikace funguje (H4 má vyšší
Sharpe než kterýkoli sleeve a o dvě třetiny nižší propad), ale na 7,7 letech není dost dat, aby se rozdíl vůči
60/40 dal prokázat, a při dvojnásobných nákladech se obrací.

Korelace sleevů (denní výnosy): A–B 0,19, A–C 0,10, B–C 0,42. Ve dnech, kdy SPY < −2 %: A–B 0,38, A–C 0,03,
**B–C 0,55**. V roce 2022: A–B −0,04, A–C 0,12, B–C 0,03. Příspěvek k riziku při rovných vahách: A 35 %, B 14 %,
**C 50 %**; rovné kapitálové váhy tedy nejsou rovné rizikové. Robustnost: inverzní volatilita 0,85 (stejné jako
rovné váhy), B+C bez krypta 0,75.

## Co z toho plyne

1. **Nejsilnější nový poznatek je negativní u krypta.** Ensemble s top 20 rotací nepřidal nic proti jednoduché
   BTC verzi (0,90 vs 1,15). Ensemble je pod BTC hold (0,96), BTC-only verze nad ním, ale ta je sekundární a kritéria
   se na ní nesoudí. Trend v kryptu snižuje propad (12 % vs 77 %), ne Sharpe. Navazuje na výzkum 9.
2. **H1b je skutečný, ale malý diverzifikátor.** Rozumná korelace se SPY, vítězí v obou krizích, citlivost na
   náklady je malá. Absolutní výnos je ale nízký (3,6 % CAGR) a přes náhodné časování jde jen těsně.
3. **H2 je robustní vůči všemu, co jsme zkusili** (náklady ×2, plnění na open, varianty vstupu a výstupu, koš
   i počet pozic), a post-publikační úsek 2009+ vychází stejně jako celek. Největší riziko forwardu: ×3 náklady
   vyprázdní polovinu edge a H2 vyžaduje denní obchodování 4 slotů.
4. **Overlaye nedělají nic.** TOM efekt na SPY po 2001 není významný; vol targeting a crash filtr dlouhopoziční
   proud téměř neovlivnily (v long-only trendovém proudu není co filtrovat).
5. **Kombinace pomáhá riziku, ne výnosu.** Za cenu ~4 p. b. CAGR (vs 60/40) dostaneme třetinový propad.

## Post hoc (není důkaz)

- BTC-only s vol sizingem (1,15) překonalo ensemble top 20. Top 10 (1,07) bylo lepší než top 20 a top 40. To je
  ex post pozorování, v tomto výzkumu se z něj nevybírá.
- Po 2022 H1a prakticky nevydělává (2022 −6 %, 2025 −3 %, 2026 −1 %); býčí roky 2020–2021 nesou většinu zisku.
- H4 by mohla vypadat lépe s váhami podle rizika (C nese polovinu rizika), ale to je nový pokus.

## Haircut (rešerše: McLean–Pontiff)

| | Backtest Sharpe | Očekávaný live (0,5×) |
|---|---|---|
| H1b | 0,77 | ≈ 0,38 |
| H2 | 0,49 | ≈ 0,25 |

## Návrh forward testu (výzkum 20, jen návrh – neimplementováno)

Samostatná pre-registrace podle výzkumu 18. Pevná pravidla H1b a H2 z tohoto výzkumu (parametry, náklady, koše),
T0 = první den po commitu, minimální délka 12 měsíců, laťka: Sharpe > 0 a průměrný obchod > 0 při nákladech ×1,5.
Sharadar už není; zdroj ETF cen je potřeba ověřit (Yahoo + Nasdaq ve večerním botu, rozdíly od Sharadaru jsou tam
změřené) – především total return s dividendami a přesnost close. H4 ani H1a do forwardu nejdou. H1b a H2 se
vyhodnocují zvlášť, ne jako kombinace.

## Omezení

Žádný holdout; survivorship v koši ETF (všechna žijí dodnes, QQQ je ex post vítěz); krypto sleeve má v roce 2019 jen
5–17 členů z 20 kvůli podmínce stáří; kalendář krypta je zarovnán na US dny (posun < 1 den); TOM a vol targeting
overlay se počítají na výnosových řadách se zjednodušenými náklady; pro rozptyl DSR se používá Sharpe přes všech 34
konfigurací, kriteriální N je 7 (výklad `N_meth` v plánu).
