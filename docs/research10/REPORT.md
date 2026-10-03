# Výzkum 10 – doplňující se strategie a short strana (report)

Stav: **uzavřeno 2026-10-03, všechny tři části zamítnuty podle pre-registrovaných kritérií.**
Pre-registrace: [`PREREGISTRATION.md`](PREREGISTRATION.md) (commit `972fa28`, před prvním výpočtem).
Výsledky: [`results/`](results/).

## Otázka

Dají se najít dvě strategie, které se doplňují (když jedna ztrácí, druhá vydělává), případně jedna long
a druhá short? A dá se pak předpovědět, kterou z nich právě držet?

## Verdikt v jedné větě

Dvojice long-only strategií se doplňují jen tehdy, když jedna z nich jsou dluhopisy, a to je obyčejné
60/40. Short strana funguje jako pojistka: vydělá v roce 2008 nebo 2022, mezi tím stojí peníze. Přepínání
má obrovský teoretický strop, ale jednoduché signály z něj nezachytí prakticky nic.

| Část | Otázka | Výsledek |
|---|---|---|
| **A – dvojice rukávů** | Existují dvojice rukávů výzkumu 4, které se doplňují, a kolik z ideálního přepínání zachytí naivní přepínač? | **P_A zamítnuto (1 ze 4 kritérií).** Bez dluhopisů je nejnižší korelace dvojice v krizi +0,53, nic se nedoplňuje. Oracle přepínač má medián Sharpe 1,70 vs statická kombinace 0,54, naivní přepínače zachytí **medián 1–2 %** tohoto rozdílu. |
| **B – SPY + TSMOM long/short** | Zlepší 50 % SPY + 50 % trend long/short na 9 ETF Sharpe proti SPY a 60/40, přidá short něco? | **Ne (1 z 5).** Kontrola 2013–2019: Sharpe 0,93 vs SPY 1,13 a 60/40 1,34; verze bez shortu je lepší (1,02). TSMOM LS vydělal +11 % v roce 2008, v letech 2010–2019 většinou ztrácel. |
| **C – krypto long/short** | Zlepší trend se shortem přes perp Sharpe BTC/ETH proti držení a proti filtru do hotovosti? | **Ne (3 z 5).** P_C = HS_20: holdout ΔSharpe +0,21, 90% CI [−0,31; +0,75], DSR 0,02. Ve vývoji bylo všech 8 variant se shortem horších než filtr do hotovosti, na holdoutu naopak většina lepší, tj. znaménko efektu je nestabilní. |

## A – dvojice rukávů (jen data ≤ 2019)

**Výběr 2001–2012 (pre-registrováno).** 75 způsobilých rukávů, 2 609 dvojic z různých rodin, 409 s oběma
Sharpe ≥ 0,3. Nejlepší statické kombinace jsou low-vol + quality/value (Sharpe 0,80 vs SPY 0,23), korelace
dvojic +0,55 až +0,75. **Nejnižší korelace v krizi mezi kvalifikovanými dvojicemi je +0,53.** Long-only akciové
rukávy jsou pořád akcie a v krizi padají spolu.

Pozor: pravidlo „data po celé okno“ vyřadilo dluhopisové ETF (rukáv IEF/TLT začíná až 2003), což
pre-registrace nečekala (zmínila jen GLD a DBC). Definice krize (SPY > 15 % pod maximem) pokryla 1 801 z ~3 000
dní, takže „korelace v krizi“ je spíš korelace v medvědí fázi.

**Kolik dá přepínání (měsíčně, mezi dvěma rukávy, 409 dvojic):**

| | 10 % | medián | 90 % |
|---|---|---|---|
| Statická 50/50 (Sharpe, měsíčně) | 0,43 | 0,54 | 0,69 |
| Oracle (vždy ten lepší příští měsíc) | 1,23 | 1,70 | 2,19 |
| mom1 (vítěz minulého měsíce) | 0,37 | 0,55 | 0,78 |
| mom12 (vítěz za 12 měsíců) | 0,36 | 0,55 | 0,72 |
| Zachycení mom1 = (mom1 − statická) / (oracle − statická) | −11 % | **+1 %** | +16 % |
| Zachycení mom12 | −14 % | **+2 %** | +18 % |

Na úrovni celé knihovny: oracle top-1 ze 75 rukávů Sharpe 3,85, rovné váhy 0,28, mom1 top-1 0,02, mom12 top-1 0,19.
Strop je obrovský a naivní signály ho nedosáhnou ani zčásti. To odpovídá výzkumu 4: LightGBM tam měl rank IC
0,034 a ani to nestačilo.

**Kontrola 2013–2019, P_A = lowvol_252_top20 + quality_value_top20:** CAGR 9,8 %, Sharpe 0,82, propad 17,9 %
vs SPY 14,6 % / 1,13 / 19,3 % a 60/40 9,8 % / 1,35 / 10,5 %. Kritéria: Sharpe vs SPY ✗, vs 60/40 ✗,
propad vs SPY ✓, oba podúseky vs 60/40 ✗. **Zamítnuto.** DSR výběru (N = 2 609) 0,85.

**Post hoc (nepre-registrováno, jen popisně): okno výběru 2004–2012, aby se dostaly dluhopisy.** Top dvojice jsou
všechny low-vol akcie + IEF/TLT s korelací −0,30 až −0,36 (v krizi −0,31). Na kontrole 2013–2019 má
lowvol_63_top20 + IEF Sharpe 1,45 vs 60/40 1,35, propad 7 %, a formálně splní všechna 4 kritéria A. Je to ale
**obyčejné defenzivní portfolio akcie/dluhopisy** (CAGR 6,5 % vs 60/40 9,8 %), výběr je post hoc, low-vol
ansámbl na období 2020–2026 selhal (výzkum 1) a v roce 2022 padaly akcie i dluhopisy současně. Nejde o nový
zdroj výnosu, jen o diverzifikaci, kterou 60/40 už má.

## B – 50 % SPY + 50 % TSMOM na 9 ETF (jen data ≤ 2019)

| Kandidát (výběr 2004–2012) | TSMOM Sharpe | korelace s SPY | 50/50 Sharpe výběr | 50/50 Sharpe kontrola 2013–2019 |
|---|---|---|---|---|
| LS_L3_eq (**P_B**) | 0,49 | −0,38 | 0,66 | 0,93 |
| LS_L12_iv | 0,64 | −0,29 | 0,59 | 1,10 |
| LO_L3_eq (kontrola P_LO) | 1,10 | +0,38 | 0,61 | 1,02 |
| LO_L12_iv | 1,23 | +0,20 | 0,61 | 1,15 |
| SPY | – | – | 0,33 | 1,13 |
| 60/40 | – | – | 0,57 | 1,34 |

Short strana dělá přesně to, co se od ní čeká: korelace s SPY je záporná a v roce 2008 vydělal P_B +11,2 %
(SPY −36,8 %). V letech 2010–2019 ale ztrácel v šesti letech z deseti (2012 −8,4 %, 2016 −6,3 %). Ve všech
šesti párech LS/LO je na kontrole lepší verze bez shortu. Kritéria P_B: vs SPY ✗, vs 60/40 ✗, vs LO ✗,
zátěž nákladů ✗, propad ✓. **Zamítnuto.**

## C – krypto long/short (vývoj 2018-04 → 2021-12, holdout 2022-01 → 2026-08 jednou)

**Vývoj:** všech 8 variant mělo nižší Sharpe než filtr do hotovosti se stejným n (ΔSharpe −0,00 až −0,78).
P_C = HS_20 (Sharpe 1,46), tj. podle §2 pre-registrace trendový filtr v poloviční velikosti.

**Holdout (vault `runs/vault_r10`, hash `c0c4d945…`):**

| | Sharpe | CAGR | Max. propad |
|---|---|---|---|
| P_C = HS_20 | 0,55 | +9,2 % | 19,3 % |
| HOLD (BTC/ETH 50/50) | 0,34 | +3,3 % | 68,2 % |
| LC_20 (filtr do hotovosti) | 0,36 | +6,7 % | 38,5 % |

ΔSharpe vs HOLD +0,21, 90% CI [−0,31; +0,75] ✗; vs LC_20 +0,19 ✓; oba podúseky ✓ (+0,11, +0,27); zátěž
nákladů i fundingu ✓ (+0,08, +0,02); DSR 0,02 ✗. **Zamítnuto.**

Popisně (nevybírá se podle toho): na holdoutu byla většina variant se shortem lepší než filtr do hotovosti
(LS_50 0,90 vs LC_50 0,81; LS_200 0,87 vs 0,76), ve vývoji naopak všechny horší. Short v kryptu vydělává
v medvědím roce (2022, 2026) a ztrácí v býčím a bočním (2018–2021). Jestli pomůže, rozhoduje to, jaké roky
v období jsou, ne vlastnost strategie. Všechny varianty HS výrazně snížily propad (19–23 % vs 68 %), ale to
dělá z velké části už poloviční expozice.

## Co z toho plyne pro původní nápad

1. **„Doplňující se dvojice“ v akciích = akcie + státní dluhopisy.** Mezi long-only akciovými strategiemi nic
   nového není. Kombinace low-vol + dluhopisy vypadá dobře, ale je to 60/40 v jiném obalu.
2. **Short strana je pojistka, ne zdroj výnosu.** V trhu, který dlouhodobě roste, platí prémii. Vyplatí se jen
   v letech velkého propadu, které nejde dopředu poznat. To je přesně ten problém přepínání.
3. **ML přepínač nemá z čeho stavět.** Rozdíl mezi oracle a statickou kombinací je velký (Sharpe +1,2), ale
   jednoduché signály ho nezachytí ani z 2 % a LightGBM ve výzkumu 4 také ne. Pokud se má ML zkoušet znovu, musí
   mít **nové informace** (fundamenty, toky, pozicování), ne minulé výnosy strategií. Laťkou je statická
   kombinace, ne trh.
4. Na forward test nepostupuje nic. Kdo chce sledovat krypto short, může HS_50 nebo LS_50 přidat do paper bota
   jako popisnou variantu, ne jako kandidáta.
