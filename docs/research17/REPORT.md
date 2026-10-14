# Výzkum 17 – LEAD na altcoinech (BTC vede, altcoin zaostává): výsledek

2026-10-14 · pre-registrace v1.0 · 15m data Binance (snapshot `binance_2026-10-03`) · test 2020-05 → 2026-09
otevřen jednou (commit kódu `c54c782f` a oprava formulace, otisk metodiky `e0697059…`), bez pádu a bez rerunu.

## Verdikt

**Splněno 6 ze 6 kritérií → kandidát na forward paper test (§10).** Pevné pravidlo z výzkumu 16 (BTC za 1 h
≥ 3σ nahoru a altcoin zaostává → long altcoinu na 4 h) vydělalo na 105 altcoinech z měsíčního top 10 podle objemu:
**Sharpe 0,80, CAGR +7,4 %, max. propad 14,8 %**, 2260 obchodů v 946 událostech BTC, expozice 24 % dnů. Je to
kvalifikace, ne potvrzení: marže jsou tenké (viz níže) a potvrdit to může jen forward.

| Kritérium (§8) | Hodnota | |
|---|---|---|
| 1. Sharpe > 0 a 90% CI průměrného obchodu > 0 (bootstrap po událostech) | 0,80; +0,21 % [+0,06; +0,36] % | ✓ |
| 2. B_RND_EVT, p < 0,05 | p = 0,001 (medián nuly −0,65, p95 −0,06) | ✓ |
| 3. Náklady ×2 (20 bps na stranu) | Sharpe **0,06** | ✓ (těsně) |
| 4. Plnění o svíčku později | 0,61 | ✓ |
| 5. Podúseky 2020-05..2022-12 / 2023-01..2026-09 | 1,39 / **0,26** | ✓ |
| 6. Šíře: bez nejlepšího (XRP) / podíl kladných altcoinů (≥ 10 obchodů) | 0,73 / 70 % z 47 | ✓ |

DSR: N = 1 → 0,98; kumulativně N = 47 (proxy rozptylu z výzkumu 16) → ≈ 0. B_HOLD (rovné váhy členů, bez
nákladů) Sharpe 0,38, CAGR −4,5 %, propad 95 %. Referenčně ETH (perp, váha 1): Sharpe 0,94, 134 obchodů.

## Co z toho plyne

**1. Efekt je na altcoinech stejný jako na ETH a nula ho jasně odlišuje od náhodného načasování.** Náhodné
přesunutí událostí v rámci měsíce má medián Sharpe −0,65 (altcoiny v testu ztrácely), skutečný výběr okamžiku
+0,80. Kladný je průměr obchodu (+0,21 %, dolní mez CI +0,06 %) i u 70 % altcoinů s ≥ 10 obchody. Po vyřazení
XRP (největší příspěvek) je Sharpe 0,73, edge tedy nestojí na jedné minci.

**2. Zdroj je lead-lag, ne pouhé momentum, ale jen v úzkém nastavení.** Bez podmínky zaostávání (B_NOLAG) je
Sharpe 0,38 a propad 48 %; zaostávání tedy přidává. Pravidlo je ale **citlivé na prahy**: s K = 2 (6032 obchodů)
je Sharpe −0,14 a **short verze (LS) −0,57**. Edge je jen long po velkých skocích BTC (≥ 3σ). Je to přesně ten
typ závislosti na parametru, před kterým varuje §2 pre-registrace, a podmínka, že K = 3 platilo už na ETH ve
výzkumu 16, nemění, že hranice mezi „funguje“ a „nefunguje“ je blízko.

**3. Náklady jsou limit.** Při 20 bps na stranu je Sharpe 0,06 (CAGR +0,1 %), při 7,5 bps 0,98. Reálné vstupy
do altcoinů po skoku BTC (rychlý trh, tenčí kniha) mohou být blíž horní hodnotě. Hlavní riziko forwardu.

**4. Stabilita v čase je slabší, než vypadá.** Sharpe po letech: 2020 2,1, 2021 3,2, **2022 −1,2**, 2023 0,6,
2024 1,1, **2025 −1,6**, **2026 −0,5**. Druhý podúsek (2023–2026) má Sharpe jen 0,26. Většina výsledku pochází
z býčího 2020–21. Funding nehraje roli (bez fundingu 0,83), perp BTC jako vůdce 0,62.

**5. Obchody nejsou nezávislé.** 2260 obchodů je 946 událostí (průměr 2,4 altcoinu na událost, maximum 10).
Interval je proto po událostech a je užší než u ETH samotného (81 obchodů ve výzkumu 16) jen díky většímu počtu
událostí, ne díky počtu obchodů.

## Robustnost (Sharpe, jen hlášení)

| Varianta | Sharpe | Obchodů | CAGR | Propad |
|---|---|---|---|---|
| základ (K = 3, long, 10 bps) | 0,80 | 2260 | +7,4 % | 14,8 % |
| náklady 7,5 bps / 20 bps | 0,98 / 0,06 | 2260 | +9,3 % / +0,1 % | 12,4 % / 33,9 % |
| plnění o svíčku později | 0,61 | 2260 | +5,1 % | 15,4 % |
| bez fundingu | 0,83 | 2260 | +7,8 % | 15,0 % |
| K = 2 | −0,14 | 6032 | −3,2 % | 39,0 % |
| long i short (LS) | −0,57 | 3400 | −7,6 % | 45,6 % |
| bez podmínky zaostávání | 0,38 | 8304 | +6,6 % | 47,7 % |
| BTC perp jako vůdce | 0,62 | 2313 | +5,7 % | 15,6 % |

Nejlepší příspěvky: XRP (254 obchodů), DOT, XLM, ADA, LINK, LTC. Nejhorší: TRB, WLD, HYPE, NEAR.

## Co dál (§10)

**Forward paper test** od 2026-10-01, altcoiny z měsíčního top 10 + ETH, stejná pravidla (K = 3, long, 4 h,
10 bps), vyhodnocení na datech do **2027-03-31**: Sharpe > 0 a průměrný obchod > 0 při nákladech ×1. Kritérium je
pevné od pre-registrace. Při ~ 400 obchodech ročně (2260 za 6,4 roku) vyjde za 6 měsíců řádově 150–200 obchodů
v ~ 70 událostech, což stačí jen na hrubé ano/ne, ne na interval. Doporučené doplnění před startem: změřit
skutečné náklady vstupu po skoku BTC (skluz) na reálných datech, protože rozhoduje mezi Sharpe 0,06 a 0,98.
