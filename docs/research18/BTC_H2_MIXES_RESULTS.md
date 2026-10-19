# BTC buy-and-hold + H2 mixy — výsledky

Historický běh na snapshotu `binance_2026-10-03`, období 2020-05-01 až 2026-09-30. Jde o post hoc popisný výsledek na stejných datech, ze kterých už byla vybrána H2; nelze jej považovat za nezávislé potvrzení. Plná roční data jsou v ignorovaném výstupu `runs/r18_btc_h2_mixes_history.json`.

Denní mix drží cílový poměr BTC/H2. Denní výnos H2 zachovává původní 10% váhu na jeden obchod, náklady 10 bps na stranu a funding. Výnos BTC je spot buy-and-hold. Mix nepřidává náklady na převážení mezi BTC a H2, takže výsledky jsou mírně optimistické. Žádná páka se nepoužívá.

| BTC / H2 | CAGR | Celkem | Max. drawdown | Sharpe | Nejhorší kalendářní rok |
|---:|---:|---:|---:|---:|---:|
| BTC B&H | 42,45% | +870,1% | 76,63% | 0,905 | −64,21% |
| H2 samotná | 10,29% | +87,5% | 8,34% | 1,229 | −1,89% |
| 90 / 10 | 40,83% | +801,2% | 72,43% | 0,922 | −59,58% |
| 80 / 20 | 38,78% | +720,4% | 67,63% | 0,942 | −54,54% |
| 70 / 30 | 36,34% | +631,9% | 62,17% | 0,967 | −49,10% |
| 60 / 40 | 33,52% | +540,0% | 56,01% | 0,999 | −43,24% |
| 50 / 50 | 30,34% | +448,4% | 49,07% | 1,042 | −36,98% |
| 40 / 60 | 26,85% | +360,6% | 41,31% | 1,099 | −30,31% |
| 25 / 75 | 21,07% | +241,5% | 28,03% | 1,230 | −19,59% |
| 20 / 80 | 19,02% | +205,9% | 23,15% | 1,288 | −15,83% |

## Čtení výsledku

Více H2 v tomto konkrétním mixu snižovalo historický drawdown a zvyšovalo Sharpe, ale také výrazně snižovalo CAGR. Nejvyšší Sharpe z testovaných mixů měl 20/80; nepřekonal však H2 samotnou v CAGR a jeho drawdown byl vyšší než u samotné H2. Mix 50/50 snížil maximální historický drawdown z 76,6% na 49,1%, ale CAGR klesl z 42,5% na 30,3%. Žádný testovaný mix současně nepřekonal BTC v CAGR a H2 v drawdownu.

Historický BTC měl v roce 2022 drawdown přes 64% a kalendářní výnos −64,2%; zlepšení v mixu proto neznamená ochranu kapitálu v každém propadu. Pro další praktické sledování dávají jako kompromis největší smysl 50/50 a 25/75, ale váhy se mají ověřovat dál na nových forward datech, nikoli vybírat podle tohoto historického žebříčku.
