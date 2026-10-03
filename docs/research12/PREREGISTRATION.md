# Výzkum 12 – forward test ML výběru akcií s nižším obratem (pre-registrace)

Verze 1.0 · 2026-10-03 · Stav: **pre-registrace, forward test nezačal** (čeká na obnovu předplatného Sharadaru).
Navazuje na výzkum 11 (model, příznaky, label, nákladový model). Změny po commitu = řádek v §10; změna pravidel
po prvním zamčeném rozhodnutí = konec tohoto testu a nový výzkum.

## 1. Otázka

Výzkum 11 zamítl ML výběr 50 akcií hlavně kvůli obratu 15–18× ročně (náklady 1,3–1,7 % p. a.) při slabém
rank IC ~0,025. **Překoná stejný model s pravidlem pro nízký obrat (hystereze) na datech, která v době
rozhodnutí ještě neexistovala, SPY, rovné váhy LIQ1000 a stejné pravidlo nad modelem jen z cen?**

## 2. Poctivé omezení předem

- Pravidlo hystereze a jeho parametry (50 / 200) jsou zvoleny **po** výzkumu 11, tedy se znalostí, že problém
  byl obrat. Proto se ověřují jen forward, ne na historii. Historický běh v §8 je **jen popisný** (kontrola
  implementace a obratu); jeho výsledek nic nemění a forward test se spustí v každém případě.
- **24 měsíců nic nedokáže.** Směrodatná chyba rozdílu Sharpe za 2 roky je ~0,7. Test umí odhalit
  zřetelné selhání (§7, předčasné ukončení) nebo říct „zatím neselhalo, pokračovat“. Splnění kritérií
  = pokračovat dalších 24 měsíců, ne důkaz.
- Forward data mají jednu výhodu proti backtestu: snapshot stažený v den rozhodnutí neobsahuje pozdější
  opravy fundamentů ani 13F. Je to skutečný point-in-time.

## 3. Modely (beze změny proti výzkumu 11)

- **`M_ALL_H`** (primární): příznaky P + F + I + H výzkumu 11, LightGBM `GBM_PARAMS`, seed 0.
- **`M_P_H`** (kontrola): jen cenové příznaky P.
- Trénink: všechny realizované labely od rozhodnutí 2003-01 do posledního realizovaného před dnem
  rozhodnutí. **Přetrénování v lednu** každého roku a při prvním forward rozhodnutí. Mezi tím se používá
  uložený model (soubor + sha256 v zámku).

## 4. Pravidlo portfolia (hystereze)

Každý měsíc po close prvního obchodního dne, univerzum LIQ1000 k tomu dni, akcie seřazené podle predikce
(sestupně, shoda → nižší sloupec):

1. **Ponechat** držené akcie, které jsou v univerzu a mají pořadí ≤ **200**.
2. **Doplnit** na **50** akcií nejvýše seřazenými nedrženými akciemi.
3. Rovné váhy 2 % všem 50 (měsíční vyrovnání vah zůstává).

První forward rozhodnutí začíná z prázdného portfolia (top 50). Plnění na openu následujícího obchodního
dne, náklady modelem výzkumu 4 (stupně podle likvidity). Benchmarky: **SPY** (5 bps) a **EW LIQ1000**
(rovné váhy celého univerza, měsíčně, stejné náklady), obojí od stejného prvního plnění.

## 5. Provoz každý měsíc (vše zapisuje skript, nic ručně)

1. Po close prvního obchodního dne měsíce *t* stáhnout plný snapshot Sharadaru (bez tabulky `holdings`,
   model ji nepotřebuje) a FRED DTB3, sha256 všech souborů do manifestu.
2. Pipeline `raw_to_parquet → build_bars → build_universe → build_panel → r4_build_panel`, příznaky
   výzkumu 4 a 11, predikce, hystereze.
3. **Zámek** `docs/research12/forward/<YYYY-MM>.json`: den rozhodnutí, cílové váhy obou modelů, předchozí
   držby, sha256 manifestu snapshotu, sha256 modelu, commit kódu, čas zámku (UTC). Commit + push na GitHub
   **před openem *t*+1** (13:30 UTC; časové razítko pushe je nezávislý důkaz). Zámek po openu = měsíc se
   nepočítá (portfolio drží předchozí váhy), zapíše se jako `missed`.
4. Výkon se počítá zpětně z každého dalšího snapshotu (plnění na openu *t*+1, vektorový engine).
5. Selhání stažení nebo pipeline = žádné nové rozhodnutí, drží se předchozí váhy, záznam `missed`.

Disk: po každém měsíci se smaže raw a odvozená data předchozího forward snapshotu (snapshot
`sharadar_2026-09-25` výzkumů 1–11 zůstává). Uchovávají se jen zámky, manifesty a modely.

## 6. Období

Rozhodnutí od prvního měsíce po obnově předplatného (nejdřív **2026-11-02**) po **24 rozhodnutí**.
Hodnocení po 12 a po 24 rozhodnutích; měsíčně jen popisný report.

## 7. Kritéria

**Předčasné ukončení (zamítnutí), kontrola od 12. rozhodnutí měsíčně:** kumulativní aktivní výnos
`M_ALL_H` − SPY < −15 %.

**Po 24 rozhodnutích (všechna, jinak zamítnuto):**
1. Sharpe `M_ALL_H` > Sharpe SPY,
2. Sharpe `M_ALL_H` > Sharpe EW LIQ1000,
3. Sharpe `M_ALL_H` > Sharpe `M_P_H` (fundamenty a toky přidávají),
4. obrat `M_ALL_H` ≤ 5× ročně (pravidlo dělá, co má),
5. kumulativní aktivní výnos vs SPY > 0.

Splnění = pokračovat dalších 24 měsíců se stejnými pravidly (§2).

## 8. Popisný historický běh (není kritérium)

Na snapshotu `sharadar_2026-09-25` (celá historie je spotřebovaná, výzkumy 1–11): walk-forward predikce
výzkumu 11 pro 2010-01 → 2026-08, porovnání top 50 (výzkum 11) a hystereze 50/200 pro oba modely: obrat,
náklady, CAGR, Sharpe. Slouží ke kontrole implementace a odhadu obratu.

## 9. Postup

1. Commit této pre-registrace.
2. `qlab.research12` (hystereze, forward rozhodnutí, zámek, vyhodnocení) + testy.
3. Popisný historický běh `scripts/r12_history.py`.
4. Suchý běh forward rozhodnutí na posledním měsíci snapshotu `sharadar_2026-09-25` (2026-09-01), bez zámku.
5. Po obnově předplatného: měsíční běh `scripts/r12_monthly.sh` (systemd timer).

## 10. Log změn

| Datum | Změna | Důvod |
|---|---|---|
| 2026-10-03 | Verze 1.0 | – |
| 2026-10-03 | Popisný historický běh (§8, `results/history.json`): hystereze snížila obrat M_ALL z 16,4× na 9,1× ročně (náklady 1,45 → 0,81 % p. a.), mění se ~18 z 50 akcií měsíčně; 2010–2026 Sharpe 0,59 vs SPY 0,86, EW 0,65. Pravidla se nemění. Forward test čeká na rozhodnutí o obnově předplatného. | informace pro rozhodnutí, ne kritérium |
