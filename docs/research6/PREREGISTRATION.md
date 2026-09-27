# Výzkum 6 – STR-TF jen při vysokém VIX (pre-registrace)

Verze 1.0 · 2026-09-27 · Stav: **pre-registrace, žádný výsledek zatím spočten**.
Navazuje na výzkum 5 (stejný panel, signál, simulátor, náklady a testy).
Změny po commitu tohoto dokumentu = nový pokus v registru a řádek v §8.

## 1. Otázka

Nagel (2012): výnos z krátkodobého reversalu je odměna za poskytnutí likvidity a je nejvyšší, když
je likvidita drahá (vysoká volatilita trhu). **Vydělává STR-TF po nákladech, pokud se nové pozice
otevírají jen ve dnech s vysokým VIX, a přidává toto podmínění hodnotu oproti nepodmíněné verzi?**

## 2. Poctivé omezení předem

- **Hypotéza je post hoc.** Vznikla po výsledcích výzkumu 5, kde zisk táhl rok 2000 a nejlikvidnější
  tituly. Celá historie 1999–2026 je viděná a čistý holdout neexistuje.
- Historický test proto hypotézu **nemůže potvrdit**. Může ji jen zamítnout, nebo ji kvalifikovat
  na forward test (§7).
- Období 1999–2002 (před decimalizací, dot-com) se hodnotí zvlášť a kritéria ho nepoužívají, jinak
  by výsledek zopakoval nález výzkumu 5.

## 3. Pravidlo

- Signál, univerzum, exekuce, sizing, náklady a výstupy jsou identické s výzkumem 5.
- **Brána VIX:** nové pozice se otevírají jen z rozhodnutí po close dne *t*, kdy close `^VIX`
  (Sharadar `funds`) dne *t* > **25**. Otevřené pozice se drží a uzavírají normálně.
  VIX close je znám před open *t*+1, kdy se nakupuje.
- **Primární konfigurace = default zadání STR-TF** (`entry_z` 2,0, `lookback_ret` 3, `trend_sma` 200,
  `exit_sma` 5, `max_hold` 5, `max_positions` 10, `min_adv` 20M). Konfigurace vybraná ve výzkumu 5
  je vyladěná na historii, proto je jen sekundární.
- Citlivost (hlásí se, nevybírá se): práh 20 a 30; relativní práh = VIX nad 80. percentilem
  vlastních posledních 252 dní (point in time); sekundární konfigurace výzkumu 5 s prahem 25.

## 4. Období

Jeden běh 1999-01-04 → 2026-09-25 (rozhodnutí od 1999-01-04). Hodnocení na **2003–2026**
a podúsecích 2003–2014, 2015–2019, 2020–2026. 1999–2002 se jen hlásí.

## 5. Kritéria (primární konfigurace, VIX > 25, 2003–2026); nutné splnit **všechna**

1. průměrný hrubý výnos na obchod ≥ 2 × průměrný náklad na obchod,
2. CAGR při nákladech ×2 > 0,
3. průměrný čistý výnos na obchod > 0 v každém podúseku 2003–2014, 2015–2019, 2020–2026,
4. podmínění pomáhá: čistý výnos na obchod s bránou > bez brány (stejná konfigurace, 2003–2026),
5. Sharpe > 95. percentil náhodného benchmarku (1000 simulací, stejné dny vstupu a délky držení,
   seed 0),
6. Deflated Sharpe ≥ 0,95, kde N = všechny pokusy výzkumu 5 + výzkumu 6.

Sharpe = průměr / sm. odchylka denních čistých výnosů × √252 (hotovost bez úroku, jako ve výzkumu 5).

## 6. Robustnost (jen hlášení)

Náklady ×3, vstup open *t*+2, prahy 20/30/relativní, sekundární konfigurace, výsledky po letech.

## 7. Co znamená výsledek

- Nesplní-li se cokoli z §5: hypotéza zamítnuta, konec.
- Splní-li se vše: kandidát na forward (paper) test, minimálně 6 měsíců. Výše VIX > 25 je zhruba
  20 % dní, takže 3 měsíce by nemusely obsahovat žádný signál. Nutná je obnova dat (Sharadar
  končí 2026-09-27). Historický výsledek sám o sobě není důkaz.

## 8. Log rozhodnutí

| Datum | Rozhodnutí | Důvod |
|---|---|---|
| 2026-09-27 | Pre-registrace v1.0 | – |
| 2026-09-27 | Vyhodnoceno: 5 ze 6 kritérií, C6 (DSR 0,0006) zamítá. Forward test se podle §7 nespouští. | `REPORT.md` |
