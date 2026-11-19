# Osnova posteru – MSM úloha 2

Kostra podle obecných požadavků (`ulohy/README.md`). Na poster patří obrázky, tabulky a odrážky, ne dlouhé odstavce. Grafy a tabulky bere z `output/`.

## 1. Název
Testy o poloze na datech o spánku a telefonu před spaním („Liší se chronotypy délkou spánku a únavou?“)

## 2. Autor a datum
Vít Novotný (+ TODO), TODO datum, KMA/MSM 2026/2027

## 3. Cíl a popis
- Otázka: liší se chronotypy (Morning Lark, Intermediate, Night Owl) polohou proměnných `bedtime_phone_minutes`, `sleep_latency_min`, `total_sleep_hours`, `next_day_fatigue_score`?
- Data: Kaggle (Sleep Debt & Screen Time), n = 8 500, 3 skupiny po 3 880 / 2 165 / 2 455, počet skupin a pozorování ve skupinách, proměnné a jednotky → `1_data_description.csv`, `1_scatter_matrix.png`
- Hladina významnosti α = 0,05

## 4. Použité technologie
Python 3.13 · numpy · pandas · scipy · statsmodels · matplotlib

## 5. Výstupy a závěry pro data

**Jednovýběrové testy**
- H0 / H1 pro každou proměnnou, zdůvodnění μ₀ a m₀
- tabulka p-hodnot t-testu vs. Wilcoxona → `2_one_sample_tests.csv`
- `2_qq_plots.png`, `2_means_vs_reference.png`, `2_histograms.png` (tvar rozdělení, referenční hodnoty)
- kdy se parametrický a neparametrický test liší a proč

**Dvouvýběrové testy**
- `3_pvalue_heatmap.png`, `3_difference_ci.png`, `3_boxplots.png`
- Welch vs. pooled t-test (Levene), Mann–Whitney

**Vícevýběrové testy**
- ANOVA / Welchova ANOVA / Kruskal–Wallis → `4_anova_kruskal.csv`
- post-hoc (Tukey HSD, Holm) a srovnání s testy po dvojicích → `4_comparison_with_3.csv`
- `4_violin_plots.png`, `4_simultaneous_ci.png`, `4_residual_diagnostics.png`

**Vícerozměrný přístup**
- Hotellingův T² (jedno- a dvouvýběrový), Boxův M test, MANOVA (Wilks, Pillai)
- `5_mean_profiles.png` (profily středních hodnot), `5_confidence_ellipses.png`, `5_simultaneous_t2_ci.png`, `5_t2_difference_ci.png`
- `5_centroid_distances.png` (Mahalanobisova vzdálenost skupin + p-hodnoty T²)
- diagnostika: `5_chi2_qq.png`, `5_covariance_comparison.png` (shoda kovariančních matic), `5_manova_residuals.png`
- co přidává vícerozměrný pohled oproti testům po proměnných

## 6. Závěr
- Orientačně z výstupů: délka spánku a únava se mezi chronotypy liší výrazně (Night Owl spí průměrně ≈ 5,0 h, Morning Lark ≈ 7,4 h), telefonní minuty a latence usínání se neliší (ANOVA p ≈ 0,3, Kruskal–Wallis p > 0,6)
- Neparametrické testy zamítají H0 i tam, kde t-test ne (Wilcoxon u telefonních minut), kvůli šikmosti a velkému n; normalita neplatí nikde (Shapiro p < 1e-17), `next_day_fatigue_score` má strop 10
- Při n = 8 500 je test citlivý i na zanedbatelné rozdíly → uvést i velikost účinku (η² ≈ 0,38 pro spánek, 0,17 pro únavu, < 0,001 pro zbylé dvě)
- Boxův M test zamítá shodu kovariančních matic → T² a MANOVA jen orientačně; Pillaiova stopa to snáší nejlépe
- TODO: doladit po kontrole výstupů; srovnání přístupů (po proměnných vs. vícerozměrně)

## Použití AI
TODO: stručně, k čemu a v jaké fázi (např. skeleton kódu, konzultace teorie, ladění)
