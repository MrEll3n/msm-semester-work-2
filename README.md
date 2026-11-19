# MSM-semester-work-2

Testy o poloze – jednovýběrové, dvouvýběrové a vícevýběrové testy, ANOVA, vícerozměrné testy (KMA/MSM 2026/2027, úloha 2, odevzdání **24. 11. 2026**).

## Požadavky

- Python 3.13
- knihovny `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `scikit-learn`

## Instalace

```bash
python3.13 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Alternativa přes [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Spuštění

```bash
python main.py      # nebo: uv run main.py
```

Datová sada se volí v konfiguraci na začátku `main.py` (`DATASET = "iris" | "penguins" | "csv"`). Tabulky výsledků (CSV) a grafy (PNG) se ukládají do složky `output/` jako podklad pro poster; `SHOW_PLOTS = True` grafy navíc otevře v oknech.

## Data

Použitá je sada [Sleep Debt & Screen Time: Late-Night Phone Habits](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits/data) (Kaggle, 8 500 řádků). Soubor `bedtime_screentime_sleep_debt.csv` je potřeba stáhnout a uložit jako `data.csv` do kořene projektu (`DATASET = "csv"`).

- skupiny: `chronotype` (Morning Lark, Intermediate, Night Owl)
- proměnné: `bedtime_phone_minutes`, `sleep_latency_min`, `total_sleep_hours`, `next_day_fatigue_score`

## Struktura `main.py`

| Oddíl | Bod zadání | Funkce |
|---|---|---|
| 1. Data | volba dat | `load_data`, `describe_groups`, `plot_overview` |
| 2. Jednovýběrové testy | t-test vs. Wilcoxon, normalita | `get_reference_values`, `one_sample_tests`, `plot_normality_check`, `plot_one_sample` |
| 3. Dvouvýběrové testy | všechny dvojice skupin | `two_sample_tests`, `plot_group_boxplots`, `plot_pvalue_heatmap`, `plot_difference_ci` |
| 4. Vícevýběrové testy | ANOVA, Kruskal–Wallis, post-hoc | `multi_sample_tests`, `post_hoc_tests`, `compare_pairwise_with_post_hoc`, `plot_violins`, `plot_simultaneous_ci`, `plot_residual_diagnostics` |
| 5. Vícerozměrný přístup | Hotellingův T², Boxův M, MANOVA | `hotelling_one_sample`, `hotelling_two_sample`, `box_m_test`, `multivariate_tests`, `plot_confidence_ellipses`, `plot_simultaneous_t2_intervals`, `plot_multivariate_normality` |

Funkce, které by vyhodily `NotImplementedError`, program přeskočí a vypíše `[TODO]`, takže jde spouštět i při dalších úpravách.

## Odevzdání

Do STAGu složka `Prijmeni_jmeno_02` s posterem (PDF) a kódem (`main.py`, `README.md`, `pyproject.toml`, `requirements.txt`).
