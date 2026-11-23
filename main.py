"""
MSM – semester work 2: Location tests
=====================================

Author:  Vít Novotný (+ TODO: co-author, for pairs a different one than in assignment 1)
Date:    TODO
Course:  KMA/MSM 2026/2027

Description
-----------
On a chosen multivariate dataset (>= 2 continuous variables, >= 3 groups) the program
compares the location of the variables:

    1. choice and description of the data
    2. one-sample tests    (t-test  vs. Wilcoxon)
    3. two-sample tests    (Welch t-test vs. Mann-Whitney), all pairs of groups
    4. multi-sample tests  (ANOVA / Welch ANOVA vs. Kruskal-Wallis) + post-hoc
    5. multivariate approach (Hotelling T², Box's M test, MANOVA)

Each section prints a table of results, saves it to OUTPUT_DIR as CSV
and saves the figures as PNG (material for the poster).

Usage:  python main.py   (or: uv run main.py)

Functions that raise NotImplementedError are skipped with a [TODO] message, so the program can be run
while it is being written.
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway


# ================================================== CONFIGURATION ================================================

DATASET = "csv"             # "iris" | "penguins" | "csv"
CSV_PATH = Path("data.csv")  # used only for DATASET = "csv"
CSV_GROUP_COL = "chronotype"  # name of the group column in the CSV
CSV_VARIABLES: list[str] = ["bedtime_phone_minutes", "sleep_latency_min", "total_sleep_hours", "next_day_fatigue_score"]  # continuous variables in the CSV (empty = all numeric columns)

ALPHA = 0.05                 # significance level used everywhere
OUTPUT_DIR = Path("output")  # tables (CSV) and figures (PNG) for the poster
SHOW_PLOTS = True            # True = also open matplotlib windows (like in MSM1)

# keys of the tables passed between sections (see `results` in main)
SHARED_RESULTS = {"reference", "one_sample", "two_sample", "post_hoc"}


# ================================================== HELPERS ======================================================

def save_figure(fig: plt.Figure, name: str) -> None:
    """
    Saves the figure to OUTPUT_DIR as PNG and optionally shows it.

    Parameters
    ----------
    fig: plt.Figure
        the figure to be saved
    name: str
        file name without extension, e.g. "2_qq_plots"
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def text_color(image, value: float) -> str:
    """
    Black or white text, whichever is more readable on the heat-map cell coloured by `image` for `value`
    (decided by the relative luminance of the cell colour).
    """
    r, g, b, _ = image.cmap(image.norm(value))
    return "black" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else "white"


def save_table(table: pd.DataFrame, name: str) -> None:
    """
    Prints the result table and saves it to OUTPUT_DIR as CSV.

    Parameters
    ----------
    table: pd.DataFrame
        the table of results
    name: str
        file name without extension, e.g. "2_one_sample_tests"
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))
    table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def group_pairs(groups: list[str]) -> list[tuple[str, str]]:
    """
    Returns all unordered pairs of groups, e.g. [(A, B), (A, C), (B, C)].

    Parameters
    ----------
    groups: list[str]
        names of the groups

    Returns
    -------
    list[tuple[str, str]]
        all pairs of distinct groups (k groups -> k*(k-1)/2 pairs)
    """
    return list(combinations(groups, 2))


def decision(p_value: float, alpha: float = ALPHA) -> str:
    """
    Returns the verbal decision about H0 for the given p-value.

    Parameters
    ----------
    p_value: float
        p-value of the test
    alpha: float
        significance level

    Returns
    -------
    str
        "reject H0" if p < alpha, otherwise "fail to reject H0"
    """
    return "reject H0" if p_value < alpha else "fail to reject H0"


# ================================================== 1. DATA ======================================================

def load_data() -> tuple[pd.DataFrame, list[str], str]:
    """
    Loads the chosen dataset and returns it in one common format, so that the rest of the program
    does not depend on which dataset was chosen.

    Returns
    -------
    tuple[pd.DataFrame, list[str], str]
        (data frame with one row per observation, names of the continuous variables, name of the group column)
    """
    if DATASET == "iris":
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame.rename(columns=lambda col: col.replace(" (cm)", "").replace(" ", "_"))
        df["species"] = iris.target_names[iris.target]
        df = df.drop(columns="target")
        variables = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        group_col = "species"
    elif DATASET == "penguins":
        import seaborn as sns
        df = sns.load_dataset("penguins").dropna()  # downloads the data, needs internet
        variables = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        group_col = "species"
    elif DATASET == "csv":
        df = pd.read_csv(CSV_PATH)
        group_col = CSV_GROUP_COL
        variables = CSV_VARIABLES or [col for col in df.select_dtypes("number").columns if col != group_col]
        df = df[[group_col, *variables]].dropna()
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")

    df[group_col] = df[group_col].astype(str)
    return df.reset_index(drop=True), variables, group_col


def describe_groups(df: pd.DataFrame, variables: list[str], group_col: str) -> pd.DataFrame:
    """
    Basic description of the data: number of observations, mean, median and standard deviation
    of every variable in every group (and for all data together).

    Parameters
    ----------
    df: pd.DataFrame
        the data
    variables: list[str]
        names of the continuous variables
    group_col: str
        name of the group column

    Returns
    -------
    pd.DataFrame
        one row per (variable, group)
    """
    rows = []
    for var in variables:
        for group, values in [("all", df[var]), *df.groupby(group_col)[var]]:
            rows.append({"variable": var, "group": group, "n": values.size,
                         "mean": values.mean(), "median": values.median(), "std": values.std(ddof=1)})
    return pd.DataFrame(rows)


def plot_overview(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Overview of the data: scatter matrix of all variables, coloured by group, histograms on the diagonal.
    (Can be reused as the first figure of section 5.)
    """
    import seaborn as sns
    grid = sns.pairplot(df, hue=group_col, vars=variables, diag_kind="hist",
                        plot_kws={"s": 6, "alpha": 0.3}, diag_kws={"common_norm": False, "element": "step"})
    return grid.figure


# ================================================== 2. ONE-SAMPLE TESTS ==========================================

def get_reference_values(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """
    Chooses the reference value for each variable: mu0 for the test of the mean (t-test)
    and m0 for the test of the median (Wilcoxon).

    The overall mean / median of all data is used, so every test answers "does this group differ from
    the whole population?".

    Returns
    -------
    pd.DataFrame
        columns: variable, mu0, m0
    """
    return pd.DataFrame({"variable": variables,
                         "mu0": [df[var].mean() for var in variables],
                         "m0": [df[var].median() for var in variables]})


def one_sample_tests(df: pd.DataFrame, variables: list[str], group_col: str,
                     references: pd.DataFrame) -> pd.DataFrame:
    """
    For every variable and every group tests
        parametric:      H0: mu = mu0   vs. H1: mu != mu0   (one-sample t-test)
        nonparametric:   H0: m  = m0    vs. H1: m  != m0    (Wilcoxon signed-rank test)
    and checks normality of the group (Shapiro-Wilk), which decides whether the t-test is trustworthy.

    Returns
    -------
    pd.DataFrame
        one row per (variable, group); columns
        variable, group, n, mean, CI_lower, CI_upper, mu0, t, p_t, median, m0, W, p_W, p_shapiro,
        decision_t, decision_W
    """
    ref = references.set_index("variable")
    rows = []
    for var in variables:
        mu0, m0 = ref.loc[var, "mu0"], ref.loc[var, "m0"]
        for group, values in df.groupby(group_col)[var]:
            t_test = stats.ttest_1samp(values, popmean=mu0)
            ci = t_test.confidence_interval(1 - ALPHA)
            wilcoxon = stats.wilcoxon(values - m0)
            rows.append({"variable": var, "group": group, "n": values.size,
                         "mean": values.mean(), "CI_lower": ci.low, "CI_upper": ci.high, "mu0": mu0,
                         "t": t_test.statistic, "p_t": t_test.pvalue,
                         "median": values.median(), "m0": m0, "W": wilcoxon.statistic, "p_W": wilcoxon.pvalue,
                         "p_shapiro": stats.shapiro(values).pvalue,
                         "decision_t": decision(t_test.pvalue), "decision_W": decision(wilcoxon.pvalue)})
    return pd.DataFrame(rows)


def plot_normality_check(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Normality check: Q-Q plot for every variable (rows) and group (columns).
    The Shapiro-Wilk p-value is in the title of each subplot.
    """
    groups = sorted(df[group_col].unique())
    fig, axes = plt.subplots(len(variables), len(groups), figsize=(3.5 * len(groups), 3 * len(variables)),
                             squeeze=False)
    for row, var in zip(axes, variables):
        for ax, group in zip(row, groups):
            values = df.loc[df[group_col] == group, var]
            stats.probplot(values, dist="norm", plot=ax)
            ax.set_title(f"{var}\n{group}, Shapiro p = {stats.shapiro(values).pvalue:.3g}", fontsize=9)
            ax.set_xlabel("theoretical quantiles")
            ax.set_ylabel("sample quantiles")
    return fig


def plot_one_sample(df: pd.DataFrame, variables: list[str], group_col: str,
                    references: pd.DataFrame, results: pd.DataFrame) -> plt.Figure:
    """
    Visualisation of the one-sample tests: for every variable the group means with 95% confidence
    intervals (and medians) compared with the reference line mu0 / m0.
    A "t" / "W" above a group means that the t-test / Wilcoxon test rejects H0 (p < ALPHA).
    """
    groups = sorted(df[group_col].unique())
    ref = references.set_index("variable")
    x = np.arange(len(groups))
    rows = -(-len(variables) // 2)   # 2 panels per row (fits the poster better than 1 x 4)
    fig, axes = plt.subplots(rows, 2, figsize=(9, 4 * rows), squeeze=False)
    for ax, var in zip(axes.flat, variables):
        res = results[results["variable"] == var].set_index("group").loc[groups]
        ax.errorbar(x, res["mean"], yerr=[res["mean"] - res["CI_lower"], res["CI_upper"] - res["mean"]],
                    fmt="o", capsize=4, label="mean (95% CI)")
        ax.scatter(x, res["median"], marker="s", color="C1", label="median")
        ax.axhline(ref.loc[var, "mu0"], color="k", label="mu0")
        ax.axhline(ref.loc[var, "m0"], color="gray", linestyle="--", label="m0")
        for xi, (_, r) in zip(x, res.iterrows()):
            mark = ("t" if r["p_t"] < ALPHA else "") + ("W" if r["p_W"] < ALPHA else "")
            ax.text(xi, r["CI_upper"], mark, ha="center", va="bottom", fontweight="bold")
        ax.set_xticks(x, groups, rotation=15)
        ax.set_title(var)
        ax.margins(y=0.12)   # headroom for the t / W marks above the intervals
    for ax in axes.flat[len(variables):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=8)
    return fig


def plot_group_histograms(df: pd.DataFrame, variables: list[str], group_col: str,
                          references: pd.DataFrame) -> plt.Figure:
    """
    Histograms of every variable (columns) in every group (rows) with the group mean and the reference
    values mu0 / m0 of the one-sample tests, so the distribution shape (skewness, ceiling effects) is visible.
    """
    groups = sorted(df[group_col].unique())
    ref = references.set_index("variable")
    fig, axes = plt.subplots(len(groups), len(variables), figsize=(4 * len(variables), 2.6 * len(groups)),
                             squeeze=False, sharex="col")
    for col, var in enumerate(variables):
        bins = np.histogram_bin_edges(df[var], bins=30)
        for row, group in enumerate(groups):
            ax = axes[row, col]
            values = df.loc[df[group_col] == group, var]
            ax.hist(values, bins=bins, color=f"C{row}", alpha=0.7)
            ax.axvline(values.mean(), color="k", label="group mean")
            ax.axvline(ref.loc[var, "mu0"], color="r", linestyle="--", label="mu0")
            ax.axvline(ref.loc[var, "m0"], color="gray", linestyle=":", label="m0")
            if row == 0:
                ax.set_title(var)
            if col == 0:
                ax.set_ylabel(group)
    axes[0, 0].legend(fontsize=7)
    return fig


# ================================================== 3. TWO-SAMPLE TESTS ==========================================

def two_sample_tests(df: pd.DataFrame, variables: list[str], group_col: str) -> pd.DataFrame:
    """
    For every variable and every pair of groups (A, B) tests
        parametric:      H0: mu_A = mu_B         (two-sample t-test, Welch = unequal variances)
        nonparametric:   H0: F_A = F_B (shift)   (Mann-Whitney U test)
    plus equality of variances (Levene) and the 95% CI for the difference mu_A - mu_B.
    The pooled t-test (p_t_pooled) is added for comparison with Welch.

    Returns
    -------
    pd.DataFrame
        one row per (variable, pair); columns
        variable, group_A, group_B, mean_diff, CI_lower, CI_upper, t, p_t, p_t_pooled, U, p_U, p_levene

    The p-values are NOT corrected for multiple testing -> compare with section 4.
    """
    groups = sorted(df[group_col].unique())
    rows = []
    for var in variables:
        samples = {group: values.to_numpy() for group, values in df.groupby(group_col)[var]}
        for a, b in group_pairs(groups):
            welch = stats.ttest_ind(samples[a], samples[b], equal_var=False)
            ci = welch.confidence_interval(1 - ALPHA)
            mann_whitney = stats.mannwhitneyu(samples[a], samples[b], alternative="two-sided")
            rows.append({"variable": var, "group_A": a, "group_B": b,
                         "mean_diff": samples[a].mean() - samples[b].mean(),
                         "CI_lower": ci.low, "CI_upper": ci.high, "t": welch.statistic, "p_t": welch.pvalue,
                         "p_t_pooled": stats.ttest_ind(samples[a], samples[b], equal_var=True).pvalue,
                         "U": mann_whitney.statistic, "p_U": mann_whitney.pvalue,
                         "p_levene": stats.levene(samples[a], samples[b]).pvalue})
    return pd.DataFrame(rows)


def plot_group_boxplots(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Box plots of every variable by group (one subplot per variable).
    """
    groups = sorted(df[group_col].unique())
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 4), squeeze=False)
    for ax, var in zip(axes[0], variables):
        ax.boxplot([df.loc[df[group_col] == group, var] for group in groups], tick_labels=groups)
        ax.set_title(var)
        ax.tick_params(axis="x", rotation=15)
    return fig


def plot_pvalue_heatmap(results: pd.DataFrame, variables: list[str], groups: list[str]) -> plt.Figure:
    """
    Matrix (heat map) of p-values: for every variable a k x k matrix, upper triangle Welch t-test,
    lower triangle Mann-Whitney. Cells with p < ALPHA are written in bold.
    """
    index = {group: i for i, group in enumerate(groups)}
    fig, axes = plt.subplots(1, len(variables), figsize=(4.5 * len(variables), 4.5), squeeze=False)
    for ax, var in zip(axes[0], variables):
        matrix = np.full((len(groups), len(groups)), np.nan)
        for _, r in results[results["variable"] == var].iterrows():
            i, j = index[r["group_A"]], index[r["group_B"]]
            matrix[min(i, j), max(i, j)] = r["p_t"]
            matrix[max(i, j), min(i, j)] = r["p_U"]
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        for i, j in zip(*np.where(~np.isnan(matrix))):
            p = matrix[i, j]
            ax.text(j, i, "<0.001" if p < 0.001 else f"{p:.3f}", ha="center", va="center", fontsize=9,
                    color=text_color(image, p), fontweight="bold" if p < ALPHA else "normal")
        ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right")
        ax.set_yticks(range(len(groups)), groups)
        ax.set_title(f"{var}\nupper: Welch t, lower: Mann-Whitney", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046)
    return fig


def plot_difference_ci(results: pd.DataFrame, variables: list[str]) -> plt.Figure:
    """
    Forest plot: 95% confidence intervals for the differences of means mu_A - mu_B for all pairs,
    with a vertical line at 0 (interval not containing 0 <=> t-test rejects H0).
    """
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 3.5), squeeze=False)
    for ax, var in zip(axes[0], variables):
        res = results[results["variable"] == var].reset_index(drop=True)
        y = np.arange(len(res))
        diff = res["mean_diff"]
        ax.errorbar(diff, y, xerr=[diff - res["CI_lower"], res["CI_upper"] - diff], fmt="o", capsize=4)
        ax.axvline(0, color="k", linestyle="--")
        ax.set_yticks(y, res["group_A"] + " - " + res["group_B"], fontsize=8)
        ax.set_title(var)
        ax.set_xlabel("difference of means (95% CI)")
    return fig


# ================================================== 4. MULTI-SAMPLE TESTS ========================================

def multi_sample_tests(df: pd.DataFrame, variables: list[str], group_col: str) -> pd.DataFrame:
    """
    For every variable tests equality of location in all groups at once
        parametric:      H0: mu_1 = ... = mu_k   (one-way ANOVA; Welch ANOVA if variances differ)
        nonparametric:   H0: all groups have the same distribution (Kruskal-Wallis)
    plus assumptions: equality of variances (Levene / Bartlett) and normality of residuals (Shapiro),
    and the effect size eta^2 (share of variance explained by the group).

    Returns
    -------
    pd.DataFrame
        one row per variable; columns variable, F, p_anova, F_welch, p_welch, H, p_kw, eta2,
        p_levene, p_bartlett, p_shapiro_resid

    The Shapiro-Wilk test is run on a random sample of 5000 residuals (scipy's limit for accurate p-values).
    """
    rng = np.random.default_rng(0)
    rows = []
    for var in variables:
        samples = [values.to_numpy(dtype=float) for _, values in df.groupby(group_col)[var]]
        anova = stats.f_oneway(*samples)
        welch = anova_oneway(df[var], df[group_col], use_var="unequal")
        kruskal = stats.kruskal(*samples)
        resid = ols(f"{var} ~ C({group_col})", data=df).fit().resid.to_numpy()
        ss_between = sum(s.size * (s.mean() - df[var].mean()) ** 2 for s in samples)
        ss_total = ((df[var] - df[var].mean()) ** 2).sum()
        shapiro_sample = rng.choice(resid, size=min(5000, resid.size), replace=False)
        rows.append({"variable": var, "F": anova.statistic, "p_anova": anova.pvalue,
                     "F_welch": welch.statistic, "p_welch": welch.pvalue,
                     "H": kruskal.statistic, "p_kw": kruskal.pvalue, "eta2": ss_between / ss_total,
                     "p_levene": stats.levene(*samples).pvalue, "p_bartlett": stats.bartlett(*samples).pvalue,
                     "p_shapiro_resid": stats.shapiro(shapiro_sample).pvalue})
    return pd.DataFrame(rows)


def post_hoc_tests(df: pd.DataFrame, variables: list[str], group_col: str) -> pd.DataFrame:
    """
    Multiple comparisons after ANOVA / Kruskal-Wallis for all pairs of groups:
        parametric:      Tukey HSD (simultaneous CIs for mu_A - mu_B)
        nonparametric:   pairwise Mann-Whitney with Holm correction

    Returns
    -------
    pd.DataFrame
        one row per (variable, pair); columns variable, group_A, group_B, diff, CI_lower, CI_upper,
        p_tukey, p_mw_holm (the difference is mu_A - mu_B, like in section 3)
    """
    groups = sorted(df[group_col].unique())
    rows = []
    for var in variables:
        samples = [df.loc[df[group_col] == group, var].to_numpy() for group in groups]
        tukey = stats.tukey_hsd(*samples)          # tukey.statistic[i, j] = mean_i - mean_j
        ci = tukey.confidence_interval(1 - ALPHA)
        pairs = list(combinations(range(len(groups)), 2))
        p_mw = [stats.mannwhitneyu(samples[i], samples[j], alternative="two-sided").pvalue for i, j in pairs]
        p_holm = multipletests(p_mw, method="holm")[1]
        for (i, j), p in zip(pairs, p_holm):
            rows.append({"variable": var, "group_A": groups[i], "group_B": groups[j],
                         "diff": tukey.statistic[i, j], "CI_lower": ci.low[i, j], "CI_upper": ci.high[i, j],
                         "p_tukey": tukey.pvalue[i, j], "p_mw_holm": p})
    return pd.DataFrame(rows)


def compare_pairwise_with_post_hoc(two_sample: pd.DataFrame, post_hoc: pd.DataFrame) -> pd.DataFrame:
    """
    Puts the uncorrected pairwise p-values (section 3) next to the post-hoc p-values (section 4)
    and marks the pairs where the decision differs.

    Returns
    -------
    pd.DataFrame
        columns variable, group_A, group_B, p_t (3), p_tukey (4), p_U (3), p_mw_holm (4), agree
        (agree = both parametric and both nonparametric decisions at ALPHA are the same)
    """
    merged = pd.merge(two_sample[["variable", "group_A", "group_B", "p_t", "p_U"]],
                      post_hoc[["variable", "group_A", "group_B", "p_tukey", "p_mw_holm"]],
                      on=["variable", "group_A", "group_B"])
    merged["agree"] = ((merged["p_t"] < ALPHA) == (merged["p_tukey"] < ALPHA)) & \
                      ((merged["p_U"] < ALPHA) == (merged["p_mw_holm"] < ALPHA))
    return merged


def plot_violins(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Violin plots of all groups for every variable (p-values of ANOVA and K-W in the subplot title).
    """
    groups = sorted(df[group_col].unique())
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 4), squeeze=False)
    for ax, var in zip(axes[0], variables):
        samples = [df.loc[df[group_col] == group, var].to_numpy() for group in groups]
        ax.violinplot(samples, showmedians=True)
        ax.set_xticks(range(1, len(groups) + 1), groups, rotation=15)
        ax.set_title(f"{var}\nANOVA p = {stats.f_oneway(*samples).pvalue:.3g}, "
                     f"K-W p = {stats.kruskal(*samples).pvalue:.3g}", fontsize=9)
    return fig


def plot_simultaneous_ci(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Simultaneous confidence intervals from the Tukey HSD post-hoc analysis
    (group means; groups whose intervals do not overlap differ significantly).
    """
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 3.5), squeeze=False)
    for ax, var in zip(axes[0], variables):
        pairwise_tukeyhsd(df[var], df[group_col], alpha=ALPHA).plot_simultaneous(ax=ax, xlabel=var, ylabel="")
        ax.set_title(f"Tukey HSD, {1 - ALPHA:.0%}", fontsize=9)
    return fig


def plot_residual_diagnostics(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Diagnostics of the ANOVA model for every variable (columns): Q-Q plot of residuals (normality, top)
    and residuals vs. fitted values (equality of variances, bottom).
    """
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, len(variables), figsize=(4 * len(variables), 7), squeeze=False)
    for col, var in enumerate(variables):
        model = ols(f"{var} ~ C({group_col})", data=df).fit()
        stats.probplot(model.resid, dist="norm", plot=axes[0, col])
        axes[0, col].set_title(var)
        jitter = rng.normal(0, 0.01 * np.ptp(model.fittedvalues) + 1e-9, size=len(df))
        axes[1, col].scatter(model.fittedvalues + jitter, model.resid, s=4, alpha=0.2)
        axes[1, col].axhline(0, color="k", linewidth=0.8)
        axes[1, col].set_xlabel("fitted values (group means)")
        axes[1, col].set_ylabel("residuals")
    return fig


# ================================================== 5. MULTIVARIATE APPROACH =====================================

def hotelling_one_sample(x: np.ndarray, mu0: np.ndarray) -> dict:
    """
    One-sample Hotelling T^2 test  H0: mu = mu0  (vector analogy of section 2).

        T^2 = n (x_bar - mu0)^T S^{-1} (x_bar - mu0),
        F   = (n - p) / (p (n - 1)) * T^2  ~  F(p, n - p)

    Parameters
    ----------
    x: np.ndarray
        data matrix n x p of one group
    mu0: np.ndarray
        reference vector of length p (e.g. vector of overall means)

    Returns
    -------
    dict
        {"T2": ..., "F": ..., "df1": ..., "df2": ..., "p": ...}
    """
    n, p = x.shape
    d = x.mean(axis=0) - mu0
    t2 = n * d @ np.linalg.solve(np.cov(x, rowvar=False), d)
    f_value = (n - p) / (p * (n - 1)) * t2
    return {"T2": t2, "F": f_value, "df1": p, "df2": n - p, "p": stats.f.sf(f_value, p, n - p)}


def hotelling_two_sample(x1: np.ndarray, x2: np.ndarray) -> dict:
    """
    Two-sample Hotelling T^2 test  H0: mu_1 = mu_2  (equal covariance matrices assumed, pooled S).

        S_p = ((n1 - 1) S1 + (n2 - 1) S2) / (n1 + n2 - 2)
        T^2 = n1 n2 / (n1 + n2) * d^T S_p^{-1} d,    d = x1_bar - x2_bar
        F   = (n1 + n2 - p - 1) / (p (n1 + n2 - 2)) * T^2  ~  F(p, n1 + n2 - p - 1)

    Returns
    -------
    dict
        {"T2": ..., "F": ..., "df1": ..., "df2": ..., "p": ...}
    """
    (n1, p), n2 = x1.shape, x2.shape[0]
    pooled = ((n1 - 1) * np.cov(x1, rowvar=False) + (n2 - 1) * np.cov(x2, rowvar=False)) / (n1 + n2 - 2)
    d = x1.mean(axis=0) - x2.mean(axis=0)
    t2 = n1 * n2 / (n1 + n2) * d @ np.linalg.solve(pooled, d)
    f_value = (n1 + n2 - p - 1) / (p * (n1 + n2 - 2)) * t2
    return {"T2": t2, "F": f_value, "df1": p, "df2": n1 + n2 - p - 1,
            "p": stats.f.sf(f_value, p, n1 + n2 - p - 1)}


def box_m_test(df: pd.DataFrame, variables: list[str], group_col: str) -> dict:
    """
    Box's M test of equality of covariance matrices  H0: Sigma_1 = ... = Sigma_k
    (assumption of the two-sample T^2 test and of MANOVA).

    Returns
    -------
    dict
        {"M": ..., "chi2": ..., "df": ..., "p": ...}
    """
    samples = [values[variables].to_numpy() for _, values in df.groupby(group_col)]
    k, p = len(samples), len(variables)
    n_i = np.array([s.shape[0] for s in samples])
    covs = [np.cov(s, rowvar=False) for s in samples]
    pooled = sum((n - 1) * c for n, c in zip(n_i, covs)) / (n_i.sum() - k)
    m = (n_i.sum() - k) * np.linalg.slogdet(pooled)[1] - sum((n - 1) * np.linalg.slogdet(c)[1]
                                                              for n, c in zip(n_i, covs))
    c = (np.sum(1 / (n_i - 1)) - 1 / (n_i.sum() - k)) * (2 * p ** 2 + 3 * p - 1) / (6 * (p + 1) * (k - 1))
    chi2 = (1 - c) * m
    dof = p * (p + 1) * (k - 1) / 2
    return {"M": m, "chi2": chi2, "df": dof, "p": stats.chi2.sf(chi2, dof)}


def multivariate_tests(df: pd.DataFrame, variables: list[str], group_col: str,
                       references: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Runs the multivariate analogies of sections 2-4:
        one-sample T^2 for every group, two-sample T^2 for every pair, Box's M test, MANOVA for all groups.

    Returns
    -------
    dict[str, pd.DataFrame]
        {"one_sample_T2": ..., "two_sample_T2": ..., "box_m": ..., "manova": ...}
        (the pairwise T^2 p-values are Holm-corrected in p_holm)
    """
    groups = sorted(df[group_col].unique())
    data = {group: df.loc[df[group_col] == group, variables].to_numpy() for group in groups}
    mu0 = references.set_index("variable").loc[variables, "mu0"].to_numpy()

    one_sample = pd.DataFrame([{"group": g, "n": len(data[g]), **hotelling_one_sample(data[g], mu0)}
                               for g in groups])
    one_sample["decision"] = one_sample["p"].map(decision)

    pairs = group_pairs(groups)
    two_sample = pd.DataFrame([{"group_A": a, "group_B": b, **hotelling_two_sample(data[a], data[b])}
                               for a, b in pairs])
    two_sample["p_holm"] = multipletests(two_sample["p"], method="holm")[1]
    two_sample["decision"] = two_sample["p_holm"].map(decision)

    manova = MANOVA.from_formula(" + ".join(variables) + f" ~ C({group_col})", data=df).mv_test()
    effect = next(name for name in manova.results if name != "Intercept")
    manova_table = (manova.results[effect]["stat"]
                    .rename(columns={"Value": "value", "Num DF": "df1", "Den DF": "df2",
                                     "F Value": "F", "Pr > F": "p"})
                    .rename_axis("test").reset_index())

    return {"one_sample_T2": one_sample, "two_sample_T2": two_sample,
            "box_m": pd.DataFrame([box_m_test(df, variables, group_col)]), "manova": manova_table}


def _draw_ellipse(ax: plt.Axes, mean: np.ndarray, cov: np.ndarray, radius_sq: float, **kwargs) -> None:
    """
    Draws the ellipse {z : (z - mean)^T cov^{-1} (z - mean) <= radius_sq} into the axes.
    """
    values, vectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    width, height = 2 * np.sqrt(radius_sq * values[::-1])
    ax.add_patch(Ellipse(mean, width, height, angle=angle, fill=False, **kwargs))


def plot_confidence_ellipses(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Scatter plots of pairs of variables with ellipses of constant Mahalanobis distance for every group:
    the 95% region of the data (d^2 <= chi2_{0.95}(2), solid) and the T^2 confidence region for the
    mean vector (dashed, often tiny for large n).
    """
    groups = sorted(df[group_col].unique())
    pairs = list(combinations(variables, 2))
    n_cols = min(3, len(pairs))
    n_rows = -(-len(pairs) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    for ax, (var_x, var_y) in zip(axes.ravel(), pairs):
        for i, group in enumerate(groups):
            xy = df.loc[df[group_col] == group, [var_x, var_y]].to_numpy()
            n, mean, cov = len(xy), xy.mean(axis=0), np.cov(xy, rowvar=False)
            ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.15, color=f"C{i}")
            _draw_ellipse(ax, mean, cov, stats.chi2.ppf(1 - ALPHA, 2), color=f"C{i}", label=group)
            _draw_ellipse(ax, mean, cov, 2 * (n - 1) / (n * (n - 2)) * stats.f.ppf(1 - ALPHA, 2, n - 2),
                          color="k", linestyle="--", linewidth=1)
        ax.set_xlabel(var_x)
        ax.set_ylabel(var_y)
    axes[0, 0].legend(fontsize=8)
    for ax in axes.ravel()[len(pairs):]:
        ax.set_visible(False)
    return fig


def plot_simultaneous_t2_intervals(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Simultaneous T^2 confidence intervals for the components of the mean vectors of all groups
    (compared with Bonferroni and ordinary t-intervals).

        a^T x_bar  +-  sqrt( p (n - 1) / (n - p) * F_{1-alpha}(p, n - p) ) * sqrt( a^T S a / n )

    with a = unit vectors, i.e. one interval per variable.
    """
    groups = sorted(df[group_col].unique())
    p = len(variables)
    fig, axes = plt.subplots(1, p, figsize=(4 * p, 4), squeeze=False)
    for ax, (j, var) in zip(axes[0], enumerate(variables)):
        for i, group in enumerate(groups):
            x = df.loc[df[group_col] == group, variables].to_numpy()
            n = len(x)
            se = np.sqrt(np.cov(x, rowvar=False)[j, j] / n)
            factors = {"T²": np.sqrt(p * (n - 1) / (n - p) * stats.f.ppf(1 - ALPHA, p, n - p)),
                       "Bonferroni": stats.t.ppf(1 - ALPHA / (2 * p), n - 1),
                       "t (single)": stats.t.ppf(1 - ALPHA / 2, n - 1)}
            for offset, (name, factor) in zip((-0.2, 0, 0.2), factors.items()):
                ax.errorbar(i + offset, x[:, j].mean(), yerr=factor * se, fmt="o", capsize=3, color=f"C{i}",
                            alpha=(1.0, 0.7, 0.4)[list(factors).index(name)],
                            label=name if i == 0 else None)
        ax.set_xticks(range(len(groups)), groups, rotation=15)
        ax.set_title(var)
    axes[0, 0].legend(fontsize=8, title="left to right")
    return fig


def plot_multivariate_normality(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Check of multivariate normality: chi-square Q-Q plot of squared Mahalanobis distances
    d_i^2 = (x_i - x_bar)^T S^{-1} (x_i - x_bar)  vs. quantiles of chi2(p), one subplot per group.
    """
    groups = sorted(df[group_col].unique())
    p = len(variables)
    fig, axes = plt.subplots(1, len(groups), figsize=(4.5 * len(groups), 4.5), squeeze=False)
    for ax, group in zip(axes[0], groups):
        x = df.loc[df[group_col] == group, variables].to_numpy()
        centered = x - x.mean(axis=0)
        d2 = np.sort(np.einsum("ij,ij->i", centered @ np.linalg.inv(np.cov(x, rowvar=False)), centered))
        theoretical = stats.chi2.ppf((np.arange(1, len(x) + 1) - 0.5) / len(x), p)
        ax.scatter(theoretical, d2, s=4)
        ax.axline((0, 0), slope=1, color="k", linestyle="--")
        ax.set_title(group)
        ax.set_xlabel(f"chi2({p}) quantiles")
        ax.set_ylabel("squared Mahalanobis distance")
    return fig


def plot_mean_profiles(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Profiles of the mean vectors: heat map of the group means of the standardised variables
    (z-scores of all data). Shows at a glance how the groups differ in location.
    """
    means = ((df[variables] - df[variables].mean()) / df[variables].std(ddof=1)).groupby(df[group_col]).mean()
    limit = np.abs(means.to_numpy()).max()
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(means.to_numpy(), cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    for i, j in np.ndindex(means.shape):
        ax.text(j, i, f"{means.iloc[i, j]:.2f}", ha="center", va="center",
                color=text_color(image, means.iloc[i, j]))
    ax.set_xticks(range(len(variables)), variables, rotation=20, ha="right")
    ax.set_yticks(range(len(means)), means.index)
    ax.set_title("standardised group means (z-score)")
    fig.colorbar(image, ax=ax, fraction=0.046)
    return fig


def plot_t2_difference_ci(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Simultaneous T^2 confidence intervals for the differences of the mean vectors (all pairs of groups,
    one subplot per variable, pooled covariance matrix) compared with ordinary t-intervals.
    An interval not containing 0 marks a variable in which the two groups differ.

        d_j  +-  sqrt( p (n1 + n2 - 2) / (n1 + n2 - p - 1) * F_{1-alpha}(p, n1 + n2 - p - 1) ) * sqrt( S_p,jj (1/n1 + 1/n2) )
    """
    groups = sorted(df[group_col].unique())
    data = {group: df.loc[df[group_col] == group, variables].to_numpy() for group in groups}
    pairs = group_pairs(groups)
    p = len(variables)
    rows = -(-p // 2)   # 2 panels per row (fits the poster better than 1 x 4)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3.5 * rows), squeeze=False)
    for ax, (j, var) in zip(axes.flat, enumerate(variables)):
        for y, (a, b) in enumerate(pairs):
            (n1, _), n2 = data[a].shape, data[b].shape[0]
            pooled = ((n1 - 1) * np.cov(data[a], rowvar=False) + (n2 - 1) * np.cov(data[b], rowvar=False)) / (n1 + n2 - 2)
            diff = data[a][:, j].mean() - data[b][:, j].mean()
            se = np.sqrt(pooled[j, j] * (1 / n1 + 1 / n2))
            c_t2 = np.sqrt(p * (n1 + n2 - 2) / (n1 + n2 - p - 1) * stats.f.ppf(1 - ALPHA, p, n1 + n2 - p - 1))
            c_t = stats.t.ppf(1 - ALPHA / 2, n1 + n2 - 2)
            ax.errorbar(diff, y + 0.12, xerr=c_t2 * se, fmt="o", capsize=3, color="C0",
                        label="T² (simultaneous)" if y == 0 else None)
            ax.errorbar(diff, y - 0.12, xerr=c_t * se, fmt="s", capsize=3, color="C1",
                        label="t (single)" if y == 0 else None)
        ax.axvline(0, color="k", linestyle="--")
        ax.set_yticks(range(len(pairs)), [f"{a} - {b}" for a, b in pairs], fontsize=8)
        ax.set_ylim(-0.5, len(pairs) + 0.3)   # headroom for the legend
        ax.set_title(var)
        ax.set_xlabel("difference of means")
    for ax in axes.flat[p:]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=7, loc="upper left", ncol=2)
    return fig


def plot_centroid_distances(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Matrix of squared Mahalanobis distances D^2 = d^T S_p^{-1} d between the group mean vectors (pooled S_p),
    annotated with the Holm-corrected p-value of the two-sample Hotelling T^2 test.
    """
    groups = sorted(df[group_col].unique())
    data = {group: df.loc[df[group_col] == group, variables].to_numpy() for group in groups}
    pairs = list(combinations(range(len(groups)), 2))
    tests = [hotelling_two_sample(data[groups[i]], data[groups[j]]) for i, j in pairs]
    p_holm = multipletests([t["p"] for t in tests], method="holm")[1]
    d2 = np.zeros((len(groups), len(groups)))
    labels = np.full(d2.shape, "", dtype=object)
    for (i, j), test, p in zip(pairs, tests, p_holm):
        n1, n2 = len(data[groups[i]]), len(data[groups[j]])
        d2[i, j] = d2[j, i] = test["T2"] * (n1 + n2) / (n1 * n2)
        labels[i, j] = labels[j, i] = f"D² = {d2[i, j]:.2f}\np(Holm) {'< 0.001' if p < 0.001 else f'= {p:.3f}'}"
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(d2, cmap="viridis")
    for i, j in np.ndindex(d2.shape):
        ax.text(j, i, labels[i, j], ha="center", va="center", color=text_color(image, d2[i, j]), fontsize=9)
    ax.set_xticks(range(len(groups)), groups, rotation=20, ha="right")
    ax.set_yticks(range(len(groups)), groups)
    ax.set_title("Mahalanobis distance between group mean vectors")
    fig.colorbar(image, ax=ax, fraction=0.046)
    return fig


def plot_covariance_comparison(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Visual check of equality of covariance matrices (assumption of T^2 and MANOVA, tested by Box's M):
    correlation matrix of every group, and standard deviations of the variables relative to all data.
    """
    groups = sorted(df[group_col].unique())
    p = len(variables)
    fig, axes = plt.subplots(1, len(groups) + 1, figsize=(4.5 * (len(groups) + 1), 4.5))
    for ax, group in zip(axes, groups):
        corr = np.corrcoef(df.loc[df[group_col] == group, variables].to_numpy(), rowvar=False)
        ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        for i, j in np.ndindex(corr.shape):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(p), variables, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(p), variables, fontsize=8)
        ax.set_title(f"{group}: correlations")
    overall_std = df[variables].std(ddof=1)
    width = 0.8 / len(groups)
    for i, group in enumerate(groups):
        ratio = df.loc[df[group_col] == group, variables].std(ddof=1) / overall_std
        axes[-1].bar(np.arange(p) + i * width, ratio, width, color=f"C{i}", label=group)
    axes[-1].axhline(1, color="k", linewidth=0.8)
    axes[-1].set_xticks(np.arange(p) + 0.4 - width / 2, variables, rotation=45, ha="right", fontsize=8)
    axes[-1].set_ylim(0, 1.3)   # headroom for the legend
    axes[-1].set_title("std relative to all data")
    axes[-1].legend(fontsize=8, loc="upper center", ncol=3)
    return fig


def plot_multivariate_residuals(df: pd.DataFrame, variables: list[str], group_col: str) -> plt.Figure:
    """
    Diagnostics of the MANOVA residuals (observation minus its group mean vector): chi-square Q-Q plot of
    the squared Mahalanobis distances of the residuals (multivariate normality) and their distribution
    in the groups (equality of spread) with the 95% quantile of chi2(p).
    """
    p = len(variables)
    resid = (df[variables] - df.groupby(group_col)[variables].transform("mean")).to_numpy()
    pooled = resid.T @ resid / (len(df) - df[group_col].nunique())
    d2 = np.einsum("ij,ij->i", resid @ np.linalg.inv(pooled), resid)
    fig, (ax_qq, ax_box) = plt.subplots(1, 2, figsize=(11, 4.5))
    theoretical = stats.chi2.ppf((np.arange(1, len(d2) + 1) - 0.5) / len(d2), p)
    ax_qq.scatter(theoretical, np.sort(d2), s=4)
    ax_qq.axline((0, 0), slope=1, color="k", linestyle="--")
    ax_qq.set_xlabel(f"chi2({p}) quantiles")
    ax_qq.set_ylabel("squared Mahalanobis distance of residuals")
    ax_qq.set_title("chi-square Q-Q plot of MANOVA residuals")
    groups = sorted(df[group_col].unique())
    ax_box.boxplot([d2[(df[group_col] == group).to_numpy()] for group in groups], tick_labels=groups)
    ax_box.axhline(stats.chi2.ppf(1 - ALPHA, p), color="r", linestyle="--", label="chi2 95% quantile")
    ax_box.set_ylabel("squared Mahalanobis distance of residuals")
    ax_box.set_title("spread of residuals by group")
    ax_box.legend()
    return fig


# ================================================== MAIN =========================================================

def run_section(title: str, steps: list) -> None:
    """
    Runs the steps of one section; a step that is not implemented yet (NotImplementedError) is skipped
    with a message, so the program can be run while it is being written.

    Parameters
    ----------
    title: str
        heading printed before the section
    steps: list
        functions without parameters (lambdas) that compute and save one result each
    """
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    for step in steps:
        try:
            step()
        except NotImplementedError as todo:
            print(f"  [TODO] {todo}")
        except KeyError as missing:
            if missing.args[0] not in SHARED_RESULTS:   # a real bug (e.g. wrong column name) -> show it
                raise
            print(f"  [WAITING] missing result {missing} from a previous step")


if __name__ == "__main__":

    df, variables, group_col = load_data()
    groups = sorted(df[group_col].unique())
    results: dict[str, pd.DataFrame] = {}   # tables shared between sections (e.g. section 3 is needed in section 4)

    print(f"Dataset: {DATASET}, n = {len(df)}, variables: {variables}, groups: {groups}")

    run_section("1. Data", [
        lambda: save_table(describe_groups(df, variables, group_col), "1_data_description"),
        lambda: save_figure(plot_overview(df, variables, group_col), "1_scatter_matrix"),
    ])

    def section_2_tests() -> None:
        results["reference"] = get_reference_values(df, variables)
        results["one_sample"] = one_sample_tests(df, variables, group_col, results["reference"])
        save_table(results["one_sample"], "2_one_sample_tests")

    run_section("2. One-sample tests", [
        section_2_tests,
        lambda: save_figure(plot_normality_check(df, variables, group_col), "2_qq_plots"),
        lambda: save_figure(plot_one_sample(df, variables, group_col, results["reference"], results["one_sample"]),
                            "2_means_vs_reference"),
        lambda: save_figure(plot_group_histograms(df, variables, group_col, results["reference"]),
                            "2_histograms"),
    ])

    def section_3_tests() -> None:
        results["two_sample"] = two_sample_tests(df, variables, group_col)
        save_table(results["two_sample"], "3_two_sample_tests")

    run_section("3. Two-sample tests", [
        section_3_tests,
        lambda: save_figure(plot_group_boxplots(df, variables, group_col), "3_boxplots"),
        lambda: save_figure(plot_pvalue_heatmap(results["two_sample"], variables, groups), "3_pvalue_heatmap"),
        lambda: save_figure(plot_difference_ci(results["two_sample"], variables), "3_difference_ci"),
    ])

    def section_4_tests() -> None:
        save_table(multi_sample_tests(df, variables, group_col), "4_anova_kruskal")
        results["post_hoc"] = post_hoc_tests(df, variables, group_col)
        save_table(results["post_hoc"], "4_post_hoc")
        save_table(compare_pairwise_with_post_hoc(results["two_sample"], results["post_hoc"]), "4_comparison_with_3")

    run_section("4. Multi-sample tests", [
        section_4_tests,
        lambda: save_figure(plot_violins(df, variables, group_col), "4_violin_plots"),
        lambda: save_figure(plot_simultaneous_ci(df, variables, group_col), "4_simultaneous_ci"),
        lambda: save_figure(plot_residual_diagnostics(df, variables, group_col), "4_residual_diagnostics"),
    ])

    def section_5_tests() -> None:
        for name, table in multivariate_tests(df, variables, group_col, results["reference"]).items():
            save_table(table, f"5_{name}")

    run_section("5. Multivariate approach", [
        section_5_tests,
        lambda: save_figure(plot_confidence_ellipses(df, variables, group_col), "5_confidence_ellipses"),
        lambda: save_figure(plot_simultaneous_t2_intervals(df, variables, group_col), "5_simultaneous_t2_ci"),
        lambda: save_figure(plot_multivariate_normality(df, variables, group_col), "5_chi2_qq"),
        lambda: save_figure(plot_mean_profiles(df, variables, group_col), "5_mean_profiles"),
        lambda: save_figure(plot_t2_difference_ci(df, variables, group_col), "5_t2_difference_ci"),
        lambda: save_figure(plot_centroid_distances(df, variables, group_col), "5_centroid_distances"),
        lambda: save_figure(plot_covariance_comparison(df, variables, group_col), "5_covariance_comparison"),
        lambda: save_figure(plot_multivariate_residuals(df, variables, group_col), "5_manova_residuals"),
    ])

    print(f"\nOutputs (tables and figures) are in the folder {OUTPUT_DIR.resolve()}")
