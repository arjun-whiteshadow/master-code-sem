"""Association tests between growth conditions and the Method 1 results:
each descriptor against each parameter, each parameter across the clusters,
and a regression of each descriptor on all the parameters together."""

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def benjamini_hochberg(p):
    """Adjusted p-values by the Benjamini–Hochberg step-up procedure.

    NaN entries (tests that could not be run) are left NaN and do not count
    toward the number of tests.
    """
    p = np.asarray(p, dtype=float)
    q = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_ok = np.empty(n)
    q_ok[order] = np.minimum(adjusted, 1.0)
    q[ok] = q_ok
    return q


def _finish(df, alpha):
    df["q_value"] = benjamini_hochberg(df["p_value"])
    df["Significant"] = df["q_value"] < alpha
    df["p_value"] = df["p_value"].round(4)
    df["q_value"] = df["q_value"].round(4)
    return df


def descriptor_vs_parameter(merged, descriptors, numeric, alpha):
    """Spearman correlation of every descriptor with every numeric
    parameter, adjusted over the whole grid."""
    rows = []
    for d in descriptors:
        for p in numeric:
            sub = merged[[d, p]].dropna()
            rho, pval = spearmanr(sub[d], sub[p])
            rows.append(dict(Descriptor=d, Parameter=p, N=len(sub),
                             Spearman_rho=round(float(rho), 4), p_value=float(pval)))
    return _finish(pd.DataFrame(rows), alpha)


def cluster_vs_numeric(merged, numeric, alpha):
    """Kruskal–Wallis test of each numeric parameter across the clusters,
    with each cluster's median and count, adjusted over the parameters.

    The test is left NaN when fewer than two clusters have a value.
    """
    clusters = sorted(merged["Cluster"].unique())
    rows = []
    for p in numeric:
        sub = merged[["Cluster", p]].dropna()
        groups = [sub.loc[sub["Cluster"] == c, p].to_numpy() for c in clusters]
        present = [g for g in groups if len(g)]
        stat, pval = kruskal(*present) if len(present) >= 2 else (np.nan, np.nan)
        row = dict(Parameter=p, N=len(sub), H=round(float(stat), 4), p_value=float(pval))
        for c, g in zip(clusters, groups):
            row[f"Median_Cluster_{c}"] = round(float(np.median(g)), 4) if len(g) else np.nan
            row[f"N_Cluster_{c}"] = len(g)
        rows.append(row)
    return _finish(pd.DataFrame(rows), alpha)


def cluster_vs_categorical(merged, categorical, n_perm, seed, alpha):
    """Cross-tabulation of each categorical parameter against the clusters,
    with an exact p-value for independence by permutation.

    The chi-square statistic ranks the shuffled tables; its asymptotic
    p-value is not used because the cluster sizes are small. Expected
    counts depend only on the margins, which a permutation leaves unchanged.

    Returns the test table and a long table of the observed counts.
    """
    rng = np.random.default_rng(seed)
    rows, counts = [], []
    for p in categorical:
        sub = merged[["Cluster", p]].dropna()
        ci, clusters = pd.factorize(sub["Cluster"], sort=True)
        li, levels = pd.factorize(sub[p], sort=True)
        R, C = len(clusters), len(levels)
        E = np.outer(np.bincount(ci, minlength=R), np.bincount(li, minlength=C)) / len(sub)

        def chi2(lab):
            O = np.bincount(ci * C + lab, minlength=R * C).reshape(R, C)
            return ((O - E) ** 2 / E).sum()

        observed = chi2(li)
        exceed = sum(chi2(rng.permutation(li)) >= observed - 1e-12 for _ in range(n_perm))
        rows.append(dict(Parameter=p, N=len(sub), Levels=C, Chi2=round(float(observed), 4),
                         p_value=(exceed + 1) / (n_perm + 1)))
        table = pd.crosstab(sub["Cluster"], sub[p])
        for c in table.index:
            for level in table.columns:
                counts.append(dict(Parameter=p, Cluster=int(c), Level=level,
                                   Count=int(table.loc[c, level])))
    tests = _finish(pd.DataFrame(rows, columns=["Parameter", "N", "Levels", "Chi2",
                                                 "p_value"]), alpha)
    return tests, pd.DataFrame(counts, columns=["Parameter", "Cluster", "Level", "Count"])


def descriptor_regressions(merged, descriptors, numeric):
    """Ordinary least squares of each descriptor on all retained numeric
    parameters.

    Reports the in-sample R², the leave-one-out R² (each sample predicted
    by a model fitted without it) and standardised coefficients, so that
    the parameters can be compared on one scale.
    """
    rows = []
    for d in descriptors:
        sub = merged[[d] + numeric].dropna()
        X = sub[numeric].to_numpy(dtype=float)
        y = sub[d].to_numpy(dtype=float)
        model = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
        pred = cross_val_predict(make_pipeline(StandardScaler(), LinearRegression()),
                                 X, y, cv=LeaveOneOut())
        loo_r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        row = dict(Descriptor=d, N=len(sub), R2_Fit=round(float(model.score(X, y)), 4),
                   R2_LOO=round(float(loo_r2), 4))
        coef = model[-1].coef_ / y.std()
        for p, b in zip(numeric, coef):
            row[f"Beta_{p}"] = round(float(b), 4)
        rows.append(row)
    return pd.DataFrame(rows)
