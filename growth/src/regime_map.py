"""The growth-condition window of each cluster, and the two parameters on
which the clusters are drawn."""

import pandas as pd


def choose_axes(cluster_tests, fixed=None):
    """The two numeric parameters that separate the clusters most, by the
    smallest Kruskal–Wallis p-value, unless a pair is fixed in the config.

    Returns fewer than two names when fewer are available.
    """
    if fixed:
        return list(fixed)
    ranked = cluster_tests.dropna(subset=["p_value"]).sort_values(["p_value", "Parameter"])
    return ranked["Parameter"].head(2).tolist()


def regime_windows(merged, numeric):
    """Quartiles of each numeric parameter within each cluster.

    The interquartile range is the window reported for a cluster; the full
    range is given alongside so that a wide spread is not hidden.
    """
    names = ["Min", "Q1", "Median", "Q3", "Max"]
    rows = []
    for c, g in merged.groupby("Cluster", sort=True):
        for p in numeric:
            v = g[p].dropna()
            row = dict(Cluster=int(c), Parameter=p, N=len(v))
            for name, val in zip(names, v.quantile([0, 0.25, 0.5, 0.75, 1])):
                row[name] = round(float(val), 4)
            rows.append(row)
    return pd.DataFrame(rows, columns=["Cluster", "Parameter", "N"] + names)


def map_points(merged, axes):
    """Sample, cluster and the two map coordinates, for the figure and the
    record."""
    return merged[["Sample_ID", "Cluster"] + list(axes)].copy()
