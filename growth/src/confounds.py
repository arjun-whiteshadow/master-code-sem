"""Checks on the growth log itself, run before any association test: which
parameters moved together across the runs, and which drifted with time."""

from itertools import combinations

import pandas as pd
from scipy.stats import spearmanr


def growth_dates(sample_ids):
    """Growth date from the MMDDYY prefix of each sample ID.

    NaT where the ID does not start with six digits that form a date.
    """
    ids = pd.Series(list(sample_ids), index=list(sample_ids), dtype=str)
    prefix = ids.str.extract(r"^(\d{6})")[0]
    return pd.to_datetime(prefix, format="%m%d%y", errors="coerce").rename("Growth_Date")


def parameter_correlations(merged, numeric, threshold):
    """Spearman correlation between every pair of numeric parameters.

    A pair whose |rho| reaches ``threshold`` is marked as not separable:
    the two were changed together across the runs, so an effect found for
    one cannot be told from an effect of the other.
    """
    rows = []
    for a, b in combinations(numeric, 2):
        d = merged[[a, b]].dropna()
        rho, p = spearmanr(d[a], d[b])
        rows.append(dict(Parameter_A=a, Parameter_B=b, N=len(d),
                         Spearman_rho=round(float(rho), 4), p_value=round(float(p), 4),
                         Not_Separable=abs(rho) >= threshold))
    return pd.DataFrame(rows, columns=["Parameter_A", "Parameter_B", "N", "Spearman_rho",
                                       "p_value", "Not_Separable"])


def date_drift(merged, numeric, dates, threshold):
    """Spearman correlation of each numeric parameter with growth date.

    A parameter whose |rho| reaches ``threshold`` is marked as drifting:
    its effect cannot be told from anything else that changed over the same
    period (source material, calibration, operator).
    """
    day = (dates.reindex(merged["Sample_ID"]) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    rows = []
    for col in numeric:
        d = pd.DataFrame({"x": merged[col].to_numpy(), "t": day}).dropna()
        rho, p = spearmanr(d["x"], d["t"])
        rows.append(dict(Parameter=col, N=len(d), Spearman_rho_vs_Date=round(float(rho), 4),
                         p_value=round(float(p), 4), Drifts=abs(rho) >= threshold))
    return pd.DataFrame(rows, columns=["Parameter", "N", "Spearman_rho_vs_Date",
                                       "p_value", "Drifts"])


def category_dates(merged, categorical, dates):
    """First and last growth date of each level of each categorical
    parameter, so that a level confined to one period can be seen."""
    day = dates.reindex(merged["Sample_ID"]).to_numpy()
    rows = []
    for col in categorical:
        d = pd.DataFrame({"level": merged[col].to_numpy(), "t": day}).dropna(subset=["level"])
        for level, g in d.groupby("level", sort=True):
            rows.append(dict(Parameter=col, Level=level, N=len(g),
                             First_Date=g["t"].min(), Last_Date=g["t"].max()))
    return pd.DataFrame(rows, columns=["Parameter", "Level", "N", "First_Date", "Last_Date"])
