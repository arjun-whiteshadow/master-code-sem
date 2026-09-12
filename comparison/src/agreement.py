"""Agreement between two partitions of the same samples."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             completeness_score, homogeneity_score,
                             normalized_mutual_info_score, v_measure_score)


def scores(a, b):
    """Standard partition-agreement indices, all label-permutation invariant."""
    return {
        "ARI": adjusted_rand_score(a, b),
        "NMI": normalized_mutual_info_score(a, b),
        "AMI": adjusted_mutual_info_score(a, b),
        "Homogeneity": homogeneity_score(a, b),
        "Completeness": completeness_score(a, b),
        "V_Measure": v_measure_score(a, b),
    }


def cluster_level(merged, a_col, b_col, names, levels):
    """For each cluster in ``a_col``, the ``b_col`` cluster that shares most
    samples with it, with capture (share of the A cluster) and purity (share
    of the B cluster) in percent."""
    ct = pd.crosstab(merged[a_col], merged[b_col])
    b_sizes = merged[b_col].value_counts()
    rows = []
    for i in ct.index:
        row = ct.loc[i]
        best, shared = row.idxmax(), int(row.max())
        a_size, b_size = int(row.sum()), int(b_sizes.get(best, 0))
        capture = 100 * shared / a_size if a_size else 0.0
        purity = 100 * shared / b_size if b_size else 0.0
        low = min(capture, purity)
        level = ("strong" if low >= levels["strong"] else
                 "moderate" if low >= levels["moderate"] else "weak")
        rows.append(dict(M1_Cluster=int(i), M1_Name=names.get(int(i), f"Cluster {i}"),
                         Best_CNN_Cluster=int(best), Shared=shared,
                         M1_Size=a_size, CNN_Size=b_size,
                         Capture_Percent=round(capture, 1), Purity_Percent=round(purity, 1),
                         Agreement_Level=level))
    return pd.DataFrame(rows)


def hungarian_mapping(ct):
    """Label pairing that maximises total overlap. For laying out tables and
    figures only; no score depends on it."""
    r, c = linear_sum_assignment(-ct.to_numpy())
    return pd.DataFrame({
        "M1_Cluster": [ct.index[i] for i in r],
        "Mapped_CNN_Cluster": [ct.columns[j] for j in c],
        "Overlap": [int(ct.iat[i, j]) for i, j in zip(r, c)],
    })
