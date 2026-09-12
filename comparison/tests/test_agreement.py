"""Checks on the agreement measures. Run with ``pytest`` from the comparison folder."""

import numpy as np
import pandas as pd
import pytest

from src import agreement


def test_scores_are_one_for_identical_partitions():
    a = np.array([0, 0, 1, 1, 2, 2])
    s = agreement.scores(a, a)
    assert all(abs(v - 1.0) < 1e-12 for v in s.values())


def test_scores_ignore_label_names():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([5, 5, 3, 3, 9, 9])
    assert agreement.scores(a, b)["ARI"] == 1.0
    assert agreement.scores(a, b)["NMI"] == 1.0


def test_scores_near_zero_for_random_partitions():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, size=300)
    b = rng.integers(0, 3, size=300)
    assert abs(agreement.scores(a, b)["ARI"]) < 0.05


def test_cluster_level_capture_and_purity():
    merged = pd.DataFrame({"M1_Cluster": [1, 1, 1, 1, 2, 2],
                           "CNN_Cluster": [1, 1, 1, 2, 2, 2]})
    table = agreement.cluster_level(merged, "M1_Cluster", "CNN_Cluster", {},
                                    {"strong": 70, "moderate": 40})
    row = table.set_index("M1_Cluster").loc[1]
    assert row["Best_CNN_Cluster"] == 1
    assert row["Capture_Percent"] == 75.0                # 3 of 4 in M1 cluster 1
    assert row["Purity_Percent"] == 100.0                # 3 of 3 in CNN cluster 1
    assert row["Agreement_Level"] == "strong"
    row = table.set_index("M1_Cluster").loc[2]
    assert row["Capture_Percent"] == 100.0
    assert row["Purity_Percent"] == pytest.approx(2 / 3 * 100, abs=0.05)
    assert row["Agreement_Level"] == "moderate"


def test_hungarian_mapping_picks_maximum_overlap():
    ct = pd.DataFrame([[8, 1, 0], [0, 2, 7], [1, 9, 0]],
                      index=[1, 2, 3], columns=[1, 2, 3])
    m = agreement.hungarian_mapping(ct).set_index("M1_Cluster")
    assert m.loc[1, "Mapped_CNN_Cluster"] == 1
    assert m.loc[2, "Mapped_CNN_Cluster"] == 3
    assert m.loc[3, "Mapped_CNN_Cluster"] == 2
    assert m["Overlap"].sum() == 24
