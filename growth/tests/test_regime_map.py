"""Checks on the regime-map tables. Run with ``pytest`` from the growth folder."""

import numpy as np
import pandas as pd

from src import regime_map


def test_choose_axes_takes_two_smallest_p_unless_fixed():
    tests = pd.DataFrame({"Parameter": ["T", "Ga", "As", "Sb"],
                          "p_value": [0.30, 0.01, np.nan, 0.02]})
    assert regime_map.choose_axes(tests) == ["Ga", "Sb"]
    assert regime_map.choose_axes(tests, fixed=["T", "As"]) == ["T", "As"]
    assert regime_map.choose_axes(tests.iloc[:1]) == ["T"]


def test_regime_windows_gives_quartiles_per_cluster():
    merged = pd.DataFrame({"Cluster": [1] * 5 + [2] * 3,
                           "T": [10, 20, 30, 40, 50, 100, np.nan, 300],
                           "Ga": [1, 1, 1, 1, 1, 2, 2, 2]})
    w = regime_map.regime_windows(merged, ["T", "Ga"]).set_index(["Cluster", "Parameter"])
    assert len(w) == 4
    assert w.loc[(1, "T"), ["N", "Min", "Q1", "Median", "Q3", "Max"]].tolist() == [5, 10, 20, 30, 40, 50]
    assert w.loc[(2, "T"), "N"] == 2 and w.loc[(2, "T"), "Median"] == 200
    assert w.loc[(2, "Ga"), "Q1"] == w.loc[(2, "Ga"), "Q3"] == 2


def test_map_points_keeps_only_the_axes():
    merged = pd.DataFrame({"Sample_ID": ["a", "b"], "Cluster": [1, 2],
                           "T": [1, 2], "Ga": [3, 4], "As": [5, 6]})
    pts = regime_map.map_points(merged, ["Ga", "T"])
    assert list(pts.columns) == ["Sample_ID", "Cluster", "Ga", "T"]
