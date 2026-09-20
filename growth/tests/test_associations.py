"""Checks on the association tests. Run with ``pytest`` from the growth folder."""

import numpy as np
import pandas as pd
import pytest

from src import associations

SEED = 0


@pytest.fixture
def merged():
    """Forty runs in three clusters. Density follows temperature, coverage
    follows temperature and arsenic together, diameter follows nothing.
    Temperature differs between clusters; substrate matches cluster."""
    rng = np.random.default_rng(SEED)
    cluster = np.repeat([1, 2, 3], [14, 13, 13])
    T = 500 + 40 * cluster + rng.normal(0, 8, 40)
    As = rng.normal(1.0, 0.2, 40)
    return pd.DataFrame({
        "Sample_ID": [f"S{i:02d}" for i in range(40)],
        "Cluster": cluster,
        "Density": 0.05 * T + rng.normal(0, 0.5, 40),
        "Coverage": 2 * T - 300 * As + rng.normal(0, 2, 40),
        "Diameter": rng.normal(0.2, 0.05, 40),
        "T": T,
        "As": As,
        "Substrate": np.where(cluster == 3, "GaAs", "Si"),
    })


def test_benjamini_hochberg_matches_hand_calculation():
    q = associations.benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert np.allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.20])
    q = associations.benjamini_hochberg([0.5, np.nan, 0.001])
    assert np.isnan(q[1]) and np.allclose(q[[0, 2]], [0.5, 0.002])


def test_descriptor_vs_parameter_finds_planted_relations(merged):
    t = associations.descriptor_vs_parameter(merged, ["Density", "Coverage", "Diameter"],
                                             ["T", "As"], 0.05)
    assert len(t) == 6
    t = t.set_index(["Descriptor", "Parameter"])
    assert t.loc[("Density", "T"), "Significant"]
    assert t.loc[("Coverage", "As"), "Significant"]
    assert t.loc[("Coverage", "As"), "Spearman_rho"] < 0
    assert not t.loc[("Diameter", "T"), "Significant"]
    assert (t["q_value"] >= t["p_value"]).all()


def test_cluster_vs_numeric_reports_medians_and_significance(merged):
    t = associations.cluster_vs_numeric(merged, ["T", "As"], 0.05).set_index("Parameter")
    assert t.loc["T", "Significant"] and not t.loc["As", "Significant"]
    assert t.loc["T", "Median_Cluster_1"] < t.loc["T", "Median_Cluster_2"] < t.loc["T", "Median_Cluster_3"]
    assert t.loc["T", ["N_Cluster_1", "N_Cluster_2", "N_Cluster_3"]].tolist() == [14, 13, 13]


def test_cluster_vs_numeric_is_nan_when_only_one_cluster_has_values(merged):
    merged.loc[merged["Cluster"] != 1, "As"] = np.nan
    t = associations.cluster_vs_numeric(merged, ["As"], 0.05)
    assert np.isnan(t.loc[0, "p_value"]) and not t.loc[0, "Significant"]


def test_cluster_vs_categorical_is_exact_and_deterministic(merged):
    tests, counts = associations.cluster_vs_categorical(merged, ["Substrate"], 2000, SEED, 0.05)
    again, _ = associations.cluster_vs_categorical(merged, ["Substrate"], 2000, SEED, 0.05)
    pd.testing.assert_frame_equal(tests, again)
    assert tests.loc[0, "Significant"]
    assert tests.loc[0, "p_value"] >= 1 / 2001
    assert counts["Count"].sum() == 40
    c = counts.set_index(["Cluster", "Level"])["Count"]
    assert c[(3, "GaAs")] == 13 and c[(1, "Si")] == 14


def test_cluster_vs_categorical_finds_nothing_in_shuffled_labels(merged):
    merged["Substrate"] = np.random.default_rng(1).permutation(merged["Substrate"])
    tests, _ = associations.cluster_vs_categorical(merged, ["Substrate"], 2000, SEED, 0.05)
    assert not tests.loc[0, "Significant"]


def test_descriptor_regressions_recover_planted_model(merged):
    r = associations.descriptor_regressions(merged, ["Coverage", "Diameter"], ["T", "As"])
    r = r.set_index("Descriptor")
    assert r.loc["Coverage", "R2_LOO"] > 0.9
    assert r.loc["Coverage", "Beta_T"] > 0 and r.loc["Coverage", "Beta_As"] < 0
    assert r.loc["Diameter", "R2_LOO"] < 0.2
    assert (r["R2_Fit"] >= r["R2_LOO"]).all()
