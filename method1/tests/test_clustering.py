"""Checks on the clustering rules. Run with ``pytest`` from the method1 folder."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src import clustering

SEED = 0


@pytest.fixture
def blobs():
    """Three well-separated clusters of 20 points in four dimensions."""
    X, _ = make_blobs(n_samples=60, centers=3, n_features=4, cluster_std=0.5,
                      random_state=SEED)
    return pd.DataFrame(X, index=[f"S{i:03d}" for i in range(60)],
                        columns=[f"f{j}" for j in range(4)])


def test_standardize_gives_zero_mean_unit_variance(blobs):
    Z = clustering.standardize(blobs)
    assert np.allclose(Z.mean(), 0, atol=1e-12)
    assert np.allclose(Z.std(ddof=0), 1, atol=1e-12)
    assert list(Z.index) == list(blobs.index)


def test_flag_outliers_finds_planted_value(blobs):
    Z = clustering.standardize(blobs)
    Z.loc["S005", "f1"] = 4.5
    flagged = clustering.flag_outliers(Z, threshold=3.0)
    assert len(flagged) == 1
    assert flagged.iloc[0]["Sample_ID"] == "S005"
    assert flagged.iloc[0]["Feature"] == "f1"


def test_run_pca_retains_enough_variance(blobs):
    Z = clustering.standardize(blobs)
    scores, variance, loadings = clustering.run_pca(Z, 0.80, 2, SEED)
    n_keep = scores.shape[1]
    assert variance["Cumulative_Variance_Percent"].iloc[n_keep - 1] >= 80
    if n_keep > 2:
        assert variance["Cumulative_Variance_Percent"].iloc[n_keep - 2] < 80
    assert int(variance["Retained_For_Clustering"].sum()) == n_keep
    assert loadings.shape == (4, n_keep)


def test_run_pca_respects_minimum_components(blobs):
    Z = clustering.standardize(blobs)
    scores, _, _ = clustering.run_pca(Z, 0.01, 3, SEED)
    assert scores.shape[1] == 3


def test_cluster_over_k_drops_invalid_k():
    S = np.random.default_rng(SEED).normal(size=(5, 2))
    summary, km, ward = clustering.cluster_over_k(S, [1, 2, 3, 4, 5, 6], 5, SEED)
    assert summary["k"].tolist() == [2, 3, 4]
    assert set(km) == set(ward) == {2, 3, 4}


def test_cluster_over_k_recovers_planted_clusters(blobs):
    S = clustering.standardize(blobs).to_numpy()
    summary, km, _ = clustering.cluster_over_k(S, [2, 3, 4], 10, SEED)
    assert summary.loc[summary["k"] == 3, "Cluster_Sizes"].iloc[0] == "20/20/20"
    assert summary["Silhouette_KMeans"].idxmax() == 1        # row for k = 3


def test_select_k_excludes_undersized_clusters():
    summary = pd.DataFrame({"k": [2, 3, 4],
                            "Silhouette_KMeans": [0.30, 0.50, 0.40],
                            "Min_Cluster_Size": [10, 1, 5]})
    k, text, degenerate = clustering.select_k(summary, min_cluster_size=2)
    assert k == 4 and not degenerate
    assert "k = [3]" in text


def test_select_k_breaks_ties_toward_smaller_k():
    summary = pd.DataFrame({"k": [2, 3, 4],
                            "Silhouette_KMeans": [0.40, 0.40, 0.35],
                            "Min_Cluster_Size": [5, 5, 5]})
    k, text, _ = clustering.select_k(summary, min_cluster_size=2)
    assert k == 2
    assert "Tie" in text


def test_select_k_flags_degenerate_case():
    summary = pd.DataFrame({"k": [2, 3],
                            "Silhouette_KMeans": [0.20, 0.60],
                            "Min_Cluster_Size": [1, 1]})
    k, _, degenerate = clustering.select_k(summary, min_cluster_size=2)
    assert k == 3 and degenerate


def test_subsample_stability_is_deterministic_and_high_for_clear_clusters(blobs):
    S = clustering.standardize(blobs).to_numpy()
    _, km, _ = clustering.cluster_over_k(S, [2, 3], 10, SEED)
    a = clustering.subsample_stability(S, km, 20, 0.8, 5, SEED)
    b = clustering.subsample_stability(S, km, 20, 0.8, 5, SEED)
    pd.testing.assert_frame_equal(a, b)
    assert a["k"].tolist() == [2, 3]
    assert a.loc[a["k"] == 3, "Mean_Subsample_ARI"].iloc[0] > 0.95


def test_dbscan_grid_has_one_row_per_setting(blobs):
    S = clustering.standardize(blobs).to_numpy()
    grid = clustering.dbscan_grid(S, [0.5, 1.0, 2.0], [2, 3])
    assert len(grid) == 6
    assert (grid["Num_Noise"] + grid["Num_Clusters"] >= 0).all()


def test_per_sample_fit_marks_negative_silhouette(blobs):
    S = clustering.standardize(blobs).to_numpy()
    _, km, _ = clustering.cluster_over_k(S, [3], 10, SEED)
    labels = km[3].copy()
    labels[0] = (labels[0] + 1) % 3                     # move one sample to the wrong cluster
    fit = clustering.per_sample_fit(S, labels, blobs.index)
    assert fit.loc[0, "Negative_Silhouette"]
    assert fit["Negative_Silhouette"].sum() == 1


def test_fitted_model_reproduces_scores_and_labels(blobs):
    Z = clustering.standardize(blobs)
    scores, _, loadings = clustering.run_pca(Z, 0.8, 2, SEED)
    S = scores.to_numpy()
    _, km, _ = clustering.cluster_over_k(S, [3], 10, SEED)
    labels = km[3]
    m = clustering.fitted_model(blobs, Z, loadings, S, labels)

    X = blobs[m["features"]].to_numpy()
    Z2 = (X - np.array(m["scaler_mean"])) / np.array(m["scaler_scale"])
    S2 = (Z2 - np.array(m["pca_mean"])) @ np.array(m["pca_loadings"])
    assert np.allclose(S2, S)

    centres = np.array(m["cluster_centres"])
    d = np.linalg.norm(S2[:, None, :] - centres[None, :, :], axis=2)
    assert (d.argmin(axis=1) == labels).all()
    assert len(m["max_member_distance"]) == m["k"] == 3
    assert all(d[labels == c, c].max() <= m["max_member_distance"][c] + 1e-12
               for c in range(3))


def test_leave_one_feature_out_returns_one_row_per_feature(blobs):
    Z = clustering.standardize(blobs)
    cfg = {"random_seed": SEED, "pca": {"variance_threshold": 0.8, "min_components": 2},
           "clustering": {"kmeans_n_init": 10}}
    _, km, _ = clustering.cluster_over_k(Z.to_numpy(), [3], 10, SEED)
    lofo = clustering.leave_one_feature_out(Z, cfg, 3, km[3])
    assert lofo["Feature_Removed"].tolist() == list(Z.columns)
    assert lofo["ARI_vs_Main_Result"].between(-1, 1).all()
