"""Checks on the clustering rules and the encoding tests. Run with ``pytest``
from the method2 folder."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.datasets import make_blobs

from src import clustering, confounds, images

SEED = 0


@pytest.fixture
def embeddings():
    """Three well-separated clusters of 20 samples in a 50-d embedding."""
    X, _ = make_blobs(n_samples=60, centers=3, n_features=50, cluster_std=1.0,
                      random_state=SEED)
    df = pd.DataFrame(X, columns=[f"E{j + 1}" for j in range(50)])
    df.insert(0, "Image_File", [f"S{i:03d}.tif" for i in range(60)])
    df.insert(0, "Sample_ID", [f"S{i:03d}" for i in range(60)])
    return df


@pytest.fixture
def cfg():
    return {"random_seed": SEED,
            "embedding": {"l2_normalize": True, "standard_scale": True},
            "pca": {"variance_threshold": 0.8, "min_components": 2},
            "clustering": {"k_values": [2, 3, 4, 5, 6], "kmeans_n_init": 10},
            "selection": {"min_cluster_size": 2}}


def test_process_embeddings_l2_then_scale(embeddings, cfg):
    cfg_l2 = dict(cfg, embedding={"l2_normalize": True, "standard_scale": False})
    Z, steps = clustering.process_embeddings(embeddings, cfg_l2)
    assert np.allclose(np.linalg.norm(Z.to_numpy(), axis=1), 1.0)
    assert steps == ["L2 normalisation per image"]

    Z, steps = clustering.process_embeddings(embeddings, cfg)
    assert np.allclose(Z.mean(), 0, atol=1e-12)
    assert list(Z.index) == list(embeddings["Sample_ID"])
    assert len(steps) == 2


def test_fit_embeddings_recovers_planted_clusters(embeddings, cfg):
    fit = clustering.fit_embeddings(embeddings, cfg)
    assert fit["final_k"] == 3
    assert not fit["degenerate"]
    row = fit["summary"][fit["summary"]["k"] == 3].iloc[0]
    assert row["Cluster_Sizes"] == "20/20/20"


def test_select_k_rules():
    summary = pd.DataFrame({"k": [2, 3, 4], "Silhouette_KMeans": [0.30, 0.50, 0.40],
                            "Min_Cluster_Size": [10, 1, 5]})
    assert clustering.select_k(summary, 2)[0] == 4

    summary = pd.DataFrame({"k": [2, 3], "Silhouette_KMeans": [0.40, 0.40],
                            "Min_Cluster_Size": [5, 5]})
    assert clustering.select_k(summary, 2)[0] == 2

    summary = pd.DataFrame({"k": [2, 3], "Silhouette_KMeans": [0.20, 0.60],
                            "Min_Cluster_Size": [1, 1]})
    k, _, degenerate = clustering.select_k(summary, 2)
    assert k == 3 and degenerate


def test_per_sample_fit_flags_far_point(embeddings, cfg):
    fit = clustering.fit_embeddings(embeddings, cfg)
    S = fit["scores"].to_numpy().copy()
    labels = fit["km"][3]
    S[0] += 25.0                                          # push one sample far from its centre
    table = clustering.per_sample_fit(S, labels, fit["scores"].index, sd_threshold=2.0)
    assert table.loc[0, "Outlier_Flag"]
    same_cluster = table[table["Cluster"] == table.loc[0, "Cluster"]]
    assert same_cluster["Outlier_Flag"].sum() == 1
    assert table["Distance_To_Cluster_Centre"].idxmax() == 0


def test_outlier_sensitivity_removes_only_named_samples(embeddings, cfg):
    fit = clustering.fit_embeddings(embeddings, cfg)
    table, ari = clustering.outlier_sensitivity(fit["Z"], cfg, 3, fit["km"][3], ["S000", "S001"])
    assert table["Samples_Removed"].iloc[0] == 2
    assert ari > 0.95                                     # removing two points cannot change clear clusters


def test_cross_model_agreement_pairs():
    a = np.array([0, 0, 1, 1, 2, 2])
    table = clustering.cross_model_agreement({"m1": a, "m2": a, "m3": (a + 1) % 3})
    assert len(table) == 3
    assert table.set_index(["Model_A", "Model_B"]).loc[("m1", "m2"), "ARI"] == 1.0
    assert table.set_index(["Model_A", "Model_B"]).loc[("m1", "m3"), "ARI"] == 1.0


def test_zoom_control_ratio():
    rng = np.random.default_rng(SEED)
    factors = [1.0, 2.0]
    zoom_emb = {}
    for i in range(6):
        base = rng.normal(size=8)
        base /= np.linalg.norm(base)
        zoomed = base + 0.1 * rng.normal(size=8)
        zoomed /= np.linalg.norm(zoomed)
        zoom_emb[f"S{i}"] = np.vstack([base, zoomed])
    per_image, summary, info = confounds.zoom_control(zoom_emb, factors)
    assert len(per_image) == 6
    assert summary["Zoom_Factor"].tolist() == [2.0]
    expected = summary["Mean_Zoom_Distance"].iloc[0] / summary["Mean_Between_Sample_Distance"].iloc[0]
    assert summary["Ratio_Zoom_To_Between_Sample"].iloc[0] == pytest.approx(expected, abs=1e-3)
    assert info["n_images"] == 6


def test_encoding_regression_recovers_linear_target(embeddings):
    X = embeddings.drop(columns=["Sample_ID", "Image_File"]).to_numpy()
    y = pd.Series(X[:, 0] * 2 + X[:, 1], index=embeddings["Sample_ID"])
    noise = pd.Series(np.random.default_rng(SEED).normal(size=60), index=embeddings["Sample_ID"])
    table = confounds.encoding_regression(embeddings, {"signal": y, "noise": noise}, 20, SEED)
    r2 = table.set_index("Target")["LOO_R2"]
    assert r2["signal"] > 0.9
    assert r2["noise"] < 0.3


def test_summarise_encoding():
    table = pd.DataFrame({"Target": ["log10(Magnification)", "a", "b"],
                          "LOO_R2": [0.6, 0.5, -0.1]})
    s = confounds.summarise_encoding(table, ["a", "b"])
    assert s["magnification_r2"] == 0.6
    assert s["best_descriptor_r2"] == 0.5
    assert s["uninformative"] == ["b"]


def test_crop_and_zoom_preserve_expected_sizes():
    im = Image.new("RGB", (200, 100))
    assert images.crop_bottom(im, 0.0).size == (200, 100)
    assert images.crop_bottom(im, 0.1).size == (200, 90)
    assert images.simulate_zoom(im, 2.0).size == (200, 100)
    assert images.simulate_zoom(im, 1.0) is im


def test_load_image_rgb_handles_16bit(tmp_path):
    arr = (np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) * 100)
    path = tmp_path / "img.tif"
    Image.fromarray(arr).save(path)
    im = images.load_image_rgb(path)
    assert im.mode == "RGB" and im.size == (64, 48)
    assert np.array(im).max() == 255
