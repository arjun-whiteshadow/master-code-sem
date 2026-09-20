"""Checks on sample placement. Run with ``pytest`` from the growth folder."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import predict

SEED = 0


@pytest.fixture
def fitted():
    """Three well-separated clusters, with the model written the way
    Method 1 writes it: scaler, two retained components, member-mean
    centres and the furthest member of each cluster."""
    A, _ = make_blobs(n_samples=60, centers=3, n_features=5, cluster_std=0.5,
                      random_state=SEED)
    X = pd.DataFrame(A, index=[f"S{i:03d}" for i in range(60)],
                     columns=[f"f{j}" for j in range(5)])
    scaler = StandardScaler().fit(A)
    pca = PCA(n_components=2, random_state=SEED).fit(scaler.transform(A))
    S = pca.transform(scaler.transform(A))
    labels = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(S).labels_
    centres = np.vstack([S[labels == c].mean(axis=0) for c in range(3)])
    dist = np.linalg.norm(S - centres[labels], axis=1)
    model = {
        "features": list(X.columns),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "pca_components": ["PC1", "PC2"],
        "pca_loadings": pca.components_.T.tolist(),
        "k": 3,
        "cluster_centres": centres.tolist(),
        "max_member_distance": [float(dist[labels == c].max()) for c in range(3)],
    }
    return X, pd.Series(labels + 1, index=X.index), model


def test_assign_reproduces_training_labels_within_range(fitted):
    X, clusters, model = fitted
    out = predict.assign(model, X)
    assert out["Cluster"].tolist() == clusters.tolist()
    assert out["Within_Range"].all()
    assert (out["Distance_To_Centre"] <= out["Cluster_Max_Member_Distance"]).all()


def test_assign_flags_a_sample_far_from_every_cluster(fitted):
    X, _, model = fitted
    far = pd.DataFrame([X.iloc[0] + 50], index=["far"])
    out = predict.assign(model, far)
    assert not out.loc[0, "Within_Range"]


def test_assign_uses_the_model_feature_order(fitted):
    X, clusters, model = fitted
    shuffled = X[list(reversed(X.columns))]
    assert predict.assign(model, shuffled)["Cluster"].tolist() == clusters.tolist()


def test_leave_one_out_is_perfect_on_separated_clusters(fitted):
    X, clusters, _ = fitted
    loo = predict.leave_one_out(X, clusters, n_components=2, seed=SEED)
    assert len(loo) == 60
    assert loo["Correct"].all()
    assert loo["Predicted"].tolist() == loo["Cluster"].tolist()
