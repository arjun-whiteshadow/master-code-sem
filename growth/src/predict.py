"""Placing a sample against the Method 1 clusters from its descriptors.

A sample is standardised with the saved scaler, projected with the saved
PCA loadings and assigned to the nearest cluster centre. No refitting takes
place, so the assignment refers to exactly the published clusters.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def project(model, X):
    """Scores of the rows of ``X`` in the model's retained PCA space."""
    A = X[model["features"]].to_numpy(dtype=float)
    Z = (A - np.array(model["scaler_mean"])) / np.array(model["scaler_scale"])
    return (Z - np.array(model["pca_mean"])) @ np.array(model["pca_loadings"])


def assign(model, X):
    """Nearest-centre cluster for each row of ``X``.

    ``Within_Range`` is True when the sample is no further from its centre
    than the furthest member of that cluster in the run that produced the
    model; a sample outside every range is reported but not trusted.
    """
    S = project(model, X)
    centres = np.array(model["cluster_centres"])
    d = np.linalg.norm(S[:, None, :] - centres[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    limit = np.array(model["max_member_distance"])[nearest]
    dist = d[np.arange(len(S)), nearest]
    return pd.DataFrame({
        "Sample_ID": list(X.index),
        "Cluster": nearest + 1,
        "Distance_To_Centre": np.round(dist, 4),
        "Cluster_Max_Member_Distance": np.round(limit, 4),
        "Within_Range": dist <= limit,
    })


def leave_one_out(X, clusters, n_components, seed):
    """Assignment of each training sample by a model fitted without it.

    The scaler, PCA and centres are refitted on the other samples with the
    cluster labels held fixed; the held-out sample is then assigned by
    nearest centre. This measures how far nearest-centre placement
    generalises, not whether k-means would find the same clusters again.
    """
    ids = list(X.index)
    A = X.to_numpy(dtype=float)
    lab = clusters.reindex(X.index).to_numpy()
    rows = []
    for i in range(len(ids)):
        keep = np.arange(len(ids)) != i
        mean, scale = A[keep].mean(axis=0), A[keep].std(axis=0)
        Z = (A - mean) / scale
        pca = PCA(n_components=n_components, random_state=seed).fit(Z[keep])
        S = pca.transform(Z)
        centres = np.vstack([S[keep][lab[keep] == c].mean(axis=0) for c in np.unique(lab)])
        pred = np.unique(lab)[np.linalg.norm(S[i] - centres, axis=1).argmin()]
        rows.append(dict(Sample_ID=ids[i], Cluster=int(lab[i]), Predicted=int(pred),
                         Correct=bool(pred == lab[i])))
    return pd.DataFrame(rows)
