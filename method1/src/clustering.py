"""Standardisation, PCA, clustering over k, selection of k and the
stability tests. Method 2 uses the same functions on its embeddings."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


def standardize(X):
    """Z-score each column. Returns a DataFrame with the original index."""
    Z = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    return pd.DataFrame(Z, index=X.index, columns=X.columns)


def flag_outliers(Z, threshold):
    """List every standardised value with |z| above the threshold."""
    rows = []
    for sid, row in Z.iterrows():
        for feat, val in row.items():
            if abs(val) > threshold:
                rows.append(dict(Sample_ID=sid, Feature=feat,
                                 Standardized_Value=round(float(val), 3)))
    return pd.DataFrame(rows, columns=["Sample_ID", "Feature", "Standardized_Value"])


def run_pca(Z, variance_threshold, min_components, seed):
    """PCA keeping the fewest components that reach the variance threshold.

    Returns
    -------
    scores : DataFrame
        Retained component scores, indexed like ``Z``.
    variance : DataFrame
        Explained and cumulative variance for every component, with a flag
        for the retained ones.
    loadings : DataFrame
        Feature loadings on the retained components.
    """
    n_max = min(Z.shape[0] - 1, Z.shape[1])
    pca = PCA(n_components=n_max, random_state=seed)
    scores_all = pca.fit_transform(Z.to_numpy())
    cum = np.cumsum(pca.explained_variance_ratio_)

    n_keep = int(np.searchsorted(cum, variance_threshold) + 1)
    n_keep = max(min_components, min(n_keep, n_max))

    names = [f"PC{i + 1}" for i in range(n_max)]
    variance = pd.DataFrame({
        "Component": names,
        "Explained_Variance_Percent": np.round(pca.explained_variance_ratio_ * 100, 4),
        "Cumulative_Variance_Percent": np.round(cum * 100, 4),
        "Retained_For_Clustering": [i < n_keep for i in range(n_max)],
    })
    scores = pd.DataFrame(scores_all[:, :n_keep], index=Z.index, columns=names[:n_keep])
    loadings = pd.DataFrame(pca.components_[:n_keep].T, index=Z.columns,
                            columns=names[:n_keep])
    return scores, variance, loadings


def cluster_over_k(S, k_values, n_init, seed):
    """k-means and Ward clustering at each k.

    Returns a summary table (one row per k) and two dicts of label arrays,
    ``km_labels[k]`` and ``ward_labels[k]``. Values of k that are not valid
    for the number of samples are dropped.
    """
    n = S.shape[0]
    valid = [k for k in k_values if 2 <= k <= n - 1]
    if not valid:
        raise SystemExit(f"No valid k for n = {n} samples.")

    km_labels, ward_labels, rows = {}, {}, []
    for k in valid:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(S)
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(S)
        km_labels[k], ward_labels[k] = km.labels_, ward.labels_
        sizes = np.bincount(km.labels_, minlength=k)
        rows.append({
            "k": k,
            "Silhouette_KMeans": round(float(silhouette_score(S, km.labels_)), 4),
            "Silhouette_Ward": round(float(silhouette_score(S, ward.labels_)), 4),
            "Inertia": round(float(km.inertia_), 4),
            "Cluster_Sizes": "/".join(str(s) for s in sizes),
            "Min_Cluster_Size": int(sizes.min()),
            "Max_Cluster_Size": int(sizes.max()),
            "CrossAlgorithm_ARI": round(float(adjusted_rand_score(km.labels_, ward.labels_)), 4),
        })
    return pd.DataFrame(rows), km_labels, ward_labels


def select_k(summary, min_cluster_size):
    """Highest k-means silhouette among solutions with no cluster smaller
    than ``min_cluster_size``; ties go to the smaller k.

    Returns (k, explanation, degenerate). ``degenerate`` is True when every
    solution had an undersized cluster and the rule fell back to the best
    silhouette overall.
    """
    eligible = summary[summary["Min_Cluster_Size"] >= min_cluster_size]
    degenerate = eligible.empty
    pool = summary if degenerate else eligible

    best = pool["Silhouette_KMeans"].max()
    tied = pool[np.isclose(pool["Silhouette_KMeans"], best)]
    k = int(tied["k"].min())

    if degenerate:
        text = (f"Every k gave at least one cluster with fewer than {min_cluster_size} "
                f"samples. Fell back to the highest silhouette overall ({best:.4f}) "
                f"at k = {k}; result flagged as degenerate.")
    else:
        text = (f"Highest silhouette among solutions with no cluster smaller than "
                f"{min_cluster_size}: k = {k} (silhouette {best:.4f}).")
        excluded = summary.loc[summary["Min_Cluster_Size"] < min_cluster_size, "k"].tolist()
        if excluded:
            text += f" Excluded for an undersized cluster: k = {excluded}."
        if len(tied) > 1:
            text += f" Tie between k = {tied['k'].tolist()} resolved to the smaller k."
    return k, text, degenerate


def dbscan_grid(S, eps_values, min_samples_values):
    """DBSCAN over a parameter grid. Silhouette is computed on non-noise
    points only and is NaN when fewer than two clusters are found."""
    rows, n = [], S.shape[0]
    for eps in eps_values:
        for ms in min_samples_values:
            lab = DBSCAN(eps=float(eps), min_samples=int(ms)).fit_predict(S)
            core = lab != -1
            n_clusters = len(set(lab[core]))
            n_noise = int((~core).sum())
            sil = np.nan
            if n_clusters >= 2 and core.sum() > n_clusters:
                sil = round(float(silhouette_score(S[core], lab[core])), 4)
            rows.append(dict(eps=float(eps), min_samples=int(ms),
                             Num_Clusters=n_clusters, Num_Noise=n_noise,
                             Noise_Percent=round(100.0 * n_noise / n, 1),
                             Silhouette_CorePoints=sil))
    return pd.DataFrame(rows)


def dbscan_representative(grid, S):
    """One DBSCAN solution to display: most clusters, then least noise, then
    best silhouette. Returns (grid row, labels)."""
    usable = grid[grid["Num_Clusters"] >= 2]
    if usable.empty:
        row = grid.sort_values("Num_Noise").iloc[0]
    else:
        row = usable.sort_values(["Num_Clusters", "Num_Noise", "Silhouette_CorePoints"],
                                 ascending=[False, True, False]).iloc[0]
    labels = DBSCAN(eps=float(row["eps"]), min_samples=int(row["min_samples"])).fit_predict(S)
    return row, labels


def subsample_stability(S, km_labels, n_iter, fraction, n_init, seed):
    """Re-cluster random subsets and compare with the full-data partition.

    For each k, ``n_iter`` subsets of ``fraction`` of the samples are drawn
    without replacement and clustered; the ARI against the full-data labels
    on the same samples is recorded. Returns mean and SD per k.
    """
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    m = max(3, int(round(n * fraction)))
    rows = []
    for k, full in km_labels.items():
        if k > m - 1:
            rows.append(dict(k=k, Mean_Subsample_ARI=np.nan,
                             SD_Subsample_ARI=np.nan, N_Iterations=0))
            continue
        aris = []
        for i in range(n_iter):
            idx = rng.choice(n, size=m, replace=False)
            sub = KMeans(n_clusters=k, n_init=n_init,
                         random_state=seed + i).fit_predict(S[idx])
            aris.append(adjusted_rand_score(full[idx], sub))
        rows.append(dict(k=k, Mean_Subsample_ARI=round(float(np.mean(aris)), 4),
                         SD_Subsample_ARI=round(float(np.std(aris)), 4),
                         N_Iterations=n_iter))
    return pd.DataFrame(rows)


def leave_one_feature_out(Z, cfg, k, reference):
    """Repeat PCA and k-means with each descriptor removed in turn."""
    seed = int(cfg["random_seed"])
    rows = []
    for feat in Z.columns:
        scores, _, _ = run_pca(Z.drop(columns=[feat]),
                               float(cfg["pca"]["variance_threshold"]),
                               int(cfg["pca"]["min_components"]), seed)
        lab = KMeans(n_clusters=k, n_init=int(cfg["clustering"]["kmeans_n_init"]),
                     random_state=seed).fit_predict(scores.to_numpy())
        rows.append(dict(Feature_Removed=feat, Retained_PCs=scores.shape[1],
                         ARI_vs_Main_Result=round(float(adjusted_rand_score(reference, lab)), 4)))
    return pd.DataFrame(rows)


def outlier_sensitivity(Z, cfg, k, reference, flagged_ids):
    """Repeat PCA and k-means without the flagged samples and compare."""
    flagged = sorted(set(flagged_ids))
    if not flagged:
        return pd.DataFrame([dict(Samples_Removed=0, ARI_vs_Main_Result=np.nan,
                                  Note="no sample exceeded the threshold")]), None

    seed = int(cfg["random_seed"])
    keep = [s for s in Z.index if s not in flagged]
    scores, _, _ = run_pca(Z.loc[keep], float(cfg["pca"]["variance_threshold"]),
                           int(cfg["pca"]["min_components"]), seed)
    k_used = min(k, len(keep) - 1)
    lab = KMeans(n_clusters=k_used, n_init=int(cfg["clustering"]["kmeans_n_init"]),
                 random_state=seed).fit_predict(scores.to_numpy())
    ref = pd.Series(reference, index=Z.index).loc[keep].to_numpy()
    ari = float(adjusted_rand_score(ref, lab))
    table = pd.DataFrame([dict(Samples_Removed=len(flagged),
                               Removed_Sample_IDs="; ".join(flagged),
                               k_used=k_used, ARI_vs_Main_Result=round(ari, 4),
                               Note="removed for this check only")])
    return table, ari


def cluster_profiles(X, Z, labels):
    """Per-cluster summary statistics in original units and mean z-scores."""
    raw = X.copy()
    raw["Cluster"] = labels + 1
    prof_raw = raw.groupby("Cluster").agg(["mean", "median", "std", "min", "max"])
    prof_raw.columns = ["_".join(c) for c in prof_raw.columns]

    std = Z.copy()
    std["Cluster"] = labels + 1
    prof_std = std.groupby("Cluster").mean()
    return prof_raw.reset_index(), prof_std.reset_index()


def per_sample_fit(S, labels, sample_ids):
    """Silhouette and distance to own cluster centre for each sample."""
    sil = silhouette_samples(S, labels)
    centres = np.vstack([S[labels == c].mean(axis=0) for c in np.unique(labels)])
    dist = np.linalg.norm(S - centres[labels], axis=1)
    return pd.DataFrame({
        "Sample_ID": list(sample_ids),
        "Cluster": labels + 1,
        "Silhouette": np.round(sil, 4),
        "Distance_To_Cluster_Centre": np.round(dist, 4),
        "Negative_Silhouette": sil < 0,
    })


def ward_linkage(S):
    return linkage(S, method="ward", metric="euclidean")
