"""Embedding normalisation, PCA, clustering over k, selection of k and the
stability tests.

The PCA, clustering, k-selection and subsample functions are the same as in
Method 1 so that the two methods are treated identically.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize


def process_embeddings(emb, cfg):
    """L2-normalise each embedding, then standardise each dimension.

    Returns a DataFrame indexed by Sample_ID and a list of the steps applied.
    """
    meta = [c for c in ("Sample_ID", "Image_File") if c in emb.columns]
    X = emb.drop(columns=meta).to_numpy(dtype=float)
    steps = []
    if cfg["embedding"].get("l2_normalize", True):
        X = normalize(X, norm="l2", axis=1)
        steps.append("L2 normalisation per image")
    if cfg["embedding"].get("standard_scale", True):
        X = StandardScaler().fit_transform(X)
        steps.append("standardisation per dimension")
    Z = pd.DataFrame(X, index=emb["Sample_ID"].to_numpy(),
                     columns=[f"E{i + 1}" for i in range(X.shape[1])])
    return Z, steps or ["none"]


def run_pca(Z, variance_threshold, min_components, seed):
    """PCA keeping the fewest components that reach the variance threshold."""
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
    return scores, variance


def cluster_over_k(S, k_values, n_init, seed):
    """k-means and Ward clustering at each k. Returns (summary, km_labels, ward_labels)."""
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

    Returns (k, explanation, degenerate).
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


def fit_embeddings(emb, cfg):
    """Normalise, reduce, cluster and select k for one model's embeddings.

    Returns a dict with Z, steps, scores, variance, summary, km, ward,
    final_k, reason and degenerate.
    """
    seed = int(cfg["random_seed"])
    Z, steps = process_embeddings(emb, cfg)
    scores, variance = run_pca(Z, float(cfg["pca"]["variance_threshold"]),
                               int(cfg["pca"]["min_components"]), seed)
    summary, km, ward = cluster_over_k(scores.to_numpy(), list(cfg["clustering"]["k_values"]),
                                       int(cfg["clustering"]["kmeans_n_init"]), seed)
    k, reason, degenerate = select_k(summary, int(cfg["selection"]["min_cluster_size"]))
    return dict(Z=Z, steps=steps, scores=scores, variance=variance, summary=summary,
                km=km, ward=ward, final_k=k, reason=reason, degenerate=degenerate)


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
    """Re-cluster random subsets and compare with the full-data partition
    by ARI. Returns mean and SD per k."""
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


def per_sample_fit(S, labels, sample_ids, sd_threshold):
    """Silhouette and distance to own cluster centre for each sample.
    Samples further than mean + ``sd_threshold`` SD from their centre,
    within their cluster, are flagged."""
    sil = silhouette_samples(S, labels)
    centres = np.vstack([S[labels == c].mean(axis=0) for c in np.unique(labels)])
    dist = np.linalg.norm(S - centres[labels], axis=1)
    df = pd.DataFrame({"Sample_ID": list(sample_ids), "Cluster": labels + 1,
                       "Silhouette": np.round(sil, 4),
                       "Distance_To_Cluster_Centre": np.round(dist, 4)})
    flagged = []
    for _, grp in df.groupby("Cluster"):
        d = grp["Distance_To_Cluster_Centre"]
        sd = d.std(ddof=0)
        cut = d.mean() + sd_threshold * sd if sd > 0 else np.inf
        flagged += grp.index[d > cut].tolist()
    df["Outlier_Flag"] = df.index.isin(flagged)
    df["Negative_Silhouette"] = df["Silhouette"] < 0
    return df


def outlier_sensitivity(Z, cfg, k, reference, flagged_ids):
    """Repeat PCA and k-means without the flagged samples and compare."""
    flagged = sorted(set(flagged_ids))
    if not flagged:
        return pd.DataFrame([dict(Samples_Removed=0, ARI_vs_Main_Result=np.nan,
                                  Note="no sample was flagged")]), None

    seed = int(cfg["random_seed"])
    keep = [s for s in Z.index if s not in flagged]
    scores, _ = run_pca(Z.loc[keep], float(cfg["pca"]["variance_threshold"]),
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


def cross_model_agreement(labels_by_model):
    """Pairwise ARI between the final partitions of different models."""
    names = list(labels_by_model)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ari = adjusted_rand_score(labels_by_model[a], labels_by_model[b])
            rows.append(dict(Model_A=a, Model_B=b, ARI=round(float(ari), 4)))
    return pd.DataFrame(rows)


def ward_linkage(S):
    return linkage(S, method="ward", metric="euclidean")
