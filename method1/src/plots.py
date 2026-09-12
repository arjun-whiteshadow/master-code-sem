"""Figures for Method 1. Every function saves one PNG and closes the figure."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

PALETTE = ["#065A82", "#B03A2E", "#1E8449", "#8E44AD", "#D68910", "#17A2B8"]

# Scatter plots label each point with its sample ID up to this many samples.
ANNOTATE_UP_TO = 60

# Axis labels for the descriptor columns.
LABELS = {
    "Density_NW_um2": "Density (NW/µm²)",
    "Mean_Diameter_um": "Mean diameter (µm)",
    "CV_percent": "Diameter CV (%)",
    "Merged_percent": "Merged (%)",
    "Foreground_Coverage_percent": "Coverage (%)",
    "Angle_STD_deg": "Angle SD (°)",
}


def _label(col):
    return LABELS.get(col, col.replace("_", " "))


def _save(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _colour(i):
    """Colour for cluster index ``i``; falls back to tab20 beyond the palette."""
    if i < len(PALETTE):
        return PALETTE[i]
    return plt.get_cmap("tab20")(i % 20)


def _annotate(ax, xs, ys, labels, fontsize=6):
    if len(labels) > ANNOTATE_UP_TO:
        return
    offsets = [(4, 4), (4, -9), (-4, 4), (-4, -9)]
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        ax.annotate(str(lab), (x, y), textcoords="offset points",
                    xytext=offsets[i % 4], fontsize=fontsize, alpha=0.85)


def correlation_heatmap(X, path, dpi):
    corr = X.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(8, 6.8))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), [_label(c) for c in corr.columns], rotation=45,
                  ha="right", fontsize=9)
    ax.set_yticks(range(n), [_label(c) for c in corr.index], fontsize=9)
    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else "black")
    ax.set_title("Pearson correlation between descriptors")
    fig.colorbar(im, ax=ax, label="r")
    _save(fig, path, dpi)
    return corr


def pca_variance(variance, threshold, out_dir, dpi):
    n_keep = int(variance["Retained_For_Clustering"].sum())
    x = np.arange(1, len(variance) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x, variance["Explained_Variance_Percent"], color=PALETTE[0], width=0.6)
    ax.axvline(n_keep + 0.5, ls="--", color="grey")
    ax.set_xticks(x, variance["Component"])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"Scree plot; {n_keep} components retained")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, out_dir / "PCA_Scree_Plot.png", dpi)

    cum_kept = variance["Cumulative_Variance_Percent"].iloc[n_keep - 1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, variance["Cumulative_Variance_Percent"], "o-", color=PALETTE[0], lw=2)
    ax.axhline(threshold * 100, ls="--", color=PALETTE[1], label=f"{threshold:.0%} threshold")
    ax.axvline(n_keep, ls="--", color="grey", label=f"{n_keep} retained ({cum_kept:.1f}%)")
    ax.set_xticks(x, variance["Component"])
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, out_dir / "PCA_Cumulative_Variance.png", dpi)


def pca_loadings(loadings, path, dpi):
    n = loadings.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.6), squeeze=False)
    for ax, pc in zip(axes[0], loadings.columns):
        vals = loadings[pc].sort_values()
        colours = [PALETTE[1] if v < 0 else PALETTE[0] for v in vals]
        ax.barh(range(len(vals)), vals.to_numpy(), color=colours)
        ax.set_yticks(range(len(vals)), [_label(c) for c in vals.index], fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(pc, fontsize=11)
        ax.set_xlabel("Loading")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
    fig.suptitle("PCA loadings", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path, dpi)


def k_selection(summary, final_k, thresholds, path, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ks = summary["k"].tolist()

    ax = axes[0]
    colours = [PALETTE[0] if k == final_k else "#9FB6C4" for k in ks]
    bars = ax.bar(summary["k"].astype(str), summary["Silhouette_KMeans"],
                  color=colours, width=0.6)
    for b, v, s in zip(bars, summary["Silhouette_KMeans"], summary["Cluster_Sizes"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.008, s, ha="center",
                fontsize=7, rotation=90, color="white")
    ax.axhline(thresholds["weak_silhouette"], ls="--", color=PALETTE[1], lw=1,
               label=f"weak below {thresholds['weak_silhouette']}")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean silhouette (k-means)")
    ax.set_title(f"Silhouette by k; selected k = {final_k}")
    ax.set_ylim(0, summary["Silhouette_KMeans"].max() * 1.42)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(ks, summary["Inertia"], "o-", color=PALETTE[0], lw=2)
    ax.set_xticks(ks)
    ax.set_xlabel("k")
    ax.set_ylabel("Within-cluster sum of squares")
    ax.set_title("Elbow plot")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[2]
    ax.plot(ks, summary["CrossAlgorithm_ARI"], "o-", color=PALETTE[2], lw=2,
            label="k-means vs Ward")
    ax.axhline(thresholds["weak_cross_algorithm_ari"], ls="--", color=PALETTE[1], lw=1,
               label=f"weak below {thresholds['weak_cross_algorithm_ari']}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(ks)
    ax.set_xlabel("k")
    ax.set_ylabel("Adjusted Rand index")
    ax.set_title("Cross-algorithm agreement")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, path, dpi)


def pca_scatter(scores, labels, variance, title, path, dpi, noise_label=None):
    """PC1 vs PC2 scatter. ``labels`` may be None (no colouring), 0-based
    cluster labels, or DBSCAN labels with ``noise_label`` marking noise."""
    x, y = scores["PC1"].to_numpy(), scores["PC2"].to_numpy()
    v1, v2 = variance["Explained_Variance_Percent"].iloc[:2]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    if labels is None:
        ax.scatter(x, y, s=70, color=PALETTE[0], edgecolors="k", linewidths=0.4)
    else:
        for j, c in enumerate(sorted(set(labels))):
            m = labels == c
            if noise_label is not None and c == noise_label:
                ax.scatter(x[m], y[m], s=70, c="lightgrey", marker="x", label="noise")
            else:
                name = c if noise_label is not None else c + 1
                ax.scatter(x[m], y[m], s=70, color=_colour(j),
                           edgecolors="k", linewidths=0.4, label=f"Cluster {name}")
        ax.legend(fontsize=9)
    _annotate(ax, x, y, scores.index)
    ax.set_xlabel(f"PC1 ({v1:.1f}%)")
    ax.set_ylabel(f"PC2 ({v2:.1f}%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, path, dpi)


def dendrogram_plot(link, sample_ids, final_k, path, dpi):
    fig, ax = plt.subplots(figsize=(max(10, len(sample_ids) * 0.36), 6.5))
    dendrogram(link, labels=list(sample_ids), ax=ax, leaf_rotation=90,
               leaf_font_size=8, color_threshold=link[-(final_k - 1), 2])
    ax.set_ylabel("Ward linkage distance")
    ax.set_title(f"Ward dendrogram, cut at k = {final_k}")
    _save(fig, path, dpi)


def profile_heatmap(prof_std, path, dpi):
    data = prof_std.set_index("Cluster")
    vmax = float(np.abs(data.to_numpy()).max())
    fig, ax = plt.subplots(figsize=(1.5 * len(data.columns) + 2, 1.1 * len(data) + 2.4))
    im = ax.imshow(data.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(data.columns)), [_label(c) for c in data.columns],
                  rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(data)), [f"Cluster {c}" for c in data.index], fontsize=10)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.iat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > vmax * 0.55 else "black")
    ax.set_title("Cluster profiles (mean z-score per descriptor)")
    fig.colorbar(im, ax=ax, label="Mean z-score")
    _save(fig, path, dpi)


def stability(stab, summary, final_k, thresholds, path, dpi):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.errorbar(stab["k"], stab["Mean_Subsample_ARI"], yerr=stab["SD_Subsample_ARI"],
                marker="o", lw=2, capsize=4, color=PALETTE[0],
                label="Subsample stability (mean ± SD)")
    ax.plot(summary["k"], summary["CrossAlgorithm_ARI"], "s--", color=PALETTE[2],
            lw=1.8, label="k-means vs Ward")
    ax.axhline(thresholds["weak_subsample_ari"], ls=":", color=PALETTE[1],
               label=f"weak below {thresholds['weak_subsample_ari']}")
    ax.axvline(final_k, ls="--", color="grey", alpha=0.7)
    ax.text(final_k, 1.02, f"k = {final_k}", ha="center", fontsize=9, color="grey")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("k")
    ax.set_ylabel("Adjusted Rand index")
    ax.set_title("Cluster stability")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, path, dpi)


def sample_fit(fit, path, dpi):
    q = fit.sort_values(["Cluster", "Silhouette"])
    fig, ax = plt.subplots(figsize=(9, max(5, 0.24 * len(q))))
    colours = [_colour(c - 1) for c in q["Cluster"]]
    ax.barh(range(len(q)), q["Silhouette"], color=colours)
    ax.set_yticks(range(len(q)), q["Sample_ID"], fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Silhouette")
    ax.set_title("Per-sample silhouette")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, path, dpi)


def cluster_vs_acquisition(fit, acquisition, path, dpi):
    """Wire count and, when available, magnification against cluster."""
    df = fit.merge(acquisition, on="Sample_ID", how="left")
    have_mag = "Magnification_X" in df.columns and df["Magnification_X"].notna().any()
    clusters = sorted(df["Cluster"].unique())
    fig, axes = plt.subplots(1, 1 + int(have_mag), figsize=(6.2 * (1 + int(have_mag)), 4.6),
                             squeeze=False)

    ax = axes[0][0]
    if "Count" in df.columns:
        for c in clusters:
            sub = df[df["Cluster"] == c]
            ax.scatter([c] * len(sub), sub["Count"], color=_colour(c - 1),
                       s=55, edgecolors="k", linewidths=0.4)
        ax.set_yscale("log")
        ax.set_xticks(clusters)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Nanowires measured")
        ax.set_title("Wire count by cluster")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    if have_mag:
        ax = axes[0][1]
        for c in clusters:
            sub = df[df["Cluster"] == c]
            ax.scatter([c] * len(sub), sub["Magnification_X"] / 1000,
                       color=_colour(c - 1), s=55,
                       edgecolors="k", linewidths=0.4)
        ax.set_xticks(clusters)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Magnification (kX)")
        ax.set_title("Magnification by cluster")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Acquisition variables by cluster (not used for clustering)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, path, dpi)
    return df


def contact_sheets(image_paths, assignments, out_dir, dpi, per_row=5):
    """One sheet of thumbnails per cluster. Returns the number written."""
    if not image_paths:
        return 0
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for c in sorted(assignments["Cluster"].unique()):
        ids = [s for s in assignments.loc[assignments["Cluster"] == c, "Sample_ID"]
               if s in image_paths]
        if not ids:
            continue
        rows = int(np.ceil(len(ids) / per_row))
        fig, axes = plt.subplots(rows, per_row, figsize=(3.2 * per_row, 3.2 * rows),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, sid in zip(axes.ravel(), ids):
            with Image.open(image_paths[sid]) as im:
                im.thumbnail((420, 420))
                ax.imshow(im, cmap="gray")
            ax.set_title(sid, fontsize=8)
        fig.suptitle(f"Cluster {c} ({len(ids)} samples)", fontsize=12)
        fig.tight_layout()
        _save(fig, out_dir / f"Cluster_{c}.png", min(dpi, 150))
        n += 1
    return n
