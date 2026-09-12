"""Figures for Method 2. Every function saves one PNG and closes the figure."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .images import crop_bottom, load_image_rgb

PALETTE = ["#065A82", "#B03A2E", "#1E8449", "#8E44AD", "#D68910", "#17A2B8"]

# Scatter plots label each point with its sample ID up to this many samples.
ANNOTATE_UP_TO = 60

LABELS = {
    "Density_NW_um2": "Density (NW/µm²)",
    "Mean_Diameter_um": "Mean diameter (µm)",
    "CV_percent": "Diameter CV (%)",
    "Merged_percent": "Merged (%)",
    "Foreground_Coverage_percent": "Coverage (%)",
    "Angle_STD_deg": "Angle SD (°)",
}


def _label(name):
    return LABELS.get(name, name.replace("_", " "))


def _colour(i):
    if i < len(PALETTE):
        return PALETTE[i]
    return plt.get_cmap("tab20")(i % 20)


def _save(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _annotate(ax, xs, ys, labels, fontsize=6):
    if len(labels) > ANNOTATE_UP_TO:
        return
    offsets = [(4, 4), (4, -9), (-4, 4), (-4, -9)]
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        ax.annotate(str(lab), (x, y), textcoords="offset points",
                    xytext=offsets[i % 4], fontsize=fontsize, alpha=0.85)


def pca_variance(variance, threshold, out_dir, dpi):
    n_keep = int(variance["Retained_For_Clustering"].sum())
    x = np.arange(1, len(variance) + 1)
    cum_kept = variance["Cumulative_Variance_Percent"].iloc[n_keep - 1]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, variance["Cumulative_Variance_Percent"], "o-", color=PALETTE[0], lw=2, ms=4)
    ax.axhline(threshold * 100, ls="--", color=PALETTE[1], label=f"{threshold:.0%} threshold")
    ax.axvline(n_keep, ls="--", color="grey", label=f"{n_keep} retained ({cum_kept:.1f}%)")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, out_dir / "PCA_Cumulative_Variance.png", dpi)


def k_selection(summary, final_k, thresholds, path, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ks = summary["k"].tolist()

    ax = axes[0]
    colours = [PALETTE[0] if k == final_k else "#9FB6C4" for k in ks]
    bars = ax.bar(summary["k"].astype(str), summary["Silhouette_KMeans"], color=colours, width=0.6)
    for b, v, s in zip(bars, summary["Silhouette_KMeans"], summary["Cluster_Sizes"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.004, s, ha="center",
                fontsize=7, rotation=90, color="white")
    ax.axhline(thresholds["weak_silhouette"], ls="--", color=PALETTE[1], lw=1,
               label=f"weak below {thresholds['weak_silhouette']}")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean silhouette (k-means)")
    ax.set_title(f"Silhouette by k; selected k = {final_k}")
    ax.set_ylim(0, max(summary["Silhouette_KMeans"].max() * 1.5, 0.3))
    ax.legend(fontsize=8)
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
    ax.plot(ks, summary["CrossAlgorithm_ARI"], "o-", color=PALETTE[2], lw=2, label="k-means vs Ward")
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


def scatter_2d(coords, labels, ids, title, xlabel, ylabel, path, dpi, noise_label=None):
    """2-D scatter with optional cluster colouring; DBSCAN noise is drawn as crosses."""
    x, y = coords[:, 0], coords[:, 1]
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
                ax.scatter(x[m], y[m], s=70, color=_colour(j), edgecolors="k",
                           linewidths=0.4, label=f"Cluster {name}")
        ax.legend(fontsize=9)
    _annotate(ax, x, y, ids)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, path, dpi)


def dendrogram_plot(link, ids, final_k, path, dpi):
    from scipy.cluster.hierarchy import dendrogram
    fig, ax = plt.subplots(figsize=(max(10, len(ids) * 0.36), 6.5))
    dendrogram(link, labels=list(ids), ax=ax, leaf_rotation=90, leaf_font_size=8,
               color_threshold=link[-(final_k - 1), 2])
    ax.set_ylabel("Ward linkage distance")
    ax.set_title(f"Ward dendrogram, cut at k = {final_k}")
    _save(fig, path, dpi)


def stability(stab, summary, final_k, thresholds, path, dpi):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.errorbar(stab["k"], stab["Mean_Subsample_ARI"], yerr=stab["SD_Subsample_ARI"],
                marker="o", lw=2, capsize=4, color=PALETTE[0],
                label="Subsample stability (mean ± SD)")
    ax.plot(summary["k"], summary["CrossAlgorithm_ARI"], "s--", color=PALETTE[2], lw=1.8,
            label="k-means vs Ward")
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


def model_comparison(table, path, dpi):
    """Silhouette, k-means/Ward ARI and selected k for each architecture."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(table))
    names = table["Model"]

    ax = axes[0]
    ax.bar(x, table["Silhouette"], color=PALETTE[0], width=0.6)
    for i, v in enumerate(table["Silhouette"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Mean silhouette")
    ax.set_title("Cluster separation")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1]
    colours = [PALETTE[1] if v < 0.3 else PALETTE[2] for v in table["CrossAlgorithm_ARI"]]
    ax.bar(x, table["CrossAlgorithm_ARI"], color=colours, width=0.6)
    for i, v in enumerate(table["CrossAlgorithm_ARI"]):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.3, ls="--", color="grey", lw=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("ARI (k-means vs Ward)")
    ax.set_title("Cross-algorithm agreement")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[2]
    ax.bar(x, table["Selected_k"], color=PALETTE[3], width=0.6)
    for i, (v, s) in enumerate(zip(table["Selected_k"], table["Cluster_Sizes"])):
        ax.text(i, v + 0.08, f"k = {int(v)}\n{s}", ha="center", fontsize=8)
    ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Selected k")
    ax.set_ylim(0, table["Selected_k"].max() * 1.5)
    ax.set_title("Selected number of clusters")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, path, dpi)


def confound_tests(zoom_summary, encoding, path, dpi):
    """Zoom-control curve and per-target encoding R² for the primary model."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    if len(zoom_summary):
        xs = [1.0] + zoom_summary["Zoom_Factor"].tolist()
        ys = [0.0] + zoom_summary["Mean_Zoom_Distance"].tolist()
        es = [0.0] + zoom_summary["SD_Zoom_Distance"].tolist()
        base = zoom_summary["Mean_Between_Sample_Distance"].iloc[0]
        ax.errorbar(xs, ys, yerr=es, marker="o", color=PALETTE[1], lw=2, capsize=4,
                    label="Same image, zoomed")
        ax.axhline(base, ls="--", color=PALETTE[0], lw=2,
                   label=f"Mean distance between samples ({base:.2f})")
        ax.set_xlabel("Simulated zoom factor")
        ax.set_ylabel("Embedding distance")
        ax.set_title("(a) Displacement under zoom alone")
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, "zoom control not run", ha="center", va="center")
        ax.axis("off")

    ax = axes[1]
    if len(encoding):
        e = encoding.dropna(subset=["LOO_R2"]).sort_values("LOO_R2")
        colours = [PALETTE[1] if "Magnif" in t else (PALETTE[0] if v > 0 else "#BBBBBB")
                   for t, v in zip(e["Target"], e["LOO_R2"])]
        ax.barh(range(len(e)), e["LOO_R2"], color=colours)
        ax.set_yticks(range(len(e)), [_label(t) for t in e["Target"]], fontsize=9)
        for i, v in enumerate(e["LOO_R2"]):
            ax.text(max(v, 0) + 0.015, i, f"{v:+.2f}", va="center", ha="left",
                    fontsize=9, fontweight="bold")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Leave-one-out R²")
        ax.set_title("(b) Prediction of each target from the embedding")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, "encoding regression not run", ha="center", va="center")
        ax.axis("off")

    fig.tight_layout()
    _save(fig, path, dpi)


def encoding_by_model(table, path, dpi):
    """Magnification R² against the best descriptor R² for each architecture."""
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(table))
    w = 0.38
    ax.bar(x - w / 2, table["Magnification_R2"], w, color=PALETTE[1], label="log10(magnification)")
    ax.bar(x + w / 2, table["Best_Morphology_R2"], w, color=PALETTE[0], label="Best descriptor")
    for i, (m, s) in enumerate(zip(table["Magnification_R2"], table["Best_Morphology_R2"])):
        if not np.isnan(m):
            ax.text(i - w / 2, m + 0.015, f"{m:+.2f}", ha="center", fontsize=8.5, fontweight="bold")
        if not np.isnan(s):
            ax.text(i + w / 2, s + 0.015, f"{s:+.2f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, table["Model"], rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Leave-one-out R²")
    ax.set_title("Encoding of magnification versus morphology, by architecture")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, path, dpi)


def cluster_vs_magnification(assign, path, dpi):
    if not assign["Magnification_X"].notna().any():
        return
    clusters = sorted(assign["Cluster"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for c in clusters:
        sub = assign[assign["Cluster"] == c]
        ax.scatter([c] * len(sub), sub["Magnification_X"] / 1000, color=_colour(c - 1),
                   s=60, edgecolors="k", linewidths=0.4)
    ax.set_xticks(clusters)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Magnification (kX)")
    ax.set_title("Magnification by cluster (not used for clustering)")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, path, dpi)


def input_preview(paths, crop_fraction, path, dpi, n_show=12):
    """Thumbnails of what the network receives after cropping and resizing."""
    ids = list(paths)[:n_show]
    if not ids:
        return
    ncol = min(4, len(ids))
    nrow = int(np.ceil(len(ids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.8, nrow * 3.0), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, sid in zip(axes.ravel(), ids):
        im = crop_bottom(load_image_rgb(paths[sid]), crop_fraction).resize((224, 224))
        ax.imshow(im, cmap="gray")
        ax.set_title(sid, fontsize=8)
    fig.suptitle(f"Network input (crop_bottom_fraction = {crop_fraction:.2f}, 224x224)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path, min(dpi, 150))


def contact_sheets(paths, assignments, out_dir, dpi, per_row=5):
    """One sheet of thumbnails per cluster. Returns the number written."""
    if not paths:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for c in sorted(assignments["Cluster"].unique()):
        ids = [s for s in assignments.loc[assignments["Cluster"] == c, "Sample_ID"] if s in paths]
        if not ids:
            continue
        ncol = min(per_row, len(ids))
        nrow = int(np.ceil(len(ids) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.7, nrow * 3.0), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, sid in zip(axes.ravel(), ids):
            im = load_image_rgb(paths[sid])
            im.thumbnail((420, 420))
            ax.imshow(im, cmap="gray")
            ax.set_title(sid, fontsize=8)
        fig.suptitle(f"Cluster {c} ({len(ids)} images)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        _save(fig, out_dir / f"Cluster_{c}.png", min(dpi, 150))
        n += 1
    return n
