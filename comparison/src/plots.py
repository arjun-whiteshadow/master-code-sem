"""Figures for the comparison."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#065A82", "#B03A2E", "#1E8449", "#8E44AD", "#D68910", "#17A2B8"]


def _save(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def method_quality(q1, q2, path, dpi):
    """Silhouette, k-means/Ward ARI and subsample ARI for the two methods."""
    metrics = ["Silhouette", "k-means/Ward ARI", "Subsample ARI"]
    v1 = [q1["silhouette"], q1["cross_ari"], q1["subsample"]]
    v2 = [q2["silhouette"], q2["cross_ari"], q2["subsample"]]
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(x - w / 2, v1, w, color=PALETTE[0], label="Method 1 (descriptors)")
    ax.bar(x + w / 2, v2, w, color=PALETTE[1], label="Method 2 (primary CNN)")
    for i, (a, b) in enumerate(zip(v1, v2)):
        ax.text(i - w / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + w / 2, b + 0.015, f"{b:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x, metrics, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Internal quality of each method's selected partition")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, path, dpi)


def agreement_by_model(table, weak_ari, path, dpi):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(table))
    colours = [PALETTE[1] if v < weak_ari else PALETTE[4] if v < 0.5 else PALETTE[2]
               for v in table["ARI"]]
    ax.bar(x, table["ARI"], color=colours, width=0.6)
    for i, (v, k) in enumerate(zip(table["ARI"], table["CNN_k"])):
        ax.text(i, v + 0.008, f"{v:+.3f}\n(k = {int(k)})", ha="center", fontsize=9.5,
                fontweight="bold")
    ax.axhline(weak_ari, ls="--", color="grey", lw=1)
    ax.set_xticks(x, table["CNN_Model"], rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("ARI against Method 1")
    ax.set_ylim(0, max(0.1, table["ARI"].max()) * 1.45)
    ax.set_title("Agreement between Method 1 and each architecture")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, path, dpi)


def crosstab(ct, names, model, path, dpi):
    data = ct.to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(ct.columns)), [f"CNN {c}" for c in ct.columns])
    ax.set_yticks(range(len(ct.index)),
                  [f"M1 {i}\n{names.get(int(i), '')[:26]}" for i in ct.index], fontsize=8.5)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, str(data[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if data[i, j] > data.max() / 2 else "black")
    ax.set_xlabel(f"Method 2 cluster ({model})")
    ax.set_ylabel("Method 1 cluster")
    ax.set_title("Cross-tabulation of cluster membership")
    fig.colorbar(im, ax=ax, label="Shared samples")
    _save(fig, path, dpi)


def encoding(table, path, dpi):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(table))
    w = 0.38
    ax.bar(x - w / 2, table["Magnification_R2"], w, color=PALETTE[1], label="log10(magnification)")
    ax.bar(x + w / 2, table["Best_Morphology_R2"], w, color=PALETTE[0], label="Best descriptor")
    for i, (m, s) in enumerate(zip(table["Magnification_R2"], table["Best_Morphology_R2"])):
        ax.text(i - w / 2, m + 0.012, f"{m:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w / 2, s + 0.012, f"{s:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, table["Model"], rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("Leave-one-out R²")
    ax.set_title("Encoding of magnification versus morphology, by architecture")
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, path, dpi)
