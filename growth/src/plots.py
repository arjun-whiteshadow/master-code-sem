"""Figures for the growth-condition analysis. Every function saves one PNG
and closes the figure."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

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
    "Substrate_Temperature_C": "Substrate temperature (°C)",
    "Ga_BEP_Torr": "Ga BEP (Torr)",
    "As_BEP_Torr": "As BEP (Torr)",
    "Sb_BEP_Torr": "Sb BEP (Torr)",
    "Growth_Time_min": "Growth time (min)",
}


def _label(col):
    return LABELS.get(col, col.replace("_", " "))


def _save(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _colour(i):
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


def _rho_heatmap(ax, matrix, row_labels, col_labels, marks=None):
    ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            text = f"{v:.2f}" + ("*" if marks is not None and marks[i, j] else "")
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "black")


def parameter_correlations(corr, numeric, path, dpi):
    """Spearman correlation between the growth parameters."""
    n = len(numeric)
    m = np.eye(n)
    for _, r in corr.iterrows():
        i, j = numeric.index(r["Parameter_A"]), numeric.index(r["Parameter_B"])
        m[i, j] = m[j, i] = r["Spearman_rho"]
    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 1.0 * n + 2.4))
    _rho_heatmap(ax, m, [_label(c) for c in numeric], [_label(c) for c in numeric])
    ax.set_title("Spearman correlation between growth parameters")
    _save(fig, path, dpi)


def descriptor_vs_parameter(table, descriptors, numeric, path, dpi):
    """Spearman correlation of each descriptor with each parameter; an
    asterisk marks a significant cell after adjustment."""
    m = np.full((len(descriptors), len(numeric)), np.nan)
    sig = np.zeros_like(m, dtype=bool)
    for _, r in table.iterrows():
        i, j = descriptors.index(r["Descriptor"]), numeric.index(r["Parameter"])
        m[i, j], sig[i, j] = r["Spearman_rho"], bool(r["Significant"])
    fig, ax = plt.subplots(figsize=(1.3 * len(numeric) + 3.5, 0.8 * len(descriptors) + 2.4))
    _rho_heatmap(ax, m, [_label(c) for c in descriptors], [_label(c) for c in numeric], sig)
    ax.set_title("Descriptor against growth parameter (Spearman ρ; * significant)")
    _save(fig, path, dpi)


def cluster_boxplots(merged, numeric, tests, path, dpi):
    """Each numeric parameter by cluster, with the adjusted p-value."""
    clusters = sorted(merged["Cluster"].unique())
    n = len(numeric)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.6 * rows), squeeze=False)
    q = tests.set_index("Parameter")
    for k, p in enumerate(numeric):
        ax = axes[k // cols][k % cols]
        data = [merged.loc[merged["Cluster"] == c, p].dropna().to_numpy() for c in clusters]
        ax.boxplot(data, positions=clusters, widths=0.5, showfliers=False,
                   medianprops=dict(color="black"))
        for c, v in zip(clusters, data):
            jitter = np.random.default_rng(c).uniform(-0.12, 0.12, len(v))
            ax.scatter(c + jitter, v, color=_colour(c - 1), s=28, edgecolors="k",
                       linewidths=0.4, zorder=3)
        ax.set_xticks(clusters)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(_label(p))
        qv = q.loc[p, "q_value"]
        ax.set_title(f"q = {qv:.3f}" if not np.isnan(qv) else "not tested", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.set_axisbelow(True)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    fig.suptitle("Growth parameters by morphology cluster", y=1.0)
    fig.tight_layout()
    _save(fig, path, dpi)


def regime_map(points, windows, axes_names, path, dpi):
    """Samples on the two chosen parameters, coloured by cluster, with each
    cluster's interquartile window drawn as a rectangle."""
    x, y = axes_names
    clusters = sorted(points["Cluster"].unique())
    w = windows.set_index(["Cluster", "Parameter"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for c in clusters:
        sub = points[points["Cluster"] == c]
        colour = _colour(c - 1)
        ax.scatter(sub[x], sub[y], color=colour, s=60, edgecolors="k", linewidths=0.5,
                   label=f"Cluster {c}", zorder=3)
        _annotate(ax, sub[x], sub[y], sub["Sample_ID"])
        x0, x1 = w.loc[(c, x), "Q1"], w.loc[(c, x), "Q3"]
        y0, y1 = w.loc[(c, y), "Q1"], w.loc[(c, y), "Q3"]
        if not any(np.isnan([x0, x1, y0, y1])):
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=colour,
                                   alpha=0.12, edgecolor=colour, linewidth=1.2, zorder=1))
    ax.set_xlabel(_label(x))
    ax.set_ylabel(_label(y))
    ax.set_title("Morphology clusters in growth-condition space")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    _save(fig, path, dpi)


def parameter_drift(merged, numeric, dates, path, dpi):
    """Each numeric parameter against growth date, coloured by cluster."""
    day = dates.reindex(merged["Sample_ID"])
    clusters = sorted(merged["Cluster"].unique())
    n = len(numeric)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.4 * n), sharex=True, squeeze=False)
    for k, p in enumerate(numeric):
        ax = axes[k][0]
        for c in clusters:
            m = (merged["Cluster"] == c).to_numpy()
            ax.scatter(day[m], merged.loc[m, p], color=_colour(c - 1), s=30,
                       edgecolors="k", linewidths=0.4, label=f"Cluster {c}")
        ax.set_ylabel(_label(p), fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    axes[0][0].legend(fontsize=8, ncol=len(clusters))
    axes[-1][0].set_xlabel("Growth date")
    fig.suptitle("Growth parameters over time", y=1.0)
    fig.tight_layout()
    _save(fig, path, dpi)
