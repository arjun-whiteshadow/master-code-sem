"""Text summary of a run and a record of the software versions used."""

import json
import platform
import sys
import textwrap
from datetime import datetime

import numpy as np


def software_versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "torchvision", "numpy", "pandas", "sklearn", "scipy",
                "umap", "matplotlib", "PIL", "tifffile"):
        try:
            v[mod] = __import__(mod).__version__
        except ImportError:
            v[mod] = "not installed"
    return v


def write_run_info(cfg, summary, path):
    settings = {k: v for k, v in cfg.items()
                if k not in ("root", "image_dir", "results_dir", "descriptor_path")}
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "results": summary,
        "software": software_versions(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))


def _wrap(text, indent="  "):
    return textwrap.fill(text, width=72, initial_indent=indent, subsequent_indent=indent)


def write_summary(path, r):
    """Plain-text summary. ``r`` is the dict assembled by run_method2.py."""
    thr = r["thresholds"]
    row = r["final_row"]
    sil, cross = row["Silhouette_KMeans"], row["CrossAlgorithm_ARI"]
    sub_mean, sub_sd = r["subsample_mean"], r["subsample_sd"]
    enc = r["encoding_summary"]

    L = []
    L.append("Method 2: clustering of pretrained-CNN embeddings")
    L.append("=" * 72)
    L.append(f"Generated {datetime.now():%Y-%m-%d %H:%M}")
    L.append("")

    L.append("Input")
    L.append("-" * 72)
    L.append(f"Images found: {r['n_found']}; embedded: {r['n_embedded']}")
    L.append(f"Bottom crop fraction: {r['crop_fraction']:.2f}")
    L.append(f"Device: {r['device']}")
    L.append("")

    L.append("Architectures")
    L.append("-" * 72)
    for _, m in r["model_status"].iterrows():
        note = f" ({m['Note']})" if m["Note"] else ""
        L.append(f"{m['Model']}: {m['Status']}, {m['Embedding_Dimension']}-d, "
                 f"{m['Weights']}{note}")
    L.append("")
    if len(r["model_summary"]) > 1:
        L.append("Result by architecture:")
        for _, m in r["model_summary"].iterrows():
            L.append(f"  {m['Model']:<20s} k = {int(m['Selected_k'])}, sizes {m['Cluster_Sizes']}, "
                     f"silhouette {m['Silhouette']:.3f}, k-means/Ward ARI {m['CrossAlgorithm_ARI']:.3f}")
        L.append("")
    if len(r["cross_model"]):
        L.append("Agreement between architectures (ARI):")
        for _, m in r["cross_model"].iterrows():
            L.append(f"  {m['Model_A']} vs {m['Model_B']}: {m['ARI']:+.3f}")
        L.append("")

    L.append(f"Primary model: {r['primary']}")
    L.append("-" * 72)
    L.append(f"Embedding processing: {', '.join(r['steps'])}")
    L.append(f"PCA components retained: {r['n_pcs']} "
             f"({r['variance_retained']:.1f}% of variance; threshold {r['pca_threshold']:.0%})")
    L.append("")
    for _, s in r["summary"].iterrows():
        mark = "  <- selected" if int(s["k"]) == r["final_k"] else ""
        L.append(f"k = {int(s['k'])}: silhouette {s['Silhouette_KMeans']:.3f}, "
                 f"sizes {s['Cluster_Sizes']}, k-means/Ward ARI {s['CrossAlgorithm_ARI']:.3f}{mark}")
    L.append("")
    L.append(_wrap(r["k_reason"]))
    if r["degenerate"]:
        L.append(_wrap("Result is flagged as degenerate."))
    L.append("")

    L.append("Stability")
    L.append("-" * 72)
    L.append(f"k-means vs Ward ARI: {cross:.3f}")
    L.append(f"Subsample ARI: {sub_mean:.3f} ± {sub_sd:.3f}")
    if r["outlier_ari"] is not None:
        L.append(f"ARI after removing {r['n_flagged']} flagged sample(s): {r['outlier_ari']:.3f}")
    else:
        L.append("No sample was flagged as an outlier.")
    L.append(_wrap(f"Banner crop: {r['crop_note']}", indent=""))
    L.append("")

    L.append("DBSCAN (exploratory)")
    L.append("-" * 72)
    L.append(_wrap(r["dbscan_note"]))
    L.append("")

    L.append("Encoding tests")
    L.append("-" * 72)
    if len(r["zoom_summary"]):
        for _, z in r["zoom_summary"].iterrows():
            L.append(f"Zoom {z['Zoom_Factor']:.1f}x: displacement {z['Mean_Zoom_Distance']:.3f} "
                     f"({z['Ratio_Zoom_To_Between_Sample']:.2f} of the between-sample distance)")
    else:
        L.append("Zoom control not run.")
    if len(r["encoding_table"]):
        for _, e in r["encoding_table"].iterrows():
            if not np.isnan(e["LOO_R2"]):
                L.append(f"LOO R2, {e['Target']}: {e['LOO_R2']:+.3f}")
        if enc and not np.isnan(enc["magnification_r2"]) and not np.isnan(enc["best_descriptor_r2"]):
            L.append(f"Magnification R2 {enc['magnification_r2']:+.3f} vs best descriptor "
                     f"R2 {enc['best_descriptor_r2']:+.3f}")
    else:
        L.append("Encoding regression not run.")
    L.append("")

    flags = []
    if sil < thr["weak_silhouette"]:
        flags.append(f"mean silhouette {sil:.3f} is below {thr['weak_silhouette']}")
    if cross < thr["weak_cross_algorithm_ari"]:
        flags.append(f"k-means/Ward ARI {cross:.3f} is below {thr['weak_cross_algorithm_ari']}")
    if sub_mean < thr["weak_subsample_ari"]:
        flags.append(f"subsample ARI {sub_mean:.3f} is below {thr['weak_subsample_ari']}")
    if enc and not np.isnan(enc["magnification_r2"]) and not np.isnan(enc["best_descriptor_r2"]) \
            and enc["magnification_r2"] > enc["best_descriptor_r2"]:
        flags.append("magnification is predicted better than any descriptor")
    L.append("Weakness thresholds")
    L.append("-" * 72)
    if flags:
        L.extend(f"  {f}" for f in flags)
    else:
        L.append("  None triggered.")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
