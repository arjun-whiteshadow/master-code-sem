"""Text summary of a run and a record of the software versions used."""

import json
import platform
import sys
import textwrap
from datetime import datetime


def software_versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn", "scipy", "matplotlib", "tifffile"):
        try:
            v[mod] = __import__(mod).__version__
        except ImportError:
            v[mod] = "not installed"
    return v


def write_run_info(cfg, summary, path):
    """JSON record of the configuration, headline results and library versions."""
    settings = {k: v for k, v in cfg.items()
                if k not in ("root", "feature_path", "results_dir", "image_dir")}
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "results": summary,
        "software": software_versions(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))


def write_model(model, path):
    """JSON record of the fitted transformation, from ``clustering.fitted_model``."""
    path.write_text(json.dumps(model, indent=2))


def _wrap(text, indent="  "):
    return textwrap.fill(text, width=72, initial_indent=indent, subsequent_indent=indent)


def write_summary(path, r):
    """Plain-text summary. ``r`` is the dict assembled by run_method1.py."""
    thr = r["thresholds"]
    row = r["final_row"]
    sil = row["Silhouette_KMeans"]
    cross = row["CrossAlgorithm_ARI"]
    sub_mean, sub_sd = r["subsample_mean"], r["subsample_sd"]

    L = []
    L.append("Method 1: clustering of morphology descriptors")
    L.append("=" * 72)
    L.append(f"Generated {datetime.now():%Y-%m-%d %H:%M}")
    L.append("")

    L.append("Data")
    L.append("-" * 72)
    L.append(f"Samples: {r['n_samples']}")
    L.append(f"Descriptors used ({r['n_features']}): {', '.join(r['features_used'])}")
    for feat, reason in r["features_excluded"]:
        L.append(f"Excluded: {feat} ({reason})")
    L.append("")

    L.append("PCA")
    L.append("-" * 72)
    L.append(f"Components retained: {r['n_pcs']} "
             f"({r['variance_retained']:.1f}% of variance; threshold {r['pca_threshold']:.0%})")
    L.append("")

    L.append("Selection of k")
    L.append("-" * 72)
    for _, s in r["summary"].iterrows():
        mark = "  <- selected" if int(s["k"]) == r["final_k"] else ""
        L.append(f"k = {int(s['k'])}: silhouette {s['Silhouette_KMeans']:.3f}, "
                 f"sizes {s['Cluster_Sizes']}, k-means/Ward ARI "
                 f"{s['CrossAlgorithm_ARI']:.3f}{mark}")
    L.append("")
    L.append(_wrap(r["k_reason"]))
    if r["degenerate"]:
        L.append(_wrap("Result is flagged as degenerate."))
    L.append("")

    L.append("Selected solution")
    L.append("-" * 72)
    L.append(f"k = {r['final_k']}, cluster sizes {row['Cluster_Sizes']}")
    L.append(f"Mean silhouette: {sil:.3f}")
    L.append("Cluster profiles (mean z-score):")
    prof = r["profiles_std"].round(2)
    cols = list(prof.columns[1:])
    L.append("  Cluster  " + "  ".join(f"{c[:12]:>12s}" for c in cols))
    for _, p in prof.iterrows():
        L.append(f"  {int(p['Cluster']):>7d}  " + "  ".join(f"{p[c]:>12.2f}" for c in cols))
    L.append("")

    L.append("Stability")
    L.append("-" * 72)
    L.append(f"k-means vs Ward ARI: {cross:.3f}")
    L.append(f"Subsample ARI: {sub_mean:.3f} ± {sub_sd:.3f}")
    if r["lofo"] is not None:
        lo, hi = r["lofo"]["ARI_vs_Main_Result"].min(), r["lofo"]["ARI_vs_Main_Result"].max()
        L.append(f"Leave-one-descriptor-out ARI: {lo:.3f} to {hi:.3f}")
    if r["outlier_ari"] is not None:
        L.append(f"ARI after removing {r['n_flagged']} flagged sample(s): {r['outlier_ari']:.3f}")
    else:
        L.append("No sample exceeded the outlier threshold.")
    L.append("")

    L.append("DBSCAN (exploratory)")
    L.append("-" * 72)
    L.append(_wrap(r["dbscan_note"]))
    L.append("")

    flags = []
    if sil < thr["weak_silhouette"]:
        flags.append(f"mean silhouette {sil:.3f} is below {thr['weak_silhouette']}")
    if cross < thr["weak_cross_algorithm_ari"]:
        flags.append(f"k-means/Ward ARI {cross:.3f} is below {thr['weak_cross_algorithm_ari']}")
    if sub_mean < thr["weak_subsample_ari"]:
        flags.append(f"subsample ARI {sub_mean:.3f} is below {thr['weak_subsample_ari']}")
    L.append("Weakness thresholds")
    L.append("-" * 72)
    if flags:
        L.extend(f"  {f}" for f in flags)
    else:
        L.append("  None triggered.")
    L.append("")

    L.append("Flagged values (|z| above threshold; retained in the analysis)")
    L.append("-" * 72)
    if len(r["outliers"]):
        for _, o in r["outliers"].iterrows():
            L.append(f"  {o['Sample_ID']:<14s} {o['Feature']:<30s} z = {o['Standardized_Value']:+.2f}")
    else:
        L.append("  None.")
    neg = r["sample_fit"][r["sample_fit"]["Negative_Silhouette"]]
    if len(neg):
        L.append("Samples with negative silhouette:")
        for _, s in neg.iterrows():
            L.append(f"  {s['Sample_ID']:<14s} cluster {int(s['Cluster'])}, "
                     f"silhouette {s['Silhouette']:+.3f}")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
