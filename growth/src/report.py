"""Text summary of a run and a record of the software versions used."""

import json
import platform
import sys
import textwrap
from datetime import datetime


def software_versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn", "scipy", "matplotlib"):
        try:
            v[mod] = __import__(mod).__version__
        except ImportError:
            v[mod] = "not installed"
    return v


def write_run_info(cfg, summary, path):
    """JSON record of the configuration, headline results and library versions."""
    settings = {k: v for k, v in cfg.items()
                if k not in ("root", "log_path", "m1_dir", "results_dir")}
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "results": summary,
        "software": software_versions(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))


def _wrap(text, indent="  "):
    return textwrap.fill(text, width=72, initial_indent=indent, subsequent_indent=indent)


def _section(L, title):
    L.append(title)
    L.append("-" * 72)


def write_summary(path, r):
    """Plain-text summary. ``r`` is the dict assembled by run_growth.py."""
    alpha = r["alpha"]
    L = []
    L.append("Growth conditions against the Method 1 morphology clusters")
    L.append("=" * 72)
    L.append(f"Generated {datetime.now():%Y-%m-%d %H:%M}")
    L.append("")

    _section(L, "Data")
    cov = r["coverage"]
    L.append(f"Samples in the growth log: {int(cov['In_Growth_Log'].sum())}")
    L.append(f"Samples in the Method 1 run: {int(cov['In_Method1'].sum())}")
    L.append(f"Samples analysed (in both): {int(cov['Included'].sum())}")
    only_log = cov.loc[cov["In_Growth_Log"] & ~cov["In_Method1"], "Sample_ID"].tolist()
    only_m1 = cov.loc[cov["In_Method1"] & ~cov["In_Growth_Log"], "Sample_ID"].tolist()
    if only_log:
        L.append(_wrap("In the log only: " + ", ".join(only_log)))
    if only_m1:
        L.append(_wrap("In Method 1 only: " + ", ".join(only_m1)))
    L.append(f"Numeric parameters used ({len(r['numeric'])}): {', '.join(r['numeric'])}")
    L.append(f"Categorical parameters used ({len(r['categorical'])}): "
             f"{', '.join(r['categorical']) or 'none'}")
    for _, a in r["audit"].iterrows():
        if a["Status"] in ("excluded", "missing"):
            L.append(f"Excluded: {a['Parameter']} ({a['Reason']})")
    L.append(f"Clusters: {r['k']}, sizes {r['cluster_sizes']}")
    L.append("")

    _section(L, "Checks on the log")
    tied = r["correlations"][r["correlations"]["Not_Separable"]]
    if len(tied):
        for _, t in tied.iterrows():
            L.append(f"  {t['Parameter_A']} and {t['Parameter_B']} moved together "
                     f"(rho = {t['Spearman_rho']:+.2f}); their effects cannot be separated.")
    else:
        L.append(f"  No pair of parameters with |rho| >= {r['confound_threshold']}.")
    drift = r["drift"]
    if drift is None:
        L.append("  Growth dates could not be read from the sample IDs; drift not checked.")
    else:
        d = drift[drift["Drifts"]]
        if len(d):
            for _, t in d.iterrows():
                L.append(f"  {t['Parameter']} drifted with growth date "
                         f"(rho = {t['Spearman_rho_vs_Date']:+.2f}).")
        else:
            L.append(f"  No parameter with |rho| >= {r['confound_threshold']} against growth date.")
    L.append("")

    _section(L, f"Descriptor against parameter (Spearman; BH-adjusted q < {alpha})")
    sig = r["descriptor_tests"][r["descriptor_tests"]["Significant"]]
    if len(sig):
        for _, t in sig.sort_values("q_value").iterrows():
            L.append(f"  {t['Descriptor']:<28s} {t['Parameter']:<24s} "
                     f"rho = {t['Spearman_rho']:+.2f}  q = {t['q_value']:.3f}")
    else:
        L.append("  None significant.")
    L.append("")

    _section(L, f"Parameter across clusters (Kruskal-Wallis; BH-adjusted q < {alpha})")
    for _, t in r["cluster_numeric"].iterrows():
        meds = "  ".join(f"{t[f'Median_Cluster_{c}']:g}" for c in range(1, r["k"] + 1))
        mark = "  <- differs" if t["Significant"] else ""
        q = f"{t['q_value']:.3f}" if t["q_value"] == t["q_value"] else "n/a"
        L.append(f"  {t['Parameter']:<26s} q = {q}  medians by cluster: {meds}{mark}")
    if len(r["cluster_categorical"]):
        L.append("")
        L.append(f"  Categorical (exact permutation test, {r['n_permutations']} shuffles):")
        for _, t in r["cluster_categorical"].iterrows():
            mark = "  <- differs" if t["Significant"] else ""
            L.append(f"  {t['Parameter']:<26s} q = {t['q_value']:.3f}  "
                     f"{int(t['Levels'])} levels{mark}")
    L.append("")

    _section(L, "Regression of each descriptor on all numeric parameters (OLS)")
    for _, t in r["regressions"].iterrows():
        L.append(f"  {t['Descriptor']:<28s} R2 fit = {t['R2_Fit']:.2f}  "
                 f"R2 leave-one-out = {t['R2_LOO']:.2f}")
    L.append("")

    _section(L, "Regime map")
    if r["axes"] is None:
        L.append("  Fewer than two numeric parameters; no map drawn.")
    else:
        L.append(f"  Axes: {r['axes'][0]} and {r['axes'][1]} ({r['axes_reason']})")
        w = r["windows"].set_index(["Cluster", "Parameter"])
        for c in range(1, r["k"] + 1):
            parts = []
            for p in r["axes"]:
                parts.append(f"{p} {w.loc[(c, p), 'Q1']:g}-{w.loc[(c, p), 'Q3']:g}")
            L.append(f"  Cluster {c}: " + "; ".join(parts) + "  (interquartile ranges)")
    L.append("")

    _section(L, "Placing samples by nearest cluster centre")
    loo = r["loo"]
    L.append(f"  Leave-one-out accuracy on the analysed samples: "
             f"{100 * loo['Correct'].mean():.0f}% ({int(loo['Correct'].sum())} of {len(loo)})")
    wrong = loo[~loo["Correct"]]
    for _, s in wrong.iterrows():
        L.append(f"    {s['Sample_ID']:<14s} cluster {s['Cluster']} placed in {s['Predicted']}")
    L.append("")

    _section(L, "Reading the result")
    n_sig = int(r["cluster_numeric"]["Significant"].sum() + r["cluster_categorical"]["Significant"].sum())
    if n_sig == 0:
        L.append(_wrap("No growth parameter differs between the clusters after "
                       "adjustment. The morphology groups are not explained by the "
                       "recorded conditions; either the relevant condition was not "
                       "recorded, or the groups reflect something other than growth "
                       "settings."))
    else:
        L.append(_wrap(f"{n_sig} parameter(s) differ between clusters. A cluster is a "
                       "candidate growth regime only where its window on those "
                       "parameters does not overlap the others' and the direction is "
                       "physically sensible; that judgement is made from the SEM "
                       "images and the growth record, not from these tables."))
    if len(tied):
        L.append(_wrap("Where two parameters moved together, any effect attributed to "
                       "one may belong to the other."))
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
