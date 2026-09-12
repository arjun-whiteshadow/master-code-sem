"""Text summary of the comparison."""

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
    settings = {k: v for k, v in cfg.items()
                if k not in ("root", "m1_dir", "m2_dir", "results_dir")}
    path.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "settings": settings, "results": summary, "software": software_versions(),
    }, indent=2, default=str))


def _wrap(text, indent="  "):
    return textwrap.fill(text, width=72, initial_indent=indent, subsequent_indent=indent)


def write_summary(path, r):
    by_model, quality, cl, ct = r["by_model"], r["quality"], r["cluster_level"], r["crosstab"]
    dis, enc, zoom, xm = r["disagreements"], r["encoding"], r["zoom"], r["cross_model"]

    L = []
    L.append("Comparison of Method 1 and Method 2")
    L.append("=" * 72)
    L.append(f"Generated {datetime.now():%Y-%m-%d %H:%M}")
    L.append("")
    L.append(f"Samples matched: {r['n_matched']}")
    L.append(f"Method 1 clusters: {r['m1_k']}")
    L.append(f"Method 2 architectures: {len(by_model)}; primary {r['primary']}")
    L.append("")

    L.append("Agreement with Method 1 (original labels)")
    L.append("-" * 72)
    L.append(f"{'Architecture':<22s} {'k':>3s} {'ARI':>8s} {'NMI':>7s} {'AMI':>7s}")
    for _, m in by_model.iterrows():
        L.append(f"{m['CNN_Model']:<22s} {int(m['CNN_k']):>3d} {m['ARI']:>+8.3f} "
                 f"{m['NMI']:>7.3f} {m['AMI']:>7.3f}")
    L.append("")

    L.append("Internal quality of the selected partitions")
    L.append("-" * 72)
    for _, q in quality.iterrows():
        L.append(f"{q['Measure']:<40s} M1 {str(q['Method_1']):>8s}   M2 {str(q['Method_2_primary']):>8s}")
    L.append("")

    L.append(f"Cluster-level correspondence ({r['primary']})")
    L.append("-" * 72)
    for _, c in cl.iterrows():
        L.append(f"M1 {c['M1_Cluster']} ({c['M1_Name']}) -> CNN {c['Best_CNN_Cluster']}: "
                 f"{c['Shared']}/{c['M1_Size']} shared, capture {c['Capture_Percent']:.0f}%, "
                 f"purity {c['Purity_Percent']:.0f}% [{c['Agreement_Level']}]")
    L.append("")
    L.append("Cross-tabulation (rows Method 1, columns Method 2):")
    L.append("        " + "".join(f"{'CNN ' + str(c):>8s}" for c in ct.columns))
    for i in ct.index:
        L.append(f"  M1 {i}  " + "".join(f"{int(ct.loc[i, c]):>8d}" for c in ct.columns))
    L.append("")
    L.append(f"Samples outside the main mapping: {len(dis)} of {r['n_matched']}")
    for _, d in dis.iterrows():
        L.append(f"  {d['Sample_ID']:<14s} M1 {int(d['M1_Cluster'])} -> CNN {int(d['CNN_Cluster'])} "
                 f"(mapped cluster {int(d['Expected_CNN'])})")
    L.append("")

    L.append("Encoding tests (from Method 2)")
    L.append("-" * 72)
    if len(zoom):
        for _, z in zoom.iterrows():
            L.append(f"Zoom {z['Zoom_Factor']:.1f}x: displacement {z['Mean_Zoom_Distance']:.3f} "
                     f"({z['Ratio_Zoom_To_Between_Sample']:.2f} of the between-sample distance)")
    if len(enc):
        L.append(f"{'Architecture':<22s} {'magnification R2':>17s} {'best descriptor R2':>20s}")
        for _, e in enc.iterrows():
            L.append(f"{e['Model']:<22s} {e['Magnification_R2']:>17.3f} {e['Best_Morphology_R2']:>20.3f}")
        n_win = int((enc["Magnification_R2"] > enc["Best_Morphology_R2"]).sum())
        L.append(f"Magnification predicted better than any descriptor in {n_win} of {len(enc)} architectures.")
    if len(xm):
        L.append("Agreement between architectures (ARI):")
        for _, x in xm.iterrows():
            L.append(f"  {x['Model_A']} vs {x['Model_B']}: {x['ARI']:+.3f}")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
