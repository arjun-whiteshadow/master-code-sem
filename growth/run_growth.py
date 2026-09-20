"""Growth conditions against the Method 1 morphology clusters.

Usage:
    python run_growth.py [config.yaml]

Reads the growth log and the Method 1 results named in config.yaml, writes
tables, figures and a summary to the results folder. Neither input is
modified.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import associations, confounds, growth_log, plots, predict, regime_map, report
from src.config import load_config


def main(config_path):
    cfg = load_config(config_path)
    seed = int(cfg["random_seed"])
    np.random.seed(seed)

    res = cfg["results_dir"]
    tables, figs, reports = res / "tables", res / "figures", res / "reports"
    for d in (tables, figs, reports):
        d.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg["output"]["figure_dpi"])
    alpha = float(cfg["tests"]["alpha"])
    n_perm = int(cfg["tests"]["n_permutations"])
    thr = float(cfg["confound_threshold"])

    # Inputs
    log = growth_log.load_growth_log(cfg)
    X, clusters, model = growth_log.load_method1(cfg)
    merged, coverage = growth_log.merge_with_method1(log, X, clusters)
    numeric, categorical, audit = growth_log.screen_parameters(cfg, merged)
    descriptors = list(X.columns)
    k = int(clusters.max())
    sizes = np.bincount(merged["Cluster"], minlength=k + 1)[1:]
    print(f"{len(merged)} samples in both the log and the Method 1 run "
          f"({int(coverage['In_Growth_Log'].sum())} in the log, {len(X)} in Method 1)")
    print(f"Parameters used: {', '.join(numeric + categorical)}")
    for _, a in audit.iterrows():
        if a["Status"] in ("excluded", "missing"):
            print(f"  excluded {a['Parameter']}: {a['Reason']}")
    coverage.to_csv(tables / "Growth_Log_Coverage.csv", index=False)
    audit.to_csv(tables / "Growth_Parameter_Audit.csv", index=False)
    merged.to_csv(tables / "Merged_Features_Conditions.csv", index=False)

    # Checks on the log itself
    corr = confounds.parameter_correlations(merged, numeric, thr)
    corr.to_csv(tables / "Growth_Parameter_Correlations.csv", index=False)
    for _, t in corr[corr["Not_Separable"]].iterrows():
        print(f"  {t['Parameter_A']} and {t['Parameter_B']} not separable "
              f"(rho = {t['Spearman_rho']:+.2f})")
    if len(numeric) >= 2:
        plots.parameter_correlations(corr, numeric, figs / "Growth_Parameter_Correlations.png", dpi)

    dates = confounds.growth_dates(merged["Sample_ID"])
    drift = None
    if dates.notna().any():
        drift = confounds.date_drift(merged, numeric, dates, thr)
        drift.to_csv(tables / "Growth_Parameter_Drift.csv", index=False)
        confounds.category_dates(merged, categorical, dates).to_csv(
            tables / "Growth_Category_Dates.csv", index=False)
        for _, t in drift[drift["Drifts"]].iterrows():
            print(f"  {t['Parameter']} drifts with date (rho = {t['Spearman_rho_vs_Date']:+.2f})")
        if numeric:
            plots.parameter_drift(merged, numeric, dates, figs / "Growth_Parameter_Drift.png", dpi)
    else:
        print("  no growth date could be read from the sample IDs; drift not checked")

    # Descriptor against parameter
    dtests = associations.descriptor_vs_parameter(merged, descriptors, numeric, alpha)
    dtests.to_csv(tables / "Descriptor_vs_Condition.csv", index=False)
    print(f"Descriptor-parameter pairs significant at q < {alpha}: "
          f"{int(dtests['Significant'].sum())} of {len(dtests)}")
    if numeric:
        plots.descriptor_vs_parameter(dtests, descriptors, numeric,
                                      figs / "Descriptor_vs_Condition.png", dpi)

    # Parameter across clusters
    ctests = associations.cluster_vs_numeric(merged, numeric, alpha)
    ctests.to_csv(tables / "Cluster_vs_Condition.csv", index=False)
    cat_tests, cat_counts = associations.cluster_vs_categorical(
        merged, categorical, n_perm, seed, alpha)
    cat_tests.to_csv(tables / "Cluster_vs_Category.csv", index=False)
    cat_counts.to_csv(tables / "Cluster_vs_Category_Counts.csv", index=False)
    for _, t in ctests.iterrows():
        mark = " <- differs" if t["Significant"] else ""
        print(f"  {t['Parameter']}: q = {t['q_value']}{mark}")
    for _, t in cat_tests.iterrows():
        mark = " <- differs" if t["Significant"] else ""
        print(f"  {t['Parameter']}: q = {t['q_value']} (exact){mark}")
    if numeric:
        plots.cluster_boxplots(merged, numeric, ctests, figs / "Cluster_vs_Condition.png", dpi)

    # Regression of each descriptor on all numeric parameters
    regs = associations.descriptor_regressions(merged, descriptors, numeric) if numeric \
        else pd.DataFrame(columns=["Descriptor", "N", "R2_Fit", "R2_LOO"])
    regs.to_csv(tables / "Descriptor_Regressions.csv", index=False)
    for _, t in regs.iterrows():
        print(f"  {t['Descriptor']}: leave-one-out R2 = {t['R2_LOO']:.2f}")

    # Regime windows and map
    windows = regime_map.regime_windows(merged, numeric)
    windows.to_csv(tables / "Regime_Windows.csv", index=False)
    fixed = cfg["regime_map"].get("axes") or None
    axes = regime_map.choose_axes(ctests, fixed)
    axes_reason = "fixed in config" if fixed else "smallest per-cluster p-values"
    if len(axes) < 2:
        axes = None
        print("Fewer than two numeric parameters; no regime map")
    else:
        points = regime_map.map_points(merged, axes)
        points.to_csv(tables / "Regime_Map_Points.csv", index=False)
        plots.regime_map(points, windows, axes, figs / "Regime_Map.png", dpi)
        print(f"Regime map on {axes[0]} and {axes[1]} ({axes_reason})")

    # How well nearest-centre placement generalises
    X_used = X.loc[merged["Sample_ID"]]
    loo = predict.leave_one_out(X_used, clusters, len(model["pca_components"]), seed)
    loo.to_csv(tables / "Placement_Leave_One_Out.csv", index=False)
    print(f"Leave-one-out placement accuracy: {100 * loo['Correct'].mean():.0f}%")

    # Summary and run record
    report.write_summary(reports / "Growth_Summary.txt", {
        "coverage": coverage, "audit": audit, "numeric": numeric,
        "categorical": categorical, "k": k,
        "cluster_sizes": "/".join(str(s) for s in sizes),
        "correlations": corr, "drift": drift, "confound_threshold": thr,
        "descriptor_tests": dtests, "cluster_numeric": ctests,
        "cluster_categorical": cat_tests, "n_permutations": n_perm,
        "regressions": regs, "axes": axes, "axes_reason": axes_reason,
        "windows": windows, "loo": loo, "alpha": alpha,
    })
    report.write_run_info(cfg, {
        "n_samples": len(merged), "n_in_log": int(coverage["In_Growth_Log"].sum()),
        "n_in_method1": len(X), "numeric_parameters": numeric,
        "categorical_parameters": categorical, "k": k,
        "not_separable_pairs": int(corr["Not_Separable"].sum()),
        "descriptor_pairs_significant": int(dtests["Significant"].sum()),
        "parameters_differing_by_cluster": ctests.loc[ctests["Significant"], "Parameter"].tolist()
        + cat_tests.loc[cat_tests["Significant"], "Parameter"].tolist(),
        "regime_map_axes": axes,
        "placement_loo_accuracy": round(float(loo["Correct"].mean()), 4),
    }, res / "run_info.json")
    print(f"Done. Results in {res}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    main(config)
