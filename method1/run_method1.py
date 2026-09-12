"""Method 1: cluster the morphology descriptor table.

Usage:
    python run_method1.py [config.yaml]

Reads the settings in config.yaml, writes tables, figures and a summary to
the results folder. The input table is not modified.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from src import clustering, features, plots, report
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
    thr = cfg["thresholds"]
    plots.ANNOTATE_UP_TO = int(cfg["output"].get("annotate_up_to", 60))

    # Input
    raw = features.load_feature_table(cfg)
    X, audit = features.select_features(cfg, raw)
    acquisition = features.acquisition_table(cfg, raw)
    used = list(X.columns)
    excluded = [(r["Feature"], r["Reason"]) for _, r in audit.iterrows()
                if r["Status"] == "excluded"]
    print(f"{len(X)} samples, {len(used)} descriptors")
    for feat, reason in excluded:
        print(f"  excluded {feat}: {reason}")
    audit.to_csv(tables / "Feature_Selection_Audit.csv", index=False)
    X.reset_index().to_csv(tables / "Features_Used_Raw_Units.csv", index=False)
    acquisition.to_csv(tables / "Acquisition_Variables.csv", index=False)

    # Standardise and screen for extreme values
    Z = clustering.standardize(X)
    Z.reset_index().to_csv(tables / "Features_Standardized.csv", index=False)
    z_thr = float(cfg["stability"]["outlier_z_threshold"])
    outliers = clustering.flag_outliers(Z, z_thr)
    outliers.to_csv(tables / "Flagged_Outliers.csv", index=False)
    print(f"{len(outliers)} value(s) with |z| > {z_thr} (retained)")

    corr = plots.correlation_heatmap(X, figs / "Feature_Correlation_Heatmap.png", dpi)
    corr.to_csv(tables / "Feature_Correlation_Matrix.csv")

    # PCA
    scores, variance, loadings = clustering.run_pca(
        Z, float(cfg["pca"]["variance_threshold"]),
        int(cfg["pca"]["min_components"]), seed)
    n_pcs = scores.shape[1]
    var_kept = float(variance["Cumulative_Variance_Percent"].iloc[n_pcs - 1])
    print(f"PCA: {n_pcs} components retained ({var_kept:.1f}% of variance)")
    variance.to_csv(tables / "PCA_Explained_Variance.csv", index=False)
    scores.reset_index().to_csv(tables / "PCA_Scores.csv", index=False)
    loadings.rename_axis("Feature").reset_index().to_csv(tables / "PCA_Loadings.csv", index=False)
    plots.pca_variance(variance, float(cfg["pca"]["variance_threshold"]), figs, dpi)
    plots.pca_loadings(loadings, figs / "PCA_Loadings.png", dpi)
    plots.pca_scatter(scores, None, variance, "Samples in PCA space",
                      figs / "PCA_Scatter_Unclustered.png", dpi)
    S = scores.to_numpy()

    # Clustering over k and selection
    summary, km_labels, ward_labels = clustering.cluster_over_k(
        S, list(cfg["clustering"]["k_values"]),
        int(cfg["clustering"]["kmeans_n_init"]), seed)
    for _, r in summary.iterrows():
        print(f"  k = {int(r['k'])}: silhouette {r['Silhouette_KMeans']:.3f}, "
              f"sizes {r['Cluster_Sizes']}, k-means/Ward ARI {r['CrossAlgorithm_ARI']:.3f}")
    summary.to_csv(tables / "Clustering_Comparison_All_k.csv", index=False)
    for k, lab in km_labels.items():
        pd.DataFrame({"Sample_ID": scores.index, "Cluster": lab + 1}).to_csv(
            tables / f"KMeans_Assignments_k{k}.csv", index=False)

    final_k, k_reason, degenerate = clustering.select_k(
        summary, int(cfg["selection"]["min_cluster_size"]))
    labels = km_labels[final_k]
    final_row = summary[summary["k"] == final_k].iloc[0]
    print(f"Selected k = {final_k}. {k_reason}")
    (reports / "k_selection.txt").write_text(
        f"Selected k = {final_k}\n{k_reason}\nDegenerate: {degenerate}\n")

    # Describe the selected solution
    fit = clustering.per_sample_fit(S, labels, scores.index)
    assignments = pd.DataFrame({
        "Sample_ID": scores.index, "Cluster": labels + 1,
        "PC1": S[:, 0], "PC2": S[:, 1],
    }).merge(fit.drop(columns="Cluster"), on="Sample_ID")
    assignments = assignments.merge(acquisition, on="Sample_ID", how="left")
    assignments.to_csv(tables / "Final_Cluster_Assignments.csv", index=False)

    prof_raw, prof_std = clustering.cluster_profiles(X, Z, labels)
    prof_raw.to_csv(tables / "Cluster_Profiles_Original_Units.csv", index=False)
    prof_std.to_csv(tables / "Cluster_Profiles_Standardized.csv", index=False)
    sizes = np.bincount(labels, minlength=final_k)
    pd.DataFrame({"Cluster": range(1, final_k + 1), "N_Samples": sizes,
                  "Percent": np.round(100 * sizes / len(labels), 1)}).to_csv(
        tables / "Cluster_Sizes.csv", index=False)

    plots.pca_scatter(scores, labels, variance, f"k-means, k = {final_k}",
                      figs / "PCA_Scatter_Final_KMeans.png", dpi)
    plots.pca_scatter(scores, ward_labels[final_k], variance, f"Ward, k = {final_k}",
                      figs / "PCA_Scatter_Ward.png", dpi)
    plots.profile_heatmap(prof_std, figs / "Cluster_Profile_Heatmap.png", dpi)
    plots.dendrogram_plot(clustering.ward_linkage(S), scores.index, final_k,
                          figs / "Dendrogram.png", dpi)
    plots.k_selection(summary, final_k, thr, figs / "Cluster_Number_Selection.png", dpi)
    plots.sample_fit(fit, figs / "Per_Sample_Silhouette.png", dpi)

    # Stability and sensitivity
    stab = clustering.subsample_stability(
        S, km_labels, int(cfg["stability"]["n_subsample"]),
        float(cfg["stability"]["subsample_fraction"]),
        int(cfg["stability"]["subsample_n_init"]), seed)
    stab.to_csv(tables / "Subsample_Stability.csv", index=False)
    srow = stab[stab["k"] == final_k].iloc[0]
    print(f"Subsample ARI at k = {final_k}: "
          f"{srow['Mean_Subsample_ARI']:.3f} ± {srow['SD_Subsample_ARI']:.3f}")

    lofo = clustering.leave_one_feature_out(Z, cfg, final_k, labels)
    lofo.to_csv(tables / "Leave_One_Feature_Out.csv", index=False)
    print(f"Leave-one-descriptor-out ARI: {lofo['ARI_vs_Main_Result'].min():.3f} "
          f"to {lofo['ARI_vs_Main_Result'].max():.3f}")

    osens, outlier_ari = clustering.outlier_sensitivity(
        Z, cfg, final_k, labels, outliers["Sample_ID"])
    osens.to_csv(tables / "Outlier_Sensitivity.csv", index=False)
    n_flagged = int(osens["Samples_Removed"].iloc[0])
    if outlier_ari is not None:
        print(f"ARI without {n_flagged} flagged sample(s): {outlier_ari:.3f}")
    plots.stability(stab, summary, final_k, thr, figs / "Cluster_Stability.png", dpi)

    # DBSCAN, for reference only
    dbscan_note = "not run"
    if cfg["clustering"].get("run_dbscan", True):
        grid = clustering.dbscan_grid(S, cfg["clustering"]["dbscan_eps_values"],
                                      cfg["clustering"]["dbscan_min_samples"])
        grid.to_csv(tables / "DBSCAN_Parameter_Search.csv", index=False)
        row, db_labels = clustering.dbscan_representative(grid, S)
        pd.DataFrame({"Sample_ID": scores.index, "DBSCAN_Cluster": db_labels,
                      "eps": row["eps"], "min_samples": int(row["min_samples"])}).to_csv(
            tables / "DBSCAN_Assignments.csv", index=False)
        plots.pca_scatter(scores, db_labels, variance,
                          f"DBSCAN, eps = {row['eps']}, min_samples = {int(row['min_samples'])}",
                          figs / "PCA_Scatter_DBSCAN.png", dpi, noise_label=-1)
        n_noise = int((db_labels == -1).sum())
        dbscan_note = (f"eps = {row['eps']}, min_samples = {int(row['min_samples'])}: "
                       f"{int(row['Num_Clusters'])} clusters, {n_noise} of {len(db_labels)} "
                       f"samples labelled noise ({100 * n_noise / len(db_labels):.0f}%)")
        print(f"DBSCAN: {dbscan_note}")

    # Acquisition variables against cluster membership
    merged = plots.cluster_vs_acquisition(fit, acquisition,
                                          figs / "Cluster_vs_Acquisition.png", dpi)
    merged.to_csv(tables / "Cluster_vs_Acquisition.csv", index=False)
    if merged["Magnification_X"].notna().any():
        sub = merged.dropna(subset=["Magnification_X"])
        edges = [float(e) for e in cfg["magnification_band_edges"]]
        band = pd.cut(sub["Magnification_X"], edges)
        nmi = normalized_mutual_info_score(band.astype(str), sub["Cluster"])
        print(f"NMI between magnification band and cluster: {nmi:.3f}")

    # Contact sheets, if images are available
    image_paths = features.find_images(cfg["image_dir"], list(scores.index))
    if image_paths:
        n_sheets = plots.contact_sheets(image_paths, assignments[["Sample_ID", "Cluster"]],
                                        res / "contact_sheets", dpi)
        print(f"{n_sheets} contact sheet(s) written from {len(image_paths)} images")

    # Summary and run record
    report.write_summary(reports / "Method1_Summary.txt", {
        "n_samples": len(X), "n_features": len(used),
        "features_used": used, "features_excluded": excluded,
        "n_pcs": n_pcs, "pca_threshold": float(cfg["pca"]["variance_threshold"]),
        "variance_retained": var_kept,
        "summary": summary, "final_k": final_k, "k_reason": k_reason,
        "degenerate": degenerate, "final_row": final_row, "profiles_std": prof_std,
        "subsample_mean": float(srow["Mean_Subsample_ARI"]),
        "subsample_sd": float(srow["SD_Subsample_ARI"]),
        "lofo": lofo, "outlier_ari": outlier_ari, "n_flagged": n_flagged,
        "dbscan_note": dbscan_note, "outliers": outliers, "sample_fit": fit,
        "thresholds": thr,
    })
    report.write_run_info(cfg, {
        "n_samples": len(X), "features_used": used,
        "pca_components": n_pcs, "variance_retained_percent": round(var_kept, 2),
        "final_k": final_k, "degenerate": degenerate,
        "cluster_sizes": [int(s) for s in sizes],
        "silhouette": float(final_row["Silhouette_KMeans"]),
        "cross_algorithm_ari": float(final_row["CrossAlgorithm_ARI"]),
        "subsample_ari_mean": float(srow["Mean_Subsample_ARI"]),
    }, res / "run_info.json")
    print(f"Done. Results in {res}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    main(config)
