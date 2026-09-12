"""Method 2: cluster pretrained-CNN embeddings of the SEM images.

Usage:
    python run_method2.py [config.yaml]

Reads the settings in config.yaml, embeds every image with each configured
network, clusters the primary model's embeddings with the same rules as
Method 1, runs the stability and encoding tests, and writes tables, figures
and a summary to the results folder. Images are only ever read.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src import clustering, confounds, embeddings, images, plots, report
from src.config import load_config


def main(config_path):
    cfg = load_config(config_path)
    seed = int(cfg["random_seed"])
    images.set_seeds(seed)

    res = cfg["results_dir"]
    tables, figs, reports = res / "tables", res / "figures", res / "reports"
    for d in (tables, figs, reports):
        d.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg["output"]["figure_dpi"])
    thr = cfg["thresholds"]
    primary = cfg["models"]["primary"]
    plots.ANNOTATE_UP_TO = int(cfg["output"].get("annotate_up_to", 60))

    # Image index and quality checks
    index, qc = images.build_image_index(cfg)
    index.to_csv(tables / "Image_Index.csv", index=False)
    qc.to_csv(tables / "Image_QC.csv", index=False)
    paths = images.image_paths(cfg, index)
    n_found = len(index)
    print(f"{n_found} images found, {len(paths)} readable")
    mags = index["Magnification_X"].dropna()
    if len(mags):
        print(f"Magnification {mags.min() / 1000:.1f} to {mags.max() / 1000:.1f} kX "
              f"({mags.max() / mags.min():.0f}-fold range)")
    device = images.get_device()
    print(f"Device: {device}")

    # Embeddings from every configured network
    print("Extracting embeddings")
    emb_by_model, model_status = embeddings.extract_all(paths, cfg, device)
    model_status.to_csv(tables / "Model_Extraction_Status.csv", index=False)
    if primary not in emb_by_model:
        note = model_status.loc[model_status["Model"] == primary, "Note"].iloc[0]
        raise SystemExit(f"Primary model '{primary}' produced no embeddings: {note}")
    for name, df in emb_by_model.items():
        df.to_csv(tables / f"Embeddings_{name}.csv", index=False)

    # Same clustering chain for each architecture
    fits, rows, labels_by_model = {}, [], {}
    for name, emb in emb_by_model.items():
        fit = clustering.fit_embeddings(emb, cfg)
        fits[name] = fit
        k = fit["final_k"]
        row = fit["summary"][fit["summary"]["k"] == k].iloc[0]
        labels_by_model[name] = fit["km"][k]
        pd.DataFrame({"Sample_ID": fit["scores"].index, "Cluster": fit["km"][k] + 1}).to_csv(
            tables / f"Cluster_Assignments_{name}.csv", index=False)
        rows.append({"Model": name, "Embedding_Dimension": emb.shape[1] - 2,
                     "PCs_Retained": fit["scores"].shape[1], "Selected_k": k,
                     "Cluster_Sizes": row["Cluster_Sizes"],
                     "Silhouette": row["Silhouette_KMeans"],
                     "CrossAlgorithm_ARI": row["CrossAlgorithm_ARI"],
                     "Degenerate": fit["degenerate"]})
        print(f"  {name:<20s} k = {k}, sizes {row['Cluster_Sizes']}, "
              f"silhouette {row['Silhouette_KMeans']:.3f}, "
              f"k-means/Ward ARI {row['CrossAlgorithm_ARI']:.3f}")
    model_summary = pd.DataFrame(rows)
    model_summary.to_csv(tables / "Model_Comparison_Summary.csv", index=False)

    cross_model = pd.DataFrame()
    if len(labels_by_model) > 1:
        cross_model = clustering.cross_model_agreement(labels_by_model)
        cross_model.to_csv(tables / "Cross_Model_Agreement.csv", index=False)
        plots.model_comparison(model_summary, figs / "Architecture_Comparison.png", dpi)

    # Primary model in detail
    fit = fits[primary]
    scores, variance, summary = fit["scores"], fit["variance"], fit["summary"]
    final_k, labels = fit["final_k"], fit["km"][fit["final_k"]]
    S = scores.to_numpy()
    n_pcs = scores.shape[1]
    var_kept = float(variance["Cumulative_Variance_Percent"].iloc[n_pcs - 1])
    final_row = summary[summary["k"] == final_k].iloc[0]
    print(f"{primary}: {n_pcs} PCs ({var_kept:.1f}% of variance); selected k = {final_k}")
    print(f"  {fit['reason']}")

    variance.to_csv(tables / "PCA_Explained_Variance.csv", index=False)
    scores.rename_axis("Sample_ID").reset_index().to_csv(tables / "PCA_Scores.csv", index=False)
    summary.to_csv(tables / "Clustering_Comparison_All_k.csv", index=False)
    for k, lab in fit["km"].items():
        pd.DataFrame({"Sample_ID": scores.index, "Cluster": lab + 1}).to_csv(
            tables / f"KMeans_Assignments_k{k}.csv", index=False)
    (reports / "k_selection.txt").write_text(
        f"Primary model: {primary}\nSelected k = {final_k}\n{fit['reason']}\n"
        f"Degenerate: {fit['degenerate']}\n")

    v1, v2 = variance["Explained_Variance_Percent"].iloc[:2]
    axis = (f"PC1 ({v1:.1f}%)", f"PC2 ({v2:.1f}%)")
    plots.pca_variance(variance, float(cfg["pca"]["variance_threshold"]), figs, dpi)
    plots.k_selection(summary, final_k, thr, figs / "Cluster_Number_Selection.png", dpi)
    plots.scatter_2d(S[:, :2], None, scores.index, "Samples in PCA space", *axis,
                     figs / "PCA_Scatter_Unclustered.png", dpi)
    plots.scatter_2d(S[:, :2], labels, scores.index, f"k-means, k = {final_k}", *axis,
                     figs / "PCA_Scatter_Final_KMeans.png", dpi)
    plots.scatter_2d(S[:, :2], fit["ward"][final_k], scores.index, f"Ward, k = {final_k}",
                     *axis, figs / "PCA_Scatter_Ward.png", dpi)
    plots.dendrogram_plot(clustering.ward_linkage(S), scores.index, final_k,
                          figs / "Dendrogram.png", dpi)

    # Assignments with per-sample fit and acquisition metadata
    fit_table = clustering.per_sample_fit(S, labels, scores.index,
                                          float(cfg["stability"]["outlier_sd_threshold"]))
    assign = fit_table.copy()
    assign.insert(2, "PC1", S[:, 0])
    assign.insert(3, "PC2", S[:, 1])
    assign = assign.merge(index[["Sample_ID", "Filename", "Magnification_X", "Pixel_Size_nm",
                                 "Width_px", "Height_px"]], on="Sample_ID", how="left")
    assign.to_csv(tables / "Final_Cluster_Assignments.csv", index=False)
    sizes = np.bincount(labels, minlength=final_k)
    pd.DataFrame({"Cluster": range(1, final_k + 1), "N_Samples": sizes,
                  "Percent": np.round(100 * sizes / len(labels), 1)}).to_csv(
        tables / "Cluster_Sizes.csv", index=False)
    flagged = assign.loc[assign["Outlier_Flag"], "Sample_ID"].tolist()
    print(f"  cluster sizes {'/'.join(map(str, sizes))}; {len(flagged)} sample(s) flagged")
    plots.cluster_vs_magnification(assign, figs / "Cluster_vs_Magnification.png", dpi)

    # Stability
    stab = clustering.subsample_stability(
        S, fit["km"], int(cfg["stability"]["n_subsample"]),
        float(cfg["stability"]["subsample_fraction"]),
        int(cfg["stability"]["subsample_n_init"]), seed)
    stab.to_csv(tables / "Subsample_Stability.csv", index=False)
    srow = stab[stab["k"] == final_k].iloc[0]
    print(f"Subsample ARI at k = {final_k}: "
          f"{srow['Mean_Subsample_ARI']:.3f} ± {srow['SD_Subsample_ARI']:.3f}")
    osens, outlier_ari = clustering.outlier_sensitivity(fit["Z"], cfg, final_k, labels, flagged)
    osens.to_csv(tables / "Outlier_Sensitivity.csv", index=False)
    if outlier_ari is not None:
        print(f"ARI without {len(flagged)} flagged sample(s): {outlier_ari:.3f}")
    plots.stability(stab, summary, final_k, thr, figs / "Cluster_Stability.png", dpi)

    crop_note = "not run"
    if cfg["stability"].get("run_crop_sensitivity", True):
        frac = float(cfg["stability"]["crop_sensitivity_fraction"])
        emb_c, info_c = embeddings.extract(primary, paths, cfg, device, crop_fraction=frac)
        if emb_c is not None:
            fit_c = clustering.fit_embeddings(emb_c, cfg)
            k_c = min(final_k, max(fit_c["km"]))
            ari_c = float(adjusted_rand_score(labels, fit_c["km"][k_c]))
            pd.DataFrame([{"Crop_Fraction": frac, "Selected_k": fit_c["final_k"],
                           "ARI_vs_Main_Result": round(ari_c, 4)}]).to_csv(
                tables / "Crop_Sensitivity.csv", index=False)
            crop_note = (f"removing the bottom {frac:.0%} of each image gives ARI {ari_c:.3f} "
                         f"against the main result at k = {k_c} "
                         f"(its own selected k = {fit_c['final_k']})")
        else:
            crop_note = f"failed: {info_c['error'][:80]}"
        print(f"Crop sensitivity: {crop_note}")

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
        plots.scatter_2d(S[:, :2], db_labels, scores.index,
                         f"DBSCAN, eps = {row['eps']}, min_samples = {int(row['min_samples'])}",
                         *axis, figs / "PCA_Scatter_DBSCAN.png", dpi, noise_label=-1)
        n_noise = int((db_labels == -1).sum())
        dbscan_note = (f"eps = {row['eps']}, min_samples = {int(row['min_samples'])}: "
                       f"{int(row['Num_Clusters'])} clusters, {n_noise} of {len(db_labels)} "
                       f"samples labelled noise ({100 * n_noise / len(db_labels):.0f}%)")
        print(f"DBSCAN: {dbscan_note}")

    # UMAP, for display only
    if cfg["visualization"].get("run_umap", True):
        try:
            import umap
            vis = cfg["visualization"]
            coords = umap.UMAP(n_neighbors=min(int(vis["umap_n_neighbors"]), len(scores) - 1),
                               min_dist=float(vis["umap_min_dist"]),
                               metric=str(vis["umap_metric"]),
                               random_state=seed).fit_transform(S)
            pd.DataFrame({"Sample_ID": scores.index, "UMAP1": coords[:, 0],
                          "UMAP2": coords[:, 1]}).to_csv(tables / "UMAP_Coordinates.csv", index=False)
            plots.scatter_2d(coords, labels, scores.index,
                             f"UMAP of the retained components, k-means k = {final_k}",
                             "UMAP1", "UMAP2", figs / "UMAP_Final_KMeans.png", dpi)
        except Exception as exc:
            print(f"UMAP skipped: {type(exc).__name__}: {exc}")

    # Encoding tests
    zoom_summary, zoom_info = pd.DataFrame(), None
    conf = cfg["confounds"]
    if conf.get("run_zoom_control", True):
        lo, hi = [float(v) * 1000 for v in conf["zoom_magnification_range_kx"]]
        in_band = index["Loaded_OK"] & index["Magnification_X"].between(lo, hi)
        band_ids = index.loc[in_band, "Sample_ID"].tolist()
        if len(band_ids) < int(conf.get("zoom_min_images", 5)):
            band_ids = list(paths)
            print("Zoom control: too few images in the magnification band; using all")
        factors = [float(f) for f in conf["zoom_factors"]]
        zoom_emb = embeddings.zoom_series(primary, {s: paths[s] for s in band_ids},
                                          factors, cfg, device)
        if zoom_emb:
            per_image, zoom_summary, zoom_info = confounds.zoom_control(zoom_emb, factors)
            per_image.to_csv(tables / "Zoom_Control_Distances.csv", index=False)
            zoom_summary.to_csv(tables / "Zoom_Control_Summary.csv", index=False)
            for _, z in zoom_summary.iterrows():
                print(f"Zoom {z['Zoom_Factor']:.1f}x: displacement {z['Mean_Zoom_Distance']:.3f} "
                      f"({z['Ratio_Zoom_To_Between_Sample']:.2f} of between-sample distance)")

    encoding_table, encoding_summary, encoding_by_model = pd.DataFrame(), None, pd.DataFrame()
    if conf.get("run_encoding_regression", True):
        targets, descriptors = confounds.build_targets(
            index, cfg["descriptor_path"], cfg["input"]["descriptor_id_column"],
            cfg["input"]["descriptor_columns"])
        if targets:
            n_comp = int(conf["encoding_pca_components"])
            rows = []
            for name, emb in emb_by_model.items():
                table = confounds.encoding_regression(emb, targets, n_comp, seed)
                s = confounds.summarise_encoding(table, descriptors)
                rows.append({"Model": name, "Magnification_R2": s["magnification_r2"],
                             "Best_Morphology_R2": s["best_descriptor_r2"],
                             "Mean_Morphology_R2": s["mean_descriptor_r2"]})
                if name == primary:
                    encoding_table, encoding_summary = table, s
            encoding_by_model = pd.DataFrame(rows)
            encoding_table.to_csv(tables / "Embedding_Encoding_R2.csv", index=False)
            encoding_by_model.to_csv(tables / "Encoding_R2_By_Model.csv", index=False)
            plots.encoding_by_model(encoding_by_model, figs / "Encoding_By_Architecture.png", dpi)
            for _, e in encoding_table.iterrows():
                if not pd.isna(e["LOO_R2"]):
                    print(f"LOO R2, {e['Target']}: {e['LOO_R2']:+.3f}")
        else:
            print("Encoding regression skipped: no magnification metadata and no descriptor table")
    plots.confound_tests(zoom_summary, encoding_table, figs / "Confound_Tests.png", dpi)

    # Image previews
    plots.input_preview(paths, float(cfg["preprocessing"]["crop_bottom_fraction"]),
                        figs / "Network_Input_Preview.png", dpi)
    if cfg["output"].get("make_contact_sheets", True):
        n_sheets = plots.contact_sheets(paths, assign[["Sample_ID", "Cluster"]],
                                        res / "contact_sheets", dpi)
        print(f"{n_sheets} contact sheet(s) written")

    # Summary and run record
    report.write_summary(reports / "Method2_Summary.txt", {
        "n_found": n_found, "n_embedded": len(scores), "device": str(device),
        "crop_fraction": float(cfg["preprocessing"]["crop_bottom_fraction"]),
        "model_status": model_status, "model_summary": model_summary,
        "cross_model": cross_model, "primary": primary, "steps": fit["steps"],
        "n_pcs": n_pcs, "pca_threshold": float(cfg["pca"]["variance_threshold"]),
        "variance_retained": var_kept, "summary": summary, "final_k": final_k,
        "k_reason": fit["reason"], "degenerate": fit["degenerate"], "final_row": final_row,
        "subsample_mean": float(srow["Mean_Subsample_ARI"]),
        "subsample_sd": float(srow["SD_Subsample_ARI"]),
        "outlier_ari": outlier_ari, "n_flagged": len(flagged),
        "crop_note": crop_note, "dbscan_note": dbscan_note,
        "zoom_summary": zoom_summary, "encoding_table": encoding_table,
        "encoding_summary": encoding_summary, "thresholds": thr,
    })
    report.write_run_info(cfg, {
        "primary_model": primary, "models_embedded": list(emb_by_model),
        "n_images": len(scores), "pca_components": n_pcs,
        "variance_retained_percent": round(var_kept, 2),
        "final_k": final_k, "degenerate": fit["degenerate"],
        "cluster_sizes": [int(s) for s in sizes],
        "silhouette": float(final_row["Silhouette_KMeans"]),
        "cross_algorithm_ari": float(final_row["CrossAlgorithm_ARI"]),
        "subsample_ari_mean": float(srow["Mean_Subsample_ARI"]),
        "magnification_r2": encoding_summary["magnification_r2"] if encoding_summary else None,
        "best_descriptor_r2": encoding_summary["best_descriptor_r2"] if encoding_summary else None,
    }, res / "run_info.json")
    print(f"Done. Results in {res}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    main(config)
