"""Compare the Method 1 and Method 2 partitions.

Usage:
    python run_comparison.py [config.yaml]

Reads the result tables written by run_method1.py and run_method2.py and
writes agreement tables, figures and a summary. Nothing in either results
folder is modified.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

from src import agreement, plots, report


def load_config(path):
    path = Path(path).resolve()
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    root = path.parent
    cfg["root"] = root
    cfg["m1_dir"] = (root / cfg["input"]["method1_results"]).resolve()
    cfg["m2_dir"] = (root / cfg["input"]["method2_results"]).resolve()
    cfg["results_dir"] = root / cfg["output"]["results_folder"]
    cfg["method1_cluster_names"] = {int(k): v for k, v in cfg["method1_cluster_names"].items()}
    return cfg


def read_table(folder, name):
    path = folder / "tables" / name
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the method pipelines first.")
    return pd.read_csv(path)


def read_optional(folder, name):
    path = folder / "tables" / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main(config_path):
    cfg = load_config(config_path)
    m1_dir, m2_dir = cfg["m1_dir"], cfg["m2_dir"]
    res = cfg["results_dir"]
    tables, figs, reports = res / "tables", res / "figures", res / "reports"
    for d in (tables, figs, reports):
        d.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg["output"]["figure_dpi"])
    primary = cfg["input"]["primary_model"]
    names = cfg["method1_cluster_names"]

    # Method 1 partition
    m1 = read_table(m1_dir, "Final_Cluster_Assignments.csv")[["Sample_ID", "Cluster"]]
    m1 = m1.rename(columns={"Cluster": "M1_Cluster"})
    m1["Sample_ID"] = m1["Sample_ID"].astype(str).str.strip()
    m1_k = int(m1["M1_Cluster"].nunique())

    # Method 2 partitions, one per architecture
    m2_models = read_table(m2_dir, "Model_Comparison_Summary.csv").set_index("Model")
    cnn = {}
    for name in m2_models.index:
        t = read_table(m2_dir, f"Cluster_Assignments_{name}.csv")
        t["Sample_ID"] = t["Sample_ID"].astype(str).str.strip()
        cnn[name] = t.rename(columns={"Cluster": "CNN_Cluster"})
    if primary not in cnn:
        raise SystemExit(f"No Method 2 assignments for the primary model '{primary}'.")
    print(f"Method 1: {len(m1)} samples, {m1_k} clusters")
    print(f"Method 2: {len(cnn)} architectures ({', '.join(cnn)}); primary {primary}")

    # Agreement with each architecture
    rows = []
    for name, t in cnn.items():
        merged = m1.merge(t, on="Sample_ID")
        sc = agreement.scores(merged["M1_Cluster"].to_numpy(), merged["CNN_Cluster"].to_numpy())
        rows.append({"CNN_Model": name, "Matched": len(merged),
                     "CNN_k": int(m2_models.loc[name, "Selected_k"]),
                     "CNN_Silhouette": round(float(m2_models.loc[name, "Silhouette"]), 4),
                     **{k: round(v, 4) for k, v in sc.items()}})
        print(f"  {name:<20s} ARI {sc['ARI']:+.3f}, NMI {sc['NMI']:.3f} (n = {len(merged)})")
    by_model = pd.DataFrame(rows).sort_values("ARI", ascending=False)
    by_model.to_csv(tables / "Agreement_By_Architecture.csv", index=False)
    plots.agreement_by_model(by_model, float(cfg["weak_ari"]),
                             figs / "Agreement_By_Architecture.png", dpi)

    # Primary model in detail
    merged = m1.merge(cnn[primary], on="Sample_ID")
    merged["M1_Name"] = merged["M1_Cluster"].map(lambda c: names.get(int(c), f"Cluster {c}"))
    only_m1 = set(m1["Sample_ID"]) - set(merged["Sample_ID"])
    only_cnn = set(cnn[primary]["Sample_ID"]) - set(merged["Sample_ID"])
    pd.DataFrame({"Sample_ID": sorted(only_m1) + sorted(only_cnn),
                  "Present_In": ["Method 1 only"] * len(only_m1) + ["Method 2 only"] * len(only_cnn)}
                 ).to_csv(tables / "Unmatched_Samples.csv", index=False)
    print(f"Matched {len(merged)} samples; {len(only_m1) + len(only_cnn)} unmatched")

    ct = pd.crosstab(merged["M1_Cluster"], merged["CNN_Cluster"])
    ct.to_csv(tables / "Crosstab_Primary.csv")
    plots.crosstab(ct, names, primary, figs / "Crosstab_Primary.png", dpi)
    agreement.hungarian_mapping(ct).to_csv(tables / "Display_Label_Mapping.csv", index=False)

    cl = agreement.cluster_level(merged, "M1_Cluster", "CNN_Cluster", names,
                                 cfg["agreement_levels"])
    cl.to_csv(tables / "Cluster_Level_Agreement.csv", index=False)
    for _, c in cl.iterrows():
        print(f"  M1 {c['M1_Cluster']} -> CNN {c['Best_CNN_Cluster']}: "
              f"{c['Shared']}/{c['M1_Size']} [{c['Agreement_Level']}]")

    expected = dict(zip(cl["M1_Cluster"], cl["Best_CNN_Cluster"]))
    merged["Expected_CNN"] = merged["M1_Cluster"].map(expected)
    merged["Agrees"] = merged["CNN_Cluster"] == merged["Expected_CNN"]
    merged.to_csv(tables / "Matched_Assignments_Primary.csv", index=False)
    dis = merged[~merged["Agrees"]]
    dis.to_csv(tables / "Disagreement_Samples.csv", index=False)
    print(f"  {len(dis)} of {len(merged)} samples outside the main mapping")

    # Internal quality of the two selected partitions
    m1_all = read_table(m1_dir, "Clustering_Comparison_All_k.csv")
    m1_sub = read_table(m1_dir, "Subsample_Stability.csv")
    m1_row = m1_all[m1_all["k"] == m1_k].iloc[0]
    m2_sub = read_table(m2_dir, "Subsample_Stability.csv")
    k2 = int(m2_models.loc[primary, "Selected_k"])
    q1 = dict(silhouette=float(m1_row["Silhouette_KMeans"]),
              cross_ari=float(m1_row["CrossAlgorithm_ARI"]),
              subsample=float(m1_sub.loc[m1_sub["k"] == m1_k, "Mean_Subsample_ARI"].iloc[0]))
    q2 = dict(silhouette=float(m2_models.loc[primary, "Silhouette"]),
              cross_ari=float(m2_models.loc[primary, "CrossAlgorithm_ARI"]),
              subsample=float(m2_sub.loc[m2_sub["k"] == k2, "Mean_Subsample_ARI"].iloc[0]))
    quality = pd.DataFrame([
        {"Measure": "Selected k", "Method_1": m1_k, "Method_2_primary": k2},
        {"Measure": "Mean silhouette", "Method_1": round(q1["silhouette"], 4),
         "Method_2_primary": round(q2["silhouette"], 4)},
        {"Measure": "k-means/Ward ARI", "Method_1": round(q1["cross_ari"], 4),
         "Method_2_primary": round(q2["cross_ari"], 4)},
        {"Measure": "Subsample ARI", "Method_1": round(q1["subsample"], 4),
         "Method_2_primary": round(q2["subsample"], 4)},
    ])
    quality.to_csv(tables / "Method_Quality_Comparison.csv", index=False)
    plots.method_quality(q1, q2, figs / "Method_Quality_Comparison.png", dpi)

    # Encoding results carried over from Method 2
    enc = read_optional(m2_dir, "Encoding_R2_By_Model.csv")
    if len(enc):
        enc.to_csv(tables / "Encoding_R2_By_Model.csv", index=False)
        plots.encoding(enc, figs / "Encoding_By_Architecture.png", dpi)
    zoom = read_optional(m2_dir, "Zoom_Control_Summary.csv")
    if len(zoom):
        zoom.to_csv(tables / "Zoom_Control_Summary.csv", index=False)
    xm = read_optional(m2_dir, "Cross_Model_Agreement.csv")
    if len(xm):
        xm.to_csv(tables / "Cross_Model_Agreement.csv", index=False)

    report.write_summary(reports / "Comparison_Summary.txt", {
        "n_matched": len(merged), "m1_k": m1_k, "primary": primary,
        "by_model": by_model, "quality": quality, "cluster_level": cl, "crosstab": ct,
        "disagreements": dis, "encoding": enc, "zoom": zoom, "cross_model": xm,
    })
    best = by_model.iloc[0]
    report.write_run_info(cfg, {
        "n_matched": len(merged), "method1_k": m1_k,
        "architectures": list(cnn), "primary_model": primary,
        "primary_ari": float(by_model.loc[by_model["CNN_Model"] == primary, "ARI"].iloc[0]),
        "best_model": best["CNN_Model"], "best_ari": float(best["ARI"]),
    }, res / "run_info.json")
    print(f"Done. Results in {res}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    main(config)
