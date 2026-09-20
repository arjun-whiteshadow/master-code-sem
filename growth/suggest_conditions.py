"""Growth-condition window of one Method 1 cluster, from a completed run.

Usage:
    python suggest_conditions.py CLUSTER [config.yaml]

Prints the interquartile window of each numeric parameter within the
cluster, marking the parameters on which the cluster differs from the
others, and the substrate (or other categorical) levels its members were
grown on. Parameters that do not differ between clusters are listed so
that they are not mistaken for a recommendation.
"""

import sys
from pathlib import Path

import pandas as pd

from src.config import load_config


def _read(tables, name):
    path = tables / name
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run run_growth.py first.")
    return pd.read_csv(path)


def main(cluster, config_path):
    cfg = load_config(config_path)
    tables = cfg["results_dir"] / "tables"
    windows = _read(tables, "Regime_Windows.csv")
    numeric = _read(tables, "Cluster_vs_Condition.csv").set_index("Parameter")
    counts = _read(tables, "Cluster_vs_Category_Counts.csv")

    if cluster not in set(windows["Cluster"]):
        raise SystemExit(f"Cluster {cluster} not in the run; "
                         f"clusters are {sorted(int(c) for c in windows['Cluster'].unique())}.")
    w = windows[windows["Cluster"] == cluster].set_index("Parameter")

    print(f"Cluster {cluster}: growth conditions of its {int(w['N'].max())} member(s)")
    print(f"{'Parameter':<26s} {'Q1':>12s} {'Median':>12s} {'Q3':>12s}  differs between clusters?")
    for p, r in w.iterrows():
        t = numeric.loc[p]
        verdict = f"yes (q = {t['q_value']:.3f})" if t["Significant"] else f"no (q = {t['q_value']:.3f})"
        print(f"{p:<26s} {r['Q1']:>12g} {r['Median']:>12g} {r['Q3']:>12g}  {verdict}")

    sub = counts[(counts["Cluster"] == cluster) & (counts["Count"] > 0)]
    for p, g in sub.groupby("Parameter"):
        levels = ", ".join(f"{r['Level']} ({int(r['Count'])})" for _, r in g.iterrows())
        print(f"{p:<26s} {levels}")

    print("\nOnly a parameter that differs between clusters is a candidate setting for "
          "this morphology; the others show where the members happened to be grown.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    config = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "config.yaml"
    main(int(sys.argv[1]), config)
