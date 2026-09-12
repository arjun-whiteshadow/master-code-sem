# Comparison of Method 1 and Method 2

Measures how far the descriptor-based partition (Method 1) and the
embedding-based partitions (Method 2, one per architecture) agree, and puts
the internal quality of the two selected partitions side by side. Reads only
the result tables of the two methods.

## Run

```bash
pip install -r requirements.txt
python run_comparison.py
```

`requirements-lock.txt` lists the exact package versions that produced the
committed tables.

Both method pipelines must have been run first (or their committed
`results/tables/` must be present). Runs in seconds.

## Tests

```bash
pytest
```

`tests/test_agreement.py` checks the agreement indices, cluster-level matching and Hungarian layout on synthetic data.

## What is computed

- ARI, NMI, AMI, homogeneity, completeness and V-measure between the Method 1
  partition and each architecture's Method 2 partition, on the original
  labels.
- For the primary architecture: cross-tabulation, the best-matching Method 2
  cluster for each Method 1 cluster with capture and purity, and the list of
  samples that fall outside that mapping. A Hungarian assignment is written
  for laying out figures only.
- Silhouette, k-means/Ward ARI and subsample ARI of the two selected
  partitions.
- The encoding-regression, zoom-control and cross-architecture tables from
  Method 2, copied alongside for reference.

The descriptive names of the Method 1 clusters are set in `config.yaml`.

## Code

| File | Contents |
|---|---|
| `run_comparison.py` | Entry point |
| `src/agreement.py` | Agreement indices, cluster-level matching, Hungarian layout |
| `src/plots.py` | Figures |
| `src/report.py` | Text summary and run record |
| `tests/test_agreement.py` | Unit tests |

## Outputs

Written to `results/`:

| Path | Contents |
|---|---|
| `tables/Agreement_By_Architecture.csv` | Agreement indices per architecture |
| `tables/Method_Quality_Comparison.csv` | Internal quality of the two selected partitions |
| `tables/Crosstab_Primary.csv`, `Cluster_Level_Agreement.csv` | Cluster-level correspondence with the primary model |
| `tables/Matched_Assignments_Primary.csv`, `Disagreement_Samples.csv` | Per-sample labels from both methods |
| `figures/` | PNG figures at 300 dpi |
| `reports/Comparison_Summary.txt` | Plain-text summary |
