# Method 1: clustering of morphology descriptors

Groups the 37 samples by six ImageJ-derived descriptors of the SEM image
(density, mean diameter, diameter CV, merged fraction, coverage and
orientation spread). Settings are given in `config.yaml` and explained in
`../PROTOCOL.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the feature table at `data/nanowire_features.xlsx` (one row per sample,
a `Sample` column and the six descriptor columns). The table is available from
the corresponding author on request.

## Run

```bash
python run_method1.py
```

Runs in well under a minute on a laptop; no GPU or network access is needed.
An alternative config can be passed as the first argument.

To add per-cluster contact sheets and the magnification cross-check, set
`input.image_folder` in `config.yaml` to a folder of the SEM TIFFs. Images
are read only.

## Pipeline

1. Load the feature table, check completeness, record acquisition variables.
2. Standardise descriptors to z-scores; list values with |z| > 3 (retained).
3. PCA; keep the fewest components reaching 80% cumulative variance.
4. k-means and Ward clustering at k = 2 to 6 on the retained components.
5. Select k by the silhouette rule in the protocol.
6. Subsample stability, leave-one-descriptor-out and outlier-removal checks.
7. DBSCAN parameter grid, for reference.
8. Figures, tables and a text summary.

## Code

| File | Contents |
|---|---|
| `run_method1.py` | Pipeline entry point |
| `src/config.py` | YAML loading and path resolution |
| `src/features.py` | Feature table loading, descriptor screening, TIFF magnification |
| `src/clustering.py` | Standardisation, PCA, clustering, k selection, stability tests |
| `src/plots.py` | Figures |
| `src/report.py` | Text summary and run record |

## Outputs

Written to `results/`:

| Path | Contents |
|---|---|
| `tables/Clustering_Comparison_All_k.csv` | Silhouette, inertia, sizes and k-means/Ward ARI at each k |
| `tables/Final_Cluster_Assignments.csv` | Cluster per sample, PC scores, silhouette, acquisition variables |
| `tables/Cluster_Profiles_*.csv` | Per-cluster descriptor statistics |
| `tables/Subsample_Stability.csv` | Mean and SD of subsample ARI at each k |
| `tables/Leave_One_Feature_Out.csv` | ARI against the main result with each descriptor removed |
| `tables/Outlier_Sensitivity.csv` | ARI against the main result with flagged samples removed |
| `tables/DBSCAN_*.csv` | Parameter grid and the displayed solution |
| `figures/` | PNG figures at 300 dpi |
| `reports/Method1_Summary.txt` | Plain-text summary of the run |
| `run_info.json` | Settings, headline numbers and library versions |

The committed tables are those produced by the run reported in the
manuscript. Tables listing per-sample descriptor values are not committed.
