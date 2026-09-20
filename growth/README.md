# Growth conditions against the morphology clusters

Tests whether the Method 1 morphology clusters correspond to the MBE
conditions the samples were grown under, and draws the clusters in
growth-condition space. Reads the growth log and the Method 1 results;
Method 1 must have been run first. Settings are given in `config.yaml` and
explained in `../PROTOCOL.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements-lock.txt` lists the exact package versions that produced the
committed tables.

Place the growth log at `../method1/data/growth_conditions.xlsx`, one row per
sample. The columns are defined in `DATA_DICTIONARY.md`;
`python make_log_template.py` writes an empty sheet with the sample IDs of
the current Method 1 run filled in. The log is not committed.

## Run

```bash
python run_growth.py
```

Runs in under a minute on a laptop. An alternative config can be passed as
the first argument.

Two further scripts use a completed run:

```bash
python predict_regime.py new_samples.xlsx   # place new samples against the clusters
python suggest_conditions.py 3              # condition window of cluster 3
```

`predict_regime.py` takes a table with a `Sample` column and the six
descriptor columns, standardises and projects each row with the fitted
Method 1 model (`../method1/results/model.json`) and assigns the nearest
cluster centre. A sample further from its centre than any member of that
cluster was is marked out of range. `suggest_conditions.py` prints the
interquartile window of each parameter within one cluster and says which
parameters actually differ between clusters.

## Tests

```bash
pytest
```

The tests plant known relations in synthetic logs and check that each
step recovers them: ID matching and parameter screening, the co-variation
and drift flags, the correlation, per-cluster and regression tests, the
regime windows, and nearest-centre placement.

## Pipeline

1. Load the log; match sample IDs against the Method 1 run and report both
   directions of mismatch; drop parameters that are constant or mostly blank.
2. Check the log itself: Spearman correlation between parameters, and of
   each parameter with growth date (read from the `MMDDYY` prefix of the
   sample ID). Pairs and drifts above the threshold are carried into the
   summary.
3. Spearman correlation of every descriptor with every numeric parameter,
   Benjamini–Hochberg adjusted over the grid.
4. Kruskal–Wallis test of each numeric parameter across the clusters; an
   exact permutation test of each categorical parameter against the
   clusters; adjusted within each family.
5. Ordinary least squares of each descriptor on all numeric parameters,
   with leave-one-out R².
6. Quartile window of each parameter within each cluster, and a map of the
   clusters on the two parameters that separate them most.
7. Leave-one-out check of nearest-centre placement on the analysed samples.
8. Figures, tables and a text summary.

## Code

| File | Contents |
|---|---|
| `run_growth.py` | Pipeline entry point |
| `predict_regime.py` | Place new samples against the clusters |
| `suggest_conditions.py` | Print one cluster's condition window |
| `make_log_template.py` | Empty growth log with the current sample IDs |
| `src/config.py` | YAML loading and path resolution |
| `src/growth_log.py` | Log loading, parameter screening, join to Method 1 |
| `src/confounds.py` | Parameter co-variation and date drift |
| `src/associations.py` | Correlation, per-cluster and regression tests |
| `src/regime_map.py` | Cluster windows and map axes |
| `src/predict.py` | Projection with the saved model, nearest-centre assignment |
| `src/plots.py` | Figures |
| `src/report.py` | Text summary and run record |
| `tests/` | Unit tests |

## Outputs

Written to `results/`:

| Path | Contents |
|---|---|
| `tables/Growth_Log_Coverage.csv` | Every sample from either source and whether it was analysed |
| `tables/Growth_Parameter_Audit.csv` | Each log column with its status and reason |
| `tables/Growth_Parameter_Correlations.csv` | Spearman ρ between parameters, with the not-separable flag |
| `tables/Growth_Parameter_Drift.csv` | Spearman ρ of each parameter with growth date |
| `tables/Growth_Category_Dates.csv` | First and last growth date of each categorical level |
| `tables/Descriptor_vs_Condition.csv` | Descriptor × parameter correlations with q-values |
| `tables/Cluster_vs_Condition.csv` | Kruskal–Wallis per parameter with per-cluster medians |
| `tables/Cluster_vs_Category.csv`, `_Counts.csv` | Exact test per categorical parameter and the cross-tabulation |
| `tables/Descriptor_Regressions.csv` | R², leave-one-out R² and standardised coefficients per descriptor |
| `tables/Regime_Windows.csv` | Quartiles of each parameter within each cluster |
| `tables/Placement_Leave_One_Out.csv` | Held-out nearest-centre assignment of each sample |
| `figures/` | PNG figures at 300 dpi |
| `reports/Growth_Summary.txt` | Plain-text summary of the run |
| `run_info.json` | Settings, headline numbers and library versions |

Tables listing per-sample descriptor values alongside growth conditions
(`Merged_Features_Conditions.csv`, `Regime_Map_Points.csv`) are not
committed.
