# Method 2: clustering of pretrained-CNN embeddings

Embeds each SEM image with frozen ImageNet-pretrained networks and clusters
the embeddings with the same rules as Method 1. Four architectures are run;
ResNet50 is the primary model and the others test whether the outcome
depends on the network. Two further tests ask whether the embeddings
represent the material or the acquisition magnification. Settings are given
in `config.yaml` and explained in `../PROTOCOL.md`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements-lock.txt` lists the exact package versions that produced the
committed tables; use it instead of `requirements.txt` to reproduce them.

Place the SEM images in `data/images/` (one image per sample; the filename
stem is the sample ID and must match the `Sample` column of the Method 1
table). Pretrained weights (about 250 MB in total) are downloaded by
torchvision on first use and cached; later runs are offline.

The encoding regression reads the Method 1 descriptor table from the path in
`input.descriptor_file`. It runs after clustering and does not affect any
clustering result. Leave the path empty to skip it.

## Run

```bash
python run_method2.py
```

A few minutes on a CPU or Apple-silicon laptop; a GPU is used if available.

## Pipeline

1. Index the images; record size, mode, checksum and SEM metadata
   (magnification, pixel size) from the Zeiss TIFF tag.
2. Embed every image with each configured network (full frame, the
   network's own transform, no augmentation).
3. For each network: L2-normalise, standardise, PCA to 80% variance, k-means
   and Ward at k = 2 to 6, select k by the silhouette rule. Write the
   assignments for every network.
4. Primary model in detail: per-sample fit, subsample stability,
   outlier-removal and banner-crop sensitivity, DBSCAN grid, UMAP display.
5. Synthetic zoom control and leave-one-out encoding regression.
6. Figures, tables and a text summary.

## Code

| File | Contents |
|---|---|
| `run_method2.py` | Pipeline entry point |
| `src/config.py` | YAML loading and path resolution |
| `src/images.py` | Image indexing, safe loading, cropping, zoom simulation, SEM metadata |
| `src/embeddings.py` | Model registry and frozen feature extraction |
| `src/clustering.py` | Normalisation, PCA, clustering, k selection, stability tests |
| `src/confounds.py` | Zoom control and encoding regression |
| `src/plots.py` | Figures |
| `src/report.py` | Text summary and run record |

## Outputs

Written to `results/`:

| Path | Contents |
|---|---|
| `tables/Model_Extraction_Status.csv` | Weights version, embedding size and status per network |
| `tables/Model_Comparison_Summary.csv` | Selected k, sizes, silhouette and k-means/Ward ARI per network |
| `tables/Cluster_Assignments_<model>.csv` | Final cluster per sample for each network |
| `tables/Cross_Model_Agreement.csv` | Pairwise ARI between networks |
| `tables/Clustering_Comparison_All_k.csv` | Primary model: metrics at each k |
| `tables/Final_Cluster_Assignments.csv` | Primary model: cluster, PC scores, silhouette, image metadata |
| `tables/Subsample_Stability.csv`, `Outlier_Sensitivity.csv`, `Crop_Sensitivity.csv` | Stability tests |
| `tables/Zoom_Control_*.csv` | Embedding displacement under simulated zoom |
| `tables/Embedding_Encoding_R2.csv`, `Encoding_R2_By_Model.csv` | Leave-one-out R² for magnification and each descriptor |
| `tables/Image_Index.csv`, `Image_QC.csv` | Image inventory and checks |
| `figures/` | PNG figures at 300 dpi |
| `reports/Method2_Summary.txt` | Plain-text summary of the run |
| `run_info.json` | Settings, headline numbers and library versions |

The committed tables are those produced by the run reported in the
manuscript. The raw embedding matrices (`Embeddings_<model>.csv`) are
written locally but not committed.
