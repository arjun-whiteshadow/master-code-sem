# Nanowire morphology clustering

Analysis code for the unsupervised grouping of GaAsSb nanowire SEM images
described in the accompanying manuscript. Two independent methods assign
37 MBE-grown samples to candidate morphology groups and are compared under
identical clustering rules.

| Folder | Input | Method |
|---|---|---|
| `method1/` | Six calibrated morphology descriptors per sample (ImageJ) | Standardise, PCA, k-means and Ward clustering over k = 2–6, stability tests |
| `method2/` | One SEM image per sample | Frozen ImageNet-pretrained CNN embeddings (ResNet50, EfficientNet-B0, MobileNetV3-Large, ConvNeXt-Tiny), then the same clustering chain; plus two tests of what the embeddings encode |
| `comparison/` | Outputs of the two methods | Agreement between methods and between architectures |
| `growth/` | Method 1 clusters and the MBE growth log | Tests of whether the clusters correspond to growth conditions; regime map; placement of new samples |

Every analytical setting is listed in `PROTOCOL.md` and mirrored in each
folder's `config.yaml`. The scripts apply those settings and report the
outcome; nothing in the selection of k depends on inspecting the result.

## Running

Each folder is self-contained and has its own `README.md`, `requirements.txt`
and `run_*.py` entry point. Run them in order:

```bash
cd method1    && python run_method1.py
cd ../method2 && python run_method2.py
cd ../comparison && python run_comparison.py
cd ../growth && python run_growth.py
```

Method 1, the comparison and the growth analysis run in seconds on a laptop.
Method 2 needs PyTorch and downloads pretrained weights (about 250 MB) on
first use; a CPU run takes a few minutes.

## Data availability

The morphology feature table and the SEM images are not distributed with this
repository. They are available from the corresponding author on reasonable
request. Place them at `method1/data/nanowire_features_<n>.xlsx` and
`method2/data/images/` respectively. The growth log read by `growth/` is
likewise not distributed; its columns are defined in `growth/DATA_DICTIONARY.md`.

The result tables the manuscript reports are committed under each folder's
`results/tables/`, so the numbers can be checked without re-running. Tables
that contain per-sample feature values or raw embeddings are excluded.

## Running on a different dataset

Nothing in the code is tied to the number of samples. To analyse a new image
set, add the two inputs, point the configs at them and run the three scripts
in order:

- `method1/data/nanowire_features_<n>.xlsx`: one row per sample, a `Sample`
  column and the six descriptor columns named in `method1/config.yaml`. The
  file is named by its sample count and referenced from `method1/config.yaml`
  and `method2/config.yaml`, so the committed configs always say which
  dataset the committed tables came from.
- `method2/data/images/`: one image per sample, filename stem equal to the
  `Sample` value.

Every table and figure under `results/` is regenerated. Re-run all three
scripts and commit their tables together; a partial re-run leaves the
committed results describing two different datasets. Settings that were
chosen for the 37-sample study and are worth revisiting for a larger set,
all in the `config.yaml` files: the range of `k_values`, `min_cluster_size`,
the DBSCAN `eps` grid, and the magnification band used by the zoom control.
The descriptive cluster names in `comparison/config.yaml` describe the
37-sample result and must be re-derived from the new cluster profiles.

The tag `v1.0-manuscript` marks the commit whose result tables are the ones
reported in the manuscript.

## Citation

See `CITATION.cff`.

## License

MIT. See `LICENSE`.
