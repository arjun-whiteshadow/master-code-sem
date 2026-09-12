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
```

Method 1 and the comparison run in seconds on a laptop. Method 2 needs PyTorch
and downloads pretrained weights (about 250 MB) on first use; a CPU run takes a
few minutes.

## Data availability

The morphology feature table and the SEM images are not distributed with this
repository. They are available from the corresponding author on reasonable
request. Place them at `method1/data/nanowire_features.xlsx` and
`method2/data/images/` respectively.

The result tables the manuscript reports are committed under each folder's
`results/tables/`, so the numbers can be checked without re-running. Tables
that contain per-sample feature values or raw embeddings are excluded.

## Citation

See `CITATION.cff`.

## License

MIT. See `LICENSE`.
