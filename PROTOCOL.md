# Analysis protocol

This document lists the settings used by all three pipelines and the reasons
for them. The values are mirrored in each folder's `config.yaml`; the scripts
read those files and do not contain analytical constants of their own.

The two clustering methods share one set of rules (Sections 3–5) so that any
difference in their results reflects the input representation, not the
procedure.

## 1. Samples

Thirty-seven GaAsSb nanowire samples grown by molecular beam epitaxy, one
SEM image each, acquired at 30° stage tilt (29 images) or 35° (the seven
images of the `070823` batch), as recorded in the TIFF metadata; one image
(`051623D1`) carries no metadata. No sample is excluded from either method.
Samples are matched between methods by the image filename stem (for example
`061623C10`).

## 2. Inputs

### Method 1: morphology descriptors

Six descriptors measured in ImageJ from the segmented SEM image:

| Column | Quantity | Unit |
|---|---|---|
| `Density_NW_um2` | Nanowire areal density | NW µm⁻² |
| `Mean_Diameter_um` | Mean nanowire diameter | µm |
| `CV_percent` | Coefficient of variation of diameter | % |
| `Merged_percent` | Fraction of coalesced objects | % |
| `Foreground_Coverage_percent` | Segmented material coverage | % |
| `Angle_STD_deg` | Standard deviation of in-plane orientation | ° |

Columns present in the feature table but not used, with the reason:

| Column | Reason |
|---|---|
| `Image_Area_um2`, `Count` | Already expressed by density |
| `Diameter_STD_um` | Redundant with `CV_percent` |
| `Mean_Angle_deg` | Depends on the arbitrary rotation of the sample on the stage |
| `Merged_Count` | Less comparable across samples than `Merged_percent` |
| `Parasitic_Coverage_percent`, `Length_um`, `Taper_Ratio` | Not measured for this image set |

A descriptor with fewer than 80% valid values is dropped and the exclusion is
reported. Missing values in a retained descriptor stop the run; no imputation
is performed. Because parasitic coverage, length and taper were not measured,
morphology regimes defined by those quantities cannot be resolved by Method 1.

`Count` and `Image_Area_um2` are carried through to the output tables as
acquisition variables but are never used for clustering.

### Method 2: CNN embeddings

Each image is passed once through a frozen ImageNet-pretrained network with
the classification head removed, giving the pooled feature vector. No
training, fine-tuning, augmentation or labels are involved.

| Model | Weights | Embedding size | Role |
|---|---|---|---|
| ResNet50 | torchvision `IMAGENET1K_V2` | 2048 | Primary |
| EfficientNet-B0 | torchvision `IMAGENET1K_V1` | 1280 | Robustness check |
| MobileNetV3-Large | torchvision `IMAGENET1K_V2` | 960 | Robustness check |
| ConvNeXt-Tiny | torchvision `IMAGENET1K_V1` | 768 | Robustness check |

ResNet50 is the primary model: its clustering is the one reported in detail
and compared with Method 1. The other three are run with identical settings
to test whether conclusions depend on the choice of architecture. Each model
uses its own torchvision preprocessing transform. Images are converted to
8-bit RGB in memory; 16-bit and palette TIFFs are rescaled to the 0–255 range.

Embeddings are L2-normalised per image and then standardised per dimension
before PCA.

Method 1 labels and descriptors are not read at any point during embedding
extraction, dimensionality reduction or clustering. They are used only in the
encoding regression (Section 6), which runs after clustering is complete.

## 3. Dimensionality reduction

Principal component analysis on the standardised features. The number of
components retained is the smallest that reaches at least 80% cumulative
explained variance, with a minimum of two. All retained components are used
for clustering and for the silhouette calculation; PC1–PC2 scatter plots are
for display only.

## 4. Clustering

| Setting | Value |
|---|---|
| k tested | 2, 3, 4, 5, 6 |
| Primary algorithm | k-means, `n_init = 50`, `random_state = 42` |
| Second algorithm | Agglomerative, Ward linkage, Euclidean distance |
| Exploratory | DBSCAN over a grid of `eps` and `min_samples` |
| Random seed | 42 throughout |

DBSCAN is reported for completeness and is not used to select the grouping.
Its `eps` grid differs between methods because the PCA spaces have different
scales (six standardised descriptors versus up to 2048 standardised embedding
dimensions).

## 5. Selection of k

Among the tested k, the one with the highest mean silhouette on the retained
components is selected, excluding any solution that contains a cluster of
fewer than two samples. If every solution contains such a cluster the highest
silhouette overall is taken and the result is flagged as degenerate. Ties are
resolved toward the smaller k.

The rule is applied by the code from `config.yaml`; the k it returns is
reported whatever it is, and the two methods are not required to agree.

Thresholds used to label a result as weak in the reports (they do not alter
the analysis): mean silhouette below 0.25; k-means/Ward ARI below 0.30;
mean subsample ARI below 0.50.

## 6. Stability and sensitivity tests

Run for both methods:

1. **Cross-algorithm agreement.** Adjusted Rand index (ARI) between the
   k-means and Ward partitions at each k.
2. **Subsample stability.** 200 draws of 80% of the samples without
   replacement; each draw is re-clustered (`n_init = 10`) and compared by ARI
   with the full-data labels restricted to the drawn samples. Mean and
   standard deviation are reported at each k.
3. **Outlier sensitivity.** Flagged samples are retained in the main
   analysis. The pipeline is re-run without them and the ARI against the
   main result is reported. Method 1 flags any standardised descriptor value
   with |z| > 3. Method 2 flags samples further than two standard deviations
   above the mean distance to their own cluster centre.

Method 1 only:

4. **Leave-one-descriptor-out.** The pipeline is re-run six times, each
   omitting one descriptor, and the ARI against the main result is reported.

Method 2 only:

5. **Banner crop.** Embeddings are re-extracted with the bottom 10% of each
   image removed (the region that holds the SEM information bar) and the
   result compared by ARI with the main run, which uses the full frame.
6. **Synthetic zoom control.** Each image is re-rendered at 1.4×, 2.0×, 2.8×
   and 4.0× apparent magnification by centre-cropping and resampling to the
   original size. The material content is unchanged. The embedding
   displacement caused by zoom alone is compared with the mean distance
   between different samples at native magnification. Images in the 5–15 kX
   band are used so that the zoomed views stay within the range of
   magnifications present in the dataset.
7. **Encoding regression.** Leave-one-out cross-validated ridge regression
   from the first 20 principal components of each model's embeddings to
   (a) log10 magnification read from the TIFF metadata and (b) each Method 1
   descriptor. A higher R² for magnification than for any descriptor
   indicates that the embedding represents the acquisition setting more
   strongly than the material.

## 7. Comparison

Agreement between the Method 1 partition and each architecture's Method 2
partition is measured by ARI, NMI, AMI, homogeneity, completeness and
V-measure on the original cluster labels. Cluster numbering is arbitrary in
both methods; a Hungarian assignment is computed only to lay out the
cross-tabulation and is not used in any score.

Descriptive names attached to Method 1 clusters are derived from their
measured descriptor profiles. No name is attached to a Method 2 cluster on
the basis of embeddings alone.

## 8. Reporting

Every number in the manuscript is traceable to a CSV table under
`results/tables/`. Clusters are described as candidate morphology groups;
identification with growth regimes requires inspection of the images and
correlation with growth conditions. Section 9 describes that correlation;
the manuscript's results do not depend on it.

## 9. Growth conditions

The `growth/` pipeline tests whether the Method 1 clusters correspond to the
MBE conditions recorded for each sample. It reads the growth log (one row
per sample; columns defined in `growth/DATA_DICTIONARY.md`) and the Method 1
results, and does not alter either. The rules below were written before the
growth log was available.

**Parameters.** The numeric and categorical parameters tested are listed in
`growth/config.yaml`. A parameter with fewer than 80% valid values, or with
the same value in every run, is left out and the exclusion reported. A
numeric column containing text stops the run. Samples absent from either
the log or the Method 1 run are listed and excluded from every test.

**Checks on the log.** Before any association test, Spearman correlation is
computed between every pair of numeric parameters and between each parameter
and the growth date (read from the `MMDDYY` prefix of the sample ID). A
pair with |ρ| ≥ 0.70 is reported as not separable; a parameter with
|ρ| ≥ 0.70 against date is reported as drifting. Both flags are carried
into the summary and qualify any effect found for those parameters. The
first and last growth date of each categorical level is tabulated for the
same reason.

**Tests.** Three families, each adjusted for multiple comparisons by the
Benjamini–Hochberg procedure within the family, with q < 0.05 reported as
significant:

1. Spearman correlation of each descriptor with each numeric parameter.
2. Kruskal–Wallis test of each numeric parameter across the clusters, and
   for each categorical parameter an exact test of independence from the
   cluster labels by 10,000 permutations (seed 42), the chi-square
   statistic ranking the permuted tables.
3. Ordinary least squares of each descriptor on all numeric parameters
   together, with the in-sample R² and the leave-one-out R².

**Regime map.** Each cluster's window on a parameter is the interquartile
range of its members. The clusters are drawn on the two numeric parameters
with the smallest Kruskal–Wallis p-value unless two are fixed in the
config. A cluster is described as a candidate growth regime only where it
differs from the others on at least one parameter after adjustment and the
direction is physically sensible; that judgement is made from the images
and the growth record and is not made by the code.

**Placement of new samples.** A new sample's descriptors are standardised
with the scaler, projected with the PCA loadings and assigned to the
nearest cluster centre saved by the Method 1 run (`method1/results/model.json`);
nothing is refitted. A sample further from its centre than any member of
that cluster was is reported as outside the known clusters. The same
nearest-centre rule is checked by leave-one-out on the analysed samples,
with the cluster labels held fixed, and its accuracy reported.
