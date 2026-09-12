"""Tests of what the embeddings encode. Both run after clustering and do
not feed back into it."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler, normalize


def zoom_control(zoom_emb, factors):
    """Displacement of each embedding under simulated zoom, compared with the
    distance between different samples at native magnification.

    ``zoom_emb`` maps Sample_ID to an (n_factors, d) array of L2-normalised
    embeddings whose first row is the unzoomed image.

    Returns (per-image table, summary per factor, info dict).
    """
    ids = sorted(zoom_emb)
    rows = []
    for sid in ids:
        e = zoom_emb[sid]
        for j, f in enumerate(factors[1:], start=1):
            rows.append(dict(Sample_ID=sid, Zoom_Factor=f,
                             Embedding_Distance_From_Original=float(np.linalg.norm(e[j] - e[0]))))
    per_image = pd.DataFrame(rows)

    between = pdist(np.vstack([zoom_emb[s][0] for s in ids]))
    base = float(between.mean())

    summary_rows = []
    for f in factors[1:]:
        d = per_image.loc[per_image["Zoom_Factor"] == f, "Embedding_Distance_From_Original"]
        summary_rows.append(dict(Zoom_Factor=f,
                                 Mean_Zoom_Distance=round(float(d.mean()), 4),
                                 SD_Zoom_Distance=round(float(d.std()), 4),
                                 Mean_Between_Sample_Distance=round(base, 4),
                                 Ratio_Zoom_To_Between_Sample=round(float(d.mean() / base), 3)))
    summary = pd.DataFrame(summary_rows)
    info = dict(n_images=len(ids), max_zoom=float(max(factors)),
                mean_between_sample_distance=round(base, 4),
                max_ratio=float(summary["Ratio_Zoom_To_Between_Sample"].max()))
    return per_image, summary, info


def _reduced(emb, n_components, seed):
    """L2-normalise, standardise and reduce an embedding table to PCA scores."""
    meta = [c for c in ("Sample_ID", "Image_File") if c in emb.columns]
    X = emb.drop(columns=meta).to_numpy(dtype=float)
    X = StandardScaler().fit_transform(normalize(X, norm="l2", axis=1))
    n = min(n_components, X.shape[0] - 1, X.shape[1])
    return PCA(n_components=n, random_state=seed).fit_transform(X), list(emb["Sample_ID"])


def encoding_regression(emb, targets, n_components, seed):
    """Leave-one-out ridge regression from reduced embeddings to each target.

    ``targets`` maps a name to a Series indexed by Sample_ID. Returns one row
    per target with the cross-validated R².
    """
    X, ids = _reduced(emb, n_components, seed)
    index = pd.Index(ids)
    rows = []
    for name, series in targets.items():
        y = series.reindex(index).to_numpy(dtype=float)
        ok = ~np.isnan(y)
        if ok.sum() < 5:
            rows.append(dict(Target=name, LOO_R2=np.nan, N=int(ok.sum()),
                             Note="fewer than five valid values"))
            continue
        pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 25)),
                                 X[ok], y[ok], cv=LeaveOneOut())
        rows.append(dict(Target=name, LOO_R2=round(float(r2_score(y[ok], pred)), 4),
                         N=int(ok.sum()), Note=""))
    return pd.DataFrame(rows)


def build_targets(index, descriptor_path, id_column, descriptor_columns):
    """Regression targets: log10 magnification from the image metadata and
    the Method 1 descriptors, when the descriptor table is available.

    Returns (targets dict, list of descriptor target names).
    """
    targets, descriptors = {}, []
    by_id = index.set_index("Sample_ID")
    if by_id["Magnification_X"].notna().any():
        targets["log10(Magnification)"] = np.log10(by_id["Magnification_X"])

    if descriptor_path and Path(descriptor_path).exists():
        path = Path(descriptor_path)
        table = (pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls")
                 else pd.read_csv(path))
        if id_column in table.columns:
            table[id_column] = table[id_column].astype(str).str.strip()
            table = table.set_index(id_column)
            for col in descriptor_columns:
                if col in table.columns and table[col].notna().sum() > 5:
                    targets[col] = table[col]
                    descriptors.append(col)
    return targets, descriptors


def summarise_encoding(table, descriptors):
    """Compare the magnification R² with the best and mean descriptor R²."""
    mag = table[table["Target"].str.contains("Magnification")]["LOO_R2"]
    desc = table[table["Target"].isin(descriptors)]["LOO_R2"]
    return dict(
        magnification_r2=float(mag.iloc[0]) if len(mag) else np.nan,
        best_descriptor_r2=float(desc.max()) if len(desc) else np.nan,
        mean_descriptor_r2=float(desc.mean()) if len(desc) else np.nan,
        uninformative=table.loc[table["Target"].isin(descriptors) & (table["LOO_R2"] <= 0),
                                "Target"].tolist(),
    )
