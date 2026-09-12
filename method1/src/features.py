"""Loading and screening of the morphology feature table."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Columns that may be present in the feature table but are not clustering
# inputs. Listed here so the audit table can state why each was left out.
NOT_USED = {
    "Image": "identifier",
    "Image_Area_um2": "expressed by density",
    "Count": "expressed by density",
    "Diameter_STD_um": "redundant with CV_percent",
    "Mean_Angle_deg": "depends on sample rotation on the stage",
    "Merged_Count": "less comparable across samples than Merged_percent",
    "Parasitic_Coverage_percent": "not measured for this image set",
    "Length_um": "not measured for this image set",
    "Taper_Ratio": "not measured for this image set",
    "QC_Notes": "free text",
}


def load_feature_table(cfg):
    """Read the feature table and return it with a cleaned ``Sample_ID`` column."""
    path = cfg["feature_path"]
    if not path.exists():
        raise SystemExit(f"Feature file not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    id_col = cfg["input"]["sample_id_column"]
    if id_col not in df.columns:
        raise SystemExit(f"Column '{id_col}' not found in {path.name}. "
                         f"Columns: {list(df.columns)}")
    df = df.rename(columns={id_col: "Sample_ID"})
    df["Sample_ID"] = df["Sample_ID"].astype(str).str.strip()

    dupes = df.loc[df["Sample_ID"].duplicated(), "Sample_ID"].tolist()
    if dupes:
        raise SystemExit(f"Duplicate sample IDs: {dupes}")
    return df.reset_index(drop=True)


def select_features(cfg, df):
    """Pick the configured descriptors and check their completeness.

    Returns
    -------
    X : DataFrame
        Selected descriptors indexed by Sample_ID.
    audit : DataFrame
        One row per column considered, with its status and reason.
    """
    requested = list(cfg["features"]["use"])
    min_frac = float(cfg["features"]["min_valid_fraction"])
    n = len(df)

    rows, kept = [], []
    for col in requested:
        if col not in df.columns:
            rows.append(dict(Feature=col, Status="missing", Valid_Values=0,
                             Valid_Fraction=0.0, Reason="column not in file"))
            continue
        valid = int(df[col].notna().sum())
        frac = valid / n
        if frac < min_frac:
            rows.append(dict(Feature=col, Status="excluded", Valid_Values=valid,
                             Valid_Fraction=round(frac, 3),
                             Reason=f"below {min_frac:.0%} valid values"))
        else:
            kept.append(col)
            rows.append(dict(Feature=col, Status="used", Valid_Values=valid,
                             Valid_Fraction=round(frac, 3), Reason=""))

    for col, reason in NOT_USED.items():
        if col in df.columns and col not in kept:
            valid = int(df[col].notna().sum())
            rows.append(dict(Feature=col, Status="not used", Valid_Values=valid,
                             Valid_Fraction=round(valid / n, 3), Reason=reason))
    audit = pd.DataFrame(rows)

    if len(kept) < 2:
        raise SystemExit("Fewer than two usable descriptors:\n"
                         + audit.to_string(index=False))

    X = df[kept].copy()
    n_missing = int(X.isna().sum().sum())
    if n_missing:
        bad = X.columns[X.isna().any()].tolist()
        raise SystemExit(f"{n_missing} missing value(s) in {bad}; "
                         "imputation is not performed.")
    X.index = df["Sample_ID"]
    return X, audit


def acquisition_table(cfg, df):
    """Acquisition variables per sample: count, imaged area and, when images
    are available, magnification from the SEM TIFF metadata."""
    cols = [c for c in cfg["features"].get("acquisition_columns", [])
            if c in df.columns]
    out = df[["Sample_ID"] + cols].copy()
    mags = read_magnifications(cfg["image_dir"])
    out["Magnification_X"] = out["Sample_ID"].map(mags) if mags else np.nan
    return out


def read_magnifications(folder):
    """Magnification (in X) keyed by filename stem, from Zeiss TIFF tag 34118."""
    if not folder or not Path(folder).exists():
        return {}
    try:
        import tifffile
    except ImportError:
        return {}

    mags = {}
    for p in Path(folder).iterdir():
        if p.suffix.lower() not in (".tif", ".tiff"):
            continue
        try:
            tag = tifffile.TiffFile(p).pages[0].tags[34118].value
            m = re.match(r"([\d.]+)\s*(K?)\s*X", str(tag["ap_mag"][1]))
        except Exception:
            continue
        if m:
            mags[p.stem] = float(m.group(1)) * (1000 if m.group(2) else 1)
    return mags


def find_images(folder, sample_ids):
    """Map Sample_ID to an image path for the IDs that have one."""
    if not folder or not Path(folder).exists():
        return {}
    by_stem = {}
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            by_stem.setdefault(p.stem, p)
    return {s: by_stem[s] for s in sample_ids if s in by_stem}
