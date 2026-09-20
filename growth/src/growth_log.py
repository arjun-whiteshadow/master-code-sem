"""Loading the MBE growth log and joining it to the Method 1 outputs."""

import json

import pandas as pd


def load_growth_log(cfg):
    """Read the growth log and return it with a cleaned ``Sample_ID`` column.

    Rows with no sample ID (blank lines left in the sheet) are dropped.
    """
    path = cfg["log_path"]
    if not path.exists():
        raise SystemExit(f"Growth log not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    id_col = cfg["input"]["sample_id_column"]
    if id_col not in df.columns:
        raise SystemExit(f"Column '{id_col}' not found in {path.name}. "
                         f"Columns: {list(df.columns)}")
    df = df.rename(columns={id_col: "Sample_ID"})
    df = df.dropna(subset=["Sample_ID"]).reset_index(drop=True)
    df["Sample_ID"] = df["Sample_ID"].astype(str).str.strip()

    dupes = df.loc[df["Sample_ID"].duplicated(), "Sample_ID"].tolist()
    if dupes:
        raise SystemExit(f"Duplicate sample IDs in the growth log: {dupes}")
    return df


def load_method1(cfg):
    """Descriptor table, cluster labels and fitted model from a Method 1 run.

    Returns
    -------
    X : DataFrame
        Descriptors in original units, indexed by Sample_ID.
    clusters : Series
        Cluster number per Sample_ID.
    model : dict
        Contents of ``model.json``.
    """
    d = cfg["m1_dir"]
    paths = {
        "features": d / "tables" / "Features_Used_Raw_Units.csv",
        "assignments": d / "tables" / "Final_Cluster_Assignments.csv",
        "model": d / "model.json",
    }
    for p in paths.values():
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run method1 first.")
    X = pd.read_csv(paths["features"]).set_index("Sample_ID")
    clusters = pd.read_csv(paths["assignments"]).set_index("Sample_ID")["Cluster"]
    with open(paths["model"]) as fh:
        model = json.load(fh)
    return X, clusters, model


def screen_parameters(cfg, log):
    """Pick the configured growth parameters and check their usability.

    A numeric parameter containing text stops the run, since silently
    coercing it would hide a data-entry error.

    Returns
    -------
    numeric, categorical : list
        Retained parameter names of each kind.
    audit : DataFrame
        One row per column in the log, with its status and reason.
    """
    p = cfg["parameters"]
    min_frac = float(p["min_valid_fraction"])
    n = len(log)
    kinds = {c: "numeric" for c in p["numeric"]}
    kinds.update({c: "categorical" for c in p["categorical"]})

    rows, kept = [], {"numeric": [], "categorical": []}
    for col, kind in kinds.items():
        if col not in log.columns:
            rows.append(dict(Parameter=col, Type=kind, Status="missing",
                             Valid_Values=0, Valid_Fraction=0.0, Distinct_Values=0,
                             Reason="column not in log"))
            continue
        values = log[col]
        if kind == "numeric":
            bad = values[values.notna() & pd.to_numeric(values, errors="coerce").isna()]
            if len(bad):
                raise SystemExit(f"Non-numeric value(s) in '{col}': {bad.tolist()}")
        valid = int(values.notna().sum())
        distinct = int(values.nunique())
        frac = valid / n
        if frac < min_frac:
            status, reason = "excluded", f"below {min_frac:.0%} valid values"
        elif distinct < 2:
            status, reason = "excluded", "constant across samples"
        else:
            status, reason = "used", ""
            kept[kind].append(col)
        rows.append(dict(Parameter=col, Type=kind, Status=status, Valid_Values=valid,
                         Valid_Fraction=round(frac, 3), Distinct_Values=distinct,
                         Reason=reason))

    for col in log.columns:
        if col == "Sample_ID" or col in kinds:
            continue
        rows.append(dict(Parameter=col, Type="", Status="not used",
                         Valid_Values=int(log[col].notna().sum()),
                         Valid_Fraction=round(float(log[col].notna().mean()), 3),
                         Distinct_Values=int(log[col].nunique()),
                         Reason="not in the configured parameter list"))
    audit = pd.DataFrame(rows)

    if not kept["numeric"] and not kept["categorical"]:
        raise SystemExit("No usable growth parameter:\n" + audit.to_string(index=False))
    return kept["numeric"], kept["categorical"], audit


def merge_with_method1(log, X, clusters):
    """Join the log to the descriptors and cluster labels on Sample_ID.

    Returns the merged table (samples present in both sources) and a
    coverage table listing every sample from either source with where it
    was found.
    """
    in_log, in_m1 = set(log["Sample_ID"]), set(X.index)
    ids = sorted(in_log | in_m1)
    coverage = pd.DataFrame({
        "Sample_ID": ids,
        "In_Growth_Log": [s in in_log for s in ids],
        "In_Method1": [s in in_m1 for s in ids],
    })
    coverage["Included"] = coverage["In_Growth_Log"] & coverage["In_Method1"]

    m1 = X.copy()
    m1.insert(0, "Cluster", clusters.reindex(X.index).astype(int))
    merged = m1.reset_index().merge(log, on="Sample_ID", how="inner")
    if merged.empty:
        raise SystemExit("No sample ID is shared between the growth log and the "
                         "Method 1 results; check the ID spelling in both files.")
    return merged, coverage
