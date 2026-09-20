"""Place new samples against the Method 1 clusters from their descriptors.

Usage:
    python predict_regime.py new_samples.xlsx [config.yaml]

The input has one row per sample, a ``Sample`` column and the descriptor
columns Method 1 was run on (listed in ``model.json``). The placement is
written next to the input as ``<name>_placement.csv`` and printed.

A sample marked ``Within_Range`` False is further from its nearest centre
than any member of that cluster was; report it as unplaced rather than as
a member.
"""

import sys
from pathlib import Path

import pandas as pd

from src import predict
from src.config import load_config
from src.growth_log import load_method1


def main(sample_path, config_path):
    cfg = load_config(config_path)
    _, _, model = load_method1(cfg)

    sample_path = Path(sample_path)
    if sample_path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(sample_path)
    else:
        df = pd.read_csv(sample_path)
    id_col = cfg["input"]["sample_id_column"]
    missing = [c for c in [id_col] + model["features"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing column(s) in {sample_path.name}: {missing}")
    df = df.dropna(subset=[id_col]).set_index(df[id_col].astype(str).str.strip())
    blank = df[model["features"]].isna().any(axis=1)
    if blank.any():
        raise SystemExit(f"Missing descriptor value(s) for: {df.index[blank].tolist()}")

    out = predict.assign(model, df)
    out_path = sample_path.with_name(sample_path.stem + "_placement.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    n_out = int((~out["Within_Range"]).sum())
    print(f"\n{len(out)} sample(s) placed; {n_out} outside the range of every cluster. "
          f"Written to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    config = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "config.yaml"
    main(sys.argv[1], config)
