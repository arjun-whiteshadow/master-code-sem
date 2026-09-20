"""Write an empty growth log with the sample IDs of the current Method 1 run.

Usage:
    python make_log_template.py [config.yaml]

The file is written next to the configured growth log with ``_template``
appended to the name, and is never overwritten if it already exists.
"""

import sys
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.growth_log import load_method1


def main(config_path):
    cfg = load_config(config_path)
    X, _, _ = load_method1(cfg)
    p = cfg["parameters"]
    columns = [cfg["input"]["sample_id_column"]] + p["numeric"] + p["categorical"] + ["Notes"]

    out = cfg["log_path"].with_name(cfg["log_path"].stem + "_template.xlsx")
    if out.exists():
        raise SystemExit(f"{out} already exists; not overwriting it.")
    template = pd.DataFrame(columns=columns)
    template[columns[0]] = list(X.index)
    template.to_excel(out, index=False)
    print(f"{len(template)} sample IDs written to {out}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "config.yaml"
    main(config)
