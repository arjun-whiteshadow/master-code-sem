"""Configuration loading."""

from pathlib import Path

import yaml


def load_config(path):
    """Read a YAML config file and resolve its paths relative to the file.

    Adds ``root``, ``image_dir`` and ``results_dir`` to the returned dict.
    ``descriptor_path`` is set when a descriptor file is configured,
    otherwise None.
    """
    path = Path(path).resolve()
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    root = path.parent
    cfg["root"] = root
    cfg["image_dir"] = (root / cfg["input"]["image_folder"]).resolve()
    cfg["results_dir"] = root / cfg["output"]["results_folder"]

    desc = (cfg["input"].get("descriptor_file") or "").strip()
    cfg["descriptor_path"] = (root / desc).resolve() if desc else None
    return cfg
