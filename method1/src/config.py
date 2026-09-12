"""Configuration loading."""

from pathlib import Path

import yaml


def load_config(path):
    """Read a YAML config file and resolve its paths relative to the file.

    Returns the parsed dictionary with three extra keys: ``root`` (the folder
    containing the config), ``feature_path`` and ``results_dir``.
    ``image_dir`` is set when an image folder is configured, otherwise None.
    """
    path = Path(path).resolve()
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    root = path.parent
    cfg["root"] = root
    cfg["feature_path"] = root / cfg["input"]["feature_file"]
    cfg["results_dir"] = root / cfg["output"]["results_folder"]

    folder = (cfg["input"].get("image_folder") or "").strip()
    cfg["image_dir"] = (root / folder).resolve() if folder else None
    return cfg
