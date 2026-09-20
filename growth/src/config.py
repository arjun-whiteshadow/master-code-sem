"""Configuration loading."""

from pathlib import Path

import yaml


def load_config(path):
    """Read a YAML config file and resolve its paths relative to the file.

    Returns the parsed dictionary with three extra keys: ``root`` (the folder
    containing the config), ``log_path`` and ``m1_dir``, plus ``results_dir``.
    """
    path = Path(path).resolve()
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    root = path.parent
    cfg["root"] = root
    cfg["log_path"] = (root / cfg["input"]["growth_log"]).resolve()
    cfg["m1_dir"] = (root / cfg["input"]["method1_results"]).resolve()
    cfg["results_dir"] = root / cfg["output"]["results_folder"]
    return cfg
