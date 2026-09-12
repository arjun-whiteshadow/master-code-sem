"""Image discovery, loading and SEM metadata. Images are only ever read."""

import hashlib
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Zeiss SEM TIFFs carry acquisition parameters in this private tag.
ZEISS_TAG = 34118


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_uint8(arr):
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    return ((arr - lo) / (hi - lo) * 255).round().astype(np.uint8)


def load_image_rgb(path):
    """Open an image as 8-bit RGB. 16-bit and float images are rescaled to
    their own min-max range; TIFFs that PIL cannot read fall back to tifffile."""
    path = Path(path)
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        if path.suffix.lower() not in (".tif", ".tiff"):
            raise
        import tifffile
        arr = tifffile.imread(str(path))
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        img = Image.fromarray(_to_uint8(arr))

    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        img = Image.fromarray(_to_uint8(np.array(img)))
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        img = bg
    return img if img.mode == "RGB" else img.convert("RGB")


def crop_bottom(img, fraction):
    """Remove ``fraction`` of the height from the bottom of the image."""
    if fraction <= 0:
        return img
    w, h = img.size
    return img.crop((0, 0, w, max(1, int(round(h * (1.0 - fraction))))))


def simulate_zoom(img, factor):
    """Centre-crop by ``factor`` and resample back to the original size."""
    if factor <= 1.0:
        return img
    w, h = img.size
    cw, ch = int(w / factor), int(h / factor)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


def find_images(cfg):
    exts = {e.lower() for e in cfg["input"]["extensions"]}
    folder = cfg["image_dir"]
    if not folder.exists():
        raise SystemExit(f"Image folder not found: {folder}")
    files = [p for p in sorted(folder.rglob("*"))
             if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")]
    if not files:
        raise SystemExit(f"No images with extensions {sorted(exts)} in {folder}")
    return files


def read_sem_metadata(path):
    """Magnification (X) and pixel size (nm) from the Zeiss tag, if present."""
    out = {"Magnification_X": np.nan, "Pixel_Size_nm": np.nan}
    if path.suffix.lower() not in (".tif", ".tiff"):
        return out
    try:
        import tifffile
        tag = tifffile.TiffFile(path).pages[0].tags[ZEISS_TAG].value
    except Exception:
        return out

    m = re.match(r"([\d.]+)\s*(K?)\s*X", str(tag.get("ap_mag", ("", ""))[1]))
    if m:
        out["Magnification_X"] = float(m.group(1)) * (1000 if m.group(2) else 1)
    px = tag.get("ap_image_pixel_size")
    if px:
        value, unit = float(px[1]), str(px[2]).lower()
        out["Pixel_Size_nm"] = value * 1000 if unit.startswith(("um", "µ")) else value
    return out


def build_image_index(cfg):
    """One row per image file with size, mode, checksum, readability and
    SEM metadata. Returns (index, qc_summary)."""
    folder = cfg["image_dir"]
    rows = []
    for p in find_images(cfg):
        row = {"Sample_ID": p.stem, "Filename": p.name,
               "Relative_Path": str(p.relative_to(folder)),
               "File_Size_Bytes": p.stat().st_size,
               "Width_px": np.nan, "Height_px": np.nan, "Image_Mode": "",
               "Loaded_OK": False, "Error": "", "MD5": ""}
        try:
            row["MD5"] = hashlib.md5(p.read_bytes()).hexdigest()
            with Image.open(p) as im:
                row["Image_Mode"] = im.mode
                row["Width_px"], row["Height_px"] = im.size
            load_image_rgb(p)
            row["Loaded_OK"] = True
        except Exception as exc:
            row["Error"] = f"{type(exc).__name__}: {exc}"
        row.update(read_sem_metadata(p))
        rows.append(row)
    index = pd.DataFrame(rows)

    n = len(index)
    n_ok = int(index["Loaded_OK"].sum())
    dup_ids = index.loc[index["Sample_ID"].duplicated(), "Sample_ID"].tolist()
    with_md5 = index[index["MD5"] != ""]
    dup_md5 = with_md5[with_md5["MD5"].duplicated(keep=False)]["Filename"].tolist()
    dims = index.dropna(subset=["Width_px"]).groupby(["Width_px", "Height_px"]).size()
    mags = index["Magnification_X"].dropna()

    qc = pd.DataFrame([
        ("Images found", n, ""),
        ("Readable", n_ok, ""),
        ("Unreadable", n - n_ok, "; ".join(index.loc[~index["Loaded_OK"], "Filename"])),
        ("Duplicate sample IDs", len(dup_ids), "; ".join(dup_ids)),
        ("Duplicate file content", len(dup_md5), "; ".join(dup_md5)),
        ("Distinct pixel dimensions", len(dims),
         "; ".join(f"{int(w)}x{int(h)} (n={c})" for (w, h), c in dims.items())),
        ("Magnification metadata present", int(len(mags)),
         f"{mags.min() / 1000:.1f} to {mags.max() / 1000:.1f} kX" if len(mags) else ""),
    ], columns=["Check", "Value", "Details"])

    if dup_ids:
        raise SystemExit(f"Duplicate sample IDs (filename stems): {dup_ids}")
    return index, qc


def image_paths(cfg, index):
    """Sample_ID -> absolute path for every readable image."""
    folder = cfg["image_dir"]
    ok = index[index["Loaded_OK"]]
    return {r["Sample_ID"]: folder / r["Relative_Path"] for _, r in ok.iterrows()}
