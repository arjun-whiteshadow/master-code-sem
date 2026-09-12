"""Feature extraction with frozen ImageNet-pretrained torchvision models.

Each model is loaded with its default weights, its classification head is
replaced by an identity so the forward pass returns the pooled feature
vector, and its own preprocessing transform is applied to every image.
No parameters are updated.
"""

import socket

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .images import crop_bottom, load_image_rgb, simulate_zoom

# torchvision constructor, weights enum and the attribute holding the head.
MODELS = {
    "resnet50": ("resnet50", "ResNet50_Weights", "fc"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", "classifier"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", "classifier"),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights", "classifier"),
}

# Weight downloads that stall would otherwise hang the run indefinitely.
DOWNLOAD_TIMEOUT_S = 120


def build_model(name):
    """Return (model in eval mode, transform, weights name)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    import torchvision.models as tvm

    ctor, weights_enum, head = MODELS[name]
    weights = getattr(tvm, weights_enum).DEFAULT
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(DOWNLOAD_TIMEOUT_S)
    try:
        model = getattr(tvm, ctor)(weights=weights)
    finally:
        socket.setdefaulttimeout(previous)
    setattr(model, head, nn.Identity())
    model.eval()
    return model, weights.transforms(), str(weights)


@torch.inference_mode()
def embed_images(model, transform, images, device, batch_size):
    """Forward a list of PIL images in batches; returns an (n, d) float64 array."""
    out = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([transform(im) for im in images[i:i + batch_size]])
        feats = model(batch.to(device))
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        out.append(feats.cpu().numpy().astype(np.float64))
    return np.vstack(out)


def extract(name, paths, cfg, device, crop_fraction=None):
    """Embed every image in ``paths`` (Sample_ID -> path) with one model.

    Returns (DataFrame, info). The DataFrame has Sample_ID, Image_File and
    one column per embedding dimension; it is None if the model could not
    be loaded or no image was readable, in which case ``info['error']``
    says why.
    """
    info = {"model": name, "status": "ok", "error": "", "dimension": 0,
            "weights": "", "n_images": 0, "n_failed": 0}
    try:
        model, transform, weights = build_model(name)
    except Exception as exc:
        info.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return None, info
    model.to(device)
    info["weights"] = weights

    frac = cfg["preprocessing"]["crop_bottom_fraction"] if crop_fraction is None else crop_fraction
    ids, files, images, failures = [], [], [], []
    for sid, p in paths.items():
        try:
            images.append(crop_bottom(load_image_rgb(p), frac))
            ids.append(sid)
            files.append(p.name)
        except Exception as exc:
            failures.append(f"{p.name}: {type(exc).__name__}: {exc}")
    if not images:
        info.update(status="failed", error="no readable images")
        return None, info

    emb = embed_images(model, transform, images, device, int(cfg["preprocessing"]["batch_size"]))
    if not np.isfinite(emb).all():
        info.update(status="failed", error="non-finite values in embedding")
        return None, info

    df = pd.DataFrame(emb, columns=[f"E{i + 1}" for i in range(emb.shape[1])])
    df.insert(0, "Image_File", files)
    df.insert(0, "Sample_ID", ids)
    info.update(dimension=emb.shape[1], n_images=len(ids), n_failed=len(failures),
                error="; ".join(failures))
    return df, info


def extract_all(paths, cfg, device):
    """Run every configured model. Returns ({name: DataFrame}, status table)."""
    results, rows = {}, []
    for name in cfg["models"]["candidates"]:
        print(f"  {name} ...", end=" ", flush=True)
        df, info = extract(name, paths, cfg, device)
        if df is not None:
            results[name] = df
            print(f"{info['dimension']}-d, {info['n_images']} images")
        else:
            print(f"failed: {info['error'][:80]}")
        rows.append({"Model": name, "Status": info["status"],
                     "Embedding_Dimension": info["dimension"], "Images": info["n_images"],
                     "Weights": info["weights"], "Note": info["error"][:300]})
    return results, pd.DataFrame(rows)


def zoom_series(name, paths, factors, cfg, device):
    """Embed each image at several simulated zoom factors.

    Returns {Sample_ID: (n_factors, d) array of L2-normalised embeddings},
    or None if the model cannot be loaded.
    """
    try:
        model, transform, _ = build_model(name)
    except Exception:
        return None
    model.to(device)
    frac = float(cfg["preprocessing"]["crop_bottom_fraction"])
    batch = int(cfg["preprocessing"]["batch_size"])

    out = {}
    for sid, p in paths.items():
        try:
            base = crop_bottom(load_image_rgb(p), frac)
        except Exception:
            continue
        e = embed_images(model, transform, [simulate_zoom(base, f) for f in factors],
                         device, batch)
        out[sid] = e / np.linalg.norm(e, axis=1, keepdims=True)
    return out or None
