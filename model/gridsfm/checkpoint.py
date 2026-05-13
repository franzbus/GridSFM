"""Checkpoint loading for the minimal public-release format."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download

from .model import GridTransformerBackbone


def _hash_state_dict(state_dict: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode("utf-8"))
        t = state_dict[key]
        if torch.is_tensor(t):
            t_cpu = t.detach().cpu().contiguous()
            if t_cpu.dtype == torch.bfloat16:
                t_cpu = t_cpu.to(torch.float32)
            arr = t_cpu.numpy()
            h.update(str(t.dtype).encode("utf-8"))
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(arr.tobytes())
        else:
            h.update(str(t).encode("utf-8"))
    return h.hexdigest()


def load_model(
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> GridTransformerBackbone:
    blob = torch.load(ckpt_path, weights_only=True, map_location="cpu")

    if not (isinstance(blob, dict) and "state_dict" in blob and "metadata" in blob):
        raise ValueError(
            f"{ckpt_path}: not a minimal release checkpoint. "
            f"Expected top-level keys 'state_dict' and 'metadata'. "
            f"Training-format checkpoints (with 'model_state_dict', "
            f"'optimizer_state_dict', etc.) are not loadable here; "
            f"convert them via the export script in the training repo."
        )

    state_dict = blob["state_dict"]
    meta = blob["metadata"]
    expected_hash = meta.get("hash")
    if not expected_hash:
        raise ValueError(
            f"{ckpt_path}: metadata.hash missing. Re-export with the "
            f"release-checkpoint script to embed an integrity hash."
        )

    actual_hash = _hash_state_dict(state_dict)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{ckpt_path}: state_dict hash mismatch.\n"
            f"  expected (metadata.hash): {expected_hash}\n"
            f"  computed:                 {actual_hash}\n"
            f"  Likely causes: partial/corrupted download, in-place "
            f"modification of the file, or the embedded hash being out "
            f"of sync with the weights."
        )

    arch = meta.get("arch") or {}
    model = GridTransformerBackbone(**arch)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def load_from_hf(
    repo_id: str,
    filename: str = "gridsfm_open_v1.0.pt",
    revision: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> GridTransformerBackbone:
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return load_model(ckpt_path, **kwargs)
