#!/usr/bin/env python3
import argparse
import os
import pickle
from dataclasses import is_dataclass
from typing import Any

import numpy as np


def _as_numpy(x):
    """Convert torch tensor -> numpy if needed, else pass through."""
    try:
        import torch  # optional
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy(), ("torch", x.device, x.dtype)
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x, ("numpy", None, None)
    return None, None


def _restore_from_numpy(original, arr_np, meta):
    if meta is None:
        return arr_np
    kind, device, dtype = meta
    if kind == "torch":
        import torch
        out = torch.as_tensor(arr_np, dtype=dtype)
        if device is not None:
            out = out.to(device)
        return out
    return arr_np


def apply_offset_to_frames(frames, dz: float):
    arr_np, meta = _as_numpy(frames)
    if arr_np is None:
        raise TypeError(f"Unsupported frames type: {type(frames)}")

    if arr_np.ndim != 2 or arr_np.shape[1] < 3:
        raise ValueError(f"frames must be (T, D>=3). Got {arr_np.shape}")

    before = (float(arr_np[:, 2].min()), float(arr_np[:, 2].max()))
    arr_np = arr_np.copy()
    arr_np[:, 2] += dz
    after = (float(arr_np[:, 2].min()), float(arr_np[:, 2].max()))

    return _restore_from_numpy(frames, arr_np, meta), before, after


def apply_offset(obj: Any, dz: float):
    """
    Tries to apply dz to:
      - dict["root_pos"][:,2]
      - dict["frames"][:,2] (assuming frames start with root_pos)
      - obj.root_pos or obj.frames or obj.data
    """
    # Case 1: dict-like
    if isinstance(obj, dict):
        if "root_pos" in obj:
            new_root_pos, before, after = apply_offset_to_frames(obj["root_pos"], dz)
            obj = dict(obj)
            obj["root_pos"] = new_root_pos
            return obj, "dict.root_pos", before, after

        if "frames" in obj:
            new_frames, before, after = apply_offset_to_frames(obj["frames"], dz)
            obj = dict(obj)
            obj["frames"] = new_frames
            return obj, "dict.frames", before, after

        raise KeyError("dict has neither 'root_pos' nor 'frames' keys; don't know where to apply z-offset.")

    # Case 2: dataclass or plain object with attributes
    # Try common attribute names in order.
    for attr in ("root_pos", "frames", "data"):
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            try:
                new_val, before, after = apply_offset_to_frames(val, dz)
            except Exception:
                continue

            # shallow copy if dataclass, else mutate
            if is_dataclass(obj):
                # recreate dataclass with updated field
                from dataclasses import replace
                obj2 = replace(obj, **{attr: new_val})
                return obj2, f"obj.{attr}", before, after
            else:
                setattr(obj, attr, new_val)
                return obj, f"obj.{attr}", before, after

    raise TypeError(f"Don't know how to apply offset to object of type {type(obj)}")


def main():
    ap = argparse.ArgumentParser(description="Apply a constant vertical offset (dz) to a motion .pkl file.")
    ap.add_argument("--input", required=True, help="Input .pkl path")
    ap.add_argument("--output", required=True, help="Output .pkl path (can be same as input if you want)")
    ap.add_argument("--dz", type=float, required=True, help="Vertical offset to add (meters). Positive lifts up.")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output
    dz = args.dz

    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    with open(in_path, "rb") as f:
        obj = pickle.load(f)

    obj2, target, before, after = apply_offset(obj, dz)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(obj2, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[ok] Applied dz={dz:+.6f} to {target}")
    print(f"     z range before: [{before[0]:+.4f}, {before[1]:+.4f}]")
    print(f"     z range after : [{after[0]:+.4f}, {after[1]:+.4f}]")
    print(f"     wrote: {out_path}")


if __name__ == "__main__":
    main()
