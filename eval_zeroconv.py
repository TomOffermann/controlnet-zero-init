"""Evaluate ControlNet snapshots, plain SD 1.5 and the released Canny model.

Writes per-image Canny F1 (CSV), aggregate scores (JSON) and Inception features
(NPY). summarize_eval.py computes KID against real COCO validation images.
Prompt mode uses the first caption / CFG 7.5; empty-text mode uses CFG 1.
Image index is the paired inference seed. Completed pairs are skipped on rerun.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

MODES = {"prompt": 7.5, "noprompt": 1.0}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", required=True, help="SD1.5 (diffusers folder or hub id)")
    p.add_argument("--data", required=True, type=Path, help="val dir with images/, canny/, meta.jsonl")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--name", default="real", help="result folder name, e.g. the run name, 'sd', 'released'")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path, help="training run dir: evaluates its snapshots/step_*.safetensors")
    src.add_argument("--controlnet", help="'none' (plain SD) or a diffusers ControlNet folder (released model)")
    src.add_argument("--real", action="store_true", help="only Inception features of the real val images -> "
                                                         "<out>/real/features.npy (KID reference)")
    p.add_argument("--every", type=int, default=500, help="--run: use snapshots whose step is a multiple of this")
    p.add_argument("--only", type=int, nargs="*", help="--run: only these steps (overrides --every)")
    p.add_argument("--n", type=int, default=500, help="number of val images (first n, fixed order)")
    p.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    p.add_argument("--steps", type=int, default=20, help="UniPC sampling steps")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--tol", type=int, default=2, help="edge matching tolerance in pixels")
    return p.parse_args()


# ----------------------------------------------------------------------------- metrics
def canny_f1(rgb, cond, tol):
    """Tolerance-matched edge F1 between Canny(rgb) and the condition map. Returns (f1, precision, recall)."""
    gen = cv2.Canny(rgb, 100, 200) > 0
    ref = cond > 127
    if not ref.any():
        return float("nan"), float("nan"), float("nan")
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    near_ref = cv2.dilate(ref.astype(np.uint8), k) > 0
    near_gen = cv2.dilate(gen.astype(np.uint8), k) > 0
    precision = (gen & near_ref).sum() / max(gen.sum(), 1)
    recall = (ref & near_gen).sum() / max(ref.sum(), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(f1), float(precision), float(recall)


class Inception:
    """Inception-v3 pool features as used for FID/KID (torchmetrics / torch-fidelity weights)."""

    def __init__(self, device):
        from torchmetrics.image.kid import KernelInceptionDistance
        self.net = KernelInceptionDistance(feature=2048).inception.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, images):  # list of HxWx3 uint8
        x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(self.device)
        return self.net(x).float().cpu().numpy()


# ----------------------------------------------------------------------------- data / models
def load_val(data, n):
    with open(data / "meta.jsonl") as handle:
        metas = [json.loads(line) for line in handle][:n]
    if n < 1 or len(metas) != n:
        raise ValueError(f"Requested {n} validation images, found {len(metas)} in {data}.")
    for m in metas:
        m["prompt"] = m["captions"][0]
    return metas


def device_and_dtype(name):
    if torch.cuda.is_available():
        dev = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    dtype = torch.float16 if name == "fp16" and dev != "cpu" else torch.float32
    return dev, dtype


def snapshots(args):
    """[(step, path or None)]: what to evaluate."""
    if args.controlnet is not None:
        return [(0, None)]
    found = {int(re.search(r"step_(\d+)", p.name).group(1)): p
             for p in sorted((args.run / "snapshots").glob("step_*.safetensors"))}
    steps = args.only if args.only else [s for s in found if s % args.every == 0]
    missing = [s for s in steps if s not in found]
    if missing:
        raise SystemExit(f"snapshots missing in {args.run}: {missing}")
    if not steps:
        raise SystemExit(f"No matching snapshots in {args.run}")
    return [(s, found[s]) for s in sorted(steps)]


def completed_pair(out, mode, settings):
    """A JSON marker is complete only with matching settings and readable result files."""
    marker = out / f"{mode}.json"
    try:
        saved = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(saved, dict) or not {"f1", "f1_sem", "precision", "recall", "n", "settings"} <= saved.keys():
        return False
    if saved["settings"] != settings:
        raise ValueError(f"Evaluation settings changed for {marker}. Use a separate output directory.")
    try:
        features = np.load(out / f"{mode}.npy", mmap_mode="r")
        with open(out / f"{mode}.csv", newline="") as handle:
            rows = list(csv.reader(handle))
        if features.shape != (settings["n"], 2048) or len(rows) != settings["n"] + 1:
            return False
        if rows[0] != ["id", "f1", "precision", "recall"] or any(len(row) != 4 for row in rows):
            return False
    except (OSError, ValueError, EOFError):
        return False
    return True


def build_pipeline(args, dev, dtype):
    from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionPipeline,
                           UniPCMultistepScheduler)
    sd = StableDiffusionPipeline.from_pretrained(args.pretrained, torch_dtype=dtype, safety_checker=None,
                                                 feature_extractor=None, requires_safety_checker=False)
    sd.scheduler = UniPCMultistepScheduler.from_config(sd.scheduler.config)
    if args.controlnet == "none":
        pipe, cn = sd, None
    else:
        if args.controlnet is not None:  # released model
            cn = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
        else:  # architecture from SD's UNet; weights come from each snapshot
            cn = ControlNetModel.from_unet(sd.unet).to(dtype)
        pipe = StableDiffusionControlNetPipeline(
            vae=sd.vae, text_encoder=sd.text_encoder, tokenizer=sd.tokenizer, unet=sd.unet, controlnet=cn,
            scheduler=sd.scheduler, safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)
    return pipe, cn


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    if not args.real and args.name == "real":
        raise SystemExit("--name is required (e.g. the run name, 'sd', 'released')")
    dev, dtype = device_and_dtype(args.dtype)
    metas = load_val(args.data, args.n)
    if args.real:
        out = args.out / "real" / "features.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        inception = Inception(dev)
        feats = [inception([np.asarray(Image.open(args.data / "images" / f"{m['id']}.jpg").convert("RGB"))
                            for m in metas[b:b + 32]]) for b in range(0, len(metas), 32)]
        np.save(out, np.concatenate(feats))
        print(f"real: {len(metas)} images -> {out}")
        return
    todo = []
    for step, path in snapshots(args):
        for mode in args.modes:
            settings = {"steps": args.steps, "guidance": MODES[mode], "dtype": str(dtype), "tol": args.tol,
                        "sampler": "UniPC", "snapshot": str(path) if path else args.controlnet,
                        "snapshot_mtime_ns": path.stat().st_mtime_ns if path else None,
                        "pretrained": args.pretrained, "data": str(args.data.resolve()),
                        "n": args.n, "batch": args.batch}
            if not completed_pair(args.out / args.name / f"step_{step:06d}", mode, settings):
                todo.append((step, path, mode, settings))
    print(f"{args.name}: {len(metas)} val images, {len(todo)} (snapshot, mode) pairs to do on {dev} {dtype}",
          flush=True)
    if not todo:
        return
    pipe, cn = build_pipeline(args, dev, dtype)
    inception = Inception(dev)
    conds = [cv2.imread(str(args.data / "canny" / f"{m['id']}.png"), cv2.IMREAD_GRAYSCALE) for m in metas]
    res = conds[0].shape[0]
    loaded = None
    for step, path, mode, settings in todo:
        t0 = time.time()
        if path is not None and loaded != path:
            cn.load_state_dict(load_file(str(path)))  # fp32 snapshot -> pipeline dtype
            loaded = path
        out = args.out / args.name / f"step_{step:06d}"
        out.mkdir(parents=True, exist_ok=True)
        marker = out / f"{mode}.json"
        marker.unlink(missing_ok=True)  # invalidate an incomplete pair before rewriting its files
        rows, feats = [], []
        for b0 in range(0, len(metas), args.batch):
            idx = range(b0, min(b0 + args.batch, len(metas)))
            prompts = [metas[i]["prompt"] if mode == "prompt" else "" for i in idx]
            kw = dict(num_inference_steps=args.steps, guidance_scale=MODES[mode], height=res, width=res,
                      generator=[torch.Generator("cpu").manual_seed(i) for i in idx])
            if cn is not None:
                kw["image"] = [Image.fromarray(np.repeat(conds[i][:, :, None], 3, axis=2)) for i in idx]
            images = [np.asarray(im.convert("RGB")) for im in pipe(prompts, **kw).images]
            for i, rgb in zip(idx, images):
                rows.append([metas[i]["id"], *canny_f1(rgb, conds[i], args.tol)])
            feats.append(inception(images))
        f1, prec, rec = (np.array([r[k] for r in rows], dtype=float) for k in (1, 2, 3))
        ok = ~np.isnan(f1)
        np.save(out / f"{mode}.npy", np.concatenate(feats))
        with open(out / f"{mode}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "f1", "precision", "recall"])
            w.writerows(rows)
        summary = {"name": args.name, "step": step, "mode": mode, "n": int(ok.sum()),
                   "f1": f1[ok].mean(), "f1_sem": f1[ok].std(ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else 0.0,
                   "precision": prec[ok].mean(), "recall": rec[ok].mean(),
                   "settings": settings}
        temporary = marker.with_suffix(".json.tmp")
        with temporary.open("w") as handle:
            json.dump(summary, handle, indent=1)
        temporary.replace(marker)  # publish a complete marker only after both result files exist
        print(f"  step {step:>6} {mode:8s} F1 {summary['f1']:.4f} ± {summary['f1_sem']:.4f}  "
              f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
