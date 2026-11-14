"""ControlNet initialization ablation: 14 zero convolutions, input (Z1) / output (Z2).

Within each replicate, the sample stream, condition encoder, noise and training
settings are paired across variants. Only the zero-convolution initialization varies.
Logs retain the gradient norms and weight drift used in the experiment analysis.
Use reproduce.py for the complete settings and paired seeds.
"""
import argparse
import csv
import json
import os
import shutil
import signal
import time
import zlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from safetensors.torch import load_file, save_file
from transformers import CLIPTextModel, CLIPTokenizer


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", required=True, help="SD1.5 diffusers folder or hub id")
    p.add_argument("--data_dir", required=True, help="prepared split dir with images/ meta.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--init", choices=["zero", "gauss", "default"], required=True)
    p.add_argument("--init_std", type=float, default=0.0)
    p.add_argument("--init_targets", choices=["all", "input", "output"], default="all",
                   help="side that gets --init: input = Z1, output = Z2, all = both; the other side stays zero")
    p.add_argument("--init_seed", type=int, default=1234)
    p.add_argument("--hint_seed", type=int, default=20230210,
                   help="random init of the condition encoder (new layers); paired across variants within each replicate")
    p.add_argument("--data_seed", type=int, default=0, help="controls the whole sample stream")
    p.add_argument("--max_steps", type=int, default=15_000, help="optimizer steps")
    p.add_argument("--global_batch", type=int, default=32)
    p.add_argument("--per_gpu_batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--prompt_dropout", type=float, default=0.5)
    p.add_argument("--canny_low", type=int, nargs=2, default=[50, 150], help="random low threshold range (inclusive)")
    p.add_argument("--canny_high", type=int, nargs=2, default=[150, 250], help="random high threshold range (inclusive)")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--snapshot_every", type=int, default=250, help="fp32 ControlNet weights for eval")
    p.add_argument("--ckpt_every", type=int, default=500, help="full resumable state (latest only)")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--dense_log_steps", type=int, default=200, help="log every step before this step")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--time_limit_min", type=float, default=0, help="save + exit cleanly after this many minutes")
    p.add_argument("--max_train_samples", type=int, default=0, help="debug: use only first N images")
    return p.parse_args()


# ----------------------------------------------------------------------------- data
class StreamDataset(Dataset):
    """Indexed by GLOBAL sample index g. Everything random about sample g is a function of (data_seed, g)."""

    def __init__(self, data_dir, tokenizer, resolution, data_seed, prompt_dropout, num_train_timesteps, n_limit=0,
                 canny_low=(50, 150), canny_high=(150, 250)):
        self.dir = data_dir
        self.meta = [json.loads(l) for l in open(os.path.join(data_dir, "meta.jsonl"))]
        if n_limit:
            self.meta = self.meta[:n_limit]
        self.N = len(self.meta)
        self.tok = tokenizer
        self.res = resolution
        self.seed = data_seed
        self.p_drop = prompt_dropout
        self.T = num_train_timesteps
        self.c_lo, self.c_hi = canny_low, canny_high
        self._perms = {}

    def perm(self, epoch):
        if epoch not in self._perms:
            self._perms[epoch] = np.random.default_rng([self.seed, 7, epoch]).permutation(self.N)
        return self._perms[epoch]

    def __len__(self):
        return 2**62  # virtual

    def __getitem__(self, g):
        epoch, pos = divmod(int(g), self.N)
        idx = int(self.perm(epoch)[pos])
        m = self.meta[idx]
        rng = np.random.default_rng([self.seed, 11, epoch, idx])
        caption = m["captions"][int(rng.integers(len(m["captions"])))]
        if rng.random() < self.p_drop:
            caption = ""
        img = Image.open(os.path.join(self.dir, "images", m["id"] + ".jpg")).convert("RGB")
        if img.size != (self.res, self.res):
            img = img.resize((self.res, self.res), Image.BICUBIC)
        rgb = np.asarray(img)
        # Paper (ICCV supp.): Canny "with random thresholds". Drawn per (epoch, image) -> identical in every run.
        lo = int(rng.integers(self.c_lo[0], self.c_lo[1] + 1))
        hi = int(rng.integers(max(lo + 1, self.c_hi[0]), self.c_hi[1] + 1))
        edges = cv2.Canny(rgb, lo, hi)                                    # same call as ControlNet's annotator
        pix = torch.from_numpy(rgb.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        cnd = torch.from_numpy(np.repeat(edges[:, :, None], 3, axis=2).astype(np.float32) / 255.0).permute(2, 0, 1)
        ids = self.tok(caption, max_length=self.tok.model_max_length, padding="max_length",
                       truncation=True, return_tensors="pt").input_ids[0]
        gen = torch.Generator().manual_seed(int(np.random.default_rng([self.seed, 13, g]).integers(2**62)))
        lat = self.res // 8
        noise = torch.randn(4, lat, lat, generator=gen)
        vae_eps = torch.randn(4, lat, lat, generator=gen)
        t = torch.randint(0, self.T, (), generator=gen)
        return {"pix": pix, "cond": cnd, "ids": ids, "noise": noise, "vae_eps": vae_eps, "t": t, "g": int(g)}


class RankSampler(torch.utils.data.Sampler):
    """Yields global indices for this rank. Optimizer step s, micro-step a, rank r, slot j ->
    g = s*G + (a*W + r)*B + j. Fully determined by the step, so resuming is exact."""

    def __init__(self, start_step, max_steps, global_batch, per_gpu, world, rank):
        self.s0, self.s1, self.G, self.B, self.W, self.r = start_step, max_steps, global_batch, per_gpu, world, rank
        self.A = global_batch // (per_gpu * world)

    def __iter__(self):
        for s in range(self.s0, self.s1):
            for a in range(self.A):
                base = s * self.G + (a * self.W + self.r) * self.B
                for j in range(self.B):
                    yield base + j

    def __len__(self):
        return (self.s1 - self.s0) * self.A * self.B


# ----------------------------------------------------------------------------- zero convs + parameter groups
def zero_convs(controlnet):
    """The 14 zero convs as (name, side, module). side "input" = Z1, "output" = Z2."""
    mods = [("cond_embedding.conv_out", "input", controlnet.controlnet_cond_embedding.conv_out)]
    mods += [(f"down.{i}", "output", m) for i, m in enumerate(controlnet.controlnet_down_blocks)]
    mods += [("mid", "output", controlnet.controlnet_mid_block)]
    if tuple(controlnet.config.block_out_channels) == (320, 640, 1280, 1280):  # SD1.5: 12 down + mid + hint
        assert len(mods) == 14, len(mods)
    return mods


def zero_conv_key(name):
    """Zero-conv name -> its prefix in the ControlNet state_dict."""
    if name == "cond_embedding.conv_out":
        return "controlnet_cond_embedding.conv_out"
    if name == "mid":
        return "controlnet_mid_block"
    return name.replace("down.", "controlnet_down_blocks.")


def param_group(key):
    """State-dict key -> z1 | z2 | hint (condition-encoder body, random init) | copy (copied from SD1.5)."""
    if key.startswith("controlnet_cond_embedding.conv_out"):
        return "z1"
    if key.startswith(("controlnet_down_blocks", "controlnet_mid_block")):
        return "z2"
    if key.startswith("controlnet_cond_embedding"):
        return "hint"
    return "copy"


def copy_stage(key):
    """Trainable-copy key -> stage: stem (conv_in, time embedding), down0..down3 (latent 64, 32, 16, 8 px), mid."""
    if key.startswith("down_blocks."):
        return "down" + key.split(".")[1]
    if key.startswith("mid_block"):
        return "mid"
    return "stem"


@torch.no_grad()
def init_zero_convs(controlnet, mode, std, seed, targets="all"):
    """Init the zero convs. Draws are made for every conv in a fixed order whatever `targets` is,
    then the side that is not targeted is set to exactly zero."""
    gen = torch.Generator().manual_seed(seed)
    stats = {}
    for name, side, m in zero_convs(controlnet):
        if mode == "gauss":
            m.weight.copy_(torch.randn(m.weight.shape, generator=gen) * std)
            m.bias.zero_()
        elif mode == "default":
            with torch.random.fork_rng(devices=[]):  # restores the CPU RNG afterwards
                torch.default_generator.manual_seed(seed + zlib.crc32(name.encode()) % 10_000)  # CPU only
                m.reset_parameters()
        perturbed = mode != "zero" and targets in ("all", side)
        if not perturbed:
            m.weight.zero_()
            m.bias.zero_()
        stats[name] = {"side": side, "perturbed": perturbed, "w_std": m.weight.std().item(),
                       "b_absmax": m.bias.abs().max().item(), "fan_in": m.weight[0].numel()}
    return stats


# ----------------------------------------------------------------------------- logging helpers
LOG_HEADERS = {
    "train_log.csv": ["step", "loss", "grad_z1", "grad_z2", "grad_hint", "grad_copy", "sec_per_step",
                      "samples_per_sec", "peak_mem_gb"],
    "zeroconv_log.csv": ["step", "conv", "side", "w_norm", "delta_norm", "w0_norm"],
    "copy_drift.csv": ["step", "stage", "delta_norm", "theta0_norm"],
}


def append_rows(out_dir, file, rows):
    path = os.path.join(out_dir, file)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_HEADERS[file])
        w.writerows(rows)


def trim_rows(out_dir, file, max_step):
    """On resume: drop rows written after the checkpoint (step > max_step); they are logged again."""
    path = os.path.join(out_dir, file)
    if not os.path.exists(path):
        return
    with open(path) as f:
        rows = list(csv.reader(f))
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([rows[0]] + [r for r in rows[1:] if int(r[0]) <= max_step])


@torch.no_grad()
def grad_norms(named_params):
    """L2 norm of the gradient per parameter group (z1, z2, hint, copy). Grads are already all-reduced."""
    sq = {g: [] for g in ("z1", "z2", "hint", "copy")}
    for k, p in named_params:
        if p.grad is not None:
            sq[param_group(k)].append(p.grad.detach().float().pow(2).sum())
    return {g: (torch.stack(v).sum().sqrt().item() if v else 0.0) for g, v in sq.items()}


@torch.no_grad()
def zero_conv_rows(step, controlnet, w0):
    """Per zero conv: |W_t|, |W_t - W_0|, |W_0| with W = weight and bias flattened."""
    names, sides, stats = [], [], []
    for name, side, m in zero_convs(controlnet):
        w = torch.cat([m.weight.flatten(), m.bias.flatten()]).float()
        names.append(name); sides.append(side)
        stats.append(torch.stack([w.norm(), (w - w0[name]).norm(), w0[name].norm()]))
    stats = torch.stack(stats).cpu().tolist()
    return [[step, n, s] + [f"{x:.6e}" for x in v] for n, s, v in zip(names, sides, stats)]


def copy_drift_rows(step, state, theta0):
    """|theta_t - theta_0| and |theta_0| of the trainable copy, per stage and in total (fp32, CPU)."""
    d2, n2 = {}, {}
    for k, v in state.items():
        if param_group(k) != "copy":
            continue
        s = copy_stage(k)
        d2[s] = d2.get(s, 0.0) + (v - theta0[k]).pow(2).sum().item()
        n2[s] = n2.get(s, 0.0) + theta0[k].pow(2).sum().item()
    d2["all"], n2["all"] = sum(d2.values()), sum(n2.values())
    return [[step, s, f"{d2[s] ** 0.5:.6e}", f"{n2[s] ** 0.5:.6e}"] for s in d2]


def validate_output(args):
    """Prevent a restart from silently mixing runs or appending to unrecoverable logs."""
    out = args.output_dir
    config_path = os.path.join(out, "config.json")
    checkpoint = os.path.join(out, "checkpoint", "step.json")
    has_logs = any(os.path.exists(os.path.join(out, name)) for name in LOG_HEADERS)
    if has_logs and not os.path.isfile(checkpoint):
        raise ValueError("Training logs exist but no resumable checkpoint was saved. Use a new output directory.")
    if os.path.isfile(checkpoint) and not os.path.isfile(config_path):
        raise ValueError("Checkpoint has no run configuration; use a new output directory.")
    if os.path.isfile(config_path):
        with open(config_path) as handle:
            previous = json.load(handle)["args"]
        operational = {"output_dir", "max_steps", "per_gpu_batch", "num_workers", "time_limit_min",
                       "snapshot_every", "ckpt_every", "log_every", "dense_log_steps"}
        changed = [key for key, value in vars(args).items()
                   if key not in operational and previous.get(key) != value]
        if changed:
            raise ValueError(f"Run settings changed: {', '.join(changed)}. Use a new output directory.")
    if os.path.isfile(checkpoint):
        with open(checkpoint) as handle:
            step = json.load(handle)["step"]
        if args.max_steps < step:
            raise ValueError(f"Requested {args.max_steps} updates but checkpoint is already at {step}.")


# ----------------------------------------------------------------------------- main
def main():
    t_start = time.time()  # --time_limit_min counts from here
    args = parse_args()
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = True  # fp32 weights/activations, TF32 matmuls (paper: fp32)
    torch.backends.cudnn.allow_tf32 = True
    acc = Accelerator(mixed_precision="no")
    W, rank = acc.num_processes, acc.process_index
    if int(os.environ.get("WORLD_SIZE", 1)) != W:
        raise RuntimeError(f"launched with WORLD_SIZE={os.environ.get('WORLD_SIZE')} but accelerate sees {W} process(es)")
    assert args.global_batch % (args.per_gpu_batch * W) == 0, "global_batch must be divisible by per_gpu*world"
    A = args.global_batch // (args.per_gpu_batch * W)
    dev = acc.device
    out = args.output_dir
    main_proc = acc.is_main_process

    validate_output(args)
    os.makedirs(os.path.join(out, "snapshots"), exist_ok=True)
    ckpt_dir = os.path.join(out, "checkpoint")

    # models
    tok = CLIPTokenizer.from_pretrained(args.pretrained, subfolder="tokenizer")
    text = CLIPTextModel.from_pretrained(args.pretrained, subfolder="text_encoder").to(dev)
    vae = AutoencoderKL.from_pretrained(args.pretrained, subfolder="vae").to(dev)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained, subfolder="unet")
    sched = DDPMScheduler.from_pretrained(args.pretrained, subfolder="scheduler")
    # from_unet copies SD's encoder; the condition encoder (controlnet_cond_embedding) is new and randomly
    # initialised from the global CPU RNG. A fixed --hint_seed gives every run the same draw - like the authors'
    # single control_sd15_ini.ckpt that all their trainings start from. Only the zero convs differ between runs.
    torch.default_generator.manual_seed(args.hint_seed)
    controlnet = ControlNetModel.from_unet(unet)  # trainable copy of encoder + mid block
    init_stats = init_zero_convs(controlnet, args.init, args.init_std, args.init_seed, args.init_targets)
    unet.to(dev)
    for m in (text, vae, unet):
        m.requires_grad_(False); m.eval()
    controlnet.train()
    if args.grad_checkpointing:
        controlnet.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()

    opt = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)  # PyTorch defaults, as in cldm.py
    controlnet, opt = acc.prepare(controlnet, opt)
    cn = acc.unwrap_model(controlnet)
    named_params = list(cn.named_parameters())

    # resume
    start_step = 0
    if os.path.exists(os.path.join(ckpt_dir, "step.json")):
        acc.load_state(ckpt_dir)
        start_step = json.load(open(os.path.join(ckpt_dir, "step.json")))["step"]
        acc.print(f"resumed from step {start_step}")
        if main_proc:
            trim_rows(out, "train_log.csv", start_step - 1)
            trim_rows(out, "zeroconv_log.csv", start_step - 1)
            trim_rows(out, "copy_drift.csv", start_step)
    else:
        if main_proc:
            json.dump({"args": vars(args), "zero_conv_init": init_stats, "world_size": W, "accum": A},
                      open(os.path.join(out, "config.json"), "w"), indent=2)
        save_snapshot(acc, cn, out, 0)  # the init itself = reference for W_0 and theta_0 (all ranks)
    acc.wait_for_everyone()

    # references for the logs (main process only): W_0 per zero conv (on device), theta_0 = step-0 weights (CPU)
    if main_proc:
        theta0 = load_file(os.path.join(out, "snapshots", "step_000000.safetensors"))
        w0 = {name: torch.cat([theta0[zero_conv_key(name) + ".weight"].flatten(),
                               theta0[zero_conv_key(name) + ".bias"].flatten()]).float().to(dev)
              for name, _, _ in zero_convs(cn)}
        if start_step == 0:
            append_rows(out, "copy_drift.csv", copy_drift_rows(0, theta0, theta0))

    ds = StreamDataset(args.data_dir, tok, args.resolution, args.data_seed, args.prompt_dropout,
                       sched.config.num_train_timesteps, args.max_train_samples, args.canny_low, args.canny_high)
    dl = DataLoader(ds, batch_size=args.per_gpu_batch,
                    sampler=RankSampler(start_step, args.max_steps, args.global_batch, args.per_gpu_batch, W, rank),
                    num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0,
                    prefetch_factor=4 if args.num_workers > 0 else None)
    it = iter(dl)

    stop = {"flag": False}
    signal.signal(signal.SIGUSR1, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    alphas = sched.alphas_cumprod.to(dev)
    sf = vae.config.scaling_factor

    step = start_step
    t_last = time.time()
    last_logged = start_step
    while step < args.max_steps:
        # ---- forward / backward at the current weights (state after `step` updates)
        loss_acc = 0.0
        for a in range(A):
            b = next(it)
            with torch.no_grad():
                pix = b["pix"].to(dev, non_blocking=True)
                post = vae.encode(pix).latent_dist
                lat = (post.mean + post.std * b["vae_eps"].to(dev)) * sf      # posterior sample, as in ldm
                emb = text(b["ids"].to(dev))[0]
                noise = b["noise"].to(dev)
                t = b["t"].to(dev).long()
                ab = alphas[t].view(-1, 1, 1, 1)
                noisy = ab.sqrt() * lat + (1 - ab).sqrt() * noise               # q(z_t | z_0)
            sync = a == A - 1
            ctx = acc.no_sync(controlnet) if not sync else torch.enable_grad()
            with ctx:
                down, mid = controlnet(noisy, t, encoder_hidden_states=emb,
                                       controlnet_cond=b["cond"].to(dev), return_dict=False)
                pred = unet(noisy, t, encoder_hidden_states=emb,
                            down_block_additional_residuals=down, mid_block_additional_residual=mid,
                            return_dict=False)[0]
                loss = F.mse_loss(pred, noise) / A
                acc.backward(loss)
            loss_acc += loss.detach()

        # ---- logs of this state, taken before the update
        log_now = step < args.dense_log_steps or step % args.log_every == 0
        if log_now:
            loss_mean = acc.reduce(loss_acc, reduction="mean").item()  # all ranks take part
            if main_proc:
                g = grad_norms(named_params)
                zc_rows = zero_conv_rows(step, cn, w0)

        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if log_now and main_proc:
            now = time.time()
            dt = (now - t_last) / max(1, step - last_logged)
            t_last, last_logged = now, step
            mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            append_rows(out, "train_log.csv", [[step - 1, f"{loss_mean:.6f}", f"{g['z1']:.4e}", f"{g['z2']:.4e}",
                                                f"{g['hint']:.4e}", f"{g['copy']:.4e}", f"{dt:.3f}",
                                                f"{args.global_batch / dt:.2f}", f"{mem:.2f}"]])
            append_rows(out, "zeroconv_log.csv", zc_rows)
            print(f"step {step - 1} loss {loss_mean:.4f} |g| z1 {g['z1']:.1e} z2 {g['z2']:.1e} "
                  f"hint {g['hint']:.1e} copy {g['copy']:.1e} | {args.global_batch / dt:.1f} samples/s", flush=True)

        # ---- snapshots (+ copy drift), checkpoints, clean stop
        if step % args.snapshot_every == 0 or step == args.max_steps:
            state = save_snapshot(acc, cn, out, step)
            if main_proc:
                append_rows(out, "copy_drift.csv", copy_drift_rows(step, state, theta0))
        out_of_time = args.time_limit_min and (time.time() - t_start) / 60 > args.time_limit_min
        stop_now = acc.reduce(torch.tensor(float(stop["flag"] or bool(out_of_time)), device=dev), "sum").item() > 0
        if step % args.ckpt_every == 0 or step == args.max_steps or stop_now:
            save_ckpt(acc, ckpt_dir, step)
        if stop_now:
            acc.print(f"stopping at step {step} (time limit / signal); resubmit to continue")
            break

    acc.wait_for_everyone()
    if step >= args.max_steps and main_proc:
        open(os.path.join(out, "DONE"), "w").write(str(step))
    acc.end_training()


def save_snapshot(acc, cn, out_dir, step):
    """fp32 ControlNet weights. Returns the CPU state dict on the main process (None elsewhere)."""
    acc.wait_for_everyone()
    state = None
    if acc.is_main_process:
        state = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in cn.state_dict().items()}
        path = os.path.join(out_dir, "snapshots", f"step_{step:06d}.safetensors")
        save_file(state, path + ".tmp")
        os.replace(path + ".tmp", path)
    return state


def save_ckpt(acc, ckpt_dir, step):
    tmp = ckpt_dir + ".tmp"
    acc.wait_for_everyone()
    if acc.is_main_process and os.path.exists(tmp):
        shutil.rmtree(tmp)
    acc.wait_for_everyone()
    acc.save_state(tmp)
    acc.wait_for_everyone()
    if acc.is_main_process:
        json.dump({"step": step}, open(os.path.join(tmp, "step.json"), "w"))
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.replace(tmp, ckpt_dir)
    acc.wait_for_everyone()


if __name__ == "__main__":
    main()
