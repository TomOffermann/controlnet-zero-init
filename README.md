# ControlNet zero-initialization experiments

Reproduce the Canny ControlNet initialization study: 8 settings × 3 paired seeds on COCO 2017, with frozen Stable Diffusion 1.5. This is a separate ablation study using the [ControlNet architecture](https://github.com/lllyasviel/ControlNet).

## Setup

Linux, Python 3.11, and NVIDIA GPUs with a CUDA 12.6-compatible driver:

```bash
git clone https://github.com/TomOffermann/controlnet-zero-init.git
cd controlnet-zero-init
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Download the **2017 train images, validation images, and train/val annotations** from [COCO](https://cocodataset.org/#download). Extract them into:

```text
work/coco/train2017/*.jpg
work/coco/val2017/*.jpg
work/coco/annotations/captions_train2017.json
work/coco/annotations/captions_val2017.json
```

The scripts download [SD 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), the [released Canny ControlNet](https://huggingface.co/lllyasviel/sd-controlnet-canny), and Inception weights when needed. Dataset and model terms apply. Use `--pretrained /path/to/sd15` and `--controlnet /path/to/controlnet` for local Diffusers-format models.

## Run

Start with the zero-initialized run and its final evaluation:

```bash
python reproduce.py prepare --execute
python reproduce.py train --run zero --execute
python reproduce.py evaluate --run zero --stage final --execute
python reproduce.py summarize --stage final --execute
```

For all 24 runs and both evaluation stages:

```bash
python reproduce.py train --run all --execute
python reproduce.py evaluate --run all --stage final --execute
python reproduce.py summarize --stage final --execute
python reproduce.py evaluate --run all --stage curves --execute
python reproduce.py summarize --stage curves --execute
```

Without `--execute`, commands are printed only. Training defaults to two GPUs; add `--gpus 1` to preserve batch 32 through extra gradient accumulation. Evaluation uses one GPU. Runs execute sequentially, resume from their latest checkpoint, and skip verified complete evaluation pairs. Use a separate `--workspace` if changing settings. If interrupted before the first checkpoint, restart in a fresh workspace to avoid mixing partial logs.

All settings and seeds are in [`experiment.json`](experiment.json). Run names are `zero`, `gauss_1e-4`, `gauss_1e-3`, `gauss_1e-2`, `gauss_1e-1`, `default`, `in_only_default`, and `out_only_default`; append `_seed1` or `_seed2` for the other replicates. `default` uses PyTorch convolution initialization; the side-specific variants keep the other side zero.

## Results and resources

- Training: 118,287 images at 512×512, 15,000 updates, batch 32, AdamW at 1e-5, 50% prompt dropout, FP32 with TF32 enabled. Training Canny thresholds vary per image; validation uses 100/200.
- Evaluation: UniPC 20 steps, first caption / CFG 7.5 and empty text / CFG 1. Curves use 500 images in FP16; final evaluation uses 2,000 in FP32. Metrics are tolerance-matched Canny F1 and KID against 5,000 real validation images. KID uses 100 subsets of up to 1,000 images, seed 0; its standard deviation is across subsets.
- Output: `work/eval/{final,curves}/summary.csv` contains scores per run, checkpoint and prompt mode, including plain SD and released ControlNet references. `work/runs/<name>/` contains weights, resumable state, and loss/gradient/weight-drift CSV logs.
- Original training used 2× RTX 4090 (24 GB each): approximately 21 GPU-hours per run, 510 across all 24, excluding evaluation. Keeping all 61 snapshots costs about 90 GB per run, plus checkpoints and data; allow over 2 TB for the full sweep.

The repository contains only reproduction code and configuration. Data and trained weights are regenerated. Original model revision hashes were not recorded; later downloads, hardware, and library changes can affect numerical results.

Run the small regression checks with `python -m unittest discover -s tests` (no model downloads).
