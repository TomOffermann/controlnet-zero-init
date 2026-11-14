"""Portable experiment commands. Print a plan unless --execute is supplied."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def commands(args, manifest):
    work = args.workspace.resolve()
    py = sys.executable
    code = ROOT
    records = [
        {"name": name + replica["suffix"],
         "training_args": {**manifest["training"], **variant,
                           **{k: v for k, v in replica.items() if k != "suffix"}}}
        for name, variant in manifest["variants"].items()
        for replica in manifest["replicates"]
    ]
    for record in records:
        record["training_args"].update(manifest["overrides"].get(record["name"], {}))
    chosen = records if args.run == "all" else [r for r in records if r["name"] == args.run]
    if not chosen:
        raise ValueError(f"Unknown run: {args.run}")
    if args.command == "prepare":
        for split in ("train2017", "val2017"):
            yield [py, str(code / "prepare_coco_canny.py"), "--coco_root", str(work / "coco"),
                   "--out", str(work / "data"), "--split", split, "--res", "512", "--workers", "8",
                   *(["--no_canny"] if split == "train2017" else [])]
    elif args.command == "train":
        for record in chosen:
            settings = dict(record["training_args"])
            settings.update(pretrained=args.pretrained, data_dir=str(work / "data/train"),
                            output_dir=str(work / "runs" / record["name"]), time_limit_min=0)
            if settings["global_batch"] % (settings["per_gpu_batch"] * args.gpus):
                raise ValueError("Global batch must be divisible by GPUs × per-GPU batch")
            command = [py, "-m", "accelerate.commands.launch", "--num_processes", str(args.gpus),
                       "--num_machines", "1", "--mixed_precision", "no", "--main_process_port", "29500"]
            if args.gpus > 1:
                command.append("--multi_gpu")
            command.append(str(code / "train_zeroconv.py"))
            for key, value in settings.items():
                if isinstance(value, bool):
                    if value:
                        command.append("--" + key)
                else:
                    command.extend(["--" + key, *map(str, value if isinstance(value, list) else [value])])
            yield command
    elif args.command == "evaluate":
        protocol = manifest["evaluation"][args.stage]
        common = [py, str(code / "eval_zeroconv.py"), "--pretrained", args.pretrained,
                  "--data", str(work / "data/val"), "--out", str(work / "eval" / args.stage)]
        sampling = ["--n", str(protocol["n"]), "--dtype", protocol["dtype"], "--steps", "20"]
        for record in chosen:
            yield common + sampling + ["--name", record["name"], "--run", str(work / "runs" / record["name"]),
                                       "--only", *map(str, protocol["steps"])]
        yield common + sampling + ["--name", "sd", "--controlnet", "none"]
        yield common + sampling + ["--name", "released", "--controlnet", args.controlnet]
        yield common + ["--real", "--n", "5000"]
    else:
        yield [py, str(code / "summarize_eval.py"), str(work / "eval" / args.stage)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "summarize"])
    parser.add_argument("--run", default="zero", help="name in experiment.json, or all (24 runs)")
    parser.add_argument("--workspace", type=Path, default=ROOT / "work")
    parser.add_argument("--pretrained", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--controlnet", default="lllyasviel/sd-controlnet-canny")
    parser.add_argument("--gpus", type=int, choices=[1, 2], default=2,
                        help="original training used 2; one GPU increases accumulation to preserve batch 32")
    parser.add_argument("--stage", choices=["curves", "final"], default="final")
    parser.add_argument("--execute", action="store_true", help="run commands (otherwise only print them)")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "experiment.json").read_text())
    try:
        plan = list(commands(args, manifest))
    except ValueError as error:
        parser.error(str(error))
    for command in plan:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True, cwd=ROOT)
    if not args.execute:
        print("Plan only. Add --execute to run these commands.")


if __name__ == "__main__":
    main()
