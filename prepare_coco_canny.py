"""COCO 2017 -> 512x512 image / Canny-condition pairs for ControlNet training.

Train split: images only (training computes Canny on the fly with random thresholds, as in the paper).
Val split: fixed cv2.Canny(rgb, 100, 200) maps (ControlNet demo defaults) for evaluation.
Images: resize short side to `res` (INTER_AREA) and centre-crop to res x res.

Output layout (per split):
  <out>/<split>/images/<id>.jpg      (quality 95)
  <out>/<split>/canny/<id>.png       (single channel, 0/255)
  <out>/<split>/meta.jsonl           one line per image: {"id": "...", "captions": [...]}, sorted by id

Usage:
  python prepare_coco_canny.py --coco_root work/coco --out work/data --split train2017 --workers 16
  python prepare_coco_canny.py --coco_root work/coco --out work/data --split val2017   --workers 16
"""
import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

LOW, HIGH = 100, 200


def center_crop_resize(img: np.ndarray, res: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = res / min(h, w)
    nh, nw = max(res, round(h * s)), max(res, round(w * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y0, x0 = (nh - res) // 2, (nw - res) // 2
    return img[y0:y0 + res, x0:x0 + res]


def canny(rgb_uint8: np.ndarray) -> np.ndarray:
    return cv2.Canny(rgb_uint8, LOW, HIGH)  # same call as ControlNet's annotator (RGB input)


def _work(job):
    src, dst_img, dst_cond, res = job
    if os.path.exists(dst_img) and (dst_cond is None or os.path.exists(dst_cond)):
        return True
    bgr = cv2.imread(src, cv2.IMREAD_COLOR)
    if bgr is None:
        return False
    rgb = center_crop_resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), res)
    written = cv2.imwrite(dst_img, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if dst_cond is not None:
        written = cv2.imwrite(dst_cond, canny(rgb)) and written
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", required=True, help="dir containing <split>/ and annotations/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train2017")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N images")
    ap.add_argument("--no_canny", action="store_true",
                    help="train split: skip edge maps (training computes Canny on the fly with random thresholds)")
    args = ap.parse_args()

    ann = json.load(open(os.path.join(args.coco_root, "annotations", f"captions_{args.split}.json")))
    caps = defaultdict(list)
    for a in ann["annotations"]:
        caps[a["image_id"]].append(a["caption"].strip())
    images = sorted(ann["images"], key=lambda im: im["id"])
    if args.limit:
        images = images[: args.limit]

    out = os.path.join(args.out, args.split.replace("2017", ""))
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    if not args.no_canny:
        os.makedirs(os.path.join(out, "canny"), exist_ok=True)

    jobs, metas = [], []
    for im in images:
        key = f"{im['id']:012d}"
        jobs.append((os.path.join(args.coco_root, args.split, im["file_name"]),
                     os.path.join(out, "images", key + ".jpg"),
                     None if args.no_canny else os.path.join(out, "canny", key + ".png"), args.res))
        metas.append({"id": key, "captions": caps[im["id"]]})

    with Pool(args.workers) as pool:
        ok = pool.map(_work, jobs, chunksize=64)

    n_bad = 0
    with open(os.path.join(out, "meta.jsonl"), "w") as f:
        for m, good in zip(metas, ok):
            if good and m["captions"]:
                f.write(json.dumps(m) + "\n")
            else:
                n_bad += 1
    print(f"{args.split}: wrote {len(metas) - n_bad} pairs to {out} ({n_bad} skipped)")
    if n_bad:
        raise SystemExit("Preparation incomplete: fix missing/corrupt images or captions and rerun before training.")


if __name__ == "__main__":
    main()
