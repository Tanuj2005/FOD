"""
save_leakage_examples.py
Saves N side-by-side examples (valid/test image | its closest train image) from the
near_duplicate_pairs.csv produced by eda_fod.py. Boxes are drawn from the label files.

Run:  python save_leakage_examples.py
"""
# ============================ CONFIG ============================
DATASET_ROOT = r"Runway FOD Detector.v1i.yolov11"                                   # folder with train/valid/test
PAIRS_CSV = r"eda_results\ds1_runway_fod_detector\near_duplicate_pairs.csv"
OUT_DIR = r"eda_results\ds1_runway_fod_detector\leakage_examples"
N_EXAMPLES = 20
CELL = 480                      # size of each image in the montage (px)
# ================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def index_split(root, split_names):
    for s in split_names:
        d = Path(root) / s / "images"
        if d.is_dir():
            return {p.name: p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS}
    return {}


def read_boxes(img_path):
    lp = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    boxes = []
    if lp.exists():
        for line in lp.read_text(errors="replace").splitlines():
            v = line.split()
            if len(v) == 5:
                boxes.append(tuple(map(float, v[1:])))
    return boxes


def render(img_path, caption):
    with Image.open(img_path) as im0:
        im = im0.convert("RGB")
    im = im.resize((CELL, CELL))
    d = ImageDraw.Draw(im)
    for xc, yc, w, h in read_boxes(img_path):
        d.rectangle([(xc - w / 2) * CELL, (yc - h / 2) * CELL, (xc + w / 2) * CELL, (yc + h / 2) * CELL],
                    outline=(255, 0, 0), width=3)
    d.rectangle([0, 0, CELL, 18], fill=(0, 0, 0))
    d.text((4, 3), caption, fill=(255, 255, 0))
    return im


def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PAIRS_CSV)
    pairs = pairs[pairs.relation.isin(["valid->train", "test->train"])].sort_values("hamming").reset_index(drop=True)
    if pairs.empty:
        print("No cross-split pairs found.")
        return
    print(f"{len(pairs)} cross-split near-duplicate pairs available")

    # evenly spaced over the distance range: shows clear duplicates AND borderline ones
    idx = np.unique(np.linspace(0, len(pairs) - 1, min(N_EXAMPLES, len(pairs))).astype(int))
    sample = pairs.iloc[idx]

    index = {"train": index_split(DATASET_ROOT, ["train"]),
             "valid": index_split(DATASET_ROOT, ["valid", "val", "validation"]),
             "test": index_split(DATASET_ROOT, ["test"])}

    saved = 0
    for k, r in enumerate(sample.itertuples(), 1):
        src_split = r.relation.split("->")[0]
        a = index[src_split].get(r.image)
        b = index["train"].get(r.partner)
        if a is None or b is None:
            print(f"  skipped (file not found): {r.image} / {r.partner}")
            continue
        left = render(a, f"{src_split.upper()}: {r.image[:50]}")
        right = render(b, f"TRAIN: {r.partner[:50]}")
        canvas = Image.new("RGB", (2 * CELL + 10, CELL + 24), (20, 20, 20))
        canvas.paste(left, (0, 24))
        canvas.paste(right, (CELL + 10, 24))
        ImageDraw.Draw(canvas).text((6, 6), f"example {k} | hamming distance = {r.hamming} / 256 | {r.relation}",
                                    fill=(255, 255, 255))
        canvas.save(out / f"leak_{k:02d}_{src_split}_d{r.hamming}.png")
        saved += 1
        del left, right, canvas
    print(f"Saved {saved} examples to {out}")


if __name__ == "__main__":
    main()