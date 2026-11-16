"""
eda_fod.py  -  Modular EDA for YOLO-format FOD datasets (Roboflow export layout)

Expected layout of every dataset folder:
    <dataset>/train/images, train/labels
    <dataset>/valid/images, valid/labels      ("val" also accepted)
    <dataset>/test/images,  test/labels
    <dataset>/data.yaml

HOW TO RUN (Windows, from Anaconda/CMD/PowerShell):
    pip install numpy pandas matplotlib pillow pyyaml psutil
    python eda_fod.py

Edit ONLY the CONFIG block below (paste your folder paths there).

Memory design:
  * Each dataset is analysed in its OWN subprocess. When it finishes, the OS
    frees 100% of its memory, so nothing carries over to the next dataset.
  * Images are opened one at a time, thumbnailed, and closed immediately.
  * Only small arrays (32-byte hashes, scalar stats) are kept per image.
  * Figures are closed after saving; gc.collect() runs after every stage.
  * No GPU is used, so the RTX 3050 is untouched.
"""

# =============================== CONFIG ====================================
DATASETS = {
    # "short_name": r"full\path\to\dataset_folder",
    "ds1_runway_fod_detector": r"Runway FOD Detector.v1i.yolov11",
    "ds2_fod_runway":          r"FOD Runway Dataset.v2i.yolov11",
    "ds3_foreign_object_debris": r"foreign object debris.v1i.yolov11",
}
OUTPUT_DIR = r"eda_results"

RARE_CLASS_MIN_INSTANCES = 50      # classes below this are flagged "rare"
DUP_BOX_IOU = 0.90                 # same-class boxes above this IoU = duplicate
NEAR_DUP_HAMMING = 10              # out of 256 bits; <= this = near-duplicate image
SMALL_AREA_PX = 32 ** 2            # COCO small  (< 32x32 px)
MEDIUM_AREA_PX = 96 ** 2           # COCO medium (< 96x96 px)
SPOTCHECK_N = 20                   # images in random spot-check grid
TINY_CROPS_N = 20                  # smallest boxes shown as zoomed crops
MAX_IMAGES_PIXEL_PASS = None       # e.g. 2000 to sample; None = all images
SEED = 42
# ===========================================================================

import gc
import json
import random
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")                      # no GUI, lower memory
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

try:
    import psutil
except ImportError:
    psutil = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_ALIASES = {"train": ["train"], "valid": ["valid", "val", "validation"], "test": ["test"]}
SPLIT_ORDER = ["train", "valid", "test"]
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ------------------------------ utilities ----------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mem(tag):
    if psutil:
        rss = psutil.Process().memory_info().rss / 1024 ** 2
        log(f"   [mem] {tag}: {rss:,.0f} MB")


def free():
    plt.close("all")
    gc.collect()


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    free()


# --------------------------- 1. dataset discovery --------------------------
def find_splits(root):
    root = Path(root)
    out = {}
    for canon, aliases in SPLIT_ALIASES.items():
        for a in aliases:
            d = root / a
            if d.is_dir() and (d / "images").is_dir():
                out[canon] = (d / "images", d / "labels")
                break
    return out


def load_class_names(root):
    yp = Path(root) / "data.yaml"
    if not yp.exists():
        return []
    with open(yp, encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    names = y.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda k: int(k))]
    return list(names)


# --------------------------- 2. label parsing ------------------------------
def parse_label_file(path, n_cls):
    """Returns (boxes, issues, n_polygons).
    boxes: list of (cls, xc, yc, w, h, valid_flag). Polygons are converted to bboxes."""
    boxes, issues, polys = [], [], 0
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return boxes, [("unreadable_label", str(e))], 0
    eps = 1e-3
    for ln, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            issues.append(("malformed_line", f"line {ln}"))
            continue
        if len(vals) == 4:
            xc, yc, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:       # segmentation polygon
            xs, ys = vals[0::2], vals[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            polys += 1
        else:
            issues.append(("malformed_line", f"line {ln}"))
            continue
        valid = True
        if n_cls and not (0 <= cls < n_cls):
            issues.append(("class_id_out_of_range", f"line {ln}: {cls}"))
            valid = False
        if w <= 0 or h <= 0:
            issues.append(("non_positive_size", f"line {ln}"))
            valid = False
        elif (xc - w / 2 < -eps or yc - h / 2 < -eps or
              xc + w / 2 > 1 + eps or yc + h / 2 > 1 + eps):
            issues.append(("out_of_bounds", f"line {ln}"))
            valid = False
        boxes.append((cls, xc, yc, w, h, valid))
    return boxes, issues, polys


def mark_duplicate_boxes(b, thr):
    """b: Nx5 array (cls,xc,yc,w,h). Returns bool array; True = duplicate of an earlier box."""
    n = len(b)
    if n < 2 or n > 2000:
        return np.zeros(n, bool)
    x1, x2 = b[:, 1] - b[:, 3] / 2, b[:, 1] + b[:, 3] / 2
    y1, y2 = b[:, 2] - b[:, 4] / 2, b[:, 2] + b[:, 4] / 2
    area = b[:, 3] * b[:, 4]
    iw = np.clip(np.minimum(x2[:, None], x2[None, :]) - np.maximum(x1[:, None], x1[None, :]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None, :]) - np.maximum(y1[:, None], y1[None, :]), 0, None)
    inter = iw * ih
    iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-12)
    same = b[:, 0][:, None] == b[:, 0][None, :]
    return np.triu((iou > thr) & same, 1).any(axis=0)


# ------------------------ 3. inventory + integrity -------------------------
def build_tables(root, splits, names):
    n_cls = len(names)
    img_rows, box_rows, issue_rows = [], [], []
    orphan_labels = 0
    for split in SPLIT_ORDER:
        if split not in splits:
            continue
        img_dir, lbl_dir = splits[split]
        files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else set()
        orphan_labels += len(lbl_stems - {p.stem for p in files})
        log(f"   {split}: {len(files)} images")
        for i, p in enumerate(files):
            if i and i % 1000 == 0:
                log(f"      {split}: {i}/{len(files)}")
            corrupt, W, H, mode = False, np.nan, np.nan, ""
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    W, H = im.size
                    mode = im.mode
            except Exception as e:
                corrupt = True
                issue_rows.append((split, p.name, "corrupt_image", str(e)[:100]))
            lp = lbl_dir / (p.stem + ".txt")
            has_label_file = lp.exists()
            boxes, polys = [], 0
            if has_label_file:
                boxes, issues, polys = parse_label_file(lp, n_cls)
                for kind, detail in issues:
                    issue_rows.append((split, p.name, kind, detail))
            else:
                issue_rows.append((split, p.name, "missing_label_file", ""))
            dup_flags = np.zeros(len(boxes), bool)
            if boxes:
                arr = np.array([b[:5] for b in boxes], dtype=np.float64)
                vmask = np.array([b[5] for b in boxes])
                dup_flags = np.zeros(len(boxes), bool)
                dup_flags[vmask] = mark_duplicate_boxes(arr[vmask], DUP_BOX_IOU)
                for k in np.where(dup_flags)[0]:
                    issue_rows.append((split, p.name, "duplicate_box", f"box #{k + 1}"))
            n_valid = int(sum(b[5] for b in boxes))
            img_rows.append((split, p.name, p.stem, p.stem.split(".rf.")[0], str(p),
                             W, H, mode, corrupt, has_label_file, len(boxes), n_valid, polys))
            for b, d in zip(boxes, dup_flags):
                box_rows.append((split, p.name, b[0], b[1], b[2], b[3], b[4], b[5], bool(d), W, H))
    img_df = pd.DataFrame(img_rows, columns=[
        "split", "image", "stem", "base_name", "path", "width", "height", "mode", "corrupt",
        "has_label_file", "n_boxes", "n_valid_boxes", "n_polygons"])
    box_df = pd.DataFrame(box_rows, columns=[
        "split", "image", "cls", "xc", "yc", "w", "h", "valid", "is_duplicate", "img_w", "img_h"])
    issue_df = pd.DataFrame(issue_rows, columns=["split", "image", "issue", "detail"])
    # derived box geometry (pixels at stored resolution)
    box_df["w_px"] = box_df.w * box_df.img_w
    box_df["h_px"] = box_df.h * box_df.img_h
    box_df["area_px"] = box_df.w_px * box_df.h_px
    box_df["area_pct"] = box_df.w * box_df.h * 100
    box_df["min_side_px"] = box_df[["w_px", "h_px"]].min(axis=1)
    box_df["aspect"] = box_df.w_px / box_df.h_px.replace(0, np.nan)
    box_df["size_cat"] = np.where(box_df.area_px < SMALL_AREA_PX, "small",
                                  np.where(box_df.area_px < MEDIUM_AREA_PX, "medium", "large"))
    box_df["cls_name"] = box_df.cls.map(lambda c: names[c] if 0 <= c < n_cls else f"id_{c}")
    return img_df, box_df, issue_df, orphan_labels


# ------------------- 4. pixel pass: brightness + hashes --------------------
def pixel_pass(img_df, seed):
    """One image at a time -> brightness/sharpness/saturation + 256-bit dHash."""
    n = len(img_df)
    ok_idx = np.where(~img_df.corrupt.values)[0]
    if MAX_IMAGES_PIXEL_PASS and len(ok_idx) > MAX_IMAGES_PIXEL_PASS:
        rng = np.random.default_rng(seed)
        ok_idx = np.sort(rng.choice(ok_idx, MAX_IMAGES_PIXEL_PASS, replace=False))
    stats = np.full((n, 5), np.nan, dtype=np.float32)   # bright, contrast, sat, sharp, is_gray
    hashes = np.zeros((n, 32), dtype=np.uint8)
    hashed = np.zeros(n, dtype=bool)
    paths = img_df.path.values
    for c, i in enumerate(ok_idx):
        if c and c % 1000 == 0:
            log(f"      pixel pass {c}/{len(ok_idx)}")
        try:
            with Image.open(paths[i]) as im0:
                im0.draft("RGB", (512, 512))             # fast JPEG decode, no-op otherwise
                im1 = im0.convert("RGB")
            im1.thumbnail((256, 256))
            rgb = np.asarray(im1, dtype=np.float32)
            gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            mx, mn = rgb.max(axis=2), rgb.min(axis=2)
            sat = float(((mx - mn) / (mx + 1e-6)).mean())
            lap = (4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
                   - gray[1:-1, :-2] - gray[1:-1, 2:])
            chan_diff = float(np.abs(rgb[..., 0] - rgb[..., 1]).mean()
                              + np.abs(rgb[..., 1] - rgb[..., 2]).mean())
            stats[i] = [gray.mean(), gray.std(), sat, lap.var(), float(chan_diff < 2.0)]
            g = np.asarray(im1.convert("L").resize((17, 16), RESAMPLE), dtype=np.int16)
            hashes[i] = np.packbits((g[:, 1:] > g[:, :-1]).flatten())
            hashed[i] = True
            del rgb, gray, mx, mn, lap, g, im1
        except Exception:
            pass
    out = img_df[["split", "image"]].copy()
    out["brightness"], out["contrast"], out["saturation"], out["sharpness"], out["is_gray"] = stats.T
    out["hashed"] = hashed
    free()
    return out, hashes


def min_hamming(A, B, exclude_self=False, chunk=96):
    """For each row of A: min Hamming distance to rows of B and argmin index."""
    dmin = np.full(len(A), 999, dtype=np.int32)
    arg = np.full(len(A), -1, dtype=np.int32)
    if len(A) == 0 or len(B) == 0:
        return dmin, arg
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk]
        d = POPCOUNT[np.bitwise_xor(a[:, None, :], B[None, :, :])].sum(axis=2, dtype=np.int32)
        if exclude_self:
            d[np.arange(len(a)), np.arange(s, s + len(a))] = 999
        j = d.argmin(axis=1)
        dmin[s:s + len(a)] = d[np.arange(len(a)), j]
        arg[s:s + len(a)] = j
        del d
    return dmin, arg


def duplicate_analysis(img_df, pix_df, hashes):
    """Leakage check: near-duplicate images across splits and inside train."""
    res, pairs = {}, []
    ok = pix_df.hashed.values
    idx = {s: np.where((img_df.split.values == s) & ok)[0] for s in SPLIT_ORDER}
    nearest = {}

    def record(src, dst, d, a, relation):
        for k in range(len(d)):
            if d[k] <= NEAR_DUP_HAMMING:
                pairs.append((relation, img_df.image.values[idx[src][k]],
                              img_df.image.values[idx[dst][a[k]]], int(d[k]),
                              img_df.path.values[idx[src][k]], img_df.path.values[idx[dst][a[k]]]))

    for s in ("valid", "test"):
        if len(idx[s]) and len(idx["train"]):
            d, a = min_hamming(hashes[idx[s]], hashes[idx["train"]])
            nearest[s] = d
            res[f"{s}_near_dup_of_train"] = int((d <= NEAR_DUP_HAMMING).sum())
            res[f"{s}_near_dup_of_train_pct"] = round(100 * (d <= NEAR_DUP_HAMMING).mean(), 2)
            res[f"{s}_exact_dup_of_train"] = int((d == 0).sum())
            res[f"{s}_median_dist_to_train"] = float(np.median(d))
            record(s, "train", d, a, f"{s}->train")
    if len(idx["valid"]) and len(idx["test"]):
        d, a = min_hamming(hashes[idx["test"]], hashes[idx["valid"]])
        res["test_near_dup_of_valid"] = int((d <= NEAR_DUP_HAMMING).sum())
        record("test", "valid", d, a, "test->valid")
    if len(idx["train"]) > 1:
        d, a = min_hamming(hashes[idx["train"]], hashes[idx["train"]], exclude_self=True)
        res["train_images_with_near_dup_in_train"] = int((d <= NEAR_DUP_HAMMING).sum())
        res["train_images_with_near_dup_in_train_pct"] = round(100 * (d <= NEAR_DUP_HAMMING).mean(), 2)
        nearest["train_internal"] = d
    pair_df = pd.DataFrame(pairs, columns=["relation", "image", "partner", "hamming", "path", "partner_path"])
    if len(pair_df):
        pair_df = pair_df.sort_values("hamming").reset_index(drop=True)
    return res, pair_df, nearest


# ----------------------------- 5. statistics -------------------------------
def basename_leakage(img_df):
    """Roboflow augmented copies share the name before '.rf.'; groups spanning splits = leakage."""
    g = img_df.groupby("base_name")
    n_per_base = g.size()
    n_splits = g.split.nunique()
    return {
        "unique_base_names": int(len(n_per_base)),
        "base_names_with_multiple_files": int((n_per_base > 1).sum()),
        "files_that_are_augmented_copies": int(len(img_df) - len(n_per_base)),
        "base_names_spanning_multiple_splits": int((n_splits > 1).sum()),
    }


def class_table(box_df, names, splits_present):
    vb = box_df[box_df.valid]
    rows = []
    all_ids = sorted(set(range(len(names))) | set(vb.cls.unique()))
    for c in all_ids:
        sub = vb[vb.cls == c]
        row = {"class_id": c, "class_name": names[c] if c < len(names) else f"id_{c}", "total": len(sub)}
        for s in SPLIT_ORDER:
            row[s] = int((sub.split == s).sum()) if s in splits_present else 0
        if len(sub):
            row["pct_small"] = round(100 * (sub.size_cat == "small").mean(), 1)
            row["median_area_px"] = round(float(sub.area_px.median()), 1)
            row["median_min_side_px"] = round(float(sub.min_side_px.median()), 1)
            row["median_aspect"] = round(float(sub.aspect.median()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(name, root, names, img_df, box_df, issue_df, pix_df, dup_res, orphan_labels, class_df):
    vb = box_df[box_df.valid]
    S = {"dataset": name, "root": str(root), "classes": len(names)}
    present = [s for s in SPLIT_ORDER if (img_df.split == s).any()]
    for s in SPLIT_ORDER:
        imgs = img_df[(img_df.split == s) & ~img_df.corrupt]
        S[f"{s}_images"] = int(len(imgs))
        S[f"{s}_instances"] = int((vb.split == s).sum())
        S[f"{s}_background_pct"] = round(100 * float((imgs.n_valid_boxes == 0).mean()), 2) if len(imgs) else np.nan
    S["total_images"] = int(len(img_df))
    S["total_instances"] = int(len(vb))
    S["boxes_per_image_mean"] = round(float(img_df.n_valid_boxes.mean()), 2)
    S["boxes_per_image_max"] = int(img_df.n_valid_boxes.max()) if len(img_df) else 0
    S["unique_resolutions"] = int(img_df[["width", "height"]].drop_duplicates().shape[0])
    S["most_common_resolution"] = (f"{int(img_df.width.mode().iloc[0])}x{int(img_df.height.mode().iloc[0])}"
                                   if img_df.width.notna().any() else "n/a")
    S["image_modes"] = ",".join(sorted(set(img_df["mode"].astype(str))))
    S["polygon_labels_converted"] = int(img_df.n_polygons.sum())
    for kind in ["corrupt_image", "missing_label_file", "malformed_line", "class_id_out_of_range",
                 "non_positive_size", "out_of_bounds", "duplicate_box", "unreadable_label"]:
        S[f"issue_{kind}"] = int((issue_df.issue == kind).sum())
    S["orphan_label_files"] = int(orphan_labels)
    # class balance
    counts = class_df.total[class_df.total > 0]
    S["imbalance_ratio_max_over_min"] = round(float(counts.max() / counts.min()), 1) if len(counts) else np.nan
    S["rare_classes_lt_threshold"] = ",".join(class_df[class_df.total < RARE_CLASS_MIN_INSTANCES].class_name.astype(str))
    S["n_rare_classes"] = int((class_df.total < RARE_CLASS_MIN_INSTANCES).sum())
    S["classes_missing_in_valid"] = ",".join(class_df[(class_df.valid == 0)].class_name.astype(str))
    S["classes_missing_in_test"] = ",".join(class_df[(class_df.test == 0)].class_name.astype(str))
    # sizes
    if len(vb):
        S["pct_small_objects"] = round(100 * float((vb.size_cat == "small").mean()), 1)
        S["pct_medium_objects"] = round(100 * float((vb.size_cat == "medium").mean()), 1)
        S["pct_large_objects"] = round(100 * float((vb.size_cat == "large").mean()), 1)
        S["median_box_area_pct_of_image"] = round(float(vb.area_pct.median()), 4)
        S["median_box_min_side_px"] = round(float(vb.min_side_px.median()), 1)
        S["pct_boxes_min_side_lt_8px"] = round(100 * float((vb.min_side_px < 8).mean()), 1)
        S["pct_boxes_min_side_lt_16px"] = round(100 * float((vb.min_side_px < 16).mean()), 1)
        S["median_aspect_ratio_w_over_h"] = round(float(vb.aspect.median()), 2)
    # conditions
    p = pix_df[pix_df.hashed]
    if len(p):
        S["brightness_mean"] = round(float(p.brightness.mean()), 1)
        S["brightness_std_across_images"] = round(float(p.brightness.std()), 1)
        S["pct_dark_images_lt_60"] = round(100 * float((p.brightness < 60).mean()), 1)
        S["pct_bright_images_gt_190"] = round(100 * float((p.brightness > 190).mean()), 1)
        S["pct_grayscale_images"] = round(100 * float(p.is_gray.mean()), 1)
        S["sharpness_median_lapvar"] = round(float(p.sharpness.median()), 1)
        S["pct_blurry_lapvar_lt_30"] = round(100 * float((p.sharpness < 30).mean()), 1)
    S.update(dup_res)
    S.update(basename_leakage(img_df))
    return S


def suggest_actions(S):
    A = []
    if S.get("issue_corrupt_image", 0): A.append(f"Remove {S['issue_corrupt_image']} corrupt image(s).")
    inv = sum(S.get(f"issue_{k}", 0) for k in ["malformed_line", "class_id_out_of_range", "non_positive_size", "out_of_bounds"])
    if inv: A.append(f"Fix or drop {inv} invalid box line(s) (see issues.csv).")
    if S.get("issue_duplicate_box", 0): A.append(f"Remove {S['issue_duplicate_box']} duplicate box(es).")
    if S.get("issue_missing_label_file", 0): A.append(f"{S['issue_missing_label_file']} image(s) have no label file: treated as background, verify they are truly empty.")
    if S.get("files_that_are_augmented_copies", 0) > 0: A.append(f"{S['files_that_are_augmented_copies']} files look like augmented copies: dedupe by base_name or re-download without augmentation.")
    if S.get("base_names_spanning_multiple_splits", 0): A.append("Same source image appears in multiple splits (leakage): re-split by base_name.")
    for s in ("valid", "test"):
        if S.get(f"{s}_near_dup_of_train_pct", 0) > 2:
            A.append(f"{S[f'{s}_near_dup_of_train_pct']}% of {s} images are near-duplicates of train: re-split by scene/sequence or remove them from {s}.")
    if S.get("train_images_with_near_dup_in_train_pct", 0) > 20:
        A.append("Many near-duplicate images inside train (likely video frames): consider thinning them.")
    tb = S.get("train_background_pct", 0)
    if tb == tb:
        if tb < 3: A.append(f"Train background-only images = {tb}%: add hard negatives up to ~5-10%.")
        elif tb > 15: A.append(f"Train background-only images = {tb}%: high for a recall-first model, consider subsampling to ~5-10%.")
    if S.get("pct_small_objects", 0) > 50: A.append(f"{S['pct_small_objects']}% of objects are <32x32 px: plan higher imgsz/tiling (SAHI) and expect small-object misses.")
    if S.get("pct_boxes_min_side_lt_8px", 0) > 10: A.append(f"{S['pct_boxes_min_side_lt_8px']}% of boxes have a side <8 px: near the detectability limit at this resolution.")
    if S.get("n_rare_classes", 0): A.append(f"Rare classes (<{RARE_CLASS_MIN_INSTANCES} instances): {S['rare_classes_lt_threshold']}. Merge, oversample, or document.")
    if S.get("classes_missing_in_test"): A.append(f"Classes absent from test: {S['classes_missing_in_test']}. Stratify the split.")
    if S.get("classes_missing_in_valid"): A.append(f"Classes absent from valid: {S['classes_missing_in_valid']}. Stratify the split.")
    if S.get("pct_grayscale_images", 0) > 20: A.append(f"{S['pct_grayscale_images']}% of images are grayscale: colour cues unavailable.")
    if S.get("pct_dark_images_lt_60", 0) < 2 and S.get("pct_bright_images_gt_190", 0) < 2:
        A.append("Almost no dark/very bright images: night/low-light generalisation cannot be evaluated with this data alone.")
    return A or ["No major issues detected."]


# ------------------------------- 6. plots ----------------------------------
def plot_class_distribution(class_df, out):
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(class_df) + 3), 5))
    bottom = np.zeros(len(class_df))
    for s in SPLIT_ORDER:
        ax.bar(class_df.class_name.astype(str), class_df[s], bottom=bottom, label=s)
        bottom += class_df[s].values
    ax.set_yscale("log"); ax.set_ylabel("instances (log)"); ax.set_title("Instances per class and split")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right"); ax.legend()
    save_fig(fig, out)


def plot_sizes(vb, out):
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    axs[0, 0].hist(np.log10(vb.area_pct.clip(lower=1e-5)), bins=60); axs[0, 0].set_title("Box area, log10(% of image)")
    axs[0, 1].hist(vb.w_px, bins=60, alpha=.6, label="width"); axs[0, 1].hist(vb.h_px, bins=60, alpha=.6, label="height")
    axs[0, 1].set_title("Box width/height (px)"); axs[0, 1].legend()
    axs[0, 2].hist(np.log10(vb.aspect.dropna().clip(lower=1e-2, upper=1e2)), bins=60); axs[0, 2].set_title("Aspect ratio, log10(w/h)")
    cat = pd.crosstab(vb.cls_name, vb.size_cat).reindex(columns=["small", "medium", "large"], fill_value=0)
    cat.div(cat.sum(axis=1), axis=0).mul(100).plot(kind="bar", stacked=True, ax=axs[1, 0])
    axs[1, 0].set_title("Small/medium/large % per class"); axs[1, 0].set_ylabel("%")
    axs[1, 1].hist(vb.min_side_px.clip(upper=100), bins=60); axs[1, 1].set_title("Min box side (px, clipped at 100)")
    axs[1, 1].axvline(8, color="r", ls="--"); axs[1, 1].axvline(16, color="orange", ls="--")
    for s in SPLIT_ORDER:
        sub = vb[vb.split == s]
        if len(sub):
            axs[1, 2].hist(np.log10(sub.area_pct.clip(lower=1e-5)), bins=50, histtype="step", density=True, label=s)
    axs[1, 2].set_title("Box area by split (density)"); axs[1, 2].legend()
    save_fig(fig, out)


def plot_heatmap(vb, out):
    fig, ax = plt.subplots(figsize=(6, 6))
    h = ax.hist2d(vb.xc.clip(0, 1), 1 - vb.yc.clip(0, 1), bins=40, range=[[0, 1], [0, 1]])
    fig.colorbar(h[3], ax=ax); ax.set_title("Box-centre heatmap (x, y flipped)")
    save_fig(fig, out)


def plot_conditions(pix_df, img_df, out):
    p = pix_df[pix_df.hashed]
    if p.empty:
        return
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    axs[0, 0].hist(p.brightness, bins=50); axs[0, 0].set_title("Mean brightness (0-255)")
    axs[0, 1].hist(p.contrast, bins=50); axs[0, 1].set_title("Contrast (grey std)")
    axs[0, 2].hist(p.saturation, bins=50); axs[0, 2].set_title("Mean saturation")
    axs[1, 0].hist(np.log10(p.sharpness.clip(lower=1)), bins=50); axs[1, 0].set_title("Sharpness, log10(Laplacian var)")
    axs[1, 1].hist(img_df.n_valid_boxes.clip(upper=20), bins=21); axs[1, 1].set_title("Boxes per image (clipped at 20)")
    for s in SPLIT_ORDER:
        sub = p[p.split == s]
        if len(sub):
            axs[1, 2].hist(sub.brightness, bins=40, histtype="step", density=True, label=s)
    axs[1, 2].set_title("Brightness by split"); axs[1, 2].legend()
    save_fig(fig, out)


def plot_altitude_proxy(vb, out):
    med = vb.groupby(["split", "image"]).area_pct.median()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(np.log10(med.clip(lower=1e-5)), bins=60)
    ax.set_title("Altitude/zoom proxy: per-image median box area, log10(%)")
    save_fig(fig, out)


def plot_leakage(nearest, out):
    if not nearest:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, d in nearest.items():
        if len(d):
            ax.hist(np.clip(d, 0, 128), bins=65, histtype="step", density=True, label=k)
    ax.axvline(NEAR_DUP_HAMMING, color="r", ls="--", label="near-dup threshold")
    ax.set_title("Nearest-neighbour dHash distance (lower = more similar)"); ax.legend()
    save_fig(fig, out)


# --------------------------- 7. visual spot-checks -------------------------
def _palette(i):
    cols = [(255, 0, 0), (0, 200, 0), (30, 144, 255), (255, 165, 0), (200, 0, 200), (0, 200, 200), (255, 255, 0)]
    return cols[i % len(cols)]


def draw_grid(entries, out, cols=5, cell=320):
    """entries: list of (path, [(cls_id, cls_name, xc, yc, w, h)], caption)."""
    if not entries:
        return
    rows = (len(entries) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), (30, 30, 30))
    for k, (path, boxes, cap) in enumerate(entries):
        try:
            with Image.open(path) as im0:
                im = im0.convert("RGB")
            im.thumbnail((cell, cell))
            w, h = im.size
            d = ImageDraw.Draw(im)
            for cid, cname, xc, yc, bw, bh in boxes:
                x1, y1, x2, y2 = (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h
                d.rectangle([x1, y1, x2, y2], outline=_palette(cid), width=2)
                d.text((x1 + 2, max(0, y1 - 11)), str(cname), fill=_palette(cid))
            d.text((4, 4), cap, fill=(255, 255, 255))
            canvas.paste(im, ((k % cols) * cell, (k // cols) * cell))
            del im
        except Exception:
            pass
    canvas.save(out)
    del canvas
    free()


def boxes_of(box_df, split, image):
    sub = box_df[(box_df.split == split) & (box_df.image == image) & box_df.valid]
    return [(r.cls, r.cls_name, r.xc, r.yc, r.w, r.h) for r in sub.itertuples()]


def spotcheck_random(img_df, box_df, out_dir, rng):
    ok = img_df[(~img_df.corrupt) & (img_df.n_valid_boxes > 0)]
    if ok.empty:
        return
    pick = ok.sample(min(SPOTCHECK_N, len(ok)), random_state=int(rng.integers(1_000_000)))
    entries = [(r.path, boxes_of(box_df, r.split, r.image), f"{r.split}:{r.image[:28]}") for r in pick.itertuples()]
    draw_grid(entries, out_dir / "spotcheck_random.png")
    # background-only samples (check for missed objects)
    bg = img_df[(~img_df.corrupt) & (img_df.n_valid_boxes == 0)]
    if len(bg):
        pick = bg.sample(min(10, len(bg)), random_state=int(rng.integers(1_000_000)))
        draw_grid([(r.path, [], f"BG {r.split}:{r.image[:26]}") for r in pick.itertuples()],
                  out_dir / "spotcheck_background_only.png")
    # most crowded images
    crowd = img_df.sort_values("n_valid_boxes", ascending=False).head(10)
    if crowd.n_valid_boxes.max() > 1:
        draw_grid([(r.path, boxes_of(box_df, r.split, r.image), f"{r.n_valid_boxes} boxes") for r in crowd.itertuples()],
                  out_dir / "spotcheck_most_crowded.png")


def tiny_crops(img_df, box_df, out):
    vb = box_df[box_df.valid].nsmallest(TINY_CROPS_N, "area_px")
    if vb.empty:
        return
    path_of = dict(zip(zip(img_df.split, img_df.image), img_df.path))
    cell, cols = 256, 5
    rows = (len(vb) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), (30, 30, 30))
    for k, r in enumerate(vb.itertuples()):
        try:
            with Image.open(path_of[(r.split, r.image)]) as im0:
                im = im0.convert("RGB")
            W, H = im.size
            cx, cy = r.xc * W, r.yc * H
            half = max(32, 4 * max(r.w_px, r.h_px))
            crop = im.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((cell, cell), RESAMPLE)
            d = ImageDraw.Draw(crop)
            s = cell / (2 * half)
            d.rectangle([(cell / 2) - r.w_px * s / 2, (cell / 2) - r.h_px * s / 2,
                         (cell / 2) + r.w_px * s / 2, (cell / 2) + r.h_px * s / 2], outline=(255, 0, 0), width=1)
            d.text((4, 4), f"{r.cls_name} {r.w_px:.0f}x{r.h_px:.0f}px", fill=(255, 255, 0))
            canvas.paste(crop, ((k % cols) * cell, (k // cols) * cell))
            del im, crop
        except Exception:
            pass
    canvas.save(out)
    del canvas
    free()


def duplicate_montage(pair_df, out, n=6, cell=320):
    cross = pair_df[pair_df.relation.isin(["valid->train", "test->train"])] if len(pair_df) else pair_df
    if cross is None or cross.empty:
        return
    cross = cross.head(n)
    canvas = Image.new("RGB", (2 * cell, len(cross) * cell), (30, 30, 30))
    for k, r in enumerate(cross.itertuples()):
        for j, (pth, lab) in enumerate([(r.path, f"{r.relation.split('->')[0]}: {r.image[:25]}"),
                                        (r.partner_path, f"train: {r.partner[:25]}")]):
            try:
                with Image.open(pth) as im0:
                    im = im0.convert("RGB")
                im.thumbnail((cell, cell))
                ImageDraw.Draw(im).text((4, 4), f"{lab} d={r.hamming}", fill=(255, 255, 0))
                canvas.paste(im, (j * cell, k * cell))
                del im
            except Exception:
                pass
    canvas.save(out)
    del canvas
    free()


# ----------------------------- 8. reports ----------------------------------
def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def write_dataset_report(S, class_df, actions, out):
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# EDA report: {S['dataset']}\n\n")
        f.write("## Headline numbers\n\n")
        flat = pd.DataFrame({"metric": list(S.keys()), "value": list(S.values())})
        f.write(md_table(flat) + "\n\n## Per-class table\n\n" + md_table(class_df) + "\n\n")
        f.write("## Suggested cleaning / design actions (rule-based, review them)\n\n")
        for a in actions:
            f.write(f"- {a}\n")


# ------------------------- 9. per-dataset pipeline -------------------------
def run_dataset(name, root, output_dir):
    """Runs inside its own process: everything is freed by the OS when it exits."""
    out_dir = Path(output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    log(f"=== {name} ===  {root}")
    mem("start")

    splits = find_splits(root)
    if not splits:
        log("   ERROR: no split folders with an 'images' subfolder found. Check the path.")
        return
    names = load_class_names(root)
    log(f"   splits found: {list(splits)} | classes in data.yaml: {len(names)}")

    # stage 1: inventory + integrity
    img_df, box_df, issue_df, orphan = build_tables(root, splits, names)
    if not names:
        names = [f"id_{i}" for i in range(int(box_df.cls.max()) + 1)] if len(box_df) else []
        box_df["cls_name"] = box_df.cls.map(lambda c: names[c] if c < len(names) else f"id_{c}")
    box_df.to_csv(out_dir / "boxes.csv", index=False)
    img_df.drop(columns=["path"]).to_csv(out_dir / "images.csv", index=False)
    issue_df.to_csv(out_dir / "issues.csv", index=False)
    mem("after tables"); free()

    # stage 2: pixel stats + hashes
    log("   pixel pass (brightness / sharpness / hashes)")
    pix_df, hashes = pixel_pass(img_df, SEED)
    pix_df.to_csv(out_dir / "pixel_stats.csv", index=False)
    mem("after pixel pass"); free()

    # stage 3: leakage / near-duplicates
    log("   near-duplicate / leakage analysis")
    dup_res, pair_df, nearest = duplicate_analysis(img_df, pix_df, hashes)
    pair_df.drop(columns=["path", "partner_path"], errors="ignore").to_csv(out_dir / "near_duplicate_pairs.csv", index=False)
    del hashes; free()
    mem("after duplicates")

    # stage 4: summary
    vb = box_df[box_df.valid].copy()
    class_df = class_table(box_df, names, list(splits))
    class_df.to_csv(out_dir / "class_summary.csv", index=False)
    S = build_summary(name, root, names, img_df, box_df, issue_df, pix_df, dup_res, orphan, class_df)
    actions = suggest_actions(S)
    S["suggested_actions"] = actions
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(S, f, indent=2, default=lambda o: None if o != o else (o.item() if hasattr(o, "item") else str(o)))
    write_dataset_report({k: v for k, v in S.items() if k != "suggested_actions"}, class_df, actions, out_dir / "report.md")
    free()

    # stage 5: plots
    log("   plots")
    if len(vb):
        plot_class_distribution(class_df, out_dir / "01_class_distribution.png")
        plot_sizes(vb, out_dir / "02_object_sizes.png")
        plot_heatmap(vb, out_dir / "03_box_center_heatmap.png")
        plot_altitude_proxy(vb, out_dir / "05_altitude_proxy.png")
    plot_conditions(pix_df, img_df, out_dir / "04_conditions.png")
    plot_leakage(nearest, out_dir / "06_leakage_distance.png")
    mem("after plots")

    # stage 6: visual checks
    log("   visual spot-checks")
    spotcheck_random(img_df, box_df, out_dir, rng)
    tiny_crops(img_df, box_df, out_dir / "spotcheck_tiniest_boxes.png")
    duplicate_montage(pair_df, out_dir / "spotcheck_cross_split_duplicates.png")
    mem("end")

    del img_df, box_df, issue_df, pix_df, vb, class_df, pair_df, nearest
    free()
    log(f"=== {name} done -> {out_dir}")


# ---------------------- 10. cross-dataset comparison -----------------------
COMPARE_KEYS = [
    "total_images", "train_images", "valid_images", "test_images", "total_instances", "classes",
    "most_common_resolution", "unique_resolutions", "image_modes",
    "train_background_pct", "valid_background_pct", "test_background_pct", "boxes_per_image_mean",
    "imbalance_ratio_max_over_min", "n_rare_classes", "classes_missing_in_valid", "classes_missing_in_test",
    "pct_small_objects", "pct_medium_objects", "pct_large_objects", "median_box_area_pct_of_image",
    "median_box_min_side_px", "pct_boxes_min_side_lt_8px", "pct_boxes_min_side_lt_16px",
    "issue_corrupt_image", "issue_missing_label_file", "issue_malformed_line", "issue_class_id_out_of_range",
    "issue_non_positive_size", "issue_out_of_bounds", "issue_duplicate_box", "orphan_label_files",
    "files_that_are_augmented_copies", "base_names_spanning_multiple_splits",
    "valid_near_dup_of_train_pct", "test_near_dup_of_train_pct", "train_images_with_near_dup_in_train_pct",
    "brightness_mean", "brightness_std_across_images", "pct_dark_images_lt_60", "pct_bright_images_gt_190",
    "pct_grayscale_images", "pct_blurry_lapvar_lt_30",
]


def build_comparison(output_dir):
    out = Path(output_dir)
    summaries = {}
    for name in DATASETS:
        p = out / name / "summary.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                summaries[name] = json.load(f)
    if not summaries:
        log("No summaries found, skipping comparison.")
        return
    df = pd.DataFrame({n: {k: s.get(k, "") for k in COMPARE_KEYS} for n, s in summaries.items()})
    df.index.name = "metric"
    df.to_csv(out / "comparison.csv")
    with open(out / "comparison.md", "w", encoding="utf-8") as f:
        f.write("# Dataset comparison\n\n" + md_table(df.reset_index()) + "\n\n## Suggested actions per dataset\n\n")
        for n, s in summaries.items():
            f.write(f"### {n}\n")
            for a in s.get("suggested_actions", []):
                f.write(f"- {a}\n")
            f.write("\n")
    # quick visual comparison
    keys = [("total_images", "Total images"), ("total_instances", "Instances"),
            ("pct_small_objects", "% small objects (<32px)"), ("pct_boxes_min_side_lt_8px", "% boxes side <8px"),
            ("train_background_pct", "% train background-only"), ("valid_near_dup_of_train_pct", "% valid near-dup of train"),
            ("test_near_dup_of_train_pct", "% test near-dup of train"), ("pct_grayscale_images", "% grayscale"),
            ("pct_dark_images_lt_60", "% dark images")]
    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    for ax, (k, title) in zip(axs.ravel(), keys):
        vals = []
        for s in summaries.values():
            try: vals.append(float(s.get(k, 0) or 0))
            except (TypeError, ValueError): vals.append(0.0)
        ax.bar(range(len(vals)), vals, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"][:len(vals)])
        ax.set_xticks(range(len(vals))); ax.set_xticklabels([n[:10] for n in summaries], rotation=20)
        ax.set_title(title)
    save_fig(fig, out / "comparison_plots.png")
    print("\n" + df.to_string())
    log(f"Comparison saved in {out}")


# ---------------------------------- main -----------------------------------
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")                   # required on Windows; also guarantees clean memory per dataset
    for name, root in DATASETS.items():
        if not Path(root).is_dir():
            log(f"SKIPPING {name}: folder not found -> {root}")
            continue
        proc = ctx.Process(target=run_dataset, args=(name, root, OUTPUT_DIR))
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            log(f"WARNING: {name} exited with code {proc.exitcode}")
    build_comparison(OUTPUT_DIR)
    free()


if __name__ == "__main__":
    mp.freeze_support()
    main()