"""
build_extras.py
Takes the merged dataset (output of safe_merge_ds1_ds2.py) and builds the FINAL dataset:

  OUT_ROOT/
    base/                       train|valid|test  (merged positives + negatives)  <- the "clean" dataset
    brownie/
      train_aug/zoomout/        EXTRA training data (simulated higher altitude)   <- derived from base/train only
      train_aug/weather/        EXTRA training data (synthetic fog/night/rain...) <- derived from base/train only
      eval/<condition>/         stress-test copies of base/test (fog, night, rain, altitude...), same GT
    yamls/                      data_base.yaml, data_train_plus_aug.yaml, eval_<condition>.yaml
    manifest_*.csv

Everything extra lives in its own folder, so train_eval.py can report each set separately.

Negatives (image-level imbalance fix)
  * real object-free images (REAL_BG_DIR) -> valid/test/train, deduplicated against everything (flip-aware hash)
  * if train still lacks negatives: synthetic crops of box-free regions of TRAIN images (train only, labelled synthetic)

Run:  python build_extras.py     (keep it in the same folder as safe_merge_ds1_ds2.py)
"""
# ============================ CONFIG ============================
MERGED_ROOT = r"merged_fod_dataset"        # output of safe_merge_ds1_ds2.py
REAL_BG_DIR = None                                 # folder of real object-free runway images (recommended)
OUT_ROOT = r"final_dataset"
OVERWRITE = True

NEG_FRAC_TRAIN = 0.08          # target share of background-only images in train
NEG_FRAC_EVAL = 0.40           # target share in valid and test
ALLOW_SYNTHETIC_TRAIN_NEG = True
ALLOW_SYNTHETIC_EVAL_NEG = True   # True: if real negatives are short, fill valid/test with box-free crops of images
                               # from the SAME split (flagged 'synthetic_crop' in manifest_negatives.csv)
BG_GAP = 3                     # images skipped between test/valid/train blocks of real backgrounds
NEAR_DUP_HAMMING = 10

EVAL_SOURCE_SPLIT = "test"     # stress-test variants are built from this split
ZOOM_SCALES = [1.5, 0.75, 0.5, 0.35]   # >1 = lower altitude (zoom in), <1 = higher altitude (zoom out)
TRAIN_AUG_FRACTION = 0.35      # share of train positives that get a zoom-out copy / a weather copy
SEED = 42
# ================================================================

import gc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFilter

from ds1_ds2_merge import hash_image, min_dist_flip_aware    # same flip-aware hash as the merge step

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
SPLITS = ["train", "valid", "test"]


def log(m):
    print(m, flush=True)


# ------------------------------ io helpers ---------------------------------
def list_imgs(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXTS) if Path(d).is_dir() else []


def read_boxes(lp):
    out = []
    if Path(lp).exists():
        for line in Path(lp).read_text().splitlines():
            v = line.split()
            if len(v) == 5:
                out.append(tuple(map(float, v[1:])))
    return out


def write_boxes(lp, boxes):
    Path(lp).write_text("\n".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in boxes))


def mkdirs(root):
    (Path(root) / "images").mkdir(parents=True, exist_ok=True)
    (Path(root) / "labels").mkdir(parents=True, exist_ok=True)


def save_variant(img, boxes, root, stem):
    img.save(Path(root) / "images" / f"{stem}.jpg", quality=95)
    write_boxes(Path(root) / "labels" / f"{stem}.txt", boxes)


# ------------------------- synthetic condition models ----------------------
def _arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def fog(img, rng, s):
    a = _arr(img); H, W, _ = a.shape
    beta = 0.6 + 2.4 * s
    depth = np.linspace(1.0, 0.25, H, dtype=np.float32)[:, None]            # top = far
    low = np.asarray(Image.fromarray((rng.random((6, 6)) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC),
                     dtype=np.float32) / 255
    t = np.exp(-beta * np.clip(depth + 0.25 * (low - 0.5), 0.05, 1.5))[..., None]
    return _img(a * t + 235.0 * (1 - t))


def low_light(img, rng, s):
    a = _arr(img) / 255.0
    a = (a ** (1 + 0.7 * s)) * (1 - 0.85 * s)
    a = a * (1 - 0.1 * s * np.array([1.0, 0.5, -1.0], dtype=np.float32))     # cool tint
    a = a * 255 + rng.normal(0, 3 + 9 * s, a.shape)                          # sensor noise
    return _img(a)


def rain(img, rng, s):
    base = img.convert("RGB"); W, H = base.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    for _ in range(int(300 + 1500 * s)):
        x, y, L = rng.integers(0, W), rng.integers(0, H), rng.integers(12, 32)
        d.line([(x, y), (x - L * 0.25, y + L)], fill=(205, 205, 215, int(70 + 60 * s)), width=1)
    out = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB").filter(ImageFilter.GaussianBlur(0.5 + s))
    return _img(_arr(out) * (1 - 0.2 * s))


def glare(img, rng, s):
    a = _arr(img); H, W, _ = a.shape
    cy, cx = rng.uniform(0.1, 0.9) * H, rng.uniform(0.1, 0.9) * W
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.25 * W) ** 2)))[..., None]
    return _img(a * (1 + 0.6 * s) + 200 * s * blob)


def shadow(img, rng, s):
    a = _arr(img); H, W, _ = a.shape
    pts = [(rng.uniform(-0.2, 1.2) * W, rng.uniform(-0.2, 1.2) * H) for _ in range(rng.integers(4, 7))]
    m = Image.new("L", (W, H), 0); ImageDraw.Draw(m).polygon(pts, fill=255)
    m = np.asarray(m.filter(ImageFilter.GaussianBlur(4)), dtype=np.float32)[..., None] / 255
    return _img(a * (1 - 0.65 * s * m))


def motion_blur(img, rng, s):
    a = _arr(img); k = int(5 + 20 * s) | 1; axis = int(rng.integers(0, 2))
    acc = np.zeros_like(a)
    for sh in range(-(k // 2), k // 2 + 1):
        acc += np.roll(a, sh, axis=axis)
    return _img(acc / k)


def sensor_noise(img, rng, s):
    a = _arr(img)
    return _img(a + rng.normal(0, 5 + 25 * s, a.shape))


EVAL_CONDITIONS = {                      # fixed strengths -> reproducible stress tests
    "fog_light": (fog, 0.3), "fog_medium": (fog, 0.6), "fog_heavy": (fog, 0.9),
    "dusk": (low_light, 0.4), "night": (low_light, 0.85),
    "rain": (rain, 0.6), "glare": (glare, 0.7), "shadow": (shadow, 0.7),
    "motion_blur": (motion_blur, 0.6), "sensor_noise": (sensor_noise, 0.6),
}
TRAIN_FUNCS = [fog, low_light, rain, glare, shadow, motion_blur, sensor_noise]   # random strength in train


# ------------------------------ altitude models ----------------------------
def zoom_out(img, boxes, s):
    """Simulate higher altitude: shrink and paste on a blurred copy of the same image."""
    W, H = img.size
    sw, sh = max(8, int(W * s)), max(8, int(H * s))
    canvas = img.convert("RGB").filter(ImageFilter.GaussianBlur(25))
    ox, oy = (W - sw) // 2, (H - sh) // 2
    canvas.paste(img.convert("RGB").resize((sw, sh), RESAMPLE), (ox, oy))
    nb = [((ox + x * sw) / W, (oy + y * sh) / H, w * sw / W, h * sh / H) for x, y, w, h in boxes]
    return canvas, nb


def zoom_in(img, boxes, s):
    """Simulate lower altitude: crop a 1/s window (around a box) and upscale. None if a box is cut ambiguously."""
    W, H = img.size
    cw, ch = W / s, H / s
    cx, cy = (boxes[0][0] * W, boxes[0][1] * H) if boxes else (W / 2, H / 2)
    x0, y0 = min(max(cx - cw / 2, 0), W - cw), min(max(cy - ch / 2, 0), H - ch)
    nb = []
    for x, y, w, h in boxes:
        bx1, by1, bx2, by2 = (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H
        ix = max(0.0, min(bx2, x0 + cw) - max(bx1, x0)); iy = max(0.0, min(by2, y0 + ch) - max(by1, y0))
        frac = ix * iy / max((bx2 - bx1) * (by2 - by1), 1e-9)
        if frac >= 0.95:
            nb.append(((((bx1 + bx2) / 2) - x0) / cw, (((by1 + by2) / 2) - y0) / ch, (bx2 - bx1) / cw, (by2 - by1) / ch))
        elif frac > 0.02:
            return None
    if boxes and not nb:
        return None
    out = img.convert("RGB").crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), RESAMPLE)
    return out, nb


# ------------------------------ negatives ----------------------------------
def count_split(root, split):
    imgs = list_imgs(Path(root) / split / "images")
    pos = sum(1 for p in imgs if read_boxes(Path(root) / split / "labels" / (p.stem + ".txt")))
    return len(imgs), pos


def needed_bg(pos, existing_bg, frac):
    target = int(np.ceil(pos * frac / (1 - frac)))
    return max(0, target - existing_bg)


def prepare_real_backgrounds(base_root, bg_dir):
    """Hash, dedupe among themselves and against every base image. Returns list of kept Paths (name order)."""
    files = sorted(p for p in Path(bg_dir).rglob("*") if p.suffix.lower() in IMG_EXTS)
    log(f"   real backgrounds found: {len(files)}")
    base_files = [p for s in SPLITS for p in list_imgs(Path(base_root) / s / "images")]
    Vb = np.zeros((len(base_files), 4, 32), np.uint8); okb = np.zeros(len(base_files), bool)
    for i, p in enumerate(base_files):
        try:
            Vb[i] = hash_image(p)[0]; okb[i] = True
        except Exception:
            pass
    Vb = Vb[okb]
    kept, Vk, m = [], np.zeros((len(files), 4, 32), np.uint8), 0
    drop_self = drop_base = bad = 0
    for p in files:
        try:
            v = hash_image(p)[0]
        except Exception:
            bad += 1; continue
        if m and min_dist_flip_aware(v[0][None], Vk[:m])[0][0] <= NEAR_DUP_HAMMING:
            drop_self += 1; continue
        if len(Vb) and min_dist_flip_aware(v[0][None], Vb)[0][0] <= NEAR_DUP_HAMMING:
            drop_base += 1; continue
        Vk[m] = v; m += 1; kept.append(p)
    log(f"   kept {len(kept)} | dropped: corrupt {bad}, near-dup of other backgrounds {drop_self}, "
        f"near-dup of dataset images {drop_base}")
    return kept


def synthetic_negatives(split_root, split, n, rng, manifest):
    """Box-free crops of positive images of the SAME split, saved back into that split."""
    split_root = Path(split_root)
    out_img, out_lbl = split_root / "images", split_root / "labels"
    imgs = [p for p in list_imgs(out_img) if read_boxes(out_lbl / (p.stem + ".txt"))]
    made, tries = 0, 0
    while made < n and tries < n * 30 and imgs:
        tries += 1
        p = imgs[int(rng.integers(len(imgs)))]
        boxes = read_boxes(out_lbl / (p.stem + ".txt"))
        with Image.open(p) as im0:
            im = im0.convert("RGB")
        W, H = im.size
        c = int(rng.uniform(0.4, 0.6) * min(W, H))
        x0, y0 = int(rng.integers(0, W - c + 1)), int(rng.integers(0, H - c + 1))
        mg = 0.03
        clash = any(not ((x + w / 2 + mg) * W < x0 or (x - w / 2 - mg) * W > x0 + c or
                         (y + h / 2 + mg) * H < y0 or (y - h / 2 - mg) * H > y0 + c) for x, y, w, h in boxes)
        if clash:
            continue
        name = f"synneg_{split}__{made:04d}__{p.stem}"
        im.crop((x0, y0, x0 + c, y0 + c)).resize((W, H), RESAMPLE).save(out_img / f"{name}.jpg", quality=95)
        (out_lbl / f"{name}.txt").write_text("")
        manifest.append((f"{name}.jpg", split, "synthetic_crop", p.name))
        made += 1
    return made


def build_base(merged, out_base, rng):
    manifest = []
    for s in SPLITS:
        mkdirs(out_base / s)
        for p in list_imgs(Path(merged) / s / "images"):
            shutil.copy2(p, out_base / s / "images" / p.name)
            lp = Path(merged) / s / "labels" / (p.stem + ".txt")
            shutil.copy2(lp, out_base / s / "labels" / lp.name) if lp.exists() else (out_base / s / "labels" / lp.name).write_text("")
    stat = {s: count_split(out_base, s) for s in SPLITS}
    need = {}
    for s in SPLITS:
        n, pos = stat[s]
        need[s] = needed_bg(pos, n - pos, NEG_FRAC_TRAIN if s == "train" else NEG_FRAC_EVAL)
    log("negatives needed (extra images): " + str(need))
    got = {s: 0 for s in SPLITS}
    if REAL_BG_DIR and Path(REAL_BG_DIR).is_dir():
        kept = prepare_real_backgrounds(out_base, REAL_BG_DIR)
        avail = max(0, len(kept) - 2 * BG_GAP)
        t = min(need["test"], avail // 2); v = min(need["valid"], avail - t); tr = min(need["train"], max(0, avail - t - v))
        blocks, pos_ = {}, 0
        for s, k in (("test", t), ("valid", v), ("train", tr)):
            blocks[s] = kept[pos_:pos_ + k]; pos_ += k + BG_GAP
        for s, files in blocks.items():
            for p in files:
                nm = f"bgreal__{p.stem}.jpg"
                with Image.open(p) as im0:
                    im0.convert("RGB").save(out_base / s / "images" / nm, quality=95)
                (out_base / s / "labels" / f"bgreal__{p.stem}.txt").write_text("")
                manifest.append((nm, s, "real_background", p.name))
            got[s] = len(files)
        log(f"   real backgrounds placed: {got}")
    else:
        log("   REAL_BG_DIR not set: no real negatives added (valid/test will have none!)")
    for s_, allowed in (("train", ALLOW_SYNTHETIC_TRAIN_NEG), ("valid", ALLOW_SYNTHETIC_EVAL_NEG), ("test", ALLOW_SYNTHETIC_EVAL_NEG)):
        short = need[s_] - got[s_]
        if short <= 0:
            continue
        if allowed:
            n = synthetic_negatives(out_base / s_, s_, short, rng, manifest)
            log(f"   synthetic negatives created in {s_}: {n} of {short} needed (box-free crops, flagged in manifest)")
        else:
            log(f"   WARNING: {s_} is still short by {short} negatives and synthetic ones are disabled for it.")
    pd.DataFrame(manifest, columns=["file", "split", "kind", "origin"]).to_csv(out_base.parent / "manifest_negatives.csv", index=False)


# ------------------------------ brownie sets -------------------------------
def build_eval_sets(base, brownie, rng):
    src_dir = Path(base) / EVAL_SOURCE_SPLIT
    files = list_imgs(src_dir / "images")
    rows = []
    # weather / lighting (geometry unchanged -> GT copied as is)
    for cname, (fn, s) in EVAL_CONDITIONS.items():
        root = brownie / "eval" / cname; mkdirs(root)
        r = np.random.default_rng(SEED)                      # fixed seed per condition => reproducible
        for p in files:
            boxes = read_boxes(src_dir / "labels" / (p.stem + ".txt"))
            with Image.open(p) as im0:
                out = fn(im0.convert("RGB"), r, s)
            save_variant(out, boxes, root, p.stem)
        rows.append((cname, "weather_lighting", s, len(files)))
        log(f"   eval/{cname}: {len(files)} images")
    # altitude
    for sc in ZOOM_SCALES:
        cname = f"alt_zoom_{sc:g}x"
        root = brownie / "eval" / cname; mkdirs(root)
        n_ok = n_skip = 0
        for p in files:
            boxes = read_boxes(src_dir / "labels" / (p.stem + ".txt"))
            with Image.open(p) as im0:
                im = im0.convert("RGB")
            res = zoom_out(im, boxes, sc) if sc < 1 else (zoom_in(im, boxes, sc) if sc > 1 else (im, boxes))
            if res is None:
                n_skip += 1; continue
            save_variant(res[0], res[1], root, p.stem); n_ok += 1
        rows.append((cname, "altitude", sc, n_ok))
        log(f"   eval/{cname}: {n_ok} images (skipped {n_skip} with ambiguous cut boxes)")
    pd.DataFrame(rows, columns=["condition", "type", "strength_or_scale", "images"]).to_csv(
        brownie.parent / "manifest_eval_variants.csv", index=False)
    gc.collect()


def build_train_aug(base, brownie, rng):
    tdir = Path(base) / "train"
    pos = [p for p in list_imgs(tdir / "images") if read_boxes(tdir / "labels" / (p.stem + ".txt"))]
    zroot, wroot = brownie / "train_aug" / "zoomout", brownie / "train_aug" / "weather"
    mkdirs(zroot); mkdirs(wroot)
    n_aug = max(1, int(TRAIN_AUG_FRACTION * len(pos)))
    for p in [pos[i] for i in rng.choice(len(pos), min(n_aug, len(pos)), replace=False)]:
        boxes = read_boxes(tdir / "labels" / (p.stem + ".txt"))
        with Image.open(p) as im0:
            out, nb = zoom_out(im0.convert("RGB"), boxes, float(rng.uniform(0.4, 0.7)))
        save_variant(out, nb, zroot, f"zo__{p.stem}")
    for p in [pos[i] for i in rng.choice(len(pos), min(n_aug, len(pos)), replace=False)]:
        boxes = read_boxes(tdir / "labels" / (p.stem + ".txt"))
        fn = TRAIN_FUNCS[int(rng.integers(len(TRAIN_FUNCS)))]       # random condition + random strength
        with Image.open(p) as im0:
            out = fn(im0.convert("RGB"), rng, float(rng.uniform(0.2, 0.9)))
        save_variant(out, boxes, wroot, f"wx__{fn.__name__}__{p.stem}")
    log(f"   train_aug: zoomout={len(list_imgs(zroot / 'images'))}, weather={len(list_imgs(wroot / 'images'))}")


# ------------------------------ yamls + report -----------------------------
def write_yamls(out_root):
    out_root = Path(out_root); base = (out_root / "base"); br = out_root / "brownie"
    yd = out_root / "yamls"; yd.mkdir(exist_ok=True)
    P = lambda p: Path(p).resolve().as_posix()
    def dump(name, d):
        with open(yd / name, "w", encoding="utf-8") as f:
            yaml.safe_dump({**d, "nc": 1, "names": ["FOD"]}, f, sort_keys=False)
    common = {"train": P(base / "train/images"), "val": P(base / "valid/images"), "test": P(base / "test/images")}
    dump("data_base.yaml", common)
    extra = [P(base / "train/images")] + [P(d / "images") for d in (br / "train_aug").iterdir() if d.is_dir()]
    dump("data_train_plus_aug.yaml", {**common, "train": extra})
    dump("eval_clean_test.yaml", {**common, "val": P(base / "test/images")})
    dump("eval_clean_valid.yaml", common)
    for d in sorted((br / "eval").iterdir()):
        if d.is_dir():
            dump(f"eval_{d.name}.yaml", {**common, "val": P(d / "images"), "test": P(d / "images")})


def final_report(out_root):
    base = Path(out_root) / "base"
    rows = []
    for s in SPLITS:
        n, pos = count_split(base, s)
        inst = sum(len(read_boxes(base / s / "labels" / (p.stem + ".txt"))) for p in list_imgs(base / s / "images"))
        rows.append({"set": f"base/{s}", "images": n, "with_debris": pos, "background_only": n - pos,
                     "background_pct": round(100 * (n - pos) / max(n, 1), 1), "instances": inst})
    for grp in ("train_aug", "eval"):
        for d in sorted((Path(out_root) / "brownie" / grp).iterdir()):
            imgs = list_imgs(d / "images")
            inst = sum(len(read_boxes(d / "labels" / (p.stem + ".txt"))) for p in imgs)
            nb = sum(1 for p in imgs if not read_boxes(d / "labels" / (p.stem + ".txt")))
            rows.append({"set": f"brownie/{grp}/{d.name}", "images": len(imgs), "with_debris": len(imgs) - nb,
                         "background_only": nb, "background_pct": round(100 * nb / max(len(imgs), 1), 1), "instances": inst})
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_root) / "final_dataset_summary.csv", index=False)
    print("\n" + "=" * 78 + "\nFINAL DATASET SUMMARY (every extra set is separate)\n" + "=" * 78)
    print(df.to_string(index=False))

    # leakage re-check of base after adding negatives
    V = {}
    for s in SPLITS:
        files = list_imgs(base / s / "images"); arr = np.zeros((len(files), 4, 32), np.uint8); ok = np.zeros(len(files), bool)
        for i, p in enumerate(files):
            try:
                arr[i] = hash_image(p)[0]; ok[i] = True
            except Exception:
                pass
        V[s] = arr[ok]
    print("\nLEAKAGE RE-CHECK on base (flip-aware, <=%d/256):" % NEAR_DUP_HAMMING)
    allok = True
    for a, b in (("valid", "train"), ("test", "train"), ("test", "valid")):
        d, _ = min_dist_flip_aware(V[a][:, 0, :], V[b])
        n = int((d <= NEAR_DUP_HAMMING).sum()); allok &= n == 0
        print(f"   {a} vs {b}: {n} near-duplicates (min distance {int(d.min()) if len(d) else 'n/a'})")
    print("   RESULT:", "PASS" if allok else "FAIL - remove the offending images")
    print(f"\nYAMLs for training/evaluation: {Path(out_root) / 'yamls'}")


def main():
    out = Path(OUT_ROOT)
    if out.exists() and any(out.iterdir()):
        if OVERWRITE:
            shutil.rmtree(out)
        else:
            raise SystemExit(f"OUT_ROOT exists and is not empty: {out} (set OVERWRITE=True or use a new folder)")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    base, brownie = out / "base", out / "brownie"
    log("1/4 base dataset + negatives"); build_base(MERGED_ROOT, base, rng)
    log("2/4 eval stress-test sets (from %s)" % EVAL_SOURCE_SPLIT); build_eval_sets(base, brownie, rng)
    log("3/4 train-only augmentation sets"); build_train_aug(base, brownie, rng)
    log("4/4 yamls + report"); write_yamls(out); final_report(out)
    with open(base / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": base.resolve().as_posix(), "train": "train/images", "val": "valid/images",
                        "test": "test/images", "nc": 1, "names": ["FOD"]}, f, sort_keys=False)


if __name__ == "__main__":
    main()