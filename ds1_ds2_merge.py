"""
safe_merge_ds1_ds2.py
Builds ONE leakage-free single-class ("FOD") YOLO dataset from ds1 + ds2 (+ optional background images).

Pipeline
  1. Scan ds1 (train only by default) and ds2 (all splits). ds1 valid/test are NOT used.
  2. Clean: corrupt images, invalid / out-of-bounds / duplicate boxes, polygon->bbox, all classes -> 0 "FOD".
  3. Hash every image (256-bit dHash + horizontal/vertical/both flips).
  4. Cluster duplicates ACROSS and WITHIN datasets (union-find on hash distance + same Roboflow base name).
  5. Split by CLUSTER (so a scene can never sit in two splits). valid/test are filled ONLY from ds2 (or
     optional background images); ds1 images that fall in an eval cluster are dropped, never moved.
  6. Thin train duplicates (keep up to MAX_PER_GROUP_TRAIN per cluster), one image per cluster in valid/test.
  7. Write the new dataset + manifest.csv + dropped.csv.
  8. FINAL independent leakage check on the files actually written (flip-aware hash, exact-byte hash,
     base-name overlap) -> PASS / FAIL, then print full dataset stats.

Run:  python safe_merge_ds1_ds2.py          (needs numpy pandas pillow pyyaml psutil[optional])
"""
# ============================ CONFIG ============================
DS1_ROOT = r"Runway FOD Detector.v1i.yolov11"
DS2_ROOT = r"FOD Runway Dataset.v2i.yolov11"
EXTRA_BG_DIR = None            # optional: folder of object-free runway images, e.g. r"C:\path\to\backgrounds"
OUT_ROOT = r"merged_fod_dataset"
OVERWRITE = False              # True = delete OUT_ROOT if it already exists

DS1_SPLITS_USED = ["train"]    # ds1 valid/test are ignored on purpose
NEAR_DUP_HAMMING = 10          # out of 256 bits
VALID_FRAC = 0.15              # share of eval-eligible clusters sent to valid
TEST_FRAC = 0.15               # share of eval-eligible clusters sent to test
MAX_PER_GROUP_TRAIN = 2        # images kept per duplicate-cluster in train (None = keep all)
MAX_PER_GROUP_EVAL = 1         # images kept per duplicate-cluster in valid/test
DUP_BOX_IOU = 0.90
SMALL_AREA_PX = 32 ** 2
MEDIUM_AREA_PX = 96 ** 2
SEED = 42
# ================================================================

import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

try:
    import psutil
except ImportError:
    psutil = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": ["train"], "valid": ["valid", "val", "validation"], "test": ["test"]}
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def mem(tag):
    if psutil:
        log(f"   [mem] {tag}: {psutil.Process().memory_info().rss / 1024 ** 2:,.0f} MB")


# ------------------------------ hashing ------------------------------------
def hash_image(path):
    """Returns (variants[4,32] uint8, W, H, brightness). Variants = orig, hflip, vflip, hvflip."""
    with Image.open(path) as im0:
        W, H = im0.size
        im0.draft("RGB", (512, 512))
        im1 = im0.convert("RGB")
    im1.thumbnail((256, 256))
    bright = float(np.asarray(im1.convert("L"), dtype=np.float32).mean())
    g = np.asarray(im1.convert("L").resize((17, 16), RESAMPLE), dtype=np.int16)
    var = np.stack([np.packbits((a[:, 1:] > a[:, :-1]).flatten())
                    for a in (g, g[:, ::-1], g[::-1, :], g[::-1, ::-1])])
    del im1, g
    return var, W, H, bright


def popcount(x64):
    """x64: (..., 4) uint64 -> (...) int32 bit counts."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x64).sum(axis=-1, dtype=np.int32)
    return POPCOUNT[x64.view(np.uint8)].sum(axis=-1, dtype=np.int32)


def to64(a):
    return np.ascontiguousarray(a).view(np.uint64)


def chunk_size(m_cols):
    return max(1, int(6_000_000 // (m_cols * 4)))


def min_dist_flip_aware(Aorig, Bvar):
    """Aorig: (n,32) uint8. Bvar: (m,4,32) uint8. Returns min distance (over j and flips) and argmin j."""
    n, m = len(Aorig), len(Bvar)
    dmin = np.full(n, 999, dtype=np.int32)
    arg = np.full(n, -1, dtype=np.int32)
    if n == 0 or m == 0:
        return dmin, arg
    A = to64(Aorig)                              # (n,4)
    B = to64(Bvar.reshape(m * 4, 32))            # (4m,4)
    c = chunk_size(m * 4)
    for s in range(0, n, c):
        d = popcount(A[s:s + c, None, :] ^ B[None, :, :])    # (c,4m)
        j = d.argmin(axis=1)
        dmin[s:s + c] = d[np.arange(len(d)), j]
        arg[s:s + c] = j // 4
        del d
    gc.collect()
    return dmin, arg


# ------------------------------ labels -------------------------------------
def mark_dups(b, thr):
    n = len(b)
    if n < 2 or n > 2000:
        return np.zeros(n, bool)
    x1, x2 = b[:, 0] - b[:, 2] / 2, b[:, 0] + b[:, 2] / 2
    y1, y2 = b[:, 1] - b[:, 3] / 2, b[:, 1] + b[:, 3] / 2
    area = b[:, 2] * b[:, 3]
    iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
    inter = iw * ih
    iou = inter / np.maximum(area[:, None] + area[None] - inter, 1e-12)
    return np.triu(iou > thr, 1).any(axis=0)


def load_clean_boxes(lp):
    """Returns (boxes[(xc,yc,w,h)], status, n_raw_lines, n_dup_removed, n_dropped_invalid).
    All classes collapse to a single class; boxes are clipped to the image."""
    if not Path(lp).exists():
        return [], "no_label_file", 0, 0, 0
    text = Path(lp).read_text(encoding="utf-8", errors="replace")
    raw, invalid, out = 0, 0, []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        raw += 1
        try:
            vals = [float(v) for v in parts[1:]]
            int(float(parts[0]))
        except ValueError:
            invalid += 1
            continue
        if len(vals) == 4:
            xc, yc, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            xs, ys = vals[0::2], vals[1::2]
            xc, yc, w, h = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
        else:
            invalid += 1
            continue
        x1, x2 = max(0.0, xc - w / 2), min(1.0, xc + w / 2)
        y1, y2 = max(0.0, yc - h / 2), min(1.0, yc + h / 2)
        if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
            invalid += 1
            continue
        out.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    n_dup = 0
    if len(out) > 1:
        dm = mark_dups(np.array(out), DUP_BOX_IOU)
        n_dup = int(dm.sum())
        out = [b for b, d in zip(out, dm) if not d]
    return out, "ok", raw, n_dup, invalid


# ------------------------------ scanning -----------------------------------
def scan_dataset(src, root, splits):
    rows = []
    for canon in splits:
        for a in SPLIT_ALIASES[canon]:
            d = Path(root) / a / "images"
            if d.is_dir():
                for p in sorted(d.iterdir()):
                    if p.suffix.lower() in IMG_EXTS:
                        rows.append((src, p.name, str(p), str(Path(root) / a / "labels" / (p.stem + ".txt")),
                                     canon, p.stem.split(".rf.")[0]))
                break
    return rows


def scan_bg(folder):
    rows = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            rows.append(("bg", p.name, str(p), "", "bg", f"{p.parent.name}/{p.stem}"))
    return rows


# ------------------------------ clustering ---------------------------------
class UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster_images(V, src, base, thr):
    """V: (n,4,32). Returns gid array (cluster id per image)."""
    n = len(V)
    uf = UnionFind(n)
    first = {}
    for i in range(n):                       # same source + same Roboflow base name = same scene
        k = (src[i], base[i])
        if k in first:
            uf.union(first[k], i)
        else:
            first[k] = i
    A = to64(V[:, 0, :])
    B = to64(V.reshape(n * 4, 32))
    c = chunk_size(n * 4)
    n_edges = 0
    for s in range(0, n, c):
        d = popcount(A[s:s + c, None, :] ^ B[None, :, :])
        r, col = np.nonzero(d <= thr)
        i, j = s + r, col // 4
        m = i != j
        if m.any():
            pairs = np.unique(i[m].astype(np.int64) * n + j[m])
            n_edges += len(pairs)
            for key in pairs:
                uf.union(int(key // n), int(key % n))
        del d
        if (s // c) % 50 == 0:
            log(f"      clustering {min(s + c, n)}/{n}")
    gc.collect()
    roots = np.array([uf.find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    log(f"   duplicate edges found: {n_edges:,}")
    return gid


def assign_split(df, gid, rng):
    """Cluster-level stratified split. Only clusters containing ds2/bg images can go to valid/test."""
    t = df[["src", "has_small", "bright"]].copy()
    t["gid"] = gid
    g = t.groupby("gid").agg(
        n=("src", "size"),
        eligible=("src", lambda s: bool(s.isin(["ds2", "bg"]).any())),
        has_bg=("src", lambda s: bool((s == "bg").any())),
        has_small=("has_small", "max"),
        bright=("bright", "mean"))
    g["bbin"] = 0
    el = g.eligible
    if el.sum() > 3:
        g.loc[el, "bbin"] = pd.qcut(g.loc[el, "bright"].rank(method="first"), 3, labels=False).astype(int)
    split = {gid_: "train" for gid_ in g.index}
    for _, sub in g[el].groupby(["has_small", "bbin", "has_bg"]):
        ids = list(sub.index)
        rng.shuffle(ids)
        nv, nt = int(round(VALID_FRAC * len(ids))), int(round(TEST_FRAC * len(ids)))
        for k, gid_ in enumerate(ids):
            split[gid_] = "valid" if k < nv else ("test" if k < nv + nt else "train")
    return split, g


# ------------------------------ final checks -------------------------------
def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def final_leakage_check(out_root, manifest):
    log("FINAL LEAKAGE CHECK (re-reading the files actually written)")
    paths, V = {}, {}
    for s in ("train", "valid", "test"):
        files = sorted((Path(out_root) / s / "images").iterdir())
        paths[s] = files
        arr = np.zeros((len(files), 4, 32), dtype=np.uint8)
        ok = np.zeros(len(files), bool)
        for i, p in enumerate(files):
            try:
                arr[i] = hash_image(p)[0]
                ok[i] = True
            except Exception:
                pass
        V[s] = arr[ok]
        gc.collect()
    res = {}
    thr = NEAR_DUP_HAMMING
    for a, b in (("valid", "train"), ("test", "train"), ("test", "valid")):
        d, _ = min_dist_flip_aware(V[a][:, 0, :], V[b])
        res[f"{a}_vs_{b}_near_dup_le_{thr}"] = int((d <= thr).sum())
        res[f"{a}_vs_{b}_borderline_le_{2 * thr}"] = int((d <= 2 * thr).sum())
        res[f"{a}_vs_{b}_min_distance"] = int(d.min()) if len(d) else None
    # exact bytes
    md5 = {s: {md5_of(p) for p in paths[s]} for s in paths}
    res["exact_byte_dups_valid_train"] = len(md5["valid"] & md5["train"])
    res["exact_byte_dups_test_train"] = len(md5["test"] & md5["train"])
    res["exact_byte_dups_test_valid"] = len(md5["test"] & md5["valid"])
    # same source image (base name) in several splits
    key = manifest.assign(k=manifest.src + "|" + manifest.base)
    bs = key.groupby("k").new_split.nunique()
    res["base_names_in_multiple_splits"] = int((bs > 1).sum())
    critical = [v for k, v in res.items() if ("near_dup_le" in k or k.startswith("exact") or k.startswith("base_names"))]
    res["PASS"] = bool(all(v == 0 for v in critical))
    del V, md5
    gc.collect()
    return res


def dataset_stats(out_root, manifest):
    rows = []
    for s in ("train", "valid", "test"):
        m = manifest[manifest.new_split == s]
        nb, small, med, large, min_side, area_pct, widths = [], 0, 0, 0, [], [], []
        for r in m.itertuples():
            lp = Path(out_root) / s / "labels" / (Path(r.new_name).stem + ".txt")
            lines = [l.split() for l in lp.read_text().splitlines() if l.strip()]
            nb.append(len(lines))
            for v in lines:
                w, h = float(v[3]) * r.W, float(v[4]) * r.H
                a = w * h
                small += a < SMALL_AREA_PX
                med += SMALL_AREA_PX <= a < MEDIUM_AREA_PX
                large += a >= MEDIUM_AREA_PX
                min_side.append(min(w, h))
                area_pct.append(float(v[3]) * float(v[4]) * 100)
        nb = np.array(nb) if nb else np.array([0])
        n_inst = int(nb.sum())
        ms = np.array(min_side) if min_side else np.array([np.nan])
        rows.append({
            "split": s, "images": len(m), "instances": n_inst,
            "ds2_images": int((m.src == "ds2").sum()), "ds1_images": int((m.src == "ds1").sum()),
            "bg_images": int((m.src == "bg").sum()),
            "background_only_pct": round(100 * float((nb == 0).mean()), 2),
            "boxes_per_image": round(float(nb.mean()), 2),
            "small_pct": round(100 * small / max(n_inst, 1), 1), "medium_pct": round(100 * med / max(n_inst, 1), 1),
            "large_pct": round(100 * large / max(n_inst, 1), 1),
            "min_side_lt_8px_pct": round(100 * float(np.nanmean(ms < 8)), 1),
            "min_side_lt_16px_pct": round(100 * float(np.nanmean(ms < 16)), 1),
            "median_box_area_pct": round(float(np.median(area_pct)), 3) if area_pct else None,
            "brightness_mean": round(float(m.bright.mean()), 1) if len(m) else None,
            "brightness_std": round(float(m.bright.std()), 1) if len(m) > 1 else None,
            "pct_dark_lt_60": round(100 * float((m.bright < 60).mean()), 1) if len(m) else None,
            "pct_bright_gt_190": round(100 * float((m.bright > 190).mean()), 1) if len(m) else None,
        })
    return pd.DataFrame(rows).set_index("split")


# --------------------------------- main ------------------------------------
def main():
    out = Path(OUT_ROOT)
    if out.exists() and any(out.iterdir()):
        if OVERWRITE:
            shutil.rmtree(out)
        else:
            raise SystemExit(f"OUT_ROOT exists and is not empty: {out}  (set OVERWRITE=True or choose a new folder)")
    rng = np.random.default_rng(SEED)

    # 1. scan
    rows = scan_dataset("ds1", DS1_ROOT, DS1_SPLITS_USED) + scan_dataset("ds2", DS2_ROOT, ["train", "valid", "test"])
    if EXTRA_BG_DIR:
        rows += scan_bg(EXTRA_BG_DIR)
    df = pd.DataFrame(rows, columns=["src", "name", "path", "label_path", "orig_split", "base"])
    log("scanned: " + ", ".join(f"{k}={v}" for k, v in df.src.value_counts().items()))
    dropped = []

    # 2. hash + read size/brightness one image at a time
    n = len(df)
    V = np.zeros((n, 4, 32), dtype=np.uint8)
    W = np.zeros(n, int); H = np.zeros(n, int); bright = np.full(n, np.nan)
    ok = np.zeros(n, bool)
    for i, p in enumerate(df.path.values):
        if i and i % 1000 == 0:
            log(f"   hashing {i}/{n}")
        try:
            V[i], W[i], H[i], bright[i] = hash_image(p)
            ok[i] = True
        except Exception:
            dropped.append((df.src[i], df.name[i], "corrupt_image"))
    df["W"], df["H"], df["bright"], df["ok"] = W, H, bright, ok
    mem("after hashing")

    # 3. clean labels
    boxes = [[] for _ in range(n)]
    keep = ok.copy()
    stats = dict(dup_boxes_removed=0, invalid_lines_dropped=0, no_label_file=0, images_all_boxes_invalid=0, polygons=0)
    for i in np.where(ok)[0]:
        if df.src[i] == "bg":
            continue
        b, status, raw, ndup, ninv = load_clean_boxes(df.label_path[i])
        stats["dup_boxes_removed"] += ndup
        stats["invalid_lines_dropped"] += ninv
        stats["no_label_file"] += status == "no_label_file"
        if raw > 0 and not b:
            keep[i] = False
            stats["images_all_boxes_invalid"] += 1
            dropped.append((df.src[i], df.name[i], "all_boxes_invalid"))
            continue
        boxes[i] = b
    df["n_boxes"] = [len(b) for b in boxes]
    df["has_small"] = [any(bw * W[i] * bh * H[i] < SMALL_AREA_PX for (_, _, bw, bh) in boxes[i]) for i in range(n)]
    log(f"cleaning: {stats}")

    # 4. cluster duplicates (within + across datasets, flip-aware)
    act = np.where(keep)[0]
    log(f"clustering {len(act)} images (threshold {NEAR_DUP_HAMMING}/256, flips included)")
    gid_act = cluster_images(V[act], df.src.values[act], df.base.values[act], NEAR_DUP_HAMMING)
    sizes = pd.Series(gid_act).value_counts()
    log(f"   clusters: {len(sizes)} | largest: {sizes.iloc[:5].tolist()} | singletons: {(sizes == 1).sum()}")
    if sizes.iloc[0] > 0.05 * len(act):
        log("   WARNING: a cluster holds >5% of all images (chaining). Consider lowering NEAR_DUP_HAMMING.")
    sub = df.loc[act].reset_index(drop=True)
    split_of_group, gtable = assign_split(sub, gid_act, rng)
    del V
    gc.collect(); mem("after clustering")

    # 5. choose members per cluster
    prio = {"ds2": 0, "bg": 1, "ds1": 2}
    keep_rows, new_split = [], []
    sub["gid"] = gid_act
    for gid, members in sub.groupby("gid"):
        sp = split_of_group[gid]
        idx = list(members.index)
        rng.shuffle(idx)
        if sp == "train":
            idx.sort(key=lambda k: prio[sub.src[k]])
            lim = MAX_PER_GROUP_TRAIN or len(idx)
            chosen, rest = idx[:lim], idx[lim:]
            reason = "train_duplicate_thinning"
        else:
            elig = [k for k in idx if sub.src[k] in ("ds2", "bg")]
            chosen = elig[:MAX_PER_GROUP_EVAL]
            rest = [k for k in idx if k not in chosen]
            reason = "eval_cluster_guard"
        keep_rows += chosen
        new_split += [sp] * len(chosen)
        for k in rest:
            r = "leak_guard_ds1_in_eval_cluster" if (sp != "train" and sub.src[k] == "ds1") else reason
            dropped.append((sub.src[k], sub.name[k], r))
    final = sub.loc[keep_rows].copy()
    final["new_split"] = new_split
    final["group"] = final.gid
    final["group_size"] = final.gid.map(sub.gid.value_counts())
    final["new_name"] = final.src + "__" + final.name

    # 6. write dataset
    for s in ("train", "valid", "test"):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)
    orig_index = {(df.src[i], df.name[i]): i for i in range(n)}
    for r in final.itertuples():
        shutil.copy2(r.path, out / r.new_split / "images" / r.new_name)
        i = orig_index[(r.src, r.name)]
        lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for (x, y, w, h) in boxes[i]]
        (out / r.new_split / "labels" / (Path(r.new_name).stem + ".txt")).write_text("\n".join(lines))
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": out.resolve().as_posix(), "train": "train/images", "val": "valid/images",
                        "test": "test/images", "nc": 1, "names": ["FOD"]}, f, sort_keys=False)
    manifest = final[["new_name", "src", "name", "orig_split", "base", "new_split", "group", "group_size",
                      "W", "H", "bright", "n_boxes"]].rename(columns={"name": "orig_name"})
    manifest.to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(dropped, columns=["src", "orig_name", "reason"]).to_csv(out / "dropped.csv", index=False)
    log(f"dataset written -> {out}")
    log("dropped by reason: " + str(pd.DataFrame(dropped, columns=["s", "n", "r"]).r.value_counts().to_dict()))
    del df, sub, boxes
    gc.collect(); mem("after writing")

    # 7. final leakage check + stats
    res = final_leakage_check(out, manifest)
    st = dataset_stats(out, manifest)
    report = {"leakage_check": res, "cleaning": stats, "stats": st.reset_index().to_dict(orient="records")}
    with open(out / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    st.T.to_csv(out / "final_stats.csv")

    print("\n" + "=" * 70 + "\nFINAL LEAKAGE CHECK\n" + "=" * 70)
    for k, v in res.items():
        print(f"  {k:48s} {v}")
    print("\n  RESULT:", "PASS - no leakage detected" if res["PASS"] else "FAIL - see counts above")
    print("\n" + "=" * 70 + "\nFINAL DATASET STATS\n" + "=" * 70)
    print(st.T.to_string())
    print(f"\nAll files saved in {out}  (data.yaml, manifest.csv, dropped.csv, final_report.json, final_stats.csv)")
    if len(st) and (st.background_only_pct.fillna(0) < 1).all():
        print("NOTE: almost no background-only images. Set EXTRA_BG_DIR to add hard negatives to all splits.")


if __name__ == "__main__":
    main()