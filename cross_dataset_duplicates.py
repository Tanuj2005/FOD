"""
cross_dataset_duplicates.py
Compares ds1 TRAIN against ds2 TRAIN / VALID / TEST with a perceptual hash (same 256-bit dHash
as eda_fod.py). Reports how many images overlap, saves the pairs, example montages, and a list
of ds1 train files to drop before merging.

Run:  python cross_dataset_duplicates.py
"""
# ============================ CONFIG ============================
DS1_ROOT = r"Runway FOD Detector.v1i.yolov11"
DS2_ROOT = r"FOD Runway Dataset.v2i.yolov11"
OUT_DIR = r"eda_results\ds1_vs_ds2"
NEAR_DUP_HAMMING = 10          # out of 256 bits (same as eda_fod.py)
N_EXAMPLES = 12                # montage examples per ds2 split
CELL = 400
# ================================================================

import gc
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def list_images(root, split_names):
    for s in split_names:
        d = Path(root) / s / "images"
        if d.is_dir():
            return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    return []


def dhash256(path):
    with Image.open(path) as im0:
        im0.draft("RGB", (512, 512))
        im1 = im0.convert("RGB")
    im1.thumbnail((256, 256))
    g = np.asarray(im1.convert("L").resize((17, 16), RESAMPLE), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def hash_folder(paths, cache_file):
    """Hashes one image at a time; caches result so re-runs are instant."""
    cache_file = Path(cache_file)
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        if list(z["names"]) == [p.name for p in paths]:
            print(f"   loaded cached hashes: {cache_file.name}")
            return z["hashes"], z["ok"]
    H = np.zeros((len(paths), 32), dtype=np.uint8)
    ok = np.zeros(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        if i and i % 1000 == 0:
            print(f"   hashed {i}/{len(paths)}")
        try:
            H[i] = dhash256(p)
            ok[i] = True
        except Exception:
            pass
    np.savez_compressed(cache_file, hashes=H, ok=ok, names=np.array([p.name for p in paths], dtype=object))
    gc.collect()
    return H, ok


def min_hamming(A, B, chunk=64):
    dmin = np.full(len(A), 999, dtype=np.int32)
    arg = np.full(len(A), -1, dtype=np.int32)
    if len(A) == 0 or len(B) == 0:
        return dmin, arg
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk]
        d = POPCOUNT[np.bitwise_xor(a[:, None, :], B[None, :, :])].sum(axis=2, dtype=np.int32)
        j = d.argmin(axis=1)
        dmin[s:s + len(a)] = d[np.arange(len(a)), j]
        arg[s:s + len(a)] = j
        del d
    gc.collect()
    return dmin, arg


def montage(rows, out_path, label_a, label_b):
    if not rows:
        return
    canvas = Image.new("RGB", (2 * CELL + 10, len(rows) * CELL), (20, 20, 20))
    for k, (pa, pb, d) in enumerate(rows):
        for j, (p, lab) in enumerate([(pa, label_a), (pb, label_b)]):
            try:
                with Image.open(p) as im0:
                    im = im0.convert("RGB").resize((CELL, CELL))
                ImageDraw.Draw(im).text((4, 4), f"{lab}: {Path(p).name[:40]}  d={d}", fill=(255, 255, 0))
                canvas.paste(im, (j * (CELL + 10), k * CELL))
                del im
            except Exception:
                pass
    canvas.save(out_path)
    del canvas
    gc.collect()


def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    ds1_train = list_images(DS1_ROOT, ["train"])
    ds2 = {"train": list_images(DS2_ROOT, ["train"]),
           "valid": list_images(DS2_ROOT, ["valid", "val", "validation"]),
           "test": list_images(DS2_ROOT, ["test"])}
    print(f"ds1 train: {len(ds1_train)} | ds2: " + ", ".join(f"{k}={len(v)}" for k, v in ds2.items()))

    print("hashing ds1 train")
    H1, ok1 = hash_folder(ds1_train, out / "cache_ds1_train.npz")
    summary, all_pairs = [], []
    drop = {}                                    # ds1 file -> min distance to ds2 valid/test

    for split, paths in ds2.items():
        if not paths:
            continue
        print(f"hashing ds2 {split}")
        H2, ok2 = hash_folder(paths, out / f"cache_ds2_{split}.npz")
        i1, i2 = np.where(ok1)[0], np.where(ok2)[0]

        # direction A: for each ds2 image, closest ds1-train image
        d2, a2 = min_hamming(H2[i2], H1[i1])
        # direction B: for each ds1-train image, closest ds2 image
        d1, a1 = min_hamming(H1[i1], H2[i2])

        n2 = int((d2 <= NEAR_DUP_HAMMING).sum())
        n1 = int((d1 <= NEAR_DUP_HAMMING).sum())
        summary.append({
            "ds2_split": split,
            "ds2_images": len(i2),
            "ds2_imgs_with_ds1_match": n2,
            "ds2_imgs_with_ds1_match_pct": round(100 * n2 / max(len(i2), 1), 2),
            "ds2_exact_matches_d0": int((d2 == 0).sum()),
            "ds2_median_dist_to_ds1": float(np.median(d2)) if len(d2) else np.nan,
            "ds1_train_imgs_matching_this_split": n1,
            "ds1_train_imgs_matching_pct": round(100 * n1 / max(len(i1), 1), 2),
        })
        print(f"   ds2 {split}: {n2}/{len(i2)} images have a near-duplicate in ds1 train "
              f"({100 * n2 / max(len(i2), 1):.1f}%)")

        for k in np.where(d2 <= NEAR_DUP_HAMMING)[0]:
            all_pairs.append((split, paths[i2[k]].name, ds1_train[i1[a2[k]]].name, int(d2[k]),
                              str(paths[i2[k]]), str(ds1_train[i1[a2[k]]])))
        if split in ("valid", "test"):
            for k in np.where(d1 <= NEAR_DUP_HAMMING)[0]:
                nm = ds1_train[i1[k]].name
                drop[nm] = min(drop.get(nm, 999), int(d1[k]))

        # montage: evenly spread over the distance range
        cand = np.where(d2 <= NEAR_DUP_HAMMING)[0]
        if len(cand):
            cand = cand[np.argsort(d2[cand])]
            pick = cand[np.unique(np.linspace(0, len(cand) - 1, min(N_EXAMPLES, len(cand))).astype(int))]
            montage([(str(paths[i2[k]]), str(ds1_train[i1[a2[k]]]), int(d2[k])) for k in pick],
                    out / f"examples_ds2_{split}_vs_ds1_train.png", f"ds2 {split}", "ds1 train")
        del H2, d1, a1, d2, a2
        gc.collect()

    pd.DataFrame(summary).to_csv(out / "summary_ds1_vs_ds2.csv", index=False)
    pd.DataFrame(all_pairs, columns=["ds2_split", "ds2_image", "ds1_train_image", "hamming",
                                     "ds2_path", "ds1_path"]).sort_values("hamming").to_csv(
        out / "matching_pairs.csv", index=False)
    pd.DataFrame(sorted(drop.items()), columns=["ds1_train_file_to_drop", "hamming_to_ds2_valid_or_test"]).to_csv(
        out / "ds1_train_files_to_drop.csv", index=False)

    print("\n" + pd.DataFrame(summary).to_string(index=False))
    print(f"\nds1 train files matching ds2 valid/test (to drop before merging): {len(drop)}")
    print(f"Results saved in {out}")


if __name__ == "__main__":
    main()