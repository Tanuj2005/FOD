r"""
test_eval.py  -  STANDALONE. No training, no dependency on train_eval.py.
Loads your already-trained best.pt files and runs them on the TEST sets only:
   clean test + every weather/lighting set + every altitude set (all built from the test split).

Per run and per test set it reports / saves:
   mAP50, mAP50-95, P, R (Ultralytics), ms/img
   precision, recall, TP/FP/FN, false positives per image at CONF
   threshold_sweep.csv, confusion_at_conf.png, and (clean test) sweep_curves.png + Ultralytics PR curve/confusion matrix

Run:  uv run --no-sync python test_eval.py
"""
# ============================ CONFIG ============================
FINAL_ROOT = r"C:\FOD\final_dataset"       # contains yamls\ (from build_extras.py)
SEARCH_ROOT = r"C:\FOD\runs"               # searched recursively for <run>\weights\best.pt
OUT_DIR = r"C:\FOD\test_results"           # everything is written here

DEVICE = 0
IMGSZ = 640
CONF = 0.25                                # confidence used for precision/recall/FP counts and confusion matrices
IOU_MATCH = 0.5
VAL_BATCH = 8
PRED_CHUNK = 8                             # images per predict() call (lower to 4 if GPU memory is tight)
ONLY_RUNS = None                           # e.g. ["yolo26_plus_aug"]; None = all runs found

# Optional: choose CONF on the VALID set instead of fixing it (valid is only used for this choice).
PICK_CONF_ON_VALID = False
RECALL_TARGET = 0.95                       # used only if PICK_CONF_ON_VALID: highest conf with recall >= this

SWEEP = [0.01, 0.02, 0.03, 0.05] + [round(0.10 + 0.05 * i, 2) for i in range(18)]   # 0.01 ... 0.95
EXTRA = {}                                 # extra kwargs for val()/predict(), e.g. {"end2end": False} for YOLO26 NMS head
# ================================================================

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def free_gpu():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ------------------------------ matching -----------------------------------
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def match_image(pred_xyxy, pred_conf, gt_xyxy, thr):
    if len(pred_xyxy) == 0:
        return 0, 0, len(gt_xyxy)
    pred_xyxy = pred_xyxy[np.argsort(-pred_conf)]
    iou = iou_matrix(pred_xyxy, gt_xyxy)
    used, tp = set(), 0
    for i in range(len(pred_xyxy)):
        if iou.shape[1] == 0:
            break
        j = int(np.argmax(iou[i]))
        if iou[i, j] >= thr and j not in used:
            used.add(j)
            tp += 1
    return tp, len(pred_xyxy) - tp, len(gt_xyxy) - tp


def read_gt(lp, W, H):
    out = []
    if Path(lp).exists():
        for line in Path(lp).read_text().splitlines():
            v = line.split()
            if len(v) == 5:
                x, y, w, h = map(float, v[1:])
                out.append([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
    return np.array(out, dtype=np.float64).reshape(-1, 4)


def collect_predictions(model, img_dir):
    """Predict at conf 0.001 in small chunks. Returns list of (pred_boxes, pred_conf, gt_boxes)."""
    files = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    lbl_dir = Path(img_dir).parent / "labels"
    preds = []
    for s in range(0, len(files), PRED_CHUNK):
        results = model.predict(source=[str(p) for p in files[s:s + PRED_CHUNK]], imgsz=IMGSZ, conf=0.001,
                                verbose=False, device=DEVICE, **EXTRA)
        for r in results:
            H, W = r.orig_shape
            gt = read_gt(lbl_dir / (Path(r.path).stem + ".txt"), W, H)
            if r.boxes is not None and len(r.boxes):
                pb, pc = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
            else:
                pb, pc = np.zeros((0, 4)), np.zeros(0)
            preds.append((pb, pc, gt))
        del results
    return preds


def metrics_at(preds, conf):
    TP = FP = FN = n_bg = fp_bg = 0
    for pb, pc, gt in preds:
        k = pc >= conf
        tp, fp, fn = match_image(pb[k], pc[k], gt, IOU_MATCH)
        TP += tp; FP += fp; FN += fn
        if len(gt) == 0:
            n_bg += 1; fp_bg += fp
    n = len(preds)
    P, R = TP / max(TP + FP, 1), TP / max(TP + FN, 1)
    return {"conf": conf, "images": n, "bg_images": n_bg, "TP": TP, "FP": FP, "FN": FN,
            "precision": round(P, 4), "recall": round(R, 4), "F1": round(2 * P * R / max(P + R, 1e-9), 4),
            "FPPI_all": round(FP / max(n, 1), 4), "FPPI_bg": round(fp_bg / n_bg, 4) if n_bg else np.nan}


def sweep(preds):
    return pd.DataFrame([metrics_at(preds, c) for c in SWEEP])


def save_sweep_plot(df, path, conf, title):
    fp = "FPPI_bg" if df.FPPI_bg.notna().any() else "FPPI_all"
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(df.conf, df.recall, "o-", label="recall"); ax[0].plot(df.conf, df.precision, "s-", label="precision")
    ax[0].plot(df.conf, df.F1, "^-", label="F1", alpha=.6); ax[0].axvline(conf, color="r", ls="--", label=f"conf {conf:g}")
    ax[0].set_xlabel("confidence threshold"); ax[0].set_title("Precision / recall vs confidence"); ax[0].legend()
    d = df.sort_values("recall"); r = df[df.conf == conf]
    ax[1].plot(d.recall, d.precision, "o-")
    if len(r): ax[1].plot(r.recall, r.precision, "r*", ms=16, label="operating point"); ax[1].legend()
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title(f"Precision-recall curve (IoU {IOU_MATCH})")
    ax[1].set_xlim(0, 1.02); ax[1].set_ylim(0, 1.02)
    ax[2].plot(df[fp], df.recall, "o-")
    if len(r): ax[2].plot(r[fp], r.recall, "r*", ms=16)
    ax[2].set_xlabel(f"false positives per image ({fp})"); ax[2].set_ylabel("recall"); ax[2].set_title("Recall vs false-alarm rate")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_confusion(m, path, title):
    mat = np.array([[m["TP"], m["FP"]], [m["FN"], np.nan]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(np.nan_to_num(mat), cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, "n/a\n(background)" if np.isnan(v) else f"{int(v)}", ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["FOD", "background"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["FOD", "background"])
    ax.set_xlabel("actual"); ax.set_ylabel("predicted"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ------------------------------ discovery + eval ---------------------------
def find_weights(root):
    found = {}
    for p in Path(root).rglob("best.pt"):
        if p.parent.name != "weights":
            continue
        name = p.parent.parent.name
        if name not in found or p.stat().st_mtime > found[name].stat().st_mtime:
            found[name] = p
    return found


def test_yamls(yaml_dir):
    skip = {"eval_clean_valid.yaml"}
    names = sorted(p.name for p in yaml_dir.glob("eval_*.yaml") if p.name not in skip)
    return ["eval_clean_test.yaml"] + [n for n in names if n != "eval_clean_test.yaml"]


def val_dir_of(yaml_file):
    with open(yaml_file, encoding="utf-8") as f:
        return yaml.safe_load(f)["val"]


def evaluate_run(name, weights, yaml_dir, out_root):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    run_out = Path(out_root) / name
    run_out.mkdir(parents=True, exist_ok=True)
    conf, how = CONF, "fixed (CONF in config)"
    if PICK_CONF_ON_VALID:
        vp = collect_predictions(model, val_dir_of(yaml_dir / "eval_clean_valid.yaml"))
        vdf = sweep(vp)
        ok = vdf[vdf.recall >= RECALL_TARGET]
        conf = float(ok.conf.max()) if len(ok) else float(vdf.sort_values(["recall", "conf"], ascending=[False, False]).iloc[0].conf)
        how = f"chosen on valid (recall >= {RECALL_TARGET}{'' if len(ok) else ' NOT reached; max-recall conf used'})"
        vdf.to_csv(run_out / "valid_threshold_sweep.csv", index=False)
        del vp
    log(f"   {name}: using conf {conf:g} - {how}")
    rows = []
    for yname in test_yamls(yaml_dir):
        cond = yname.replace("eval_", "").replace(".yaml", "")
        sdir = run_out / cond
        sdir.mkdir(parents=True, exist_ok=True)
        log(f"   {name} -> {cond}")
        m = model.val(data=str(yaml_dir / yname), split="val", imgsz=IMGSZ, batch=VAL_BATCH, conf=0.001,
                      device=DEVICE, plots=(cond == "clean_test"), verbose=False,
                      project=str(run_out), name=f"ultralytics_{cond}", exist_ok=True, **EXTRA)
        preds = collect_predictions(model, val_dir_of(yaml_dir / yname))
        sdf = sweep(preds)
        sdf.to_csv(sdir / "threshold_sweep.csv", index=False)
        op = metrics_at(preds, conf)
        save_confusion(op, sdir / "confusion_at_conf.png", f"{name} {cond} @ conf {conf:g}")
        if cond == "clean_test":
            save_sweep_plot(sdf, sdir / "sweep_curves.png", conf, f"{name} - clean test")
        rows.append({"run": name, "test_set": cond,
                     "mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
                     "P@maxF1": round(float(m.box.mp), 4), "R@maxF1": round(float(m.box.mr), 4),
                     "ms_inference": round(float(m.speed.get("inference", np.nan)), 2),
                     "conf_used": conf, "images": op["images"], "bg_images": op["bg_images"],
                     "TP": op["TP"], "FP": op["FP"], "FN": op["FN"],
                     "precision@conf": op["precision"], "recall@conf": op["recall"],
                     "FPPI_all": op["FPPI_all"], "FPPI_bg": op["FPPI_bg"]})
        del m, preds
        gc.collect()
    del model
    free_gpu()
    pd.DataFrame(rows).to_csv(run_out / "test_results.csv", index=False)
    with open(run_out / "conf_used.json", "w", encoding="utf-8") as f:
        json.dump({"conf": conf, "how": how, "weights": str(weights)}, f, indent=2)
    return rows


def main():
    yaml_dir = Path(FINAL_ROOT) / "yamls"
    if not (yaml_dir / "eval_clean_test.yaml").exists():
        raise SystemExit(f"yamls not found in {yaml_dir}. Check FINAL_ROOT.")
    found = find_weights(SEARCH_ROOT)
    if ONLY_RUNS:
        found = {k: v for k, v in found.items() if k in ONLY_RUNS}
    print("weights found:")
    for n, p in sorted(found.items()):
        print(f"   {n:22s} {p}")
    if not found:
        raise SystemExit(f"No best.pt found under {SEARCH_ROOT}")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name, w in sorted(found.items()):
        rows = evaluate_run(name, w, yaml_dir, OUT_DIR)
        df = pd.DataFrame(rows)
        print("\n" + "=" * 120 + f"\n{name}   (conf = {rows[0]['conf_used']:g})\n" + "=" * 120)
        print(df.drop(columns=["run"]).to_string(index=False))
        all_rows += rows
    big = pd.DataFrame(all_rows)
    big.to_csv(Path(OUT_DIR) / "test_results_all_runs.csv", index=False)
    for metric in ("mAP50", "mAP50-95", "recall@conf", "FPPI_bg", "ms_inference"):
        print("\n" + "=" * 120 + f"\nCOMPARISON  {metric}   (rows = test set, columns = run)\n" + "=" * 120)
        print(big.pivot(index="test_set", columns="run", values=metric).to_string())
    log(f"saved: {Path(OUT_DIR) / 'test_results_all_runs.csv'}")


if __name__ == "__main__":
    main()