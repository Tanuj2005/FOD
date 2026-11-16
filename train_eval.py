r"""
train_eval.py   -  YOLO11 vs YOLO26 on the final FOD dataset (output of build_extras.py)

Pipeline
  1. TRAIN   every (model x data-variant) in MODELS x DATA_VARIANTS with identical settings.
  2. THRESHOLD CHOICE on the VALID set (never on test): sweeps confidence, picks the highest confidence
     whose recall >= RECALL_TARGET (= fewest false alarms that still meet the safety recall goal).
  3. EVALUATE on every eval set separately (clean test, each weather/lighting condition, each altitude scale)
     using that chosen confidence.

Task-5 outputs saved per run in RUNS_DIR\<run>\ ...
  eval\<set>\                 Ultralytics PR / P / R / F1 curves + confusion matrix (clean_test)
  eval\<set>\threshold_sweep.csv            recall, precision, F1, FP/image for conf 0.05..0.95
  eval\<set>\sweep_curves.png               precision & recall vs conf | PR curve | recall vs FP per image
  eval\<set>\confusion_at_conf.png          confusion matrix at the chosen conf (FOD vs background)
  threshold_choice.json                     chosen conf + the justification numbers (valid) and test numbers
  eval_by_condition.csv                     one row per eval set
RUNS_DIR\eval_all_runs.csv + printed side-by-side comparisons.

Run (Windows):  pip install -U ultralytics   then   python train_eval.py
TRAIN=False evaluates existing weights only.
"""
# ============================ CONFIG ============================
FINAL_ROOT = r"final_dataset"       # OUT_ROOT of build_extras.py
RUNS_DIR = r"runs"
DEVICE = 0                                  # GPU index or "cpu"

TRAIN = True
EVALUATE = True

MODELS = {"yolo11": "yolo11s.pt", "yolo26": "yolo26s.pt"}
DATA_VARIANTS = {                           # name -> yaml in FINAL_ROOT/yamls ; delete a line to skip it
    "base":     "data_base.yaml",
    "plus_aug": "data_train_plus_aug.yaml",
}

IMGSZ = 640
EPOCHS = 60
BATCH = 8                                   # RTX 3050 4GB; lower if out-of-memory
WORKERS = 2
PATIENCE = 20
SEED = 42
AMP = True
OPTIMIZER = "auto"
AUGMENTATION = dict(
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.6,
    degrees=5.0, translate=0.1, scale=0.6, fliplr=0.5, flipud=0.0,
    mosaic=1.0, close_mosaic=10, mixup=0.0,
)

# --- threshold / metrics ---
RECALL_TARGET = 0.95                        # safety goal: recall to keep on the VALID set
AUTO_THRESHOLD = True                       # True: choose conf on valid; False: use CONF_OPERATING everywhere
CONF_OPERATING = 0.25                       # used if AUTO_THRESHOLD=False or as fallback
SWEEP = [round(x, 2) for x in [0.01, 0.02, 0.03, 0.05] + list(__import__("numpy").arange(0.10, 0.951, 0.05))]
IOU_MATCH = 0.5
EVAL_BATCH = 8
RUN_ONLY = None                             # e.g. ["yolo26_base"]

# Extra kwargs for val()/predict() per model family (e.g. {"yolo26": {"end2end": False}} to force the NMS head;
# check your installed Ultralytics docs). {} = library default.
EVAL_EXTRA = {"yolo11": {}, "yolo26": {}}
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


def build_runs():
    runs = []
    for mname, weights in MODELS.items():
        for vname, ymlfile in DATA_VARIANTS.items():
            name = f"{mname}_{vname}"
            if RUN_ONLY and name not in RUN_ONLY:
                continue
            runs.append(dict(name=name, family=mname, weights=weights, data=ymlfile, variant=vname))
    return runs


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
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_image(pred_xyxy, pred_conf, gt_xyxy, thr):
    """Greedy match by confidence. Returns (tp, fp, fn)."""
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


# ------------------- predictions once, metrics at any threshold ------------
def collect_predictions(model, img_dir, imgsz, extra):
    """One predict pass at a very low conf. Returns list of (pred_boxes, pred_conf, gt_boxes)."""
    files = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    lbl_dir = Path(img_dir).parent / "labels"
    preds = []
    for r in model.predict(source=[str(p) for p in files], imgsz=imgsz, conf=0.001, stream=True,
                           verbose=False, device=DEVICE, **extra):
        H, W = r.orig_shape
        gt = read_gt(lbl_dir / (Path(r.path).stem + ".txt"), W, H)
        if r.boxes is not None and len(r.boxes):
            pb, pc = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        else:
            pb, pc = np.zeros((0, 4)), np.zeros(0)
        preds.append((pb, pc, gt))
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


def choose_threshold(df):
    """Highest conf with recall >= target (fewest false alarms that meet the goal). Fallback: max recall."""
    ok = df[df.recall >= RECALL_TARGET]
    if len(ok):
        return float(ok.conf.max()), True
    best = df.sort_values(["recall", "conf"], ascending=[False, False]).iloc[0]
    return float(best.conf), False


def save_sweep_plot(df, path, chosen, title):
    fpcol = "FPPI_bg" if df.FPPI_bg.notna().any() else "FPPI_all"
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].plot(df.conf, df.recall, "o-", label="recall"); ax[0].plot(df.conf, df.precision, "s-", label="precision")
    ax[0].plot(df.conf, df.F1, "^-", label="F1", alpha=.6)
    ax[0].axvline(chosen, color="r", ls="--", label=f"chosen conf {chosen:g}"); ax[0].axhline(RECALL_TARGET, color="gray", ls=":")
    ax[0].set_xlabel("confidence threshold"); ax[0].set_title("Precision / recall vs confidence"); ax[0].legend()
    d = df.sort_values("recall")
    ax[1].plot(d.recall, d.precision, "o-"); r = df[df.conf == chosen]
    if len(r): ax[1].plot(r.recall, r.precision, "r*", ms=16, label="chosen")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title(f"Precision-recall curve (IoU {IOU_MATCH})")
    ax[1].set_xlim(0, 1.02); ax[1].set_ylim(0, 1.02); ax[1].legend()
    ax[2].plot(df[fpcol], df.recall, "o-")
    if len(r): ax[2].plot(r[fpcol], r.recall, "r*", ms=16)
    ax[2].set_xlabel(f"false positives per image ({fpcol})"); ax[2].set_ylabel("recall"); ax[2].set_title("Recall vs false-alarm rate")
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


# ------------------------------ train / eval -------------------------------
def train_one(run, yaml_dir):
    from ultralytics import YOLO
    log(f"TRAIN {run['name']} | {run['weights']} | {run['data']} | imgsz={IMGSZ} epochs={EPOCHS} batch={BATCH}")
    model = YOLO(run["weights"])
    model.train(data=str(yaml_dir / run["data"]), imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, workers=WORKERS,
                patience=PATIENCE, seed=SEED, amp=AMP, optimizer=OPTIMIZER, device=DEVICE, plots=True,
                project=RUNS_DIR, name=run["name"], exist_ok=True, **AUGMENTATION)
    del model
    free_gpu()


def eval_yamls(yaml_dir):
    skip = {"eval_clean_test.yaml", "eval_clean_valid.yaml"}
    return ["eval_clean_test.yaml"] + sorted(p.name for p in yaml_dir.glob("eval_*.yaml") if p.name not in skip)


def load_val_dir(yaml_file):
    with open(yaml_file, encoding="utf-8") as f:
        return yaml.safe_load(f)["val"]


def evaluate_one(run, yaml_dir):
    from ultralytics import YOLO
    run_dir = Path(RUNS_DIR) / run["name"]
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        log(f"   no weights for {run['name']} ({best}) - skipped")
        return []
    extra = EVAL_EXTRA.get(run["family"], {})
    model = YOLO(str(best))

    # ---- 1. choose the threshold on VALID ----
    vdir = run_dir / "eval" / "clean_valid"; vdir.mkdir(parents=True, exist_ok=True)
    vpreds = collect_predictions(model, load_val_dir(yaml_dir / "eval_clean_valid.yaml"), IMGSZ, extra)
    vdf = sweep(vpreds); vdf.to_csv(vdir / "threshold_sweep.csv", index=False)
    if AUTO_THRESHOLD:
        conf, reached = choose_threshold(vdf)
    else:
        conf, reached = CONF_OPERATING, None
    vrow = metrics_at(vpreds, conf)
    save_sweep_plot(vdf, vdir / "sweep_curves.png", conf, f"{run['name']} - VALID set (threshold selection)")
    save_confusion(vrow, vdir / "confusion_at_conf.png", f"{run['name']} valid @ conf {conf:g}")
    log(f"   {run['name']}: chosen conf = {conf:g} on valid "
        f"(recall {vrow['recall']}, precision {vrow['precision']}, FPPI_bg {vrow['FPPI_bg']}, "
        f"target {RECALL_TARGET} {'reached' if reached else 'NOT reached' if reached is not None else 'n/a'})")
    del vpreds
    gc.collect()

    # ---- 2. evaluate every eval set at that threshold ----
    rows = []
    for yname in eval_yamls(yaml_dir):
        cond = yname.replace("eval_", "").replace(".yaml", "")
        ydir = run_dir / "eval" / cond; ydir.mkdir(parents=True, exist_ok=True)
        log(f"   eval {run['name']} on {cond}")
        m = model.val(data=str(yaml_dir / yname), split="val", imgsz=IMGSZ, batch=EVAL_BATCH, conf=0.001,
                      device=DEVICE, plots=(cond == "clean_test"), verbose=False,
                      project=str(run_dir / "eval"), name=cond, exist_ok=True, **extra)
        preds = collect_predictions(model, load_val_dir(yaml_dir / yname), IMGSZ, extra)
        sdf = sweep(preds); sdf.to_csv(ydir / "threshold_sweep.csv", index=False)
        op = metrics_at(preds, conf)
        save_confusion(op, ydir / "confusion_at_conf.png", f"{run['name']} {cond} @ conf {conf:g}")
        if cond == "clean_test":
            save_sweep_plot(sdf, ydir / "sweep_curves.png", conf, f"{run['name']} - TEST set (threshold fixed from valid)")
        rows.append({"run": run["name"], "model": run["family"], "data": run["variant"], "eval_set": cond,
                     "mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
                     "P@maxF1": round(float(m.box.mp), 4), "R@maxF1": round(float(m.box.mr), 4),
                     "ms_inference": round(float(m.speed.get("inference", np.nan)), 2),
                     "ms_postprocess": round(float(m.speed.get("postprocess", np.nan)), 2),
                     "conf_used": conf, "images": op["images"], "bg_images": op["bg_images"],
                     "TP": op["TP"], "FP": op["FP"], "FN": op["FN"],
                     "precision@conf": op["precision"], "recall@conf": op["recall"],
                     "FPPI_all": op["FPPI_all"], "FPPI_bg": op["FPPI_bg"]})
        if cond == "clean_test":
            with open(run_dir / "threshold_choice.json", "w", encoding="utf-8") as f:
                json.dump({"recall_target": RECALL_TARGET, "chosen_conf": conf, "target_reached_on_valid": reached,
                           "valid_at_chosen_conf": {k: (None if v != v else v) for k, v in vrow.items()},
                           "test_at_chosen_conf": {k: (None if v != v else v) for k, v in op.items()},
                           "rule": "highest confidence on VALID whose recall >= target (fewest false alarms meeting the goal); "
                                   "test metrics are reported at that fixed threshold"}, f, indent=2, default=lambda o: o.item())
        del m, preds
        gc.collect()
    del model
    free_gpu()
    return rows


def main():
    yaml_dir = Path(FINAL_ROOT) / "yamls"
    runs = build_runs()
    Path(RUNS_DIR).mkdir(parents=True, exist_ok=True)
    log("runs: " + ", ".join(r["name"] for r in runs))
    if TRAIN:
        for r in runs:
            train_one(r, yaml_dir)
    if not EVALUATE:
        return
    all_rows = []
    for r in runs:
        rows = evaluate_one(r, yaml_dir)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(Path(RUNS_DIR) / r["name"] / "eval_by_condition.csv", index=False)
            print("\n" + "=" * 120 + f"\n{r['name']}   (conf chosen on valid = {rows[0]['conf_used']:g})\n" + "=" * 120)
            print(df.drop(columns=["run", "model", "data"]).to_string(index=False))
            all_rows += rows
    if all_rows:
        big = pd.DataFrame(all_rows)
        big.to_csv(Path(RUNS_DIR) / "eval_all_runs.csv", index=False)
        for metric in ("mAP50", "mAP50-95", "recall@conf", "FPPI_bg", "ms_inference"):
            print("\n" + "=" * 120 + f"\nCOMPARISON  {metric}   (rows = eval set, columns = run)\n" + "=" * 120)
            print(big.pivot(index="eval_set", columns="run", values=metric).to_string())
        log(f"saved: {Path(RUNS_DIR) / 'eval_all_runs.csv'}")


if __name__ == "__main__":
    main()