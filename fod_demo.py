"""
fod_demo.py
  --mode list      show available stress-test condition folders
  --mode video     result video (1 model, or 2 models side by side)
  --mode examples  mine true-positive / false-positive / missed examples
  --mode glare     check how many glare "misses" are physically invisible objects
  --mode grid      one image under many conditions (for the report figures)
"""
import argparse, csv, random
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path("final_dataset")
EXT = {".jpg", ".jpeg", ".png", ".bmp"}
GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 255), (0, 215, 255)
WHITE, BLACK, GREY = (255, 255, 255), (0, 0, 0), (60, 60, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX
S, BAR, FOOT = 720, 60, 36          # panel size, title bar, footer


# ---------------------------------------------------------------- helpers
def cond_dir(cond):
    return ROOT / "base" / "test" if cond == "clean_test" else ROOT / "brownie" / "eval" / cond

def list_imgs(d):
    return sorted(p for p in (d / "images").iterdir() if p.suffix.lower() in EXT)

def label_path(img):
    return img.parent.parent / "labels" / (img.stem + ".txt")

def has_obj(img):
    f = label_path(img)
    return f.exists() and f.read_text().strip() != ""

def read_gt(img, W, H):
    out = []
    f = label_path(img)
    if f.exists():
        for ln in f.read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                xc, yc, w, h = map(float, p[1:5])
                out.append([(xc-w/2)*W, (yc-h/2)*H, (xc+w/2)*W, (yc+h/2)*H])
    return np.array(out, dtype=float).reshape(-1, 4)

def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1]); bb = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)

def analyse(model, path, conf):
    """Run the model and match predictions to ground truth (greedy, IoU>=0.5)."""
    im = cv2.imread(str(path)); H, W = im.shape[:2]
    r = model.predict(str(path), conf=conf, imgsz=640, verbose=False)[0]
    P = r.boxes.xyxy.cpu().numpy(); C = r.boxes.conf.cpu().numpy()
    o = np.argsort(-C); P, C = P[o], C[o]
    G = read_gt(path, W, H)
    I = iou_mat(P, G); used, tp = set(), []
    for i in range(len(P)):
        j, best = -1, 0.5
        for k in range(len(G)):
            if k not in used and I[i, k] >= best:
                j, best = k, I[i, k]
        tp.append(j >= 0)
        if j >= 0:
            used.add(j)
    fn = [k for k in range(len(G)) if k not in used]
    return im, P, C, G, tp, fn

def label(img, text, org, color, scale=0.6):
    (w, h), b = cv2.getTextSize(text, FONT, scale, 2)
    x, y = org; y = max(y, h + 4)
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + b), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), FONT, scale,
                BLACK if color == YELLOW else WHITE, 2, cv2.LINE_AA)

def render(model, title, path, conf):
    """Annotated panel: green = correct, red = false alarm, yellow = missed."""
    im, P, C, G, tp, fn = analyse(model, path, conf)
    H, W = im.shape[:2]; sx, sy = S / W, S / H
    im = cv2.resize(im, (S, S))
    for k in fn:
        x1, y1, x2, y2 = (G[k] * [sx, sy, sx, sy]).astype(int)
        cv2.rectangle(im, (x1, y1), (x2, y2), YELLOW, 2); label(im, "MISSED", (x1, y1), YELLOW, 0.5)
    for b, c, ok in zip(P, C, tp):
        x1, y1, x2, y2 = (b * [sx, sy, sx, sy]).astype(int); col = GREEN if ok else RED
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2); label(im, f"FOD {c:.2f}", (x1, y1), col)
    n_tp = int(sum(tp)); n_fp = len(tp) - n_tp
    bar = np.zeros((BAR, S, 3), np.uint8)
    cv2.putText(bar, title, (8, 24), FONT, 0.7, WHITE, 2, cv2.LINE_AA)
    cv2.putText(bar, f"conf>={conf:.2f}   TP={n_tp}  FP={n_fp}  FN={len(fn)}", (8, 50), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    return np.vstack([bar, im]), (n_tp, n_fp, len(fn))

def finish(frame, cond):
    foot = np.zeros((FOOT, frame.shape[1], 3), np.uint8)
    cv2.putText(foot, f"set: {cond}", (8, 24), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    x = frame.shape[1] - 520
    for txt, col in [("correct", GREEN), ("false alarm", RED), ("missed", YELLOW)]:
        cv2.rectangle(foot, (x, 10), (x + 16, 26), col, -1)
        cv2.putText(foot, txt, (x + 22, 24), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        x += 170
    return np.vstack([frame, foot])


# ---------------------------------------------------------------- modes
def video(a):
    models = [YOLO(w) for w in a.weights]
    names = a.names or [Path(w).parts[-3] for w in a.weights]
    size = (S * len(models), BAR + S + FOOT)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, size)
    rng, log = random.Random(a.seed), []
    for cond in a.conditions:
        imgs = list_imgs(cond_dir(cond))
        pos = [p for p in imgs if has_obj(p)]; neg = [p for p in imgs if not has_obj(p)]
        k_neg = min(len(neg), max(1, round(a.n * 0.3)))          # ~30% empty runway frames
        pick = rng.sample(pos, min(len(pos), a.n - k_neg)) + rng.sample(neg, k_neg)
        rng.shuffle(pick)
        card = np.zeros((size[1], size[0], 3), np.uint8)
        cv2.putText(card, f"Test condition: {cond}", (40, size[1]//2 - 10), FONT, 1.2, WHITE, 2, cv2.LINE_AA)
        cv2.putText(card, " vs ".join(names) + f"   (conf >= {a.conf})", (40, size[1]//2 + 40), FONT, 0.8, WHITE, 1, cv2.LINE_AA)
        for _ in range(int(a.fps * 1.5)):
            vw.write(card)
        for p in pick:
            panels = []
            for m, nm in zip(models, names):
                f, c = render(m, nm, p, a.conf); panels.append(f); log.append([cond, nm, p.name, *c])
            fr = finish(np.hstack(panels), cond)
            for _ in range(int(a.fps * a.hold)):
                vw.write(fr)
    vw.release()
    with open(Path(a.out).with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "model", "image", "TP", "FP", "FN"]); w.writerows(log)
    print("saved", a.out)

def examples(a):
    model = YOLO(a.weights[0]); name = (a.names or ["model"])[0].replace(" ", "_")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, saved = [], {}
    for cond in a.conditions:
        for p in list_imgs(cond_dir(cond)):
            f, (tp, fp, fn) = render(model, name, p, a.conf)
            rows.append([cond, p.name, tp, fp, fn])
            kinds = (["tp"] if tp and not fp and not fn else []) + (["fp"] if fp else []) + (["fn"] if fn else [])
            for k in kinds:
                n = saved.get((cond, k), 0)
                if n < a.per_kind:
                    saved[(cond, k)] = n + 1
                    cv2.imwrite(str(out / f"{name}_{cond}_{k}{n+1}.png"), finish(f, cond))
    with open(out / f"{name}_all_images.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "image", "TP", "FP", "FN"]); w.writerows(rows)
    for cond in a.conditions:
        r = [x for x in rows if x[0] == cond]
        print(f"{cond}: {len(r)} images | with FP: {sum(x[3] > 0 for x in r)} | with FN: {sum(x[4] > 0 for x in r)}")

def glare(a):
    model = YOLO(a.weights[0]); tot = sat_n = miss = miss_sat = 0
    for p in list_imgs(cond_dir(a.conditions[0])):
        im, P, C, G, tp, fn = analyse(model, p, a.conf)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        for k, (x1, y1, x2, y2) in enumerate(G.astype(int)):
            crop = g[max(y1, 0):y2, max(x1, 0):x2]
            sat = crop.size > 0 and (crop > 250).mean() > 0.5       # >50% of the box is saturated
            m = k in fn
            tot += 1; sat_n += sat; miss += m; miss_sat += (m and sat)
    print(f"boxes={tot} | mostly saturated={sat_n} | missed={miss} | missed AND saturated={miss_sat}")
    print(f"recall (all boxes)       = {(tot - miss) / tot:.3f}")
    print(f"recall (visible boxes)   = {(tot - sat_n - (miss - miss_sat)) / max(tot - sat_n, 1):.3f}")

def grid(a):
    ref = [p for p in list_imgs(cond_dir("clean_test")) if has_obj(p)][a.index]
    conds = a.conditions if "clean_test" in a.conditions else ["clean_test"] + a.conditions
    tiles = []
    for c in conds:
        imgs = list_imgs(cond_dir(c)); m = [p for p in imgs if ref.stem in p.stem]
        if not m:
            print(f"WARNING {c}: could not match {ref.stem}; using index {a.index} (check it is the same scene)")
        p = m[0] if m else imgs[a.index]
        im = cv2.imread(str(p)); H, W = im.shape[:2]
        for b in read_gt(p, W, H):
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GREEN, 2)
        im = cv2.resize(im, (320, 320)); label(im, c, (4, 24), GREY, 0.6); tiles.append(im)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows)); print("saved", a.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["list", "video", "examples", "glare", "grid"])
    ap.add_argument("--weights", nargs="+"); ap.add_argument("--names", nargs="+")
    ap.add_argument("--conditions", nargs="+", default=["clean_test"])
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--fps", type=int, default=24); ap.add_argument("--hold", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--per_kind", type=int, default=3)
    ap.add_argument("--out", default="demo_out/fod_demo.mp4"); ap.add_argument("--out_dir", default="demo_out/examples")
    a = ap.parse_args()
    if a.mode == "list":
        print(sorted(d.name for d in (ROOT / "brownie" / "eval").iterdir() if d.is_dir()))
    else:
        {"video": video, "examples": examples, "glare": glare, "grid": grid}[a.mode](a)