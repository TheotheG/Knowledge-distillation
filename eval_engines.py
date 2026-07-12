#!/usr/bin/env python3
"""
eval_engines.py — apples-to-apples precision eval, ALL precisions via TensorRT.
---------------------------------------------------------------
Runs FP32 / FP16 / INT8 *engines* through the SAME polygraphy/TensorRT runtime
so differences are purely precision (no PyTorch-vs-TRT framework confound).

TORCH-FREE: needs only polygraphy + pillow + numpy → runs on the Legion AND on
the Jetson. (On the Jetson: `pip install polygraphy pillow` if missing; the
`tensorrt` python bindings already ship with JetPack.)

Same preprocessing + label policy as eval_precisions.py / dump_preds.py:
  Resize(224,224, bilinear) -> /255 -> ImageNet normalize ; labels: -1->0, blank->0

Optionally dumps preds_<tag>_<prec>.npz (probs/labels/patient_ids/pathologies)
so bootstrap_auroc_ci.py can compute CIs on the INT8 engine too.

Examples:
  # Legion (dynamic engines) or Jetson (static engines) — same command
  python eval_engines.py --tag baseline --platform jetson \
      --fp32-engine engines/baseline_fp32.engine \
      --fp16-engine engines/baseline_fp16.engine \
      --int8-engine engines/baseline_int8.engine \
      --data-root /home/user/CXpertData --batch 1 --dump-npz
"""
import argparse
import csv
import os
import re

import numpy as np
from PIL import Image

# Best-effort: where CUDA comes from pip wheels (e.g. the Legion conda env),
# importing torch preloads libcudart so polygraphy's TrtRunner can find it.
# On the Jetson (no torch) the JetPack system CUDA at /usr/local/cuda is used
# instead, so this missing import is harmless.
try:
    import torch  # noqa: F401
except Exception:
    pass

PATHOLOGIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
COMPETITION = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def fast_auc(y_true, y_score):
    """Tie-aware AUROC via Mann-Whitney U (matches sklearn), numpy-only."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = float((y_true == 1).sum())
    n_neg = float((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    uniq, inv, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts          # 0-based start of each tie group
    avg_rank = start + (counts + 1) / 2.0        # 1-based average rank per group
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = avg_rank[inv]
    sum_pos = ranks[y_true == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_valid(data_root, valid_csv):
    """Returns (paths, labels[N,14] float32, patient_ids[N])."""
    valid_csv = valid_csv or os.path.join(data_root, "valid.csv")
    paths, rows = [], []
    with open(valid_csv, newline="") as f:
        for row in csv.DictReader(f):
            paths.append(row["Path"])
            rows.append(row)
    def val(row, name):
        v = row.get(name, "")
        if v is None or v == "":
            return 0.0
        x = float(v)
        return 0.0 if x == -1.0 else x     # U-Zeros + blank->0
    labels = np.array([[val(r, p) for p in PATHOLOGIES] for r in rows], dtype=np.float32)
    def pid(p):
        m = re.search(r"(patient\d+)", str(p))
        return m.group(1) if m else str(p)
    patient_ids = np.array([pid(p) for p in paths])
    return paths, labels, patient_ids


def resolve(data_root, p):
    return os.path.join(data_root, str(p).replace("CheXpert-v1.0-small/", ""))


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0   # HWC in [0,1]
    a = (a - MEAN) / STD
    return np.transpose(a, (2, 0, 1))               # CHW


def run_engine(engine_path, imgs, batch=1):
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner
    out = np.zeros((len(imgs), len(PATHOLOGIES)), dtype=np.float32)
    with TrtRunner(EngineFromBytes(BytesFromPath(engine_path))) as r:
        for i in range(0, len(imgs), batch):
            b = np.ascontiguousarray(imgs[i:i + batch])
            out[i:i + len(b)] = r.infer({"input": b})["logits"]
    return out


def class_aucs(logits, labels):
    probs = 1.0 / (1.0 + np.exp(-logits))
    d = {}
    for j, name in enumerate(PATHOLOGIES):
        if len(np.unique(labels[:, j])) >= 2:
            d[name] = fast_auc(labels[:, j], probs[:, j])
    return d, probs.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--platform", default="", help="label only, e.g. legion / jetson")
    ap.add_argument("--fp32-engine", default=None)
    ap.add_argument("--fp16-engine", default=None)
    ap.add_argument("--int8-engine", default=None)
    ap.add_argument("--data-root", default="/home/user/CXpertData")
    ap.add_argument("--valid-csv", default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dump-npz", action="store_true",
                    help="save preds_<tag>_<prec>[_<platform>].npz for bootstrap CIs")
    args = ap.parse_args()

    engines = [("FP32", args.fp32_engine), ("FP16", args.fp16_engine),
               ("INT8", args.int8_engine)]
    engines = [(k, v) for k, v in engines if v]
    if not engines:
        raise SystemExit("[!] give at least one of --fp32/--fp16/--int8-engine")

    paths, labels, patient_ids = load_valid(args.data_root, args.valid_csv)
    imgs = np.stack([preprocess(resolve(args.data_root, p)) for p in paths]).astype(np.float32)
    print(f"[{args.tag}{('/' + args.platform) if args.platform else ''}] "
          f"loaded {len(imgs)} images.\n")

    results = {}
    suffix = f"_{args.platform}" if args.platform else ""
    for prec, eng in engines:
        logits = run_engine(eng, imgs, args.batch)
        results[prec], probs = class_aucs(logits, labels)
        if args.dump_npz:
            out = f"preds_{args.tag}_{prec.lower()}{suffix}.npz"
            np.savez(out, probs=probs, labels=labels, patient_ids=patient_ids,
                     pathologies=np.array(PATHOLOGIES), tag=np.array(f"{args.tag}_{prec.lower()}"))
            print(f"    [npz] {out}")

    cols = list(results.keys())
    print(f"\n{'Class':28s}" + "".join(f"{c:>9s}" for c in cols))
    print("-" * (28 + 9 * len(cols)))
    for name in PATHOLOGIES:
        if all(name in results[c] for c in cols):
            print(f"{name:28s}" + "".join(f"{results[c][name]:9.4f}" for c in cols))
    print("-" * (28 + 9 * len(cols)))
    mac = lambda c: np.nanmean(list(results[c].values()))
    mac5 = lambda c: np.nanmean([results[c][k] for k in COMPETITION if k in results[c]])
    print(f"{'MACRO (14)':28s}" + "".join(f"{mac(c):9.4f}" for c in cols))
    print(f"{'MACRO (5 competition)':28s}" + "".join(f"{mac5(c):9.4f}" for c in cols))


if __name__ == "__main__":
    main()
