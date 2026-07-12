#!/usr/bin/env python3
"""
blur_robustness.py — input-robustness των student engines σε Gaussian blur.
---------------------------------------------------------------
Wrap-άρει τον inference core του eval_engines.py (ίδιο preprocessing, ίδιος
polygraphy/TRT runtime, ίδιο label policy) και προσθέτει loop πάνω σε σ.

Για κάθε engine × σ: τρέχει το επίσημο valid set (234) και μετράει MACRO(5) AUROC
-> καμπύλη AUROC vs σ (input-robustness). ΟΧΙ PyTorch: τρέχει σε Legion & Jetson.

Blur: εφαρμόζεται ΜΕΤΑ το resize(224) και ΠΡΙΝ το normalize, μέσω
PIL.ImageFilter.GaussianBlur(radius=σ) — στο Pillow το radius ΕΙΝΑΙ το Gaussian std,
οπότε σ -> radius απευθείας. Post-resize => το σ είναι σε pixels στο input resolution
του μοντέλου, άρα συγκρίσιμο σε όλες τις εικόνες (ImageNet-C convention). Scipy-free.

Έξοδος:
  raw/robustness/blur_<device>.npz        probs[n_eng,n_sigma,234,14] + sigmas + labels...
  raw/robustness/robust_summary_<device>.csv   (device,variant,precision,sigma,macro5,macro14)
  raw/robustness/robust_aud_<device>.csv        area-under-degradation ανά engine
  + live MACRO(5) vs σ table ανά engine στο terminal (για validation)

Παράδειγμα (ίδιο command Legion & Jetson — auto-fallback στο featHardKD naming):
  python blur_robustness.py --device jetson --engine-dir engines \
      --variants baseline logitKD featureKD --precisions fp16 int8 \
      --sigmas 0 1 2 3 4 --data-root /home/user/CXpertData
"""
import argparse
import csv
import os
import re

import numpy as np
from PIL import Image, ImageFilter

# Best-effort: στο Legion conda env το import torch προφορτώνει libcudart ώστε ο
# TrtRunner της polygraphy να βρει το CUDA. Στο Jetson (χωρίς torch) χρησιμοποιεί
# το system CUDA του JetPack — το missing import είναι αβλαβές.
try:
    import torch  # noqa: F401
except Exception:
    pass

def _trapz(y, x):
    """numpy 2.0 -> trapezoid, numpy 1.x -> trapz."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))

PATHOLOGIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
COMPETITION = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
COMP_IDX = [PATHOLOGIES.index(c) for c in COMPETITION]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---- AUROC (tie-aware, numpy-only· ίδιο με eval_engines) ----
def fast_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = float((y_true == 1).sum())
    n_neg = float((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score, kind="mergesort")
    uniq, inv, counts = np.unique(y_score[order], return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts
    avg_rank = start + (counts + 1) / 2.0
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = avg_rank[inv]
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---- I/O (ίδιο με eval_engines) ----
def load_valid(data_root, valid_csv):
    valid_csv = valid_csv or os.path.join(data_root, "valid.csv")
    paths, rows = [], []
    with open(valid_csv, newline="") as f:
        for row in csv.DictReader(f):
            paths.append(row["Path"]); rows.append(row)
    def val(row, name):
        v = row.get(name, "")
        if v is None or v == "":
            return 0.0
        x = float(v)
        return 0.0 if x == -1.0 else x            # U-Zeros + blank->0
    labels = np.array([[val(r, p) for p in PATHOLOGIES] for r in rows], dtype=np.float32)
    def pid(p):
        m = re.search(r"(patient\d+)", str(p))
        return m.group(1) if m else str(p)
    patient_ids = np.array([pid(p) for p in paths])
    return paths, labels, patient_ids


def resolve(data_root, p):
    return os.path.join(data_root, str(p).replace("CheXpert-v1.0-small/", ""))


def find_engine(engine_dir, variant, prec):
    """{dir}/{variant}_{prec}.engine, με auto-fallback στο Legion abbreviated naming."""
    cand = os.path.join(engine_dir, f"{variant}_{prec}.engine")
    if os.path.exists(cand):
        return cand
    alt = variant.replace("featureHardKD", "featHardKD")   # Legion tag
    cand2 = os.path.join(engine_dir, f"{alt}_{prec}.engine")
    if os.path.exists(cand2):
        return cand2
    raise SystemExit(f"[!] engine not found: {cand}\n    (also tried {cand2})")


# ---- blur + normalize ----
def blur_uint8(resized_pil, sigma):
    """Λίστα από PIL(224,224) -> uint8 stack [N,224,224,3] με Gaussian blur σ."""
    if sigma <= 0:
        arrs = [np.asarray(im, dtype=np.uint8) for im in resized_pil]
    else:
        arrs = [np.asarray(im.filter(ImageFilter.GaussianBlur(radius=float(sigma))),
                           dtype=np.uint8) for im in resized_pil]
    return np.stack(arrs)                                   # [N,224,224,3]


def normalize_batch(u8):
    """[N,224,224,3] uint8 -> [N,3,224,224] float32 (ImageNet normalize)."""
    a = u8.astype(np.float32) / 255.0
    a = (a - MEAN) / STD                                    # broadcast στα κανάλια
    return np.ascontiguousarray(np.transpose(a, (0, 3, 1, 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, help="legion / jetson (για output naming)")
    ap.add_argument("--engine-dir", default="engines")
    ap.add_argument("--variants", nargs="+",
                    default=["baseline", "logitKD", "featureKD"])
    ap.add_argument("--precisions", nargs="+", default=["fp16", "int8"])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0, 1, 2, 3, 4])
    ap.add_argument("--data-root", default="/home/user/CXpertData")
    ap.add_argument("--valid-csv", default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out-dir", default="raw/robustness")
    args = ap.parse_args()

    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner

    os.makedirs(args.out_dir, exist_ok=True)
    sigmas = list(args.sigmas)
    engines = [(v, p, find_engine(args.engine_dir, v, p))
               for v in args.variants for p in args.precisions]

    # 1) valid set + resize μία φορά (cache PIL, ~35MB)
    paths, labels, patient_ids = load_valid(args.data_root, args.valid_csv)
    resized = [Image.open(resolve(args.data_root, p)).convert("RGB")
               .resize((224, 224), Image.BILINEAR) for p in paths]
    N = len(resized)
    print(f"[{args.device}] {N} εικόνες | engines={len(engines)} | σ={sigmas}\n")

    # 2) blur μία φορά ανά σ (κοινό για όλα τα engines) -> uint8, ~35MB/σ
    blurred_u8 = [blur_uint8(resized, s) for s in sigmas]

    # 3) inference: κάθε engine φορτώνεται ΜΙΑ φορά, iterate τα σ μέσα
    n_eng, n_sig = len(engines), len(sigmas)
    probs_all = np.zeros((n_eng, n_sig, N, len(PATHOLOGIES)), dtype=np.float32)
    macro5 = np.full((n_eng, n_sig), np.nan)
    macro14 = np.full((n_eng, n_sig), np.nan)

    for ei, (variant, prec, path) in enumerate(engines):
        with TrtRunner(EngineFromBytes(BytesFromPath(path))) as r:
            for si in range(n_sig):
                X = normalize_batch(blurred_u8[si])
                for i in range(0, N, args.batch):
                    b = np.ascontiguousarray(X[i:i + args.batch])
                    logits = r.infer({"input": b})["logits"]
                    probs_all[ei, si, i:i + len(b)] = 1.0 / (1.0 + np.exp(-logits))
        # metrics ανά σ
        for si in range(n_sig):
            pr = probs_all[ei, si]
            per = [fast_auc(labels[:, j], pr[:, j]) for j in range(len(PATHOLOGIES))
                   if len(np.unique(labels[:, j])) >= 2]
            macro14[ei, si] = np.nanmean(per)
            macro5[ei, si] = np.nanmean([fast_auc(labels[:, j], pr[:, j]) for j in COMP_IDX])

        # live table + area-under-degradation (AUD) για validation
        clean = macro5[ei, 0]
        aud = _trapz(clean - macro5[ei], sigmas)             # ↑AUD = λιγότερο robust
        print(f"[{variant}/{prec}]  clean(σ=0)={clean:.4f}  "
              f"σ={sigmas[-1]}->{macro5[ei, -1]:.4f}  AUD={aud:.4f}")
        print("   " + "  ".join(f"σ{sigmas[si]:g}={macro5[ei, si]:.4f}"
                                for si in range(n_sig)) + "\n")

    # 4) npz (τα πάντα για T-ROBUST / F-ROBUST / μελλοντικό CI)
    npz = os.path.join(args.out_dir, f"blur_{args.device}.npz")
    np.savez(npz,
             probs=probs_all, sigmas=np.array(sigmas, dtype=np.float32),
             labels=labels, patient_ids=patient_ids,
             variants=np.array([v for v, _, _ in engines]),
             precisions=np.array([p for _, p, _ in engines]),
             pathologies=np.array(PATHOLOGIES), device=np.array(args.device))
    print(f"[npz] {npz}")

    # 5) summary + AUD csv
    scsv = os.path.join(args.out_dir, f"robust_summary_{args.device}.csv")
    with open(scsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "variant", "precision", "sigma", "macro5", "macro14"])
        for ei, (v, p, _) in enumerate(engines):
            for si, s in enumerate(sigmas):
                w.writerow([args.device, v, p, s,
                            f"{macro5[ei, si]:.6f}", f"{macro14[ei, si]:.6f}"])
    acsv = os.path.join(args.out_dir, f"robust_aud_{args.device}.csv")
    with open(acsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "variant", "precision", "macro5_clean",
                    "macro5_maxblur", "aud"])
        for ei, (v, p, _) in enumerate(engines):
            w.writerow([args.device, v, p, f"{macro5[ei, 0]:.6f}",
                        f"{macro5[ei, -1]:.6f}",
                        f"{_trapz(macro5[ei, 0] - macro5[ei], sigmas):.6f}"])
    print(f"[csv] {scsv}\n[csv] {acsv}")


if __name__ == "__main__":
    main()