"""
eval_teacher_vit.py
---------------------------------------------------------------
Αξιολόγηση του ViT teacher (HuggingFace ViTForImageClassification) στο
επίσημο CheXpert valid (234 εικόνες). Χρησιμοποιεί το ΔΙΚΟ του image processor
(AutoImageProcessor) -> εγγυημένα το ίδιο preprocessing με το training,
οπότε ΔΕΝ μαντεύουμε mean/std (π.χ. 0.5 vs ImageNet).

Σκοπός: το "ταβάνι" (teacher ceiling) για σύγκριση με τον distilled student.
Ίδιες 234 εικόνες, ίδια metric με το student eval -> apples-to-apples.

Sanity gate: τα per-class AUROC πρέπει να ΑΝΑΠΑΡΑΓΟΥΝ τα νούμερα που έστειλε
ο συμφοιτητής σου (Cardiomegaly~0.79, Edema~0.92, Pleural Effusion~0.93...).
Αν ταιριάζουν, το preprocessing είναι σωστό και το ceiling αξιόπιστο.

Εγκατάσταση:
    pip install -U transformers safetensors

Χρήση:
    python eval_teacher_vit.py \
        --model_dir vit-chest-xray-full-labels-epoch-4 \
        --valid_csv data/CheXpert-v1.0-small/valid.csv \
        --data_root data
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score

PATHOLOGIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
COMPETITION = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]


def resolve_path(p, data_root):
    """Βρίσκει την εικόνα δοκιμάζοντας μερικά πιθανά roots (ανθεκτικό σε path prefixes)."""
    cands = [
        os.path.join(data_root, p),
        os.path.join(data_root, p.replace("CheXpert-v1.0-small/", "")),
        p,
    ]
    for c in cands:
        c = os.path.expanduser(c)
        if os.path.exists(c):
            return c
    return None


def _to_list(v):
    if v is None:
        return None
    try:
        return [float(x) for x in v]
    except TypeError:
        return float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="φάκελος με config.json + model.safetensors + preprocessor_config.json")
    ap.add_argument("--valid_csv", required=True)
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_json", default="teacher_vit_eval.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from transformers import ViTForImageClassification, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = ViTForImageClassification.from_pretrained(args.model_dir).eval().to(device)

    # Δείξε το preprocessing που ΟΝΤΩΣ εφαρμόζεται (για να το γράψεις στο paper)
    mean = _to_list(getattr(processor, "image_mean", None))
    std = _to_list(getattr(processor, "image_std", None))
    print(f"Processor normalization -> mean={mean}  std={std}")

    df = pd.read_csv(args.valid_csv)
    y = (df[PATHOLOGIES].apply(pd.to_numeric, errors="coerce")
         .fillna(0.0).values.astype(np.float32))
    paths = df["Path"].tolist()
    print(f"Valid images: {len(paths)}")

    r0 = resolve_path(paths[0], args.data_root)
    print(f"First image resolved -> {r0}")
    if r0 is None:
        raise SystemExit("[X] Δεν βρέθηκε η 1η εικόνα. Διόρθωσε το --data_root "
                         f"(Path στο CSV: {paths[0]})")

    all_logits = []
    with torch.no_grad():
        for i in range(0, len(paths), args.batch_size):
            bp = paths[i:i + args.batch_size]
            imgs = [Image.open(resolve_path(p, args.data_root)).convert("RGB") for p in bp]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            all_logits.append(logits.float().cpu())
    probs = torch.sigmoid(torch.cat(all_logits)).numpy()

    # ---- per-class AUROC ----
    print(f"\n{'Class':<28}{'AUROC':>8}")
    print("-" * 36)
    aucs = {}
    for i, name in enumerate(PATHOLOGIES):
        yc = y[:, i]
        if len(np.unique(yc)) < 2:      # χρειάζονται και οι δύο κλάσεις
            print(f"{name:<28}{'nan':>8}")
            continue
        a = float(roc_auc_score(yc, probs[:, i]))
        aucs[name] = a
        print(f"{name:<28}{a:>8.4f}")

    comp = [aucs[k] for k in COMPETITION if k in aucs]
    macro_comp = float(np.mean(comp)) if comp else float("nan")
    macro_all = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    print("-" * 36)
    print(f"{'macro (5 competition)':<28}{macro_comp:>8.4f}")
    print(f"{'macro (all computable)':<28}{macro_all:>8.4f}")

    json.dump({"per_class": aucs,
               "macro_5competition": macro_comp,
               "macro_all_computable": macro_all,
               "normalization": {"mean": mean, "std": std},
               "n_images": len(paths)},
              open(args.out_json, "w"), indent=2)
    print(f"\n[OK] Αποθηκεύτηκε -> {args.out_json}")


if __name__ == "__main__":
    main()
