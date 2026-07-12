"""
dump_preds.py
---------------------------------------------------------------
Τρέχει έναν STUDENT (EfficientNet-B0, 14-class CheXpert) στο ΕΠΙΣΗΜΟ valid set
(234 εικόνες) και αποθηκεύει probabilities + labels σε ένα .npz.

ΓΙΑΤΙ ξεχωριστό script: αποσυνδέει το inference (μία φορά ανά μοντέλο) από τη
στατιστική ανάλυση (bootstrap). Έτσι το bootstrap_auroc_ci.py δουλεύει πάνω στα
αποθηκευμένα predictions και το ξανατρέχεις όσες φορές θες (π.χ. 10k vs 50k
resamples, image- vs patient-level) ΧΩΡΙΣ να ξαναπερνάς εικόνες από το δίκτυο.

Ίδιο ακριβώς preprocessing & label policy με το eval_precisions.py:
  - Resize(224) -> ToTensor -> ImageNet normalize
  - labels: replace(-1 -> 0) [U-Zeros], fillna(0)  (το επίσημο valid ούτως ή άλλως
    είναι certain, δεν έχει -1· το κρατάμε για ασφάλεια/συνέπεια)

Το ίδιο script δουλεύει ΓΙΑ ΟΛΟΥΣ τους students (baseline / logitKD / featureKD),
αφού είναι όλοι EfficientNet-B0 με ίδιο preprocessing.

Χρήση:
    conda activate chexpert
    python dump_preds.py --tag baseline --ckpt efficientnet-b0-epoch-5.pth
    python dump_preds.py --tag logitKD  --ckpt student-efficientnet-b0-epoch-5_lg.pth
    # αργότερα, όταν έχεις το feature-KD checkpoint:
    python dump_preds.py --tag featureKD --ckpt student-efficientnet-b0_featkd.pth
"""
import os
import re
import argparse

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms

PATHOLOGIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_sd(path):
    """Ξεδιπλώνει 'state_dict'/'model'/'module.' (ίδια λογική με τα άλλα scripts)."""
    ck = torch.load(path, map_location="cpu")
    if isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
    elif isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict):
        sd = ck["model"]
    else:
        sd = ck
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def build_model(ckpt, arch, device):
    model = timm.create_model(arch, pretrained=False,
                              num_classes=len(PATHOLOGIES), exportable=True)
    miss, unexp = model.load_state_dict(load_sd(ckpt), strict=False)
    if miss:
        print("[!] MISSING keys:", miss)
    if unexp:
        print("[!] UNEXPECTED keys:", unexp)
    if not miss and not unexp:
        print("[OK] Το state_dict ταιριάζει τέλεια με την αρχιτεκτονική.")
    return model.eval().to(device)


@torch.no_grad()
def run_model(model, imgs, device, batch=32):
    """imgs: float32 [N,3,224,224] -> probs float32 [N,14] (μετά από sigmoid)."""
    out = []
    for i in range(0, len(imgs), batch):
        xb = torch.from_numpy(imgs[i:i + batch]).to(device)
        logits = model(xb)
        out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="baseline / logitKD / featureKD")
    ap.add_argument("--ckpt", required=True, help="το .pth του student")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--data-root", default="/home/user/CXpertData",
                    help="φάκελος που περιέχει valid.csv + τις εικόνες")
    ap.add_argument("--valid-csv", default=None)
    ap.add_argument("--out", default=None, help="default: preds_<tag>.npz")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | tag={args.tag}")

    valid_csv = args.valid_csv or os.path.join(args.data_root, "valid.csv")
    out_path = args.out or f"preds_{args.tag}.npz"

    df = pd.read_csv(valid_csv)

    # labels: U-Zeros + blank->0 (ίδιο με eval_precisions)
    labels = df[PATHOLOGIES].replace(-1.0, 0.0).fillna(0.0).values.astype(np.float32)

    # patient id (για προαιρετικό patient-level bootstrap)
    def pid(p):
        m = re.search(r"(patient\d+)", str(p))
        return m.group(1) if m else str(p)
    patient_ids = np.array([pid(p) for p in df["Path"].values])

    # φόρτωσε εικόνες (ίδιο path-resolve με eval_precisions)
    resolve = lambda p: os.path.join(args.data_root, str(p).replace("CheXpert-v1.0-small/", ""))
    imgs = np.stack([_tf(Image.open(resolve(p)).convert("RGB")).numpy()
                     for p in df["Path"].values]).astype(np.float32)
    print(f"[{args.tag}] Φορτώθηκαν {len(imgs)} εικόνες.")

    model = build_model(args.ckpt, args.arch, device)
    probs = run_model(model, imgs, device, batch=args.batch)

    np.savez(
        out_path,
        probs=probs.astype(np.float32),
        labels=labels.astype(np.float32),
        patient_ids=patient_ids,
        pathologies=np.array(PATHOLOGIES),
        tag=np.array(args.tag),
    )
    print(f"[OK] Αποθηκεύτηκε -> {out_path}")

    # μικρό sanity: full-sample macro(5) για να δεις ότι βγάζει ~αναμενόμενο νούμερο.
    # (προαιρετικό· αν το bootstrap_auroc_ci.py δεν είναι δίπλα, απλώς το παραλείπουμε)
    try:
        from bootstrap_auroc_ci import fast_auc, COMPETITION
        comp_idx = [PATHOLOGIES.index(c) for c in COMPETITION]
        aucs = [fast_auc(labels[:, j], probs[:, j]) for j in comp_idx]
        print(f"[{args.tag}] full-sample MACRO(5 competition) = {np.nanmean(aucs):.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
