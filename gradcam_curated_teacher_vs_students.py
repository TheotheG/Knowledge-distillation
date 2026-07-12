"""
gradcam_curated_teacher_vs_students.py
---------------------------------------------------------------
CURATED Grad-CAM: ViT TEACHER vs students (baseline / logitKD / featureKD).
Για κάθε παθολογία επιλέγει ΑΥΤΟΜΑΤΑ ground-truth ΘΕΤΙΚΕΣ + πιο σίγουρες εικόνες.

!!! Ο TEACHER είναι HuggingFace ViTForImageClassification (φάκελος safetensors) !!!
  - Φορτώνεται μέσω vit_teacher_loader.load_vit_teacher (ΟΧΙ timm, ΟΧΙ .pth).
  - Έχει ΔΙΚΟ του normalization (π.χ. 0.5/0.5/0.5) -> ΔΙΑΦΟΡΕΤΙΚΟ από τους students
    (ImageNet). Γι' αυτό ΚΑΘΕ μοντέλο κανονικοποιεί ΜΟΝΟ του την ίδια εικόνα.
  - Grad-CAM target: vit.encoder.layer[-1].layernorm_before (+ reshape tokens->14x14).

Οι students (EfficientNet-B0) φορτώνονται από .pth, ImageNet norm, target conv_head.

ΠΡΟΫΠΟΘΕΣΗ: το vit_teacher_loader.py στον ΙΔΙΟ φάκελο (~/chexpert-project).
Απαιτήσεις: pip install matplotlib pandas transformers safetensors

Χρήση:
    python gradcam_curated_teacher_vs_students.py \
        --vit-dir        vit-chest-xray-full-labels-epoch-4 \
        --baseline-ckpt  checkpoints/efficientnet-b0-epoch-5.pth \
        --logitkd-ckpt   checkpoints/logitKD_efficientnet_b0.pth \
        --featurekd-ckpt checkpoints/featureKD.pth \
        --data-root /home/user/archive \
        --valid-csv /home/user/archive/valid.csv \
        --competition --per-class 3 --rank-by ViT --mask-corners 0.18
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vit_teacher_loader import load_vit_teacher   # ίδιος φάκελος


PATHOLOGIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
COMPETITION = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================ STUDENTS (CNN) ============================
def load_state_dict(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        sd = ckpt
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in sd.items()}


def build_efficientnet(ckpt_path, device, label):
    model = timm.create_model("efficientnet_b0", pretrained=False,
                              num_classes=len(PATHOLOGIES))
    m, u = model.load_state_dict(load_state_dict(ckpt_path, device), strict=False)
    if m or u:
        print(f"[{label}] MISSING={m[:4]} UNEXPECTED={u[:4]}")
    else:
        print(f"[{label}] [OK] state_dict ταιριάζει.")
    return model.eval().to(device)


# ============================ GRAD-CAM (CNN + ViT) ============================
def make_vit_reshape(num_prefix=1):
    """[B, num_prefix+N, D] -> [B, D, h, w] πετώντας τα prefix (CLS) tokens."""
    def _reshape(tensor):
        n = tensor.shape[1] - num_prefix
        h = w = int(round(n ** 0.5))
        assert h * w == n, f"Τα patch tokens ({n}) δεν σχηματίζουν τετράγωνο grid."
        r = tensor[:, num_prefix:, :].reshape(tensor.shape[0], h, w, tensor.shape[2])
        return r.permute(0, 3, 1, 2).contiguous()
    return _reshape


class GradCAM:
    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model
        self.reshape_transform = reshape_transform
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[0, class_idx].backward()
        acts, grads = self.activations, self.gradients
        if self.reshape_transform is not None:
            acts = self.reshape_transform(acts)
            grads = self.reshape_transform(grads)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        prob = float(torch.sigmoid(logits)[0, class_idx])
        return cam.cpu().numpy(), prob


# ============================ PREPROCESSING (per-model norm) ============================
def apply_center_crop(pil, frac):
    w, h = pil.size
    cw, ch = int(round(w * frac)), int(round(h * frac))
    left, top = (w - cw) // 2, (h - ch) // 2
    return pil.crop((left, top, left + cw, top + ch))


def apply_mask_corners(arr, frac):
    h, w = arr.shape[:2]
    s = int(round(frac * h))
    fill = float(np.median(arr))
    out = arr.copy()
    out[:s, :s] = fill
    out[:s, w - s:] = fill
    return out


def preprocess_disp(pil, img_size, mask_corners=None, center_crop=None):
    """ΚΟΙΝΗ εικόνα-εμφάνισης [0,1] (ίδια pixels για ΟΛΑ τα μοντέλα)."""
    pil = pil.convert("RGB")
    if center_crop:
        pil = apply_center_crop(pil, center_crop)
    pil = pil.resize((img_size, img_size), Image.BILINEAR)
    disp = np.asarray(pil, dtype=np.float32) / 255.0
    if mask_corners:
        disp = apply_mask_corners(disp, mask_corners)
    return disp


def disp_to_input(disp, mean, std, device):
    """Κανονικοποιεί την ΙΔΙΑ εικόνα με το normalization ΤΟΥ ΚΑΘΕ μοντέλου."""
    arr = (disp - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float().to(device)


# ============================ PATH / CSV ============================
def resolve_image_path(data_root, csv_path_value):
    data_root = os.path.expanduser(data_root)
    clean_path = csv_path_value.replace("CheXpert-v1.0-small/", "").lstrip("/")
    final_path = os.path.join(data_root, clean_path)
    return final_path


def patient_id(csv_path_value):
    for part in csv_path_value.split("/"):
        if part.startswith("patient"):
            return part
    return os.path.basename(os.path.dirname(csv_path_value))


def _slug(s):
    return "".join(ch if ch.isalnum() else "-" for ch in s).strip("-")


def build_name2idx(id2label):
    """Χαρτογράφηση όνομα-κλάσης -> index ΣΤΗΝ ΕΞΟΔΟ ΤΟΥ ΜΟΝΤΕΛΟΥ.
       Αν το id2label δεν ταιριάζει με PATHOLOGIES, το class_idx θα στόχευε λάθος κλάση."""
    if not id2label:
        return {n: i for i, n in enumerate(PATHOLOGIES)}
    label2id = {v: int(k) for k, v in id2label.items()}
    if all(n in label2id for n in PATHOLOGIES):
        return {n: label2id[n] for n in PATHOLOGIES}
    print("[!] id2label δεν καλύπτει όλες τις PATHOLOGIES -> χρήση default σειράς.")
    return {n: i for i, n in enumerate(PATHOLOGIES)}


# ============================ PLOT (Original + N μοντέλα) ============================
def save_class_figure(class_name, cases, model_labels, out_path):
    n = len(cases)
    nrows = 1 + len(model_labels)
    fig, axes = plt.subplots(nrows, n, figsize=(4 * n, 4 * nrows))
    axes = np.array(axes).reshape(nrows, n)

    for j, case in enumerate(cases):
        disp = case["disp"]
        axes[0, j].imshow(disp)
        axes[0, j].set_title(case["id"], fontsize=10)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        for r, lbl in enumerate(model_labels, start=1):
            ax = axes[r, j]
            ax.imshow(disp)
            ax.imshow(case["cams"][lbl], cmap="jet", alpha=0.45)
            ax.set_xlabel(f"p={case['probs'][lbl]:.2f}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    row_labels = ["Original"] + model_labels
    for r in range(nrows):
        axes[r, 0].set_ylabel(row_labels[r], fontsize=12, fontweight="bold")

    fig.suptitle(f"{class_name}  —  ground-truth positive cases",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================ MAIN ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit-dir", default=None,
                    help="φάκελος teacher (HF: config.json + model.safetensors + preprocessor_config.json)")
    ap.add_argument("--baseline-ckpt", default=None)
    ap.add_argument("--logitkd-ckpt", default=None)
    ap.add_argument("--featurekd-ckpt", default=None)
    ap.add_argument("--data-root", default="~/CXpertData", help="φάκελος που περιέχει CheXpert-v1.0-small/valid.csv + εικόνες")
    ap.add_argument("--valid-csv", default=None)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--competition", action="store_true")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--rank-by", default="mean",
                    help="ViT/baseline/logitKD/featureKD/mean (default mean)")
    ap.add_argument("--min-prob", type=float, default=0.0)
    ap.add_argument("--mask-corners", type=float, default=None)
    ap.add_argument("--center-crop", type=float, default=None)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--outdir", default="gradcam_teacher_vs_students_out")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    # ---- Χτίσιμο entries: ViT (HF) + students (CNN), σε σταθερή σειρά ----
    entries = []

    if args.vit_dir:
        info = load_vit_teacher(args.vit_dir, device)
        print(f"[ViT] framework={info['framework']} norm={info['mean'].tolist()} "
              f"img={info['img_size']}")
        if info["img_size"] != args.img_size:
            print(f"[!] ViT img_size={info['img_size']} != {args.img_size}. "
                  f"Χρησιμοποιώ {args.img_size} για όλους (ViT-B/16 θέλει 224).")
        entries.append({
            "label": "ViT",
            "model": info["model"],
            "cam": GradCAM(info["model"], info["target"], make_vit_reshape(info["num_prefix"])),
            "mean": info["mean"], "std": info["std"],
            "name2idx": build_name2idx(info.get("id2label")),
        })

    for label, ckpt in [("baseline", args.baseline_ckpt),
                        ("logitKD", args.logitkd_ckpt),
                        ("featureKD", args.featurekd_ckpt)]:
        if ckpt is None:
            continue
        model = build_efficientnet(ckpt, device, label)
        entries.append({
            "label": label,
            "model": model,
            "cam": GradCAM(model, model.conv_head, None),
            "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
            "name2idx": {n: i for i, n in enumerate(PATHOLOGIES)},
        })

    if len(entries) < 2:
        raise ValueError("Δώσε τουλάχιστον 2 μοντέλα (π.χ. --vit-dir + ένα student).")

    # reorder index: model-output -> canonical PATHOLOGIES σειρά (για ranking/gt)
    for e in entries:
        e["reorder"] = [e["name2idx"][name] for name in PATHOLOGIES]

    model_labels = [e["label"] for e in entries]
    print(f"Μοντέλα: {model_labels}")

    # ---- valid csv / ground truth ----
    data_root = os.path.expanduser(args.data_root)
    valid_csv = args.valid_csv or os.path.join(data_root, "CheXpert-v1.0-small", "valid.csv")
    valid_csv = os.path.expanduser(valid_csv)
    if not os.path.exists(valid_csv):
        raise FileNotFoundError(f"Δεν βρέθηκε valid.csv: {valid_csv}\n"
                                f"Δώσε σωστό --valid-csv ή --data-root.")
    df = pd.read_csv(valid_csv)
    gt = df[PATHOLOGIES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    img_paths = [resolve_image_path(data_root, p) for p in df["Path"].values]
    print(f"valid.csv: {len(df)} εικόνες\n")

    if args.classes:
        classes = [c.strip() for c in args.classes.split(",")]
    else:
        classes = COMPETITION
    canon_idx = {n: i for i, n in enumerate(PATHOLOGIES)}

    # ---- ΒΗΜΑ 1: probs (canonical σειρά) για ΟΛΟ το valid, ΟΛΑ τα μοντέλα ----
    print("Inference σε όλο το valid set (per-model normalization)...")
    all_probs = {e["label"]: np.zeros((len(df), len(PATHOLOGIES)), dtype=np.float32)
                 for e in entries}
    for i, ip in enumerate(img_paths):
        disp = preprocess_disp(Image.open(ip), args.img_size,
                               args.mask_corners, args.center_crop)
        for e in entries:
            x = disp_to_input(disp, e["mean"], e["std"], device)
            with torch.no_grad():
                raw = torch.sigmoid(e["model"](x))[0].cpu().numpy()
            all_probs[e["label"]][i] = raw[e["reorder"]]   # -> canonical σειρά
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)}")
    print("  ...ολοκληρώθηκε.\n")

    if args.rank_by == "mean":
        rank_probs = np.mean([all_probs[l] for l in model_labels], axis=0)
    elif args.rank_by in all_probs:
        rank_probs = all_probs[args.rank_by]
    else:
        raise ValueError(f"--rank-by '{args.rank_by}' μη διαθέσιμο. "
                         f"Επιλογές: {model_labels + ['mean']}")
    print(f"Κατάταξη cases κατά: {args.rank_by}\n")

    # ---- ΒΗΜΑ 2 & 3: ανά κλάση -> positives -> top-N -> Grad-CAM (όλα τα μοντέλα) ----
    for cname in classes:
        ci = canon_idx[cname]
        pos_idx = np.where(gt[cname].values == 1.0)[0]
        if len(pos_idx) == 0:
            print(f"[skip] {cname}: 0 ground-truth positives.")
            continue

        order = pos_idx[np.argsort(-rank_probs[pos_idx, ci])]
        if args.min_prob > 0.0:
            order = [i for i in order if rank_probs[i, ci] >= args.min_prob]
        chosen_idx = list(order[:args.per_class])
        if not chosen_idx:
            print(f"[skip] {cname}: κανένα positive πάνω από min-prob={args.min_prob}.")
            continue

        cases = []
        for i in chosen_idx:
            disp = preprocess_disp(Image.open(img_paths[i]), args.img_size,
                                   args.mask_corners, args.center_crop)
            cams, probs = {}, {}
            for e in entries:
                x = disp_to_input(disp, e["mean"], e["std"], device)
                model_ci = e["name2idx"][cname]     # index ΣΤΗΝ ΕΞΟΔΟ ΤΟΥ ΜΟΝΤΕΛΟΥ
                cam, p = e["cam"](x, model_ci)
                cams[e["label"]] = cam
                probs[e["label"]] = p
            cases.append({"id": patient_id(df["Path"].values[i]),
                          "disp": disp, "cams": cams, "probs": probs})

        out_name = f"curated_{'-'.join(model_labels)}_{_slug(cname)}.png"
        out_path = os.path.join(args.outdir, out_name)
        save_class_figure(cname, cases, model_labels, out_path)

        line = "  ".join(f"{c['id']}[" + ",".join(f"{l}={c['probs'][l]:.2f}"
                         for l in model_labels) + "]" for c in cases)
        print(f"[OK] {cname:18s} -> {out_name}")
        print(f"      {line}")

    print(f"\nΈτοιμα. Αποθηκεύτηκαν στο: {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
