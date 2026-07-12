"""
gradcam_curated_compare_chexpert.py
---------------------------------------------------------------
CURATED Grad-CAM σύγκριση δύο student checkpoints (baseline vs logitKD/featureKD),
όπου οι εικόνες ΔΕΝ δίνονται από εσένα αλλά ΕΠΙΛΕΓΟΝΤΑΙ ΑΥΤΟΜΑΤΑ ανά κλάση:

    για κάθε παθολογία -> κράτα εικόνες που είναι GROUND-TRUTH ΘΕΤΙΚΕΣ (από valid.csv)
    -> τις κατατάσσει κατά confidence του μοντέλου -> κρατάει τις top-N.

ΓΙΑΤΙ: το Grad-CAM κανονικοποιεί κάθε χάρτη στο [0,1] ανεξάρτητα από το p. Σε
εικόνα ΑΡΝΗΤΙΚΗ για μια κλάση (ή με p~0.2) ο χάρτης απλώς μεγεθύνει θόρυβο και
«κοιτάζει» εκτός θώρακα. Επιλέγοντας positive + high-confidence, το CAM πέφτει
σε πραγματικό εύρημα -> paper-ready figures.

ΠΡΟΑΙΡΕΤΙΚΑ (κατά confounders τύπου burned-in κειμένου «AP PORT» / δείκτη «L»):
    --mask-corners 0.18   -> μασκάρει πάνω-αριστερή & πάνω-δεξιά γωνία (fill=median)
    --center-crop  0.85   -> κρατάει το κεντρικό 85% (crop) πριν το resize
Και τα δύο εφαρμόζονται ΤΑΥΤΟΧΡΟΝΑ στην εικόνα-εμφάνισης και στην είσοδο του
μοντέλου, ώστε το CAM να αντιστοιχεί σε αυτό που βλέπεις.

Έξοδος: ΕΝΑ figure ανά κλάση, 3 σειρές (Original / A / B) × N στήλες (τα cases).

Τρέχει ΜΟΝΟ στο laptop (δεν αγγίζει το Jetson).

Απαιτήσεις: pip install matplotlib pandas

Χρήση:
    python gradcam_curated_compare_chexpert.py \
        --ckpt-a checkpoints/efficientnet-b0-epoch-5.pth --variant-a baseline \
        --ckpt-b checkpoints/logitKD_efficientnet_b0.pth --variant-b logitKD \
        --data-root ~/chexpert-project/data \
        --competition --per-class 3 --rank-by mean \
        --mask-corners 0.18
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


PATHOLOGIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
COMPETITION = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================ CHECKPOINT ============================
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


def build_model(ckpt_path, device, label=""):
    model = timm.create_model("efficientnet_b0", pretrained=False,
                              num_classes=len(PATHOLOGIES))
    missing, unexpected = model.load_state_dict(load_state_dict(ckpt_path, device),
                                                strict=False)
    tag = f"[{label}] " if label else ""
    if missing:
        print(f"{tag}[!] MISSING keys:", missing)
    if unexpected:
        print(f"{tag}[!] UNEXPECTED keys:", unexpected)
    if not missing and not unexpected:
        print(f"{tag}[OK] state_dict ταιριάζει τέλεια.")
    return model.eval().to(device)


def resolve_target_layer(model, name=None):
    if name:
        mod = model
        for part in name.split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        return mod
    if hasattr(model, "conv_head"):
        return model.conv_head
    return model.blocks[-1]


# ============================ GRAD-CAM ============================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
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
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        prob = float(torch.sigmoid(logits)[0, class_idx])
        return cam.cpu().numpy(), prob


# ============================ PREPROCESSING (με προαιρετικό mask/crop) ============================
def apply_center_crop(pil, frac):
    """Κρατάει το κεντρικό frac της αρχικής εικόνας (πριν το resize)."""
    w, h = pil.size
    cw, ch = int(round(w * frac)), int(round(h * frac))
    left, top = (w - cw) // 2, (h - ch) // 2
    return pil.crop((left, top, left + cw, top + ch))


def apply_mask_corners(arr, frac):
    """Γεμίζει πάνω-αριστερή & πάνω-δεξιά γωνία (τετράγωνα frac*H) με την median τιμή.
       Χτυπάει τα burned-in metadata (κείμενο «AP PORT», δείκτης «L») χωρίς σκληρό
       μαύρο τετράγωνο που θα δημιουργούσε δικό του edge-artifact."""
    h, w = arr.shape[:2]
    s = int(round(frac * h))
    fill = float(np.median(arr))
    out = arr.copy()
    out[:s, :s] = fill        # top-left
    out[:s, w - s:] = fill    # top-right
    return out


def preprocess(pil, img_size, mask_corners=None, center_crop=None):
    """Επιστρέφει (disp_arr HxWx3 [0,1], x_tensor normalized). ΙΔΙΑ pixels και στα δύο,
       ώστε το CAM να αντιστοιχεί ακριβώς στην εικόνα που εμφανίζεται."""
    pil = pil.convert("RGB")
    if center_crop:
        pil = apply_center_crop(pil, center_crop)
    pil = pil.resize((img_size, img_size), Image.BILINEAR)
    disp = np.asarray(pil, dtype=np.float32) / 255.0
    if mask_corners:
        disp = apply_mask_corners(disp, mask_corners)
    arr = (disp - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
    return disp, x


@torch.no_grad()
def predict_probs(model, x):
    return torch.sigmoid(model(x))[0].cpu().numpy()


# ============================ PATH / CSV ============================
def resolve_image_path(data_root, csv_path_value):
    """Χτίζει το πλήρες path της εικόνας με ανοχή στο 'CheXpert-v1.0-small/' prefix."""
    data_root = os.path.expanduser(data_root)
    cands = [
        os.path.join(data_root, csv_path_value),
        os.path.join(data_root, csv_path_value.replace("CheXpert-v1.0-small/", "")),
        os.path.join(data_root, "CheXpert-v1.0-small",
                     csv_path_value.replace("CheXpert-v1.0-small/", "")),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]  # best guess· θα βγάλει καθαρό error αργότερα


def patient_id(csv_path_value):
    for part in csv_path_value.split("/"):
        if part.startswith("patient"):
            return part
    return os.path.basename(os.path.dirname(csv_path_value))


def _slug(s):
    return "".join(ch if ch.isalnum() else "-" for ch in s).strip("-")


# ============================ PLOT (3 σειρές: Original / A / B) ============================
def save_class_figure(class_name, cases, label_a, label_b, out_path):
    n = len(cases)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    axes = np.array(axes).reshape(3, n)
    row_labels = ["Original", label_a, label_b]

    for j, case in enumerate(cases):
        disp = case["disp"]
        # row 0: original
        ax = axes[0, j]
        ax.imshow(disp)
        ax.set_title(case["id"], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        # row 1: model A overlay
        ax = axes[1, j]
        ax.imshow(disp); ax.imshow(case["cam_a"], cmap="jet", alpha=0.45)
        ax.set_xlabel(f"p={case['pa']:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        # row 2: model B overlay
        ax = axes[2, j]
        ax.imshow(disp); ax.imshow(case["cam_b"], cmap="jet", alpha=0.45)
        ax.set_xlabel(f"p={case['pb']:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for r in range(3):
        axes[r, 0].set_ylabel(row_labels[r], fontsize=12, fontweight="bold")

    fig.suptitle(f"{class_name}  —  ground-truth positive cases",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================ MAIN ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--label-a", default="baseline")
    ap.add_argument("--label-b", default="logitKD")
    ap.add_argument("--variant-a", default=None,
                    choices=["baseline", "logitKD", "featureKD"],
                    help="υπερισχύει του --label-a (featureKD = feature-level distillation)")
    ap.add_argument("--variant-b", default=None,
                    choices=["baseline", "logitKD", "featureKD"],
                    help="υπερισχύει του --label-b")
    ap.add_argument("--data-root", default="~/CXpertData",
                    help="φάκελος που περιέχει το CheXpert-v1.0-small/")
    ap.add_argument("--valid-csv", default=None,
                    help="default: {data-root}/CheXpert-v1.0-small/valid.csv")
    ap.add_argument("--classes", default=None,
                    help="ρητή λίστα (π.χ. \"Cardiomegaly,Edema\"). Default: 5 competition.")
    ap.add_argument("--competition", action="store_true",
                    help="χρησιμοποίησε τις 5 competition παθολογίες (default συμπεριφορά)")
    ap.add_argument("--per-class", type=int, default=3,
                    help="πόσα positive cases ανά κλάση (default 3)")
    ap.add_argument("--rank-by", default="mean", choices=["a", "b", "mean"],
                    help="κατά ποιου μοντέλου το confidence να γίνει η κατάταξη")
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="κράτα μόνο cases με prob >= αυτό (μετά την κατάταξη)")
    ap.add_argument("--mask-corners", type=float, default=None,
                    help="frac (π.χ. 0.18) -> μασκάρει πάνω γωνίες (burned-in text)")
    ap.add_argument("--center-crop", type=float, default=None,
                    help="frac (π.χ. 0.85) -> κρατάει κεντρικό crop πριν το resize")
    ap.add_argument("--target-layer", default=None)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--outdir", default="gradcam_curated_out")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    label_a = args.variant_a or args.label_a
    label_b = args.variant_b or args.label_b
    print(f"A = {label_a}   B = {label_b}")
    if args.mask_corners:
        print(f"[preproc] mask-corners frac={args.mask_corners}")
    if args.center_crop:
        print(f"[preproc] center-crop frac={args.center_crop}")

    # ---- κλάσεις ----
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",")]
        for c in classes:
            if c not in PATHOLOGIES:
                raise ValueError(f"Άγνωστη παθολογία: '{c}'")
    else:
        classes = COMPETITION  # default

    # ---- valid.csv ----
    data_root = os.path.expanduser(args.data_root)
    valid_csv = args.valid_csv or os.path.join(data_root, "CheXpert-v1.0-small", "valid.csv")
    valid_csv = os.path.expanduser(valid_csv)
    if not os.path.exists(valid_csv):
        raise FileNotFoundError(f"Δεν βρέθηκε valid.csv: {valid_csv}\n"
                                f"Δώσε σωστό --data-root ή --valid-csv.")
    df = pd.read_csv(valid_csv)
    # ground truth: το επίσημο valid είναι expert-labeled 0/1 (χωρίς uncertains).
    gt = df[PATHOLOGIES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    print(f"valid.csv: {len(df)} εικόνες\n")

    # ---- μοντέλα + cam engines ----
    model_a = build_model(args.ckpt_a, device, label=label_a)
    model_b = build_model(args.ckpt_b, device, label=label_b)
    cam_a = GradCAM(model_a, resolve_target_layer(model_a, args.target_layer))
    cam_b = GradCAM(model_b, resolve_target_layer(model_b, args.target_layer))
    name2idx = {n: i for i, n in enumerate(PATHOLOGIES)}

    # ---- ΒΗΜΑ 1: probs για ΟΛΟ το valid, και τα δύο μοντέλα (no grad) ----
    print("Inference σε όλο το valid set (για κατάταξη)...")
    probs_a = np.zeros((len(df), len(PATHOLOGIES)), dtype=np.float32)
    probs_b = np.zeros((len(df), len(PATHOLOGIES)), dtype=np.float32)
    img_paths = [resolve_image_path(data_root, p) for p in df["Path"].values]
    for i, ip in enumerate(img_paths):
        _, x = preprocess(Image.open(ip), args.img_size,
                          args.mask_corners, args.center_crop)
        x = x.to(device)
        probs_a[i] = predict_probs(model_a, x)
        probs_b[i] = predict_probs(model_b, x)
    print("  ...ολοκληρώθηκε.\n")

    rank_probs = {"a": probs_a, "b": probs_b, "mean": (probs_a + probs_b) / 2.0}[args.rank_by]

    # ---- ΒΗΜΑ 2 & 3: ανά κλάση -> positives -> top-N -> Grad-CAM -> figure ----
    for cname in classes:
        ci = name2idx[cname]
        pos_idx = np.where(gt[cname].values == 1.0)[0]
        if len(pos_idx) == 0:
            print(f"[skip] {cname}: 0 ground-truth positives στο valid set.")
            continue

        order = pos_idx[np.argsort(-rank_probs[pos_idx, ci])]  # ταξινόμηση κατά confidence
        if args.min_prob > 0.0:
            order = [i for i in order if rank_probs[i, ci] >= args.min_prob]
        chosen_idx = list(order[:args.per_class])
        if not chosen_idx:
            print(f"[skip] {cname}: κανένα positive πάνω από min-prob={args.min_prob}.")
            continue

        cases = []
        for i in chosen_idx:
            disp, x = preprocess(Image.open(img_paths[i]), args.img_size,
                                 args.mask_corners, args.center_crop)
            x = x.to(device)
            ca, pa = cam_a(x, ci)
            cb, pb = cam_b(x, ci)
            cases.append({"id": patient_id(df["Path"].values[i]),
                          "disp": disp, "cam_a": ca, "pa": pa, "cam_b": cb, "pb": pb})

        out_name = f"curated_{label_a}_vs_{label_b}_{_slug(cname)}.png"
        out_path = os.path.join(args.outdir, out_name)
        save_class_figure(cname, cases, label_a, label_b, out_path)

        summary = "  ".join(f"{c['id']}({label_a}={c['pa']:.2f},{label_b}={c['pb']:.2f})"
                            for c in cases)
        print(f"[OK] {cname:18s} -> {out_name}")
        print(f"      {summary}")

    print(f"\nΈτοιμα. Αποθηκεύτηκαν στο: {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
