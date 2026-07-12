"""
gradcam_chexpert.py
---------------------------------------------------------------
Grad-CAM για τον STUDENT (timm EfficientNet-B0, 14-class CheXpert, multi-label).

Δείχνει ΠΟΙΑ περιοχή της ακτινογραφίας κοίταξε το μοντέλο για να αποφασίσει
για μια συγκεκριμένη παθολογία. Τρέχει ΜΟΝΟ στο laptop (δεν αγγίζει το Jetson).

ΓΙΑΤΙ per-class: το πρόβλημα είναι multi-label (14 ανεξάρτητα sigmoid outputs),
οπότε ΔΕΝ υπάρχει «η πρόβλεψη». Για κάθε εικόνα διαλέγουμε ποια/ποιες
παθολογίες θέλουμε να οπτικοποιήσουμε (default: top-3 πιο ενεργοποιημένες).

Το --tag σε βοηθά να τρέξεις το ΙΔΙΟ script για baseline και logitKD και να
βγάλεις ΣΥΓΚΡΙΣΙΜΕΣ εικόνες (δυνατό figure: το distilled «κοιτάζει» πιο σωστά).

Απαιτήσεις (ήδη στο chexpert env, εκτός ίσως matplotlib):
    pip install matplotlib

Χρήση:
    # μία εικόνα, top-3 παθολογίες, baseline student
    python gradcam_chexpert.py \
        --ckpt checkpoints/efficientnet-b0-epoch-5.pth \
        --image ~/chexpert-project/data/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg \
        --tag baseline

    # ρητά επιλεγμένες κλάσεις, distilled student
    python gradcam_chexpert.py \
        --ckpt checkpoints/logitKD_efficientnet_b0.pth \
        --image path/to/xray.jpg \
        --classes "Cardiomegaly,Edema,Pleural Effusion" \
        --tag logitKD

    # ολόκληρος φάκελος εικόνων, οι 5 competition κλάσεις
    python gradcam_chexpert.py \
        --ckpt checkpoints/efficientnet-b0-epoch-5.pth \
        --image ~/chexpert-project/data/CheXpert-v1.0-small/valid/patient64541/study1/ \
        --competition --tag baseline
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")  # headless save, χωρίς παράθυρο
import matplotlib.pyplot as plt


# ----- Οι 14 παθολογίες (ΙΔΙΑ σειρά με training/export/eval) -----
PATHOLOGIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
COMPETITION = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================ CHECKPOINT ============================
def load_state_dict(ckpt_path, device="cpu"):
    """Ίδια λογική με το export script: ξεδιπλώνει 'state_dict'/'model'/'module.'."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        sd = ckpt  # σκέτο state_dict (η δική σου περίπτωση)
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in sd.items()}


def build_model(ckpt_path, device):
    # exportable=False -> κανονικό SiLU (ίδιο με το training/eval· τα βάρη είναι ίδια ούτως ή άλλως)
    model = timm.create_model("efficientnet_b0", pretrained=False,
                              num_classes=len(PATHOLOGIES))
    sd = load_state_dict(ckpt_path, device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[!] MISSING keys:", missing)
    if unexpected:
        print("[!] UNEXPECTED keys:", unexpected)
    if not missing and not unexpected:
        print("[OK] Το state_dict ταιριάζει τέλεια με την αρχιτεκτονική.")
    return model.eval().to(device)


def resolve_target_layer(model, name=None):
    """
    Επιστρέφει το conv layer πάνω στο οποίο 'κρεμάμε' τα hooks.
    Default: conv_head (τελευταίο conv πριν το pooling -> feature map 7x7x1280).
    Εναλλακτικές που δίνουν επίσης καλά CAM: 'blocks' (τελευταίο MBConv stage).
    """
    if name:
        # π.χ. --target-layer conv_head  ή  --target-layer blocks
        mod = model
        for part in name.split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        print(f"[cam] target layer (user): {name} -> {mod.__class__.__name__}")
        return mod
    # auto: προτίμησε conv_head, αλλιώς το τελευταίο block
    if hasattr(model, "conv_head"):
        print("[cam] target layer (auto): conv_head")
        return model.conv_head
    print("[cam] target layer (auto): blocks[-1]")
    return model.blocks[-1]


# ============================ GRAD-CAM ============================
class GradCAM:
    """
    Κλασικό Grad-CAM (Selvaraju et al. 2017) με forward+backward hooks.
    - forward hook  : αποθηκεύει τα activations A του target layer  [1, C, h, w]
    - backward hook : αποθηκεύει τα gradients dScore/dA             [1, C, h, w]
    weights = global-average-pool(gradients)  ->  σημασία κάθε feature map
    CAM = ReLU( Σ_k weight_k * A_k )  ->  upscale στο 224x224, normalize [0,1]
    """
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
        """x: [1,3,224,224] normalized tensor. Επιστρέφει (cam_2d[np], prob[float])."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                 # [1, 14] (raw logits)
        score = logits[0, class_idx]           # στοχεύουμε το logit ΤΗΣ κλάσης (όχι softmax)
        score.backward()

        grads = self.gradients                 # [1, C, h, w]
        acts = self.activations                # [1, C, h, w]
        weights = grads.mean(dim=(2, 3), keepdim=True)          # GAP -> [1, C, 1, 1]
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # [1, 1, h, w]
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # -> [0,1]
        prob = float(torch.sigmoid(logits)[0, class_idx])
        return cam.cpu().numpy(), prob


# ============================ IMAGE I/O ============================
def make_transforms(img_size=224):
    norm_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    # για την ΕΜΦΑΝΙΣΗ θέλουμε την εικόνα ΧΩΡΙΣ normalization, στο [0,1]
    disp_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return norm_tf, disp_tf


def gather_images(path):
    """Δέχεται είτε ένα αρχείο είτε φάκελο. Επιστρέφει λίστα από jpg/png paths."""
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return [path]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(path, "**", e), recursive=True))
    return sorted(files)


# ============================ ΟΝΟΜΑΤΑ ΑΡΧΕΙΩΝ (χωρίς overwrite) ============================
def _slug(s, sep=""):
    """Κράτα μόνο αλφαριθμητικά· τα υπόλοιπα -> sep (default: τίποτα)."""
    return "".join(ch if ch.isalnum() else sep for ch in s)


def path_signature(img_path, n=3):
    """Μοναδικό, αναγνώσιμο signature από τα τελευταία n path components
    (π.χ. patient64541-study1-view1-frontal). Έτσι ΔΕΝ γίνεται overwrite μεταξύ
    διαφορετικών ασθενών/studies που τυχαίνει να έχουν ίδιο όνομα εικόνας."""
    parts = [p for p in os.path.normpath(os.path.expanduser(img_path)).split(os.sep) if p]
    tail = parts[-n:] if len(parts) >= n else parts
    sig = os.path.splitext("_".join(tail))[0]  # πέτα την κατάληξη (.jpg) από το τελευταίο κομμάτι
    return _slug(sig, sep="-")


def classes_signature(chosen):
    """Signature από τις επιλεγμένες κλάσεις -> διαφορετικές κλάσεις = διαφορετικό αρχείο."""
    return "-".join(_slug(c) for c in chosen)


# ============================ PLOT ============================
def save_figure(disp_img, cams, titles, out_path):
    """disp_img: HxWx3 [0,1]. cams: λίστα από 2D arrays. Σώζει πρωτότυπο + overlays."""
    n = len(cams)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4.4))
    if n == 0:
        axes = [axes]

    axes[0].imshow(disp_img)
    axes[0].set_title("Original X-ray", fontsize=11)
    axes[0].axis("off")

    for ax, cam, title in zip(axes[1:], cams, titles):
        ax.imshow(disp_img)
        ax.imshow(cam, cmap="jet", alpha=0.45)  # heatmap overlay
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================ MAIN ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="π.χ. efficientnet-b0-epoch-5.pth")
    ap.add_argument("--image", required=True, help="αρχείο εικόνας Ή φάκελος")
    ap.add_argument("--tag", default="student",
                    help="custom prefix στα output filenames (π.χ. baseline / logitKD)")
    ap.add_argument("--variant", default=None,
                    choices=["baseline", "logitKD", "featureKD"],
                    help="γνωστή παραλλαγή· αν δοθεί, γίνεται το prefix (υπερισχύει του --tag). "
                         "featureKD = feature-level distillation.")
    ap.add_argument("--classes", default=None,
                    help="ρητή λίστα παθολογιών χωρισμένη με κόμμα "
                         "(π.χ. \"Cardiomegaly,Edema\"). Αν λείπει -> top-k.")
    ap.add_argument("--competition", action="store_true",
                    help="χρησιμοποίησε τις 5 competition παθολογίες")
    ap.add_argument("--topk", type=int, default=3,
                    help="πόσες κλάσεις να δείξεις όταν δεν δίνεις --classes")
    ap.add_argument("--target-layer", default=None,
                    help="όνομα layer για τα hooks (default: conv_head)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--outdir", default="gradcam_out")
    ap.add_argument("--cpu", action="store_true", help="ανάγκασε CPU")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    tag = args.variant or args.tag          # --variant υπερισχύει (baseline/logitKD/featureKD)
    print(f"Variant/tag: {tag}")

    model = build_model(args.ckpt, device)
    target_layer = resolve_target_layer(model, args.target_layer)
    cam_engine = GradCAM(model, target_layer)

    norm_tf, disp_tf = make_transforms(args.img_size)
    name2idx = {n: i for i, n in enumerate(PATHOLOGIES)}

    # ---- ποιες κλάσεις (αν είναι ρητές) ----
    fixed_classes = None
    if args.competition:
        fixed_classes = COMPETITION
    elif args.classes:
        fixed_classes = [c.strip() for c in args.classes.split(",")]
        for c in fixed_classes:
            if c not in name2idx:
                raise ValueError(f"Άγνωστη παθολογία: '{c}'.\nΕπιτρεπτές: {PATHOLOGIES}")

    images = gather_images(args.image)
    if not images:
        raise FileNotFoundError(f"Δεν βρέθηκαν εικόνες στο: {args.image}")
    print(f"Βρέθηκαν {len(images)} εικόνα(ες).\n")

    for img_path in images:
        pil = Image.open(img_path).convert("RGB")
        x = norm_tf(pil).unsqueeze(0).to(device)                    # [1,3,224,224]
        disp = disp_tf(pil).permute(1, 2, 0).numpy()                # HxWx3 [0,1]

        # πρώτα ένα forward για να δούμε τις πιθανότητες (χωρίς grad)
        with torch.no_grad():
            probs = torch.sigmoid(model(x))[0].cpu().numpy()

        if fixed_classes is not None:
            chosen = fixed_classes
        else:
            top = np.argsort(-probs)[:args.topk]                    # top-k πιο ενεργές
            chosen = [PATHOLOGIES[i] for i in top]

        cams, titles = [], []
        for cname in chosen:
            idx = name2idx[cname]
            cam, prob = cam_engine(x, idx)                          # ΕΝΑ backward ανά κλάση
            cams.append(cam)
            titles.append(f"{cname}\np={prob:.2f}")

        # ΜΟΝΑΔΙΚΟ όνομα: tag + (patient/study/view) + επιλεγμένες κλάσεις
        # -> διαφορετικός ασθενής Ή διαφορετικές κλάσεις = διαφορετικό αρχείο (όχι overwrite)
        out_name = f"{tag}_{path_signature(img_path)}_{classes_signature(chosen)}_gradcam.png"
        out_path = os.path.join(args.outdir, out_name)
        save_figure(disp, cams, titles, out_path)

        preds = ", ".join(f"{c}={probs[name2idx[c]]:.2f}" for c in chosen)
        print(f"[OK] {os.path.basename(img_path):40s} -> {out_name}")
        print(f"      {preds}")

    print(f"\nΈτοιμα. Αποθηκεύτηκαν στο: {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
