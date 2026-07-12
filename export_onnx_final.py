"""
export_onnx_final.py
---------------------------------------------------------------
Τελικό export του STUDENT (timm EfficientNet-B0, 14-class CheXpert) -> ONNX,
με parity check ΠΑΝΩ ΣΕ ΠΡΑΓΜΑΤΙΚΗ ΑΚΤΙΝΟΓΡΑΦΙΑ (όχι θόρυβο).

Αντικαθιστά το παλιό export_efficientnet_onnx.py.

Χρήση (με μια πραγματική εικόνα από το CheXpert για να επικυρωθεί σωστά):
    python export_onnx_final.py \
        --ckpt efficientnet-b0-epoch-5.pth \
        --onnx efficientnet_b0_chexpert.onnx \
        --image /home/user/archive/valid/patient64541/study1/view1_frontal.jpg
"""
import argparse
import numpy as np
import torch
import timm

PATHOLOGIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        sd = ckpt
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def build_model(ckpt_path):
    model = timm.create_model("efficientnet_b0", pretrained=False,
                              num_classes=len(PATHOLOGIES), exportable=True)
    missing, unexpected = model.load_state_dict(load_state_dict(ckpt_path), strict=False)
    if missing or unexpected:
        print(f"[!] missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[OK] Το state_dict ταιριάζει τέλεια με την αρχιτεκτονική.")
    return model.eval()


def real_image(path, img_size=224):
    from PIL import Image
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx", default="efficientnet_b0_chexpert.onnx")
    ap.add_argument("--image", default=None,
                    help="Πραγματική ακτινογραφία CheXpert για το parity check")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = build_model(args.ckpt)

    # ---- Export ----
    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    kw = dict(input_names=["input"], output_names=["logits"],
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
              opset_version=args.opset, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, args.onnx, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, args.onnx, **kw)
    print(f"[OK] Εξήχθη ONNX -> {args.onnx}  (opset {args.opset})")

    import onnx
    onnx.checker.check_model(onnx.load(args.onnx))
    print("[OK] onnx.checker: το γράφημα είναι έγκυρο.")

    # ---- Parity check ΠΑΝΩ ΣΕ ΠΡΑΓΜΑΤΙΚΗ ΕΙΚΟΝΑ ----
    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    if args.image:
        x = real_image(args.image, args.img_size)
        src = "πραγματική ακτινογραφία"
    else:
        # fallback: λείο, χαμηλόσυχνο input (proxy εικόνας) — ΟΧΙ καθαρός θόρυβος
        import torch.nn.functional as F
        torch.manual_seed(0)
        x = F.interpolate(torch.randn(1, 3, 28, 28), size=(args.img_size, args.img_size),
                          mode="bicubic", align_corners=False)
        x = (x - x.mean()) / (x.std() + 1e-6)
        src = "smooth synthetic (χωρίς --image)"

    with torch.no_grad():
        t = model(x).numpy()
    o = sess.run(None, {"input": x.numpy()})[0]
    d_logits = float(np.abs(t - o).max())
    mag = float(np.abs(t).max())

    print(f"\n[parity] input: {src}")
    print(f"         max|logit| = {mag:.2f}   max|Δ logits| = {d_logits:.3e}")
    if d_logits < 1e-3 and mag < 50:
        print("[OK] ✓ Parity πέρασε σε ρεαλιστικό input — το ONNX είναι ΣΩΣΤΟ.")
        probs = 1.0 / (1.0 + np.exp(-np.clip(t[0], -30, 30)))
        top = np.argsort(-probs)[:5]
        print("     Top-5 προβλέψεις: " +
              ", ".join(f"{PATHOLOGIES[i]}={probs[i]:.2f}" for i in top))
    elif d_logits < 1e-3:
        print("[⚠] Parity ok αλλά μεγάλα logits — δες FP16 στο Jetson με προσοχή.")
    else:
        print("[!] Parity FAIL σε ρεαλιστικό input — στείλε το output.")
        if not args.image:
            print("    (Δοκίμασε ξανά με --image <πραγματική ακτινογραφία>.)")


if __name__ == "__main__":
    main()
