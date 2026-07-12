"""
build_efficiency_table.py
---------------------------------------------------------------
Efficiency σύγκριση: teacher (ViT-B/16) vs student (EfficientNet-B0).
Αναφέρει: #params, GMACs, GFLOPs (=2xGMACs), και on-disk μέγεθος για όποια
checkpoints/engines του δώσεις.

ΣΗΜΑΝΤΙΚΟ: params & FLOPs εξαρτώνται ΜΟΝΟ από την αρχιτεκτονική (όχι τα βάρη),
οπότε δεν φορτώνουμε checkpoint γι' αυτά. Τα file sizes διαβάζονται από τα
πραγματικά αρχεία που περνάς στο --files.

Εγκατάσταση (μία φορά, μέσα στο chexpert env):
    pip install fvcore

Χρήση:
    python build_efficiency_table.py \
        --files student_ckpt=checkpoints/efficientnet-b0-epoch-5.pth \
                student_fp32=engines/baseline_fp32.engine \
                student_fp16=engines/baseline_fp16.engine \
                student_int8=engines/baseline_int8.engine \
                teacher_ckpt=checkpoints/vit_b16_chexpert_best.pth
"""
import argparse
import os
import torch
import timm

NUM_CLASSES = 14


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_macs(model, img_size=224):
    """Επιστρέφει MACs (fvcore: 1 MAC == 1 'flop' στη μέτρησή του)."""
    x = torch.randn(1, 3, img_size, img_size)
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)   # τα gelu/softmax/layernorm είναι
        fca.uncalled_modules_warnings(False)  # αμελητέα -> τα σιωπούμε
        return fca.total()  # <-- MACs
    except Exception as e:
        print(f"[!] fvcore απέτυχε ({e}).")
        print("    Δοκίμασε: pip install fvcore   (ή ptflops ως εναλλακτική)")
        return None


def human(nbytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def build(kind):
    if kind == "teacher":
        return timm.create_model("vit_base_patch16_224", pretrained=False,
                                 num_classes=NUM_CLASSES).eval()
    return timm.create_model("efficientnet_b0", pretrained=False,
                             num_classes=NUM_CLASSES).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--files", nargs="*", default=[],
                    help="ζευγάρια label=path (π.χ. student_int8=engines/baseline_int8.engine)")
    args = ap.parse_args()

    # ---- Params + FLOPs (μόνο αρχιτεκτονική) ----
    print(f"\n{'model':<22}{'params (M)':>12}{'GMACs':>9}{'GFLOPs':>9}")
    print("-" * 52)
    rows = {}
    for key, disp in [("teacher", "ViT-B/16 (teacher)"),
                      ("student", "EffNet-B0 (student)")]:
        m = build(key)
        total, _ = count_params(m)
        macs = count_macs(m, args.img_size)
        gmacs = macs / 1e9 if macs else float("nan")
        print(f"{disp:<22}{total/1e6:>12.2f}{gmacs:>9.2f}{2*gmacs:>9.2f}")
        rows[key] = (total, gmacs)

    if "teacher" in rows and "student" in rows:
        pr = rows["teacher"][0] / rows["student"][0]
        fr = rows["teacher"][1] / rows["student"][1]
        print("-" * 52)
        print(f"{'ratio (T/S)':<22}{pr:>12.1f}{fr:>9.1f}{fr:>9.1f}"
              f"   (x μικρότερος ο student)")

    # ---- File sizes ----
    if args.files:
        print(f"\n{'artifact':<24}{'size':>12}")
        print("-" * 36)
        for item in args.files:
            if "=" not in item:
                continue
            label, path = item.split("=", 1)
            path = os.path.expanduser(path)
            if os.path.exists(path):
                print(f"{label:<24}{human(os.path.getsize(path)):>12}")
            else:
                print(f"{label:<24}{'ΔΕΝ ΒΡΕΘΗΚΕ':>12}   ({path})")


if __name__ == "__main__":
    main()
