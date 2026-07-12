"""
calibration data για INT8 quantization με polygraphy.
Δίνει batches πραγματικών εικόνων ώστε το TensorRT να μάθει τα INT8 scales.
"""
import os, glob, random
import numpy as np
from PIL import Image
from torchvision import transforms


IMAGES_DIR = os.path.expanduser("/home/user/CXpertData/train")
N_CALIB    = 300     # 200-500 είναι τυπικό
BATCH      = 8
SEED       = 42

_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                                   # /255, CHW, RGB
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),             # ImageNet
])

def _paths():
    p = glob.glob(os.path.join(IMAGES_DIR, "**", "*_frontal.jpg"), recursive=True)
    if not p:  # fallback σε οποιοδήποτε jpg
        p = glob.glob(os.path.join(IMAGES_DIR, "**", "*.jpg"), recursive=True)
    random.Random(SEED).shuffle(p)
    return p[:N_CALIB]

def load_data():   # το polygraphy ψάχνει αυτό το όνομα by default
    paths = _paths()
    assert paths, f"Καμία εικόνα στο {IMAGES_DIR} — έλεγξε το path."
    print(f"[calib] {len(paths)} εικόνες, batch={BATCH}")
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        batch = np.stack([_tf(Image.open(x).convert("RGB")).numpy() for x in chunk])
        yield {"input": batch.astype(np.float32)}