import os
import torch
import numpy as np
from tqdm import tqdm

PT_DIR = "/home/lhe09/projects/rsna_spine/outputs/roi_dataset"
OUT_DIR = "/home/lhe09/projects/rsna_spine/outputs/roi_dataset_npy"

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(os.listdir(PT_DIR))

N = len(files)

print("Total samples:", N)

# -----------------------------
# level mapping
# -----------------------------

level_map = {
    "L1/L2":0,
    "L2/L3":1,
    "L3/L4":2,
    "L4/L5":3,
    "L5/S1":4
}

# -----------------------------
# memmap arrays 생성
# -----------------------------

images = np.memmap(
    os.path.join(OUT_DIR, "images.npy"),
    dtype=np.float16,
    mode="w+",
    shape=(N,9,224,224)
)

labels = np.memmap(
    os.path.join(OUT_DIR, "labels.npy"),
    dtype=np.int64,
    mode="w+",
    shape=(N,5)
)

coords = np.memmap(
    os.path.join(OUT_DIR, "coords.npy"),
    dtype=np.float32,
    mode="w+",
    shape=(N,2)
)

levels = np.memmap(
    os.path.join(OUT_DIR, "levels.npy"),
    dtype=np.int64,
    mode="w+",
    shape=(N,)
)

# -----------------------------
# 변환 시작
# -----------------------------

for i,f in enumerate(tqdm(files)):

    path = os.path.join(PT_DIR, f)

    data = torch.load(path, map_location="cpu")

    # image
    img = data["image"].cpu().numpy().astype(np.float16)
    images[i] = img

    # label
    labels[i] = data["label"].cpu().numpy().astype(np.int64)

    # coord
    coords[i] = data["coord"].cpu().numpy().astype(np.float32)

    # level
    levels[i] = level_map[data["level"]]

# -----------------------------
# flush (중요)
# -----------------------------

images.flush()
labels.flush()
coords.flush()
levels.flush()

print("Conversion finished!")
