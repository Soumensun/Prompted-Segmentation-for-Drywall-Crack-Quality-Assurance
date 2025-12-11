# split_dataset.py
import os, random
from pathlib import Path

# --- CONFIG ---
DATA_ROOT = Path("/content/data/cracks")     # <-- change if needed
IMG_DIR = DATA_ROOT / "images"
MASK_DIR = DATA_ROOT / "masks"
SPLITS = (0.8, 0.1, 0.1)           # train, val, test
SEED = 42
# ---------------

random.seed(SEED)
img_files = sorted([
    f for f in IMG_DIR.iterdir()
    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
])
n = len(img_files)
n_train = int(SPLITS[0]*n)
n_val   = int(SPLITS[1]*n)
n_test  = n - n_train - n_val

random.shuffle(img_files)
splits = {
    "train": img_files[:n_train],
    "val":   img_files[n_train:n_train+n_val],
    "test":  img_files[n_train+n_val:]
}

for split_name, files in splits.items():
    txt_path = DATA_ROOT / f"{split_name}.txt"
    with open(txt_path, "w") as f:
        for img in files:
            mask_name = img.with_suffix(".png").name
            line = f"images/{img.name} masks/{mask_name}\n"
            f.write(line)
    print(f"Wrote {len(files)} pairs to {txt_path}")

print(f"Total images: {n} | Train {n_train} | Val {n_val} | Test {n_test}")
