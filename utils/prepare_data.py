"""
Run this once to download and set up the dataset.

Dataset sources (pick ONE):
  Option A — Kaggle (recommended, easiest):
    kaggle datasets download -d warcoder/tyre-quality-classification
  Option B — Mendeley TyreNet (best quality, manual download):
    https://data.mendeley.com/datasets/32b5vfj6tc/1

After downloading, run:
    python utils/prepare_data.py
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

RAW_DIR = Path("data/raw")
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
TEST_DIR = Path("data/test")

CLASSES = ["good", "defective"]
SPLIT = (0.70, 0.15, 0.15)
IMG_SIZE = (224, 224)
SEED = 42


def setup_dirs():
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in CLASSES:
            (split / cls).mkdir(parents=True, exist_ok=True)
    print("Directory structure created.")


def find_images(directory: Path) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in directory.rglob("*") if p.suffix.lower() in exts]


def split_and_copy(class_name: str, images: list):
    random.seed(SEED)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    dirs = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}

    for split_name, imgs in splits.items():
        dest_dir = dirs[split_name] / class_name
        for i, img_path in enumerate(tqdm(imgs, desc=f"{class_name}/{split_name}")):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                dest = dest_dir / f"{class_name}_{i:04d}.jpg"
                img.save(dest, "JPEG", quality=95)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    print(f"{class_name}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


def main():
    if not RAW_DIR.exists():
        print(f"""
No raw data found at {RAW_DIR}/

To get the dataset:

  OPTION A — Kaggle (easiest):
    1. Install kaggle CLI:  pip install kaggle
    2. Set up API key:      https://www.kaggle.com/settings → Create New Token
    3. Place kaggle.json in ~/.kaggle/
    4. Run:
         kaggle datasets download -d warcoder/tyre-quality-classification
         unzip tyre-quality-classification.zip -d data/raw/

  OPTION B — Mendeley TyreNet (1,698 real service station images):
    1. Go to: https://data.mendeley.com/datasets/32b5vfj6tc/1
    2. Download and extract to data/raw/
    3. Ensure structure:
         data/raw/good/   ← good tyre images
         data/raw/defective/ ← defective tyre images

Then re-run: python utils/prepare_data.py
""")
        return

    setup_dirs()

    for cls in CLASSES:
        src = RAW_DIR / cls
        if not src.exists():
            print(f"Warning: {src} not found — check your folder names match 'good' and 'defective'")
            continue
        images = find_images(src)
        print(f"Found {len(images)} images in {src}")
        split_and_copy(cls, images)

    total_train = sum(len(list((TRAIN_DIR / c).iterdir())) for c in CLASSES)
    total_val = sum(len(list((VAL_DIR / c).iterdir())) for c in CLASSES)
    total_test = sum(len(list((TEST_DIR / c).iterdir())) for c in CLASSES)
    print(f"\nDataset ready: {total_train} train / {total_val} val / {total_test} test")


if __name__ == "__main__":
    main()
