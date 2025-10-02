!apt-get update -qq
!apt-get install -y -qq libzbar0
from google.colab import drive
drive.mount('/content/drive')
import os, glob, shutil
PROJECT_DIR = "/content/drive/MyDrive/multiqr-hackathon"

os.makedirs(PROJECT_DIR, exist_ok=True)

import os, textwrap

# try expected layout
DATA_DIR = os.path.join(PROJECT_DIR, "data")
train_images = os.path.join(DATA_DIR, "train_images", "images")
val_images   = os.path.join(DATA_DIR, "val_images", "images")

# Fallback: if dataset has single folder 'images' and 'labels', try to detect
if not os.path.exists(train_images):
    # look for possible folders
    possible = []
    for root, dirs, files in os.walk(DATA_DIR):
        if 'images' in dirs and 'labels' in dirs:
            possible.append(root)
    if possible:
        # pick first
        base = possible[0]
        train_images = os.path.join(base, "images")
        val_images = train_images  # fallback if no val split
        print("Auto-detected images folder:", base)
    else:
        print("WARNING: Could not auto-detect train/val folders under", DATA_DIR)
        print("Make sure dataset structure is: data/train_images/images, data/train_images/labels, data/val_images/images, data/val_images/labels")
from pathlib import Path

def check_pairing(images_dir, labels_dir, limit_show=5):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    imgs = sorted([p.name for p in images_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    labs = sorted([p.name for p in labels_dir.glob("*.txt")])
    print(f"\n{images_dir} -> images: {len(imgs)}, labels: {len(labs)}")
    print(" sample images:", imgs[:limit_show])
    print(" sample labels:", labs[:limit_show])
    img_bases = {p.stem for p in images_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]}
    lab_bases = {p.stem for p in labels_dir.glob("*.txt")}
    missing_labels = sorted(list(img_bases - lab_bases))[:10]
    missing_images = sorted(list(lab_bases - img_bases))[:10]
    print(" missing labels for images (examples):", missing_labels)
    print(" missing images for labels (examples):", missing_images)

# try to run check for train and val if they exist
if os.path.exists(os.path.join(DATA_DIR, "train_images", "images")):
    check_pairing(os.path.join(DATA_DIR, "train_images", "images"), os.path.join(DATA_DIR, "train_images", "labels"))
if os.path.exists(os.path.join(DATA_DIR, "val_images", "images")):
    check_pairing(os.path.join(DATA_DIR, "val_images", "images"), os.path.join(DATA_DIR, "val_images", "labels"))
# WARNING: training can take long. Adjust epochs/batch for Colab time limits.
from ultralytics import YOLO
import os

WEIGHTS_PRETRAIN = "yolov8n.pt"   # use yolov8s.pt or yolov8m.pt if you have stronger GPU
PROJECT_RUNS = os.path.join(PROJECT_DIR, "runs")
os.makedirs(PROJECT_RUNS, exist_ok=True)

# training settings — tune as needed
epochs = 60
imgsz = 640
batch = 16
name = "train_qr"

print("Starting training (this prints Ultralytics logs)...")
model = YOLO(WEIGHTS_PRETRAIN)
model.train(
    data=os.path.join(PROJECT_DIR, "dataset.yaml"),
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    workers=4,
    project=PROJECT_RUNS,
    name=name,
    augment=True,   # enable augmentation
)
print("Training finished. Check runs at:", os.path.join(PROJECT_RUNS, name))
