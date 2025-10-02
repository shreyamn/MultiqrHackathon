import json, os, cv2, numpy as np
from pathlib import Path

def iou(boxA, boxB):
    # boxes [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter/union if union>0 else 0

# load predictions
with open(SUB_DET) as f:
    preds = json.load(f)

# gt folder
GT_LABELS_DIR = os.path.join(DATA_DIR, "val_images", "labels")
total_pred = 0
tp = 0
for p in preds:
    img_id = p["image_id"]
    img_path_candidates = list(Path(INFER_INPUT).glob(img_id+"*"))
    if not img_path_candidates:
        continue
    img_path = str(img_path_candidates[0])
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    # load gt labels
    gt_file = os.path.join(GT_LABELS_DIR, img_id + ".txt")
    gts=[]
    if os.path.exists(gt_file):
        for ln in open(gt_file):
            parts = ln.strip().split()
            if len(parts) < 5: continue
            _, xc, yc, ww, hh = map(float, parts[:5])
            xc *= w; yc *= h; ww *= w; hh *= h
            x1 = max(0, xc - ww/2); y1 = max(0, yc - hh/2)
            x2 = min(w-1, xc + ww/2); y2 = min(h-1, yc + hh/2)
            gts.append([x1,y1,x2,y2])
    # compare preds
    for q in p["qrs"]:
        total_pred += 1
        box = q["bbox"]
        if any(iou(box, gt) >= 0.5 for gt in gts):
            tp += 1

precision = tp / total_pred if total_pred>0 else 0.0
print(f"Detected {total_pred} boxes, true positives (IoU>=0.5): {tp}, precision ≈ {precision:.3f}")
