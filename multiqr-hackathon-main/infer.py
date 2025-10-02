import os, re, json, cv2
from pathlib import Path
from ultralytics import YOLO
from pyzbar.pyzbar import decode

# 1️⃣ Set project root
PROJECT_DIR = "/content/drive/MyDrive/multiqr-hackathon"

# 2️⃣ Try to detect dataset folder
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Try common layouts
infer_input = None
candidates = [
    os.path.join(DATA_DIR, "val_images", "images"),
    os.path.join(DATA_DIR, "val_images"),
    os.path.join(DATA_DIR, "images"),
]
for c in candidates:
    if os.path.exists(c):
        infer_input = c
        break

if not infer_input:
    raise FileNotFoundError(f"Could not find val_images in {DATA_DIR}. Check your folder structure!")

print("Inference input folder:", infer_input)

# 3️⃣ Weights
WEIGHTS_BEST = os.path.join(PROJECT_DIR, "runs/train_qr3/weights/best.pt")
print("Using weights:", WEIGHTS_BEST)

# 4️⃣ Outputs
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
SUB_DET = os.path.join(OUTPUTS_DIR, "submission_detection_1.json")
SUB_DEC = os.path.join(OUTPUTS_DIR, "submission_decoding_2.json")

# 5️⃣ QR classifier helper
def classify_qr(value: str) -> str:
    if not value: return "unknown"
    v = value.strip().upper()
    if (v and v[0].isdigit()) or v.startswith("B"):
        return "batch_number"
    if "MFR" in v or "MANU" in v or v.startswith("MF"):
        return "manufacturer"
    if "DIST" in v or re.match(r"^D\d+$", v):
        return "distributor"
    if "REG" in v or "FDA" in v or "DRUG" in v:
        return "regulator"
    return "unknown"

# 6️⃣ Load model
model = YOLO(WEIGHTS_BEST)
CONF = 0.35
PAD = 8

preds = []

# 7️⃣ Run inference
for img_file in sorted(os.listdir(infer_input)):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img_path = os.path.join(infer_input, img_file)
    img = cv2.imread(img_path)
    if img is None:
        continue

    res = model(img_path, conf=CONF, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, 'xyxy') else []
    qrs = []

    for b in boxes:
        x1,y1,x2,y2 = map(int, b.tolist())
        entry = {"bbox":[float(x1), float(y1), float(x2), float(y2)]}
        decoded_val = ""

        h, w = img.shape[:2]
        x1c = max(0, x1-PAD); y1c = max(0, y1-PAD)
        x2c = min(w-1, x2+PAD); y2c = min(h-1, y2+PAD)
        crop = img[y1c:y2c, x1c:x2c]

        try:
            db = decode(crop)
            if db:
                decoded_val = db[0].data.decode("utf-8")

            if not decoded_val:
                detector = cv2.QRCodeDetector()
                val, pts, _ = detector.detectAndDecode(crop)
                if val:
                    decoded_val = val

            if not decoded_val and crop.size > 0:
                h_c, w_c = crop.shape[:2]
                if min(h_c,w_c) < 200:
                    scale = int(max(1, 200/min(h_c,w_c)))
                    big = cv2.resize(crop, (w_c*scale, h_c*scale), interpolation=cv2.INTER_LINEAR)
                    db2 = decode(big)
                    if db2:
                        decoded_val = db2[0].data.decode("utf-8")
        except:
            decoded_val = ""

        if decoded_val:
            entry["value"] = decoded_val
            entry["type"] = classify_qr(decoded_val)
        qrs.append(entry)

    preds.append({"image_id": os.path.splitext(img_file)[0], "qrs": qrs})

# 8️⃣ Save JSON outputs
with open(SUB_DET, "w") as f:
    json.dump(preds, f, indent=4)
print("Saved detection JSON:", SUB_DET)

with open(SUB_DEC, "w") as f:
    json.dump(preds, f, indent=4)
print("Saved decoding JSON:", SUB_DEC)

# 9️⃣ Save sample visualizations
VIS_DIR = os.path.join(OUTPUTS_DIR, "visuals")
os.makedirs(VIS_DIR, exist_ok=True)

for p in preds[:10]:
    candidates = list(Path(infer_input).glob(p["image_id"]+"*"))
    if not candidates: continue
    img = cv2.imread(str(candidates[0]))
    for q in p["qrs"]:
        x1,y1,x2,y2 = map(int, q["bbox"])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        lbl = q.get("value", "")
        if lbl:
            cv2.putText(img, lbl[:30], (x1, max(y1,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
    cv2.imwrite(os.path.join(VIS_DIR, p["image_id"]+".jpg"), img)

print("Saved sample visualizations:", VIS_DIR)
