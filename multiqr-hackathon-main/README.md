# Multi-QR Hackathon

This project detects **multiple QR codes** in medicine pack images using **YOLOv8**.  
It supports **training, inference, and evaluation** in a clean and modular setup.

> **Note:** All QR code annotations were created using [Roboflow](https://roboflow.com/) and exported in YOLO format.
# *This project was succesfully executed on Google Colab* #
---

## Project Structure

- `README.md` – Project description and usage guide  
- `requirements.txt` – Python dependencies  
- `dataset.yaml` – YOLO dataset configuration  
- `train.py` – Script to train the YOLO model  
- `infer.py` – Run inference on new images  
- `evaluate.py` – Evaluate model performance  
- `data/` – Dataset folder with sample images for quick testing  
  - `demo_images/` – Small sample images  
- `outputs/` – predictions, JSON files, and annotated visuals
  - `visuals/` - predicted QR-codes for images
  - `submission_detection_1.json`
  - `submission_decoding_2.json`
- `runs` -
    - `train_qr3` has `weights` and more graphs
- `src/`    
  - `utils.py` – Utility functions  

---
**Dataset structure:**  
- `train_images/images` → Training images  
- `train_images/labels` → Training labels  
- `val_images/images` → Validation images  
- `val_images/labels` → Validation labels  

---

## Setup

Clone the repository and navigate into the project directory. Create a Python virtual environment and install dependencies. The project requires a dataset in YOLO format, with images and labels organized into `train_images` and `val_images` folders. The dataset configuration is specified in `dataset.yaml`, which sets the paths, number of classes, and class names. If no dataset is available, sample images can be placed in `data/demo_images`.

---

- Each image must have a corresponding `.txt` label file.
- Each label line format: `<class_id> <x_center> <y_center> <width> <height>` (normalized to [0,1]).

---

### 2️⃣ Create `dataset.yaml`

Specify dataset paths, number of classes, and class names. Example:
```
yaml
path: data
train: train_images/images
val: val_images/images
nc: 1
names: ['qrcode']
```
## Training

Training is run via `train.py`. The default setup uses pretrained YOLOv8 weights (`yolov8n.pt`), runs for 60 epochs with image size 640 and batch size 16. Outputs such as model weights, logs, and metrics are stored under `multiqr-hackathon/runs/train_qr/`. Training uses images annotated with **Roboflow** to ensure accurate bounding boxes.
-`if running on bash` -
```
-python -m venv venv
-source venv/bin/activate  # Linux/Mac
-venv\Scripts\activate     # Windows
-pip install -r requirements.txt
```


## Inference

Inference can be run using `infer.py` on validation or demo images. The script detects all QR codes in the images, saves bounding box predictions in JSON format (`submission_detection_1.json`, `submission_decoding_2.json`) and saves sample annotated images in `outputs/visuals/`. This allows quick verification of model performance.
- run on colab or`for bash`
```
- python train.py --weights yolov8n.pt --data dataset.yaml --epochs 60 --imgsz 640 --batch 16 --project runs --name train_qr --augment
```



## Evaluation

Evaluation is done with `evaluate.py`, which compares model predictions against ground truth labels and calculates metrics such as **precision** (IoU ≥ 0.5). This provides a quantitative assessment of the detection performance.
 -`on bash`
  ```
 - python evaluate.py
  ```


## Outputs

After training and inference, the outputs include:

- `submission_detection_1.json` – Bounding boxes for detected QR codes  
- `submission_decoding_2.json` – Bounding boxes plus decoded QR values  
- `visuals/` – Annotated images showing QR code detection results
-`on bash`
```
-python infer.py --weights runs/train_qr3/weights/best.pt --source data/demo_images --output outputs --conf 0.35 --pad 8
 ```



## Quick Demo

To quickly test the pipeline, run inference on demo images placed in `data/demo_images`. This demonstrates the full detection workflow without requiring a full dataset.


## Workflow Summary

1. Clone the repository  
2. Install dependencies  
3. Prepare dataset (annotated via **Roboflow**)  
4. Train the YOLOv8 model  
5. Run inference on images  
6. Evaluate results using provided metrics  
7. Inspect outputs in the `outputs/` folder  

This modular workflow ensures that adding more data, retraining, or testing new images is straightforward and repeatable.
## Notes

- Ensure dataset follows YOLO format with corresponding labels for each image.  
- Use data augmentation for better detection accuracy.  
- Adjust training parameters based on available hardware.  
- QR code classification logic can be customized for your dataset.

