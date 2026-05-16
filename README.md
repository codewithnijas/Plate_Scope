# 🚗 License Plate Detection & Owner Lookup

A computer vision web application that automatically detects vehicle license plates from uploaded images, extracts the plate number using OCR, and retrieves the registered owner's details from a database.

---

## How It Works

1. User uploads a car image via the web UI or API
2. **YOLOv8** detects the license plate region in the image
3. The plate crop is extracted, upscaled, and passed to **EasyOCR**
4. The recognized plate number is cleaned and matched against the vehicle database
5. Owner details are returned if a match is found

---

## Stack

| Component | Tool |
|-----------|------|
| Plate Detection | YOLOv8 (`best.pt`) |
| Text Recognition | EasyOCR |
| Backend API | FastAPI |
| Image Processing | OpenCV, NumPy |
| Data | Pandas (in-memory) |
| Frontend | HTML/JS (Jinja2) |

---

## Dataset

Training data sourced from Kaggle:  
🔗 [Car Number Plate Dataset — YOLO Format](https://www.kaggle.com/datasets/sujaymann/car-number-plate-dataset-yolo-format)

The dataset is pre-annotated in YOLO format with bounding boxes around license plates, ready to use directly for YOLOv8 training.

---

## Setup

### 1. Install dependencies
```bash
pip install ultralytics easyocr fastapi uvicorn opencv-python pandas pillow
```

### 2. Add your trained model
Place `best.pt` in the project root. To train your own, see the [Training](#training) section below.

### 3. Run the server
```bash
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Project Structure

```
project/
├── main.py                  # FastAPI app + detection logic
├── best.pt                  # YOLOv8 trained weights
├── train_plate_detector.ipynb  # Colab training notebook
└── static/
    └── templates/
        └── index.html       # Web UI
```

---



## Training

A ready-to-use Google Colab notebook (`train_plate_detector.ipynb`) is included.

**Steps:**
1. Upload the notebook to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Upload your dataset to Google Drive in this structure:
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```
4. Update `DATASET_DIR` in the notebook to point to your Drive folder
5. Run all cells — `best.pt` is saved back to your Drive automatically
6. Download `best.pt` and place it in the project root

The notebook also includes sample testing cells that run the full detection + OCR pipeline on validation images so you can verify quality before deploying.

---

## OCR Pipeline

The OCR step runs EasyOCR at **4 different scale factors** (2×, 3×, 4×, 6×) on each plate crop using Lanczos interpolation, and picks the result with the most characters. This handles small or low-resolution plate crops where a single scale can drop characters.

---

## Notes

- **Database is in-memory** — all records reset on server restart. For production, replace `_DB` with PostgreSQL, SQLite, or any persistent store.
- **OCR accuracy** is heavily dependent on input image resolution. Blurry or very small plates will produce unreliable reads.
