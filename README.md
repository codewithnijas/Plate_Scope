# 🚗 License Plate Detection & Owner Lookup

Detects license plates in images using YOLOv8, reads the plate text with EasyOCR, and looks up the owner from an in-memory database.

---

## Stack
- **YOLOv8** — plate detection (`best.pt`)
- **EasyOCR** — plate text recognition
- **FastAPI** — REST API
- **OpenCV / NumPy / Pandas**

---

## Setup

```bash
pip install ultralytics easyocr fastapi uvicorn opencv-python pandas pillow
```

Place your trained `best.pt` in the project root, then:

```bash
uvicorn main:app --reload
```

---

## Project Structure

```
project/
├── main.py
├── best.pt
└── static/
    └── templates/
        └── index.html
```

---

## Training

Use the provided `train_plate_detector.ipynb` in Google Colab (T4 GPU).  
After training, download `best.pt` from your Drive and replace the one in the project root.

---

## Notes
- Database is **in-memory** — data resets on restart. Replace `_DB` with a real DB for production.
- OCR accuracy depends on image resolution. Higher resolution = better results.
`