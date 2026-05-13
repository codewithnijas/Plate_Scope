import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import io
import re
import cv2
import numpy as np
import pandas as pd
import easyocr
reader = easyocr.Reader(['en'], gpu=False)

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from PIL import Image

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Plate Detection & Owner Lookup API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static/templates")

# ── Load model once at startup ────────────────────────────────────────────────
model = YOLO("best.pt")

# ── In-memory vehicle database ────────────────────────────────────────────────
_DB = pd.DataFrame({
    "reg_number":  ["KL07AB1234", "DL7CN5617", "MH20EJ0364", "TN99F2378", "KA05MK7788"],
    "owner_name":  ["Rahul Menon", "Anjali Sharma", "Suresh Patil", "Meera Krishnan", "Arjun Gowda"],
    "house_name":  ["Green Villa", "Shanti Nivas", "Sai Residency", "Blue Haven", "Lakshmi Nilayam"],
    "place":       ["Ernakulam", "Delhi", "Pune", "Chennai", "Bengaluru"],
    "phone":       ["9876543210", "9123456780", "9988776655", "9765432100", "9654321098"],
})


# ── Helper: process one image ─────────────────────────────────────────────────
def detect_and_lookup(img_bgr: np.ndarray) -> dict:
    results = model(img_bgr, conf=0.1)
    detections = []

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            plate_crop = img_bgr[y1:y2, x1:x2]

            # Pre-process for OCR
            plate_up = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(plate_up, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            plate_large = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            results_ocr = reader.readtext(plate_large, detail=0)
            raw_text = ' '.join(results_ocr)
            plate_text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())

            # DB lookup
            match = _DB[_DB["reg_number"] == plate_text]
            owner_info = match.to_dict(orient="records")[0] if not match.empty else None

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "raw_ocr": raw_text.strip(),
                "plate_text": plate_text,
                "found": owner_info is not None,
                "owner": owner_info,
            })

    return {"detections": detections, "count": len(detections)}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)  # ← UNCHANGED preserves full quality
    if img_bgr.dtype != np.uint8:
      img_bgr = (img_bgr / 256).astype(np.uint8)  # 16-bit → 8-bit
    if len(img_bgr.shape) == 2:
      img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)  # grayscale → BGR
    elif img_bgr.shape[2] == 4:
      img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)  # RGBA → BGR

    result = detect_and_lookup(img_bgr)
    return JSONResponse(content=result)


@app.get("/vehicles")
async def list_vehicles():
    """Return all vehicles in the database."""
    return JSONResponse(content=_DB.to_dict(orient="records"))


@app.get("/vehicles/{reg_number}")
async def lookup_vehicle(reg_number: str):
    """Direct registration-number lookup."""
    match = _DB[_DB["reg_number"] == reg_number.upper()]
    if match.empty:
        raise HTTPException(status_code=404, detail="Vehicle not found.")
    return JSONResponse(content=match.to_dict(orient="records")[0])


@app.post("/vehicles")
async def add_vehicle(payload: dict):
    """Add a new vehicle record to the in-memory database."""
    global _DB
    required = {"reg_number", "owner_name", "house_name", "place", "phone"}
    if not required.issubset(payload):
        raise HTTPException(status_code=422, detail=f"Required fields: {required}")
    reg = payload["reg_number"].upper()
    if reg in _DB["reg_number"].values:
        raise HTTPException(status_code=409, detail="Registration number already exists.")
    _DB = pd.concat([_DB, pd.DataFrame([{**payload, "reg_number": reg}])], ignore_index=True)
    return JSONResponse(content={"message": "Vehicle added successfully.", "reg_number": reg})

@app.post("/detect-path")
async def detect_from_path(payload: dict):
    img_bgr = cv2.imread(payload["path"])
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not read file.")
    result = detect_and_lookup(img_bgr)
    return JSONResponse(content=result)