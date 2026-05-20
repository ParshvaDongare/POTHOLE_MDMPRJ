import os
import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pothole_detection_pipeline as pipeline
from pathlib import Path

from fastapi.responses import FileResponse

app = FastAPI(title="Pothole Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# Serve all static files from Vite build except when we just hit root directly
frontend_dir = Path(__file__).resolve().parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dir / "assets")), name="assets")

@app.get("/")
async def serve_frontend():
    if frontend_dir.exists():
        return FileResponse(str(frontend_dir / "index.html"))
    return {"error": "Frontend not built yet. Run npm run build in frontend folder."}

MODEL_PATH = Path(__file__).resolve().parent / "best.pt"
yolo_model = None
midas_model = None
midas_transform = None
device = None

@app.on_event("startup")
def load_models():
    global yolo_model, midas_model, midas_transform, device
    print("Loading models into FastAPI server...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = pipeline.load_trained_model(MODEL_PATH)
    midas_model, midas_transform = pipeline.load_midas_model(device)
    print("Models loaded successfully!")

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        return {"error": "Invalid image format"}

    image_height, image_width = image_bgr.shape[:2]

    # 1. YOLO segmentation
    results = yolo_model.predict(
        source=image_bgr, 
        conf=pipeline.DEFAULT_CONFIDENCE_THRESHOLD, 
        save=False, 
        verbose=False
    )
    detection_result = results[0]

    # 2. Extract features
    pothole_data = pipeline.extract_pothole_features(detection_result, image_bgr)
    
    # 3. Add MiDaS depth
    pothole_data, _ = pipeline.add_depth_information(
        image_bgr, pothole_data, midas_model, midas_transform, device
    )
    
    # 4. Severity computation
    pothole_data = pipeline.assign_severity_labels(pothole_data, image_width, image_height)
    
    # 5. Build clean JSON output
    clean_potholes = []
    total_area_ratio = 0.0
    
    for pothole in pothole_data:
        mask = pothole.get("mask")
        polygon = []
        if mask is not None and np.count_nonzero(mask) > 0:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                # Keep original coordinates for drawing on full-size canvas
                simplified = pipeline.approximate_contour(contour)
                polygon = [{"x": int(pt[0][0]), "y": int(pt[0][1])} for pt in simplified]
        
        if not polygon: # Fallback
            x1, y1, x2, y2 = pothole["bbox"]
            polygon = [{"x": x1, "y": y1}, {"x": x2, "y": y1}, {"x": x2, "y": y2}, {"x": x1, "y": y2}]

        total_area_ratio += float(pothole.get("area_ratio", 0.0))
        
        clean_potholes.append({
            "id": pothole["id"],
            "severity": pothole.get("severity", "Low"),
            "severity_score": float(pothole.get("severity_score", 0.0)),
            "area_ratio": float(pothole.get("area_ratio", 0.0)),
            "size_label": pothole.get("size_label", "Unknown"),
            "confidence": float(pothole.get("confidence", 0.0)),
            "normalized_depth": float(pothole.get("normalized_depth", 0.0)),
            "raw_depth": float(pothole.get("raw_depth", 0.0)),
            "polygon": polygon
        })

    road_condition = pipeline.summarize_road_condition(pothole_data)
    average_area_ratio = total_area_ratio / max(1, len(pothole_data))
    high_count = sum(1 for p in clean_potholes if p["severity"] == "High")

    return {
        "road_condition": road_condition,
        "summary": {
            "pothole_count": len(clean_potholes),
            "average_area_ratio": average_area_ratio,
            "high_severity_count": high_count
        },
        "potholes": clean_potholes
    }
