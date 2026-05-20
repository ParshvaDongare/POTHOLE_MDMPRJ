import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import pothole_detection_pipeline_v3 as pipeline
from road_agent import build_agent_assessment

app = FastAPI(title="Pothole Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dir / "assets")), name="assets")


@app.get("/")
async def serve_frontend():
    if frontend_dir.exists():
        return FileResponse(str(frontend_dir / "index.html"))
    return {"error": "Frontend not built yet. Run npm run build in frontend folder."}


MODEL_PATH = Path(__file__).resolve().parent / "best.pt"
LIVE_CONFIDENCE_THRESHOLD = 0.35
LIVE_ENSEMBLE_SIZES = [640, 1024, 1280]
yolo_model = None
midas_model = None
midas_transform = None
device = None


def run_live_detection_ensemble(model, image_bgr):
    """Favor recall for portal uploads by combining detections from multiple scales."""
    merged = []
    for imgsz in LIVE_ENSEMBLE_SIZES:
        results = model.predict(
            source=image_bgr,
            conf=LIVE_CONFIDENCE_THRESHOLD,
            imgsz=imgsz,
            save=False,
            verbose=False,
        )
        merged.extend(
            pipeline.extract_pothole_features(
                results[0],
                image_bgr,
                min_conf=LIVE_CONFIDENCE_THRESHOLD,
            )
        )
    return pipeline.remove_duplicate_detections(merged)


def remap_crop_detections(crop_detections, full_shape, top, left):
    full_h, full_w = full_shape[:2]
    remapped = []
    for pothole in crop_detections:
        bbox = pothole.get("bbox", [0, 0, 0, 0])
        new_entry = dict(pothole)
        new_entry["bbox"] = [
            int(bbox[0] + left),
            int(bbox[1] + top),
            int(bbox[2] + left),
            int(bbox[3] + top),
        ]
        mask = pothole.get("mask")
        if mask is not None:
            full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
            crop_h, crop_w = mask.shape[:2]
            full_mask[top:top + crop_h, left:left + crop_w] = mask.astype(np.uint8)
            new_entry["mask"] = full_mask
        remapped.append(new_entry)
    return remapped


def run_live_detection_with_distant_scan(model, image_bgr):
    """Combine full-frame detections with an upper-road crop pass for distant potholes."""
    merged = run_live_detection_ensemble(model, image_bgr)
    height, width = image_bgr.shape[:2]
    crop = image_bgr[0:max(int(height * 0.7), 1), 0:width]
    crop_detections = run_live_detection_ensemble(model, crop)
    merged.extend(remap_crop_detections(crop_detections, image_bgr.shape, 0, 0))
    return pipeline.remove_duplicate_detections(merged)


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

    pothole_data = run_live_detection_with_distant_scan(yolo_model, image_bgr)
    pothole_data, _ = pipeline.add_depth_information(
        image_bgr, pothole_data, midas_model, midas_transform, device
    )
    pothole_data = pipeline.assign_severity_labels(pothole_data, image_width, image_height)
    pothole_data = pipeline.filter_live_detections(pothole_data)

    clean_potholes = []
    total_area_ratio = 0.0

    for pothole in pothole_data:
        mask = pothole.get("mask")
        polygon = []
        if mask is not None and np.count_nonzero(mask) > 0:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                contour = max(contours, key=cv2.contourArea)
                simplified = pipeline.approximate_contour(contour)
                polygon = [{"x": int(pt[0][0]), "y": int(pt[0][1])} for pt in simplified]

        if not polygon:
            x1, y1, x2, y2 = pothole["bbox"]
            polygon = [
                {"x": x1, "y": y1},
                {"x": x2, "y": y1},
                {"x": x2, "y": y2},
                {"x": x1, "y": y2},
            ]

        total_area_ratio += float(pothole.get("area_ratio", 0.0))
        clean_potholes.append(
            {
                "id": pothole["id"],
                "severity": pothole.get("severity", "Low"),
                "severity_score": float(pothole.get("severity_score", 0.0)),
                "area_ratio": float(pothole.get("area_ratio", 0.0)),
                "size_label": pothole.get("size_label", "Unknown"),
                "confidence": float(pothole.get("confidence", 0.0)),
                "normalized_depth": float(pothole.get("normalized_depth", 0.0)),
                "raw_depth": float(pothole.get("raw_depth", 0.0)),
                "polygon": polygon,
            }
        )

    road_condition = pipeline.summarize_road_condition(pothole_data)
    average_area_ratio = total_area_ratio / max(1, len(pothole_data))
    high_count = sum(1 for pothole in clean_potholes if pothole["severity"] == "High")
    summary = {
        "pothole_count": len(clean_potholes),
        "average_area_ratio": average_area_ratio,
        "total_area_ratio": total_area_ratio,
        "high_severity_count": high_count,
    }
    agent_assessment = build_agent_assessment(clean_potholes, road_condition, summary)

    return {
        "road_condition": road_condition,
        "summary": summary,
        "agent_assessment": agent_assessment,
        "potholes": clean_potholes,
    }
