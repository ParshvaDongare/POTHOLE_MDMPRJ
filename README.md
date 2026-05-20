# Pothole Detection & Road Damage Analysis System

A full-stack AI-powered pothole detection web application using YOLOv8 segmentation, Depth Anything V2 depth estimation, and a React + FastAPI interface. Upload a road image and get pothole detection, severity scoring, depth analysis, and a road condition summary.

## Features

- Real-time pothole detection via YOLOv8 segmentation
- Depth estimation using Depth Anything V2
- Severity scoring based on area, depth, and shape metrics
- Road condition summary for each uploaded image
- Polygon overlays for pothole visualization
- FastAPI backend with a React frontend
- Extended v3 analysis pipeline with clustering, ANN/CNN models, and generative augmentation

## Project Structure

```text
DSprj3.0/
|-- backend.py                        # FastAPI server using the v3 pipeline
|-- pothole_detection_pipeline.py     # Compatibility shim re-exporting v3
|-- pothole_detection_pipeline_v3.py  # Canonical latest pipeline
|-- best.pt                           # Trained YOLOv8 segmentation weights
|-- yolov8s-seg.pt                    # YOLOv8s-seg base model
|-- sam_vit_b_01ec64.pth              # SAM checkpoint for XML -> segmentation conversion
|-- requirements.txt                  # Python dependencies
|-- nixpacks.toml                     # Deployment config
|-- frontend/                         # Vite + React frontend
|-- images/                           # Dataset images
|-- annotations/                      # Pascal VOC XML annotations
`-- PROJECT_DOCUMENTATION.md          # Extended project notes
```

## Setup

### Prerequisites

- Python 3.12 recommended
- Node.js 18+

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### Run the backend

```bash
uvicorn backend:app --host 0.0.0.0 --port 8080
```

Open [http://localhost:8080](http://localhost:8080).

## API

### `POST /detect`

Send a `multipart/form-data` request with the `image` field.

Example response:

```json
{
  "road_condition": "Poor",
  "summary": {
    "pothole_count": 3,
    "average_area_ratio": 0.042,
    "high_severity_count": 1
  },
  "potholes": [
    {
      "id": 1,
      "severity": "High",
      "severity_score": 0.78,
      "area_ratio": 0.06,
      "size_label": "Large",
      "confidence": 0.91,
      "normalized_depth": 0.65,
      "raw_depth": 142.3,
      "polygon": [{"x": 120, "y": 340}]
    }
  ]
}
```

## Model Stack

| Component | Model | Purpose |
|---|---|---|
| Detection | YOLOv8s-seg (`best.pt`) | Pothole localization and segmentation |
| Depth | Depth Anything V2 Small | Monocular depth estimation |
| Severity | Rule-based scoring | Area + depth + shape severity labeling |
| Extended analysis | SVM, RF, K-Means, DBSCAN, MLP, EfficientNetV2-S, VAE | Research and downstream analysis |

## Notes

- `backend.py` now uses `pothole_detection_pipeline_v3.py` as the active pipeline.
- `pothole_detection_pipeline.py` is kept as a compatibility alias for older imports.
- The SAM checkpoint is only needed for XML-to-YOLO segmentation label generation and training workflows.
- The v3 pipeline expects Python 3.12 because TensorFlow support is limited on newer versions.

## Author

Parshva Dongare  
[GitHub](https://github.com/ParshvaDongare)
