# 🛣️ Pothole Detection & Road Damage Analysis System

A full-stack AI-powered pothole detection web application using **YOLOv8 segmentation**, **MiDaS depth estimation**, and a **React + FastAPI** interface. Upload a road image and get instant pothole detection with severity scoring, depth analysis, and road condition assessment.

---

## 📸 Features

- **Real-time pothole detection** via YOLOv8 segmentation model
- **Depth estimation** using MiDaS (monocular depth) for 3D severity analysis
- **Severity scoring** — Low / Medium / High per pothole based on area + depth
- **Road condition summary** — Overall road health report
- **Polygon overlays** drawn on the original image
- **REST API** (FastAPI) with CORS support
- **React frontend** with live detection results

---

## 🗂️ Project Structure

```
DSprj3.0/
├── backend.py                        # FastAPI server — detection API
├── pothole_detection_pipeline.py     # Core pipeline (YOLO + MiDaS + severity)
├── pothole_detection_pipeline_v3.py  # Extended pipeline (EfficientNet, VAE, DBSCAN)
├── best.pt                           # Trained YOLOv8 segmentation weights
├── yolov8s-seg.pt                    # YOLOv8s-seg base model (auto-downloaded)
├── requirements.txt                  # Python dependencies
├── nixpacks.toml                     # Railway/Render deployment config
├── frontend/                         # Vite + React frontend
│   ├── src/
│   │   ├── App.jsx                   # Main UI component
│   │   └── index.css                 # Styles
│   ├── package.json
│   └── vite.config.js
├── images/                           # Dataset images (local only, gitignored)
├── annotations/                      # Pascal VOC XML annotations (local only)
└── PROJECT_DOCUMENTATION.md         # Full technical documentation
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Clone the repo
```bash
git clone https://github.com/ParshvaDongare/POTHOLE_MDMPRJ.git
cd POTHOLE_MDMPRJ
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install & build the frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Run the backend
```bash
uvicorn backend:app --host 0.0.0.0 --port 8080
```

Open **http://localhost:8080** in your browser.

---

## 🚀 API Usage

### `POST /detect`
Upload an image for pothole detection.

**Request:** `multipart/form-data` with field `image`

**Response:**
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
      "polygon": [{"x": 120, "y": 340}, ...]
    }
  ]
}
```

---

## 🧠 Model Architecture

| Component | Model | Purpose |
|---|---|---|
| Detection | YOLOv8s-seg (`best.pt`) | Pothole localization + segmentation |
| Depth | MiDaS DPT-Large | Monocular depth map estimation |
| Severity | Rule-based scoring | Area ratio + depth → severity label |
| Extended (v3) | EfficientNetV2-S + DBSCAN + VAE | Feature extraction, clustering, synthetic augmentation |

---

## 🌐 Deployment

This project is configured for **Railway** deployment via `nixpacks.toml`.

> ⚠️ **Note:** `sam_vit_b_01ec64.pth` (SAM model, 375MB) and the dataset (`images/`, `annotations/`) are excluded from the repo. The SAM model is downloaded automatically at runtime when using the v3 pipeline.

### Deploy to Railway
1. Connect the GitHub repo to Railway
2. Set start command: `uvicorn backend:app --host 0.0.0.0 --port $PORT`
3. Railway auto-detects nixpacks config

---

## 📊 Dataset

- **665 road images** with Pascal VOC XML annotations
- Pothole bounding boxes + segmentation masks
- Images sourced from real-world road surveys

---

## 🛠️ Tech Stack

**Backend:** Python · FastAPI · PyTorch · Ultralytics YOLOv8 · OpenCV · MiDaS  
**Frontend:** React · Vite · JavaScript  
**Deployment:** Railway (nixpacks) · Uvicorn

---

## 👤 Author

**Parshva Dongare**  
[GitHub](https://github.com/ParshvaDongare)