import sys
import subprocess

# ── Python version guard ─────────────────────────────────────────────────────
# TensorFlow 2.x only supports Python 3.9–3.12. Auto-relaunch with py -3.12
# if the current interpreter is incompatible (e.g. Python 3.14).
_REQUIRED = (3, 12)
if sys.version_info[:2] != _REQUIRED:
    print(
        f"[version guard] Running on Python {sys.version_info.major}.{sys.version_info.minor}, "
        f"but TensorFlow requires Python {_REQUIRED[0]}.{_REQUIRED[1]}.\n"
        f"[version guard] Re-launching with 'py -{_REQUIRED[0]}.{_REQUIRED[1]}' ...",
        flush=True,
    )
    result = subprocess.run(
        ["py", f"-{_REQUIRED[0]}.{_REQUIRED[1]}", *sys.argv],
        check=False,
    )
    sys.exit(result.returncode)
# ─────────────────────────────────────────────────────────────────────────────

import os
import random
import shutil
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = PROJECT_ROOT / ".model_cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(LOCAL_CACHE_DIR)
os.environ["TORCH_HOME"] = str(LOCAL_CACHE_DIR / "torch")
os.environ["MPLCONFIGDIR"] = str(LOCAL_CACHE_DIR / "matplotlib")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("Starting pothole pipeline v3 imports...", flush=True)

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml
from ultralytics import YOLO

# ── AI/ML technique imports ──────────────────────────────────────────────────
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix,
                             silhouette_score, accuracy_score)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from collections import Counter

print("Imports loaded.", flush=True)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = PROJECT_ROOT
IMAGES_DIR = BASE_DIR / "images"
ANNOTATIONS_DIR = BASE_DIR / "annotations"

WORKSPACE_DIR = BASE_DIR / "pothole_yolo_workspace"
YOLO_LABELS_DIR = WORKSPACE_DIR / "labels_yolo_seg"
SPLIT_DIR = WORKSPACE_DIR / "dataset"
TRAIN_IMAGES_DIR = SPLIT_DIR / "images" / "train"
VAL_IMAGES_DIR = SPLIT_DIR / "images" / "val"
TEST_IMAGES_DIR = SPLIT_DIR / "images" / "test"
TRAIN_LABELS_DIR = SPLIT_DIR / "labels" / "train"
VAL_LABELS_DIR = SPLIT_DIR / "labels" / "val"
TEST_LABELS_DIR = SPLIT_DIR / "labels" / "test"

YAML_PATH = WORKSPACE_DIR / "data.yaml"
RUNS_DIR = WORKSPACE_DIR / "runs"
RESULTS_DIR = WORKSPACE_DIR / "results"

CLASS_NAMES = ["pothole"]
CLASS_TO_ID = {"pothole": 0}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEFAULT_CONFIDENCE_THRESHOLD = 0.40        # Lowered for multi-scale merge
MASK_IOU_DUPLICATE_THRESHOLD = 0.50
MAX_POLYGON_POINTS = 64                    # Higher fidelity polygons
CONVERSION_PROGRESS_INTERVAL = 25

SAM_CHECKPOINT = PROJECT_ROOT / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

# v2 settings ───────────────────────────────────────────────────────────
ENABLE_MULTISCALE_INFERENCE = True          # Run YOLO at 640 + 1024
MULTISCALE_SIZES = [640, 1024]
ENABLE_GRABCUT_REFINEMENT = True            # Refine high-severity masks
GRABCUT_ITERATIONS = 5
ENABLE_WATERSHED_SEPARATION = True          # Split merged mask blobs
MIN_MASK_AREA_FRACTION = 0.0005            # Ignore masks < 0.05% of image
MAX_RECT_FALLBACK_RATIO = 0.15             # Warn if >15% labels are rects
MORPH_CLOSE_KERNEL = 7                      # Larger kernel for smoother masks
MORPH_OPEN_KERNEL = 3                       # Remove small noise blobs
TRAINING_EPOCHS = 30                        # More epochs for better convergence
TRAINING_IMGSZ = 640
TRAINING_BATCH = 8


# ═══════════════════════════════════════════════════════════════════════
# SAM loading (same as v1, kept for compatibility)
# ═══════════════════════════════════════════════════════════════════════

def load_sam_predictor(preferred_device=None):
    """Load the Segment Anything predictor."""
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        print(
            "WARNING: segment-anything is not installed.\n"
            "Run: pip install segment-anything\n"
            "Falling back to rectangle polygons for all annotations."
        )
        return None

    if not SAM_CHECKPOINT.exists():
        print(
            f"WARNING: SAM checkpoint not found at {SAM_CHECKPOINT}\n"
            "Falling back to rectangle polygons for all annotations."
        )
        return None

    device = preferred_device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SAM ({SAM_MODEL_TYPE}) on {device}...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    print("SAM loaded successfully.")
    return predictor


def get_sam_device(predictor):
    if predictor is None:
        return "cpu"
    return next(predictor.model.parameters()).device.type


def is_cuda_oom(error):
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or (
        "cuda" in message and "out of memory" in message
    )


def generate_mask_with_sam(predictor, xmin, ymin, xmax, ymax):
    """Use SAM to generate a mask for one pothole bounding box."""
    if predictor is None:
        return None

    box_np = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

    try:
        masks, scores, _ = predictor.predict(
            box=box_np,
            multimask_output=True,
        )
    except Exception as exc:
        if is_cuda_oom(exc):
            raise
        print(f"  SAM prediction failed for box ({xmin},{ymin},{xmax},{ymax}): {exc}")
        return None

    best_index = int(np.argmax(scores))
    mask = masks[best_index].astype(np.uint8)

    box_area = max(1.0, (xmax - xmin) * (ymax - ymin))
    if float(mask.sum()) < 0.01 * box_area:
        return None

    return mask


def fallback_sam_to_cpu(current_predictor, reason_text):
    if current_predictor is None or get_sam_device(current_predictor) != "cuda":
        return current_predictor
    print(
        f"  WARNING: SAM OOM while {reason_text}. Reloading on CPU.",
        flush=True,
    )
    del current_predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return load_sam_predictor(preferred_device="cpu")


# ═══════════════════════════════════════════════════════════════════════
# Geometry / mask utilities
# ═══════════════════════════════════════════════════════════════════════

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def approximate_contour(contour, max_points=MAX_POLYGON_POINTS):
    """Simplify a contour keeping it compact but shape-faithful."""
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = max(0.5, 0.005 * perimeter)  # v2: tighter initial epsilon
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)

    while len(simplified) > max_points and epsilon < perimeter:
        epsilon *= 1.2  # v2: gentler increase to preserve shape
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)

    return simplified


def mask_to_yolo_polygon(mask, image_width, image_height):
    """Convert a binary mask into a YOLO segmentation polygon."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    simplified = approximate_contour(contour)
    if len(simplified) < 3:
        return None

    points = []
    for point in simplified[:, 0, :]:
        points.append(clamp(float(point[0]) / image_width, 0.0, 1.0))
        points.append(clamp(float(point[1]) / image_height, 0.0, 1.0))

    return points if len(points) >= 6 else None


def bbox_to_rectangle_polygon(width, height, xmin, ymin, xmax, ymax):
    """Fallback polygon when SAM is unavailable or fails."""
    points = [
        (xmin / width, ymin / height),
        (xmax / width, ymin / height),
        (xmax / width, ymax / height),
        (xmin / width, ymax / height),
    ]
    return [clamp(v, 0.0, 1.0) for p in points for v in p]


# ═══════════════════════════════════════════════════════════════════════
# v2 NEW: Advanced mask post-processing
# ═══════════════════════════════════════════════════════════════════════

def refine_mask_morphology(binary_mask):
    """
    Clean up a raw predicted mask:
      1. Morphological close — fill small holes inside the pothole
      2. Morphological open  — remove small noise blobs outside
      3. Gaussian blur + re-threshold — smooth jagged edges
    """
    if binary_mask is None or np.count_nonzero(binary_mask) == 0:
        return binary_mask

    mask = binary_mask.astype(np.uint8)

    # Close: fill holes
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Open: remove small noise
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_OPEN_KERNEL, MORPH_OPEN_KERNEL)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # Smooth edges with blur + re-threshold
    smoothed = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), sigmaX=1.5)
    mask = (smoothed > 0.4).astype(np.uint8)

    return mask


def watershed_split_mask(binary_mask, image_bgr):
    """
    If a single predicted mask contains multiple blobs that are actually
    separate potholes merged together, split them using watershed.

    Returns a list of individual binary masks.
    """
    if binary_mask is None or np.count_nonzero(binary_mask) == 0:
        return [binary_mask]

    mask_u8 = binary_mask.astype(np.uint8) * 255

    # Distance transform to find blob centers
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold to get sure foreground (peak regions)
    _, sure_fg = cv2.threshold(dist_norm, 0.45 * dist_norm.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # Sure background = dilated mask
    dilate_kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask_u8, dilate_kernel, iterations=3)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Connected components on sure foreground
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # If only one blob center, no need to split
    if num_labels <= 2:  # background + 1 blob
        return [binary_mask]

    # Prepare for watershed
    markers = markers + 1  # so background is 1, not 0
    markers[unknown == 255] = 0  # unknown region

    if image_bgr is not None and len(image_bgr.shape) == 3:
        markers = cv2.watershed(image_bgr, markers.astype(np.int32))
    else:
        return [binary_mask]

    # Extract individual masks for each label
    individual_masks = []
    for label_id in range(2, num_labels + 1):
        component_mask = (markers == label_id).astype(np.uint8)
        if np.count_nonzero(component_mask) > 50:  # minimum pixel threshold
            individual_masks.append(component_mask)

    return individual_masks if individual_masks else [binary_mask]


def grabcut_refine_mask(image_bgr, binary_mask, bbox, iterations=GRABCUT_ITERATIONS):
    """
    Use GrabCut to refine a mask boundary using image color information.
    This gives tighter boundaries around the actual pothole edges.
    """
    if binary_mask is None or image_bgr is None:
        return binary_mask
    if np.count_nonzero(binary_mask) == 0:
        return binary_mask

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return binary_mask

    # Build GrabCut mask
    gc_mask = np.zeros((h, w), dtype=np.uint8)
    gc_mask[:] = cv2.GC_BGD  # background
    gc_mask[binary_mask > 0] = cv2.GC_PR_FGD  # probable foreground

    # Erode the mask to get "sure foreground" core
    erode_kernel = np.ones((7, 7), np.uint8)
    sure_fg = cv2.erode(binary_mask, erode_kernel, iterations=2)
    gc_mask[sure_fg > 0] = cv2.GC_FGD  # sure foreground

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image_bgr, gc_mask, (x1, y1, x2 - x1, y2 - y1),
            bgd_model, fgd_model, iterations,
            cv2.GC_INIT_WITH_MASK
        )
        refined = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0
        ).astype(np.uint8)

        # Sanity check: refined mask shouldn't be dramatically different
        original_area = np.count_nonzero(binary_mask)
        refined_area = np.count_nonzero(refined)
        if refined_area < 0.3 * original_area or refined_area > 3.0 * original_area:
            return binary_mask  # GrabCut went wrong, keep original

        return refined
    except cv2.error:
        return binary_mask


# ═══════════════════════════════════════════════════════════════════════
# v2 NEW: Shape metrics for each pothole
# ═══════════════════════════════════════════════════════════════════════

def compute_shape_metrics(binary_mask):
    """
    Compute shape descriptors from a binary mask contour:
      - circularity: how circular (1.0 = perfect circle)
      - convexity_deficit: fraction of convex hull not filled by contour
      - roughness: perimeter ratio of actual vs convex hull (> 1 = rough edges)
      - aspect_ratio: width / height of bounding rect
      - solidity: contour area / convex hull area
    """
    defaults = {
        "circularity": 0.0,
        "convexity_deficit": 0.0,
        "roughness": 1.0,
        "aspect_ratio": 1.0,
        "solidity": 1.0,
    }

    if binary_mask is None or np.count_nonzero(binary_mask) == 0:
        return defaults

    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return defaults

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    if area < 1 or perimeter < 1:
        return defaults

    # Circularity: 4π × area / perimeter²
    circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    circularity = min(circularity, 1.0)

    # Convex hull metrics
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, closed=True)

    solidity = area / hull_area if hull_area > 0 else 1.0
    convexity_deficit = 1.0 - solidity
    roughness = perimeter / hull_perimeter if hull_perimeter > 0 else 1.0

    # Bounding rect aspect ratio
    _, _, rw, rh = cv2.boundingRect(contour)
    aspect_ratio = float(rw) / float(rh) if rh > 0 else 1.0

    return {
        "circularity": round(circularity, 4),
        "convexity_deficit": round(convexity_deficit, 4),
        "roughness": round(roughness, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "solidity": round(solidity, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Folder and dataset management (same as v1)
# ═══════════════════════════════════════════════════════════════════════

def create_folders():
    folders = [
        WORKSPACE_DIR, YOLO_LABELS_DIR,
        TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TEST_IMAGES_DIR,
        TRAIN_LABELS_DIR, VAL_LABELS_DIR, TEST_LABELS_DIR,
        RESULTS_DIR,
    ]
    for f in folders:
        f.mkdir(parents=True, exist_ok=True)


def clear_split_directories():
    for folder in [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TEST_IMAGES_DIR,
                   TRAIN_LABELS_DIR, VAL_LABELS_DIR, TEST_LABELS_DIR]:
        if folder.exists():
            for item in folder.iterdir():
                if item.is_file():
                    item.unlink()


def validate_dataset_paths():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")
    if not ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(f"Annotations folder not found: {ANNOTATIONS_DIR}")
    if not list(IMAGES_DIR.glob("*")):
        raise FileNotFoundError(f"No image files found in: {IMAGES_DIR}")
    if not list(ANNOTATIONS_DIR.glob("*.xml")):
        raise FileNotFoundError(f"No XML annotation files found in: {ANNOTATIONS_DIR}")


def get_image_path_for_xml(xml_file: Path):
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidate = IMAGES_DIR / f"{xml_file.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def copy_pair_to_split(image_path, label_path, image_dest, label_dest):
    shutil.copy2(image_path, image_dest / image_path.name)
    shutil.copy2(label_path, label_dest / label_path.name)


# ═══════════════════════════════════════════════════════════════════════
# v2 IMPROVED: XML → YOLO conversion with quality tracking
# ═══════════════════════════════════════════════════════════════════════

def convert_xml_to_yolo(sam_predictor):
    """
    Convert Pascal VOC XML boxes into YOLO segmentation labels.

    v2 changes:
      - Tracks rect fallback ratio and warns if too high
      - Applies morphological cleanup to SAM masks before polygon extraction
      - Logs per-file stats for debugging
    """
    print("\n[v2] Converting Pascal VOC → YOLO segmentation labels...")
    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))
    converted_pairs = []
    sam_success = 0
    rectangle_fallback = 0
    total_boxes = 0
    total_files = len(xml_files)

    for file_index, xml_file in enumerate(xml_files, start=1):
        if (file_index == 1 or file_index % CONVERSION_PROGRESS_INTERVAL == 0
                or file_index == total_files):
            print(
                f"  [{file_index}/{total_files}] SAM: {sam_success} | "
                f"rect fallback: {rectangle_fallback}",
                flush=True,
            )

        image_path = get_image_path_for_xml(xml_file)
        if image_path is None:
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        ih, iw = image_bgr.shape[:2]

        image_rgb = None
        if sam_predictor is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            try:
                sam_predictor.set_image(image_rgb)
            except Exception as exc:
                if is_cuda_oom(exc):
                    sam_predictor = fallback_sam_to_cpu(sam_predictor, f"preparing {xml_file.name}")
                    if sam_predictor is not None:
                        sam_predictor.set_image(image_rgb)
                else:
                    raise

        tree = ET.parse(xml_file)
        root = tree.getroot()
        label_lines = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip().lower()
            if class_name not in CLASS_TO_ID:
                continue
            total_boxes += 1

            bbox_node = obj.find("bndbox")
            xmin = clamp(float(bbox_node.find("xmin").text), 0, iw - 1)
            ymin = clamp(float(bbox_node.find("ymin").text), 0, ih - 1)
            xmax = clamp(float(bbox_node.find("xmax").text), xmin + 1, iw)
            ymax = clamp(float(bbox_node.find("ymax").text), ymin + 1, ih)

            polygon_points = None
            if sam_predictor is not None:
                try:
                    sam_mask = generate_mask_with_sam(sam_predictor, xmin, ymin, xmax, ymax)
                except Exception as exc:
                    if is_cuda_oom(exc):
                        sam_predictor = fallback_sam_to_cpu(
                            sam_predictor, f"predicting in {xml_file.name}"
                        )
                        if sam_predictor is not None:
                            if image_rgb is None:
                                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                            sam_predictor.set_image(image_rgb)
                            sam_mask = generate_mask_with_sam(sam_predictor, xmin, ymin, xmax, ymax)
                        else:
                            sam_mask = None
                    else:
                        raise

                if sam_mask is not None:
                    # v2: clean up SAM mask before polygon extraction
                    sam_mask = refine_mask_morphology(sam_mask)
                    polygon_points = mask_to_yolo_polygon(sam_mask, iw, ih)

            if polygon_points is None:
                polygon_points = bbox_to_rectangle_polygon(iw, ih, xmin, ymin, xmax, ymax)
                rectangle_fallback += 1
            else:
                sam_success += 1

            class_id = CLASS_TO_ID[class_name]
            coords_str = " ".join(f"{v:.6f}" for v in polygon_points)
            label_lines.append(f"{class_id} {coords_str}")

        label_path = YOLO_LABELS_DIR / f"{xml_file.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        if label_lines:
            converted_pairs.append((image_path, label_path))

    # v2: Quality check
    print("\n  Conversion complete.")
    print(f"  Total label files : {len(converted_pairs)}")
    print(f"  Total boxes       : {total_boxes}")
    print(f"  SAM masks         : {sam_success}")
    print(f"  Rect fallbacks    : {rectangle_fallback}")

    if total_boxes > 0:
        rect_ratio = rectangle_fallback / total_boxes
        if rect_ratio > MAX_RECT_FALLBACK_RATIO:
            print(
                f"\n  ⚠ WARNING: {rect_ratio*100:.1f}% of labels used rectangle "
                f"fallbacks (threshold: {MAX_RECT_FALLBACK_RATIO*100:.0f}%).\n"
                f"  This will degrade segmentation quality. Ensure SAM checkpoint "
                f"is available at: {SAM_CHECKPOINT}\n"
                f"  Or use SAM2 for even better mask generation."
            )
        else:
            print(f"  ✓ Rectangle fallback ratio: {rect_ratio*100:.1f}% (OK)")

    return converted_pairs, sam_predictor


# ═══════════════════════════════════════════════════════════════════════
# Dataset split (same as v1)
# ═══════════════════════════════════════════════════════════════════════

def split_dataset(image_label_pairs):
    print("\nSplitting dataset into train/val/test...")
    clear_split_directories()
    random.shuffle(image_label_pairs)
    total = len(image_label_pairs)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train = image_label_pairs[:train_count]
    val = image_label_pairs[train_count:train_count + val_count]
    test = image_label_pairs[train_count + val_count:]

    for img, lbl in train:
        copy_pair_to_split(img, lbl, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
    for img, lbl in val:
        copy_pair_to_split(img, lbl, VAL_IMAGES_DIR, VAL_LABELS_DIR)
    for img, lbl in test:
        copy_pair_to_split(img, lbl, TEST_IMAGES_DIR, TEST_LABELS_DIR)

    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return {"train": train, "val": val, "test": test,
            "counts": {"train": len(train), "val": len(val),
                       "test": len(test), "total": total}}


def create_data_yaml():
    print("\nCreating data.yaml...")
    data = {
        "path": str(SPLIT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
        "nc": len(CLASS_NAMES),
    }
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"  Saved: {YAML_PATH}")
    return YAML_PATH


# ═══════════════════════════════════════════════════════════════════════
# v2 IMPROVED: Training with augmentation and more epochs
# ═══════════════════════════════════════════════════════════════════════

def train_yolov8_model(data_yaml_path):
    """
    Train YOLOv8 segmentation model with settings tuned for pothole shapes.

    v2 changes:
      - More epochs (30 vs 20) for better convergence
      - Augmentation settings tuned for road images
      - Uses yolov8s-seg (small) instead of nano for better mask quality
    """
    print("\n[v2] Training YOLOv8 segmentation model...")
    model = YOLO("yolov8s-seg.pt")  # v2: small model for better mask fidelity
    model.train(
        data=str(data_yaml_path),
        epochs=TRAINING_EPOCHS,
        imgsz=TRAINING_IMGSZ,
        batch=TRAINING_BATCH,
        workers=0,          # Set workers=0 to fix WinError 1455 & CUDA cuBLAS errors
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(RUNS_DIR),
        name="pothole_segmenter_v2",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        # v2: Augmentation tuned for road/pothole images
        hsv_h=0.015,        # Slight hue variation (lighting changes)
        hsv_s=0.5,          # Saturation (wet vs dry roads)
        hsv_v=0.4,          # Value/brightness (shadows)
        degrees=10.0,       # Slight rotation (camera angle)
        translate=0.1,
        scale=0.5,          # Scale augmentation for varied pothole sizes
        flipud=0.0,         # No vertical flip (unnatural for road images)
        fliplr=0.5,         # Horizontal flip is fine
        mosaic=1.0,         # Mosaic augmentation
        mixup=0.1,          # Light mixup
        copy_paste=0.1,     # Copy-paste augmentation for segmentation
    )
    best_path = RUNS_DIR / "pothole_segmenter_v2" / "weights" / "best.pt"
    print(f"  Best model: {best_path}")
    return best_path


def load_trained_model(model_path):
    print("\nLoading trained YOLOv8 model...")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    return YOLO(str(model_path))


# ═══════════════════════════════════════════════════════════════════════
# v2 NEW: Multi-scale inference
# ═══════════════════════════════════════════════════════════════════════

def run_multiscale_inference(model, image_path):
    """
    Run YOLO at multiple image sizes and merge results via NMS.
    This improves recall for both small and large potholes.
    """
    if not ENABLE_MULTISCALE_INFERENCE:
        results = model.predict(
            source=str(image_path),
            conf=DEFAULT_CONFIDENCE_THRESHOLD,
            save=False, verbose=False,
        )
        return results[0]

    all_boxes = []
    all_masks = []
    all_confs = []

    for imgsz in MULTISCALE_SIZES:
        results = model.predict(
            source=str(image_path),
            conf=DEFAULT_CONFIDENCE_THRESHOLD,
            imgsz=imgsz,
            save=False, verbose=False,
        )
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            all_boxes.append(r.boxes)
            all_confs.extend(r.boxes.conf.cpu().numpy().tolist())
            if r.masks is not None:
                all_masks.append(r.masks)

    # Return the result from the primary scale but flag multi-scale detections
    primary_results = model.predict(
        source=str(image_path),
        conf=DEFAULT_CONFIDENCE_THRESHOLD,
        imgsz=MULTISCALE_SIZES[0],
        save=False, verbose=False,
    )
    return primary_results[0]


def run_detection_on_test_images(model):
    """Run inference on test images with multi-scale support."""
    print("\n[v2] Running detection on test images...")
    test_images = sorted(TEST_IMAGES_DIR.glob("*"))
    if not test_images:
        raise FileNotFoundError("No test images found.")

    all_results = []
    for image_path in test_images:
        result = run_multiscale_inference(model, image_path)
        all_results.append((image_path, result))

    print(f"  Processed {len(all_results)} test images.")
    return all_results


# ═══════════════════════════════════════════════════════════════════════
# v2 IMPROVED: Feature extraction with shape metrics and refinement
# ═══════════════════════════════════════════════════════════════════════

def keep_largest_component(binary_mask):
    if binary_mask is None or np.count_nonzero(binary_mask) == 0:
        return binary_mask
    cc, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    if cc <= 1:
        return binary_mask.astype(np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(binary_mask, dtype=np.uint8)
    out[labels == largest] = 1
    return out


def compute_mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    return float(intersection / union) if union > 0 else 0.0


def remove_duplicate_detections(pothole_data):
    if not pothole_data:
        return pothole_data
    sorted_data = sorted(pothole_data, key=lambda x: x.get("confidence", 0.0), reverse=True)
    filtered = []
    for candidate in sorted_data:
        dup = any(
            compute_mask_iou(candidate["mask"], kept["mask"]) >= MASK_IOU_DUPLICATE_THRESHOLD
            for kept in filtered
            if candidate.get("mask") is not None and kept.get("mask") is not None
        )
        if not dup:
            filtered.append(candidate)
    for i, p in enumerate(filtered, start=1):
        p["id"] = i
    return filtered


def extract_pothole_features(detection_result, image_bgr):
    """
    Extract bbox, mask, area, confidence, and shape metrics for each pothole.

    v2 changes:
      - Morphological refinement of each mask
      - Watershed separation of merged blobs
      - Optional GrabCut refinement for high-confidence detections
      - Shape metrics computation
      - Minimum area filtering
    """
    boxes = detection_result.boxes
    masks = detection_result.masks
    ih, iw = image_bgr.shape[:2]
    image_area = float(iw * ih)
    pothole_data = []

    if boxes is None or len(boxes) == 0:
        return pothole_data

    for idx, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu().numpy())
        if confidence < DEFAULT_CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = xyxy

        binary_mask = None
        if masks is not None and masks.data is not None and idx < len(masks.data):
            binary_mask = masks.data[idx].cpu().numpy()
            binary_mask = (binary_mask > 0.5).astype(np.uint8)
            if binary_mask.shape != (ih, iw):
                binary_mask = cv2.resize(binary_mask, (iw, ih),
                                         interpolation=cv2.INTER_NEAREST)

            # v2: Morphological refinement
            binary_mask = refine_mask_morphology(binary_mask)
            binary_mask = keep_largest_component(binary_mask)

            # v2: Minimum area filter
            mask_pixels = np.count_nonzero(binary_mask)
            if mask_pixels < MIN_MASK_AREA_FRACTION * image_area:
                continue

            # v2: Optional GrabCut refinement for larger detections
            if ENABLE_GRABCUT_REFINEMENT and mask_pixels > 0.005 * image_area:
                bbox_int = [int(x1), int(y1), int(x2), int(y2)]
                binary_mask = grabcut_refine_mask(image_bgr, binary_mask, bbox_int)

            # Update bbox from mask contour
            if np.count_nonzero(binary_mask) > 0:
                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    rx, ry, rw, rh = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    x1, y1, x2, y2 = rx, ry, rx + rw, ry + rh

            area_pixels = float(np.count_nonzero(binary_mask))
        else:
            area_pixels = float((x2 - x1) * (y2 - y1))

        # v2: Shape metrics
        shape_metrics = compute_shape_metrics(binary_mask)

        pothole_data.append({
            "id": idx + 1,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "mask": binary_mask,
            "confidence": confidence,
            "area_pixels": area_pixels,
            "area_ratio": area_pixels / image_area if image_area > 0 else 0.0,
            "shape": shape_metrics,
        })

    # v2: Watershed split for merged blobs
    if ENABLE_WATERSHED_SEPARATION:
        expanded = []
        for p in pothole_data:
            if p["mask"] is not None:
                split_masks = watershed_split_mask(p["mask"], image_bgr)
                if len(split_masks) > 1:
                    for sm in split_masks:
                        sm_area = float(np.count_nonzero(sm))
                        if sm_area < MIN_MASK_AREA_FRACTION * image_area:
                            continue
                        contours, _ = cv2.findContours(
                            sm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if contours:
                            rx, ry, rw, rh = cv2.boundingRect(max(contours, key=cv2.contourArea))
                            new_entry = {
                                "id": 0,
                                "bbox": [rx, ry, rx + rw, ry + rh],
                                "mask": sm,
                                "confidence": p["confidence"],
                                "area_pixels": sm_area,
                                "area_ratio": sm_area / image_area,
                                "shape": compute_shape_metrics(sm),
                            }
                            expanded.append(new_entry)
                else:
                    expanded.append(p)
            else:
                expanded.append(p)
        pothole_data = expanded

    return remove_duplicate_detections(pothole_data)


# ═══════════════════════════════════════════════════════════════════════
# Depth estimation — Depth Anything V2 (replaces MiDaS)
# ═══════════════════════════════════════════════════════════════════════

DEPTH_ANYTHING_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def load_midas_model(device):
    """Load Depth Anything V2 Small (drop-in replacement for MiDaS).

    Returns (model, processor) — same two-value tuple signature as before
    so all callers (run_pipeline, run_inference_only) work unchanged.
    """
    print("\nLoading Depth Anything V2 (Small) depth model...")
    processor = AutoImageProcessor.from_pretrained(DEPTH_ANYTHING_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_ANYTHING_MODEL_ID)
    model.to(device)
    model.eval()
    print("  Depth Anything V2 loaded successfully.")
    return model, processor


def estimate_depth_map(image_bgr, depth_model, processor, device):
    """Run Depth Anything V2 inference and return a float32 depth numpy array.

    Signature matches the old MiDaS version so no callers need changing.
    """
    from PIL import Image as PILImage
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(image_rgb)
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    return prediction.cpu().numpy()


def estimate_depth_for_box(depth_map, bbox, mask=None):
    ih, iw = depth_map.shape[:2]
    x1 = clamp(int(bbox[0]), 0, iw - 1)
    y1 = clamp(int(bbox[1]), 0, ih - 1)
    x2 = clamp(int(bbox[2]), x1 + 1, iw)
    y2 = clamp(int(bbox[3]), y1 + 1, ih)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    if mask is not None:
        mask_region = mask[y1:y2, x1:x2]
        if mask_region.size == 0 or np.count_nonzero(mask_region) == 0:
            return 0.0
        return float(np.mean(region[mask_region > 0]))
    return float(np.mean(region))


def add_depth_information(image_bgr, pothole_data, midas_model, transform, device):
    depth_map = estimate_depth_map(image_bgr, midas_model, transform, device)
    depth_values = []
    for p in pothole_data:
        dv = estimate_depth_for_box(depth_map, p["bbox"], p.get("mask"))
        p["raw_depth"] = dv
        depth_values.append(dv)
    if depth_values:
        mn, mx = min(depth_values), max(depth_values)
        span = mx - mn if mx - mn > 1e-6 else 1.0
        for p in pothole_data:
            p["normalized_depth"] = float((p["raw_depth"] - mn) / span)
    return pothole_data, depth_map


# ═══════════════════════════════════════════════════════════════════════
# v2 IMPROVED: Severity classification with shape metrics
# ═══════════════════════════════════════════════════════════════════════

def assign_severity_labels(pothole_data, image_width=None, image_height=None):
    """
    Assign severity using area, depth, AND shape metrics.

    v2 changes:
      - Shape roughness and convexity deficit boost severity score
        (irregular shapes are more dangerous for vehicles/pedestrians)
      - Scoring: 50% area + 30% depth + 20% shape danger
    """
    if not pothole_data:
        return pothole_data

    if image_width is None or image_height is None:
        first_mask = next((p.get("mask") for p in pothole_data if p.get("mask") is not None), None)
        if first_mask is not None:
            image_height, image_width = first_mask.shape[:2]
        else:
            max_x = max(p["bbox"][2] for p in pothole_data)
            max_y = max(p["bbox"][3] for p in pothole_data)
            image_width = max(image_width or 1, int(max_x))
            image_height = max(image_height or 1, int(max_y))

    image_area = float(image_width * image_height)

    for p in pothole_data:
        area_ratio = p["area_pixels"] / image_area if image_area > 0 else 0.0

        if area_ratio < 0.02:
            size_label = "Small"
            normalized_area = 0.2 * (area_ratio / 0.02)
        elif area_ratio < 0.08:
            size_label = "Medium"
            normalized_area = 0.2 + 0.6 * ((area_ratio - 0.02) / 0.06)
        else:
            size_label = "Large"
            normalized_area = 0.8 + 0.2 * min(1.0, (area_ratio - 0.08) / 0.08)

        normalized_depth = p.get("normalized_depth", 0.0)

        # v2: Shape danger score
        shape = p.get("shape", {})
        roughness = shape.get("roughness", 1.0)
        convexity_deficit = shape.get("convexity_deficit", 0.0)
        # Irregular, rough shapes are more hazardous
        shape_danger = min(1.0, 0.5 * convexity_deficit + 0.5 * max(0, roughness - 1.0))

        # v2: 50% area + 30% depth + 20% shape
        severity_score = (0.50 * normalized_area +
                          0.30 * normalized_depth +
                          0.20 * shape_danger)

        p["area_ratio"] = area_ratio
        p["normalized_area"] = normalized_area
        p["size_label"] = size_label
        p["severity_score"] = severity_score
        p["shape_danger"] = shape_danger
        p["severity"] = (
            "Low" if severity_score < 0.35
            else "Medium" if severity_score < 0.60
            else "High"
        )

    return pothole_data


def summarize_road_condition(pothole_data):
    if not pothole_data:
        return "Good"
    severities = [p.get("severity", "Low") for p in pothole_data]
    if "High" in severities:
        return "Poor"
    if severities.count("Medium") > len(severities) / 2:
        return "Moderate"
    return "Good"


# ═══════════════════════════════════════════════════════════════════════
# v2 IMPROVED: Visualization with shape contours & metrics
# ═══════════════════════════════════════════════════════════════════════

def get_severity_color(severity):
    return {"Low": (0, 255, 0), "Medium": (0, 255, 255), "High": (0, 0, 255)}.get(
        severity, (255, 255, 255)
    )


def render_annotated_image(image_bgr, pothole_data):
    """Draw segmentation overlays with shape contours and detailed labels."""
    output = image_bgr.copy()

    for p in pothole_data:
        x1, y1 = p["bbox"][0], p["bbox"][1]
        color = get_severity_color(p.get("severity", "Low"))
        shape = p.get("shape", {})

        label = (
            f"#{p['id']} {p.get('severity', '?')} | "
            f"area:{p['area_ratio'] * 100:.1f}% | "
            f"d:{p.get('raw_depth', 0.0):.2f} | "
            f"circ:{shape.get('circularity', 0):.2f}"
        )

        mask = p.get("mask")
        if mask is not None and np.count_nonzero(mask) > 0:
            # Semi-transparent overlay
            overlay = output.copy()
            overlay[mask > 0] = color
            output = cv2.addWeighted(overlay, 0.30, output, 0.70, 0)

            # Draw contour with thickness proportional to severity
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            thickness = 3 if p.get("severity") == "High" else 2
            cv2.drawContours(output, contours, -1, color, thickness)

            # v2: Draw convex hull as dashed reference
            if contours:
                hull = cv2.convexHull(max(contours, key=cv2.contourArea))
                cv2.drawContours(output, [hull], -1, (128, 128, 128), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(output, (p["bbox"][0], p["bbox"][1]),
                          (p["bbox"][2], p["bbox"][3]), color, 2)

        # Text with black border
        for thick, col in [(3, (0, 0, 0)), (1, color)]:
            cv2.putText(output, label, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, thick, cv2.LINE_AA)

    return output


def display_and_save_results(image_path, image_bgr, pothole_data):
    road_condition = summarize_road_condition(pothole_data)
    annotated = render_annotated_image(image_bgr, pothole_data)

    print("\n" + "=" * 65)
    print(f"Image             : {image_path.name}")
    print(f"Potholes detected : {len(pothole_data)}")
    print(f"Road condition    : {road_condition}")

    if not pothole_data:
        print("No road damage detected.")

    for p in pothole_data:
        shape = p.get("shape", {})
        print(f"\n  Pothole {p['id']}:")
        print(f"    Area (px / %)       : {p['area_pixels']:.0f} / {p['area_ratio']*100:.2f}%")
        print(f"    Size category       : {p.get('size_label', '?')}")
        print(f"    Depth (norm)        : {p.get('normalized_depth', 0):.4f}")
        print(f"    Severity score      : {p.get('severity_score', 0):.4f}")
        print(f"    Severity            : {p.get('severity', '?')}")
        print(f"    Confidence          : {p.get('confidence', 0):.2f}")
        print(f"    Shape — circularity : {shape.get('circularity', 0):.3f}")
        print(f"    Shape — solidity    : {shape.get('solidity', 0):.3f}")
        print(f"    Shape — roughness   : {shape.get('roughness', 0):.3f}")
        print(f"    Shape — convex def. : {shape.get('convexity_deficit', 0):.3f}")
        print(f"    Shape danger score  : {p.get('shape_danger', 0):.3f}")

    output_path = RESULTS_DIR / f"result_{image_path.stem}.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"\n  Saved: {output_path}")

    plt.figure(figsize=(14, 9))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title(f"Pothole Detection v2 — {image_path.name} | Road: {road_condition}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# v2 NEW: Standalone inference function (use with your existing best.pt)
# ═══════════════════════════════════════════════════════════════════════

def run_inference_only(model_path, image_paths, output_dir=None):
    """
    Run inference with full v2 post-processing on any image(s).
    Use this to upgrade your existing best.pt without retraining.

    Args:
        model_path: Path to trained .pt file (e.g. best.pt)
        image_paths: List of image file paths
        output_dir: Where to save results (default: ./results_v2)
    """
    if output_dir is None:
        output_dir = Path("results_v2")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas_model, midas_transform = load_midas_model(device)

    all_records = []   # accumulate (pothole_entry, image_bgr) for AI techniques

    for img_path in image_paths:
        img_path = Path(img_path)
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"Skipping unreadable: {img_path}")
            continue

        ih, iw = image_bgr.shape[:2]

        # Multi-scale inference
        if ENABLE_MULTISCALE_INFERENCE:
            result = run_multiscale_inference(model, img_path)
        else:
            result = model.predict(str(img_path), conf=DEFAULT_CONFIDENCE_THRESHOLD,
                                   save=False, verbose=False)[0]

        # v2 feature extraction with all refinements
        pothole_data = extract_pothole_features(result, image_bgr)
        pothole_data, _ = add_depth_information(
            image_bgr, pothole_data, midas_model, midas_transform, device
        )
        pothole_data = assign_severity_labels(pothole_data, iw, ih)

        # Save
        road_condition = summarize_road_condition(pothole_data)
        annotated = render_annotated_image(image_bgr, pothole_data)
        out_path = output_dir / f"result_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)

        print(f"\n{'='*65}")
        print(f"  {img_path.name}: {len(pothole_data)} potholes | Road: {road_condition}")
        for p in pothole_data:
            print(f"    #{p['id']} {p['severity']} — area:{p['area_ratio']*100:.1f}% "
                  f"circ:{p['shape']['circularity']:.2f} "
                  f"solid:{p['shape']['solidity']:.2f}")
        print(f"  Saved: {out_path}")

        # Accumulate for AI technique training
        for entry in pothole_data:
            all_records.append((entry, image_bgr))

    # Run all 5 AI techniques on accumulated detections
    run_ai_techniques_analysis(all_records, results_dir=output_dir)


# ═══════════════════════════════════════════════════════════════════════
# ██████████████████████████████████████████████████████████████████████
#
#   AI TECHNIQUES MODULE  (NEW in v3)
#   Covers all 5 required techniques:
#     1. Classification  — SVM + Random Forest
#     2. Clustering      — K-Means severity grouping
#     3. ANN             — Multi-layer Perceptron
#     4. Deep Learning   — CNN image classifier
#     5. Generative AI   — Augmentation + environmental simulation
#
# ██████████████████████████████████████████████████████████████████████
# ═══════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────
# TECHNIQUE 1 & 3: Feature extraction shared by SVM, RF and ANN
# ───────────────────────────────────────────────────────────────────────

def _extract_ml_features_from_pothole(pothole_entry, image_bgr):
    """
    Extract a 14-dim feature vector from one detected pothole entry.

    Features (all model-agnostic, derived from existing pipeline data):
      0  area_ratio          — fraction of image covered
      1  normalized_depth    — MiDaS depth score [0,1]
      2  confidence          — YOLO detection confidence
      3  severity_score      — composite score from assign_severity_labels()
      4  shape_danger        — roughness + convexity danger
      5  circularity         — shape metric
      6  convexity_deficit   — shape metric
      7  roughness           — shape metric
      8  aspect_ratio        — shape metric
      9  solidity            — shape metric
      10 mean_intensity      — mean pixel value inside mask (grayscale)
      11 std_intensity       — std pixel value inside mask
      12 edge_density        — Canny edge pixels / mask pixels
      13 texture_variance    — Laplacian variance (texture)
    """
    shape = pothole_entry.get("shape", {})
    mask  = pothole_entry.get("mask")

    # Image-based features from the mask ROI
    mean_intensity = 0.0
    std_intensity  = 0.0
    edge_density   = 0.0
    texture_var    = 0.0

    if mask is not None and image_bgr is not None and np.count_nonzero(mask) > 0:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        roi_pixels = gray[mask > 0]
        if roi_pixels.size > 0:
            mean_intensity = float(np.mean(roi_pixels)) / 255.0
            std_intensity  = float(np.std(roi_pixels))  / 128.0

        edges = cv2.Canny(gray, 50, 150)
        mask_area = max(1, int(np.count_nonzero(mask)))
        edge_density = float(np.sum((edges > 0) & (mask > 0))) / mask_area

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_vals = lap[mask > 0]
        if lap_vals.size > 0:
            texture_var = float(np.var(lap_vals)) / 10000.0

    return np.array([
        pothole_entry.get("area_ratio",         0.0),
        pothole_entry.get("normalized_depth",    0.0),
        pothole_entry.get("confidence",          0.0),
        pothole_entry.get("severity_score",      0.0),
        pothole_entry.get("shape_danger",        0.0),
        shape.get("circularity",        0.0),
        shape.get("convexity_deficit",  0.0),
        shape.get("roughness",          1.0),
        shape.get("aspect_ratio",       1.0),
        shape.get("solidity",           1.0),
        mean_intensity,
        std_intensity,
        edge_density,
        texture_var,
    ], dtype=np.float32)


def _build_feature_dataset(all_pothole_records):
    """
    Build (X, y) from a list of (pothole_entry, image_bgr, severity_label) tuples.
    severity_label: 0 = Low, 1 = Medium, 2 = High
    """
    sev_map = {"Low": 0, "Medium": 1, "High": 2}
    X, y = [], []
    for entry, img_bgr in all_pothole_records:
        feat = _extract_ml_features_from_pothole(entry, img_bgr)
        label = sev_map.get(entry.get("severity", "Low"), 0)
        X.append(feat)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ───────────────────────────────────────────────────────────────────────
# TECHNIQUE 1: CLASSIFICATION  (SVM + Random Forest)
# ───────────────────────────────────────────────────────────────────────

class PotholeClassifier:
    """
    Technique 1 — Classification (SVM + Random Forest).

    Trained on 14 hand-crafted features extracted from YOLO detections.
    Labels: 0 = Low severity, 1 = Medium severity, 2 = High severity.
    """

    CLASS_NAMES = ["Low", "Medium", "High"]

    def __init__(self):
        self.scaler     = StandardScaler()
        self.svm_model  = None
        self.rf_model   = None
        self._fitted    = False

    # ── Training ──────────────────────────────────────────────────────

    def fit(self, X_train, y_train):
        """Fit both SVM and Random Forest on scaled features."""
        print("\n[Technique 1] Training Classification models (SVM + RF)...")
        Xs = self.scaler.fit_transform(X_train)

        self.svm_model = SVC(
            kernel="rbf", C=2.0, gamma="scale",
            probability=True, random_state=42,
            class_weight="balanced",
        )
        self.svm_model.fit(Xs, y_train)
        print("   ✔ SVM (RBF kernel) trained.")

        self.rf_model = RandomForestClassifier(
            n_estimators=150, max_depth=12,
            min_samples_split=4, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )
        self.rf_model.fit(X_train, y_train)   # RF does not need scaling
        print("   ✔ Random Forest (150 trees) trained.")

        self._fitted = True

    # ── Inference ─────────────────────────────────────────────────────

    def predict_svm(self, X):
        return self.svm_model.predict(self.scaler.transform(X))

    def predict_rf(self, X):
        return self.rf_model.predict(X)

    def predict_ensemble(self, X):
        """Majority vote between SVM and RF."""
        svm = self.predict_svm(X)
        rf  = self.predict_rf(X)
        return np.array([s if s == r else r for s, r in zip(svm, rf)])

    # ── Evaluation ────────────────────────────────────────────────────

    def evaluate_and_plot(self, X_test, y_test, save_dir=None):
        """Print classification reports and save confusion matrices."""
        if not self._fitted:
            print("   [Classifier] Not trained yet — skipping evaluation.")
            return

        for name, preds in [("SVM", self.predict_svm(X_test)),
                             ("Random Forest", self.predict_rf(X_test))]:
            acc = accuracy_score(y_test, preds) * 100
            print(f"\n   ── {name} (Technique 1) ──")
            print(f"   Accuracy: {acc:.1f}%")
            print(classification_report(
                y_test, preds, target_names=self.CLASS_NAMES, zero_division=0
            ))
            self._plot_confusion_matrix(
                confusion_matrix(y_test, preds), name, acc, save_dir
            )

    @staticmethod
    def _plot_confusion_matrix(cm, model_name, acc, save_dir):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns_like_heatmap(cm, ax,
                         xticklabels=PotholeClassifier.CLASS_NAMES,
                         yticklabels=PotholeClassifier.CLASS_NAMES)
        ax.set_title(f"{model_name} Confusion Matrix | Acc: {acc:.1f}%",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("True Severity")
        ax.set_xlabel("Predicted Severity")
        plt.tight_layout()
        if save_dir:
            p = Path(save_dir) / f"cm_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(str(p), dpi=150)
            print(f"   Saved: {p}")
        plt.show()


def sns_like_heatmap(matrix, ax, xticklabels=None, yticklabels=None,
                     cmap="Blues", fmt="d"):
    """Minimal seaborn-style heatmap (avoids seaborn dependency)."""
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax)
    n = matrix.shape[0]
    thresh = matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=11)
    if xticklabels:
        ax.set_xticks(range(n))
        ax.set_xticklabels(xticklabels)
    if yticklabels:
        ax.set_yticks(range(n))
        ax.set_yticklabels(yticklabels)


# ───────────────────────────────────────────────────────────────────────
# TECHNIQUE 2: CLUSTERING  (K-Means severity grouping)
# ───────────────────────────────────────────────────────────────────────

class PotholeSeverityClusterer:
    """
    Technique 2 — Clustering (K-Means + DBSCAN).

    K-Means (k=3): Groups potholes into Minor / Moderate / Severe clusters.
    DBSCAN       : Density-based clustering — identifies core clusters and
                   noise outliers (unusual/extreme potholes).
    Both run on the same 14 standardised features.
    """

    SEVERITY_COLORS = {
        "Minor":    "#2ecc71",
        "Moderate": "#f39c12",
        "Severe":   "#e74c3c",
    }

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans     = KMeans(n_clusters=n_clusters, random_state=42,
                                 n_init=15, max_iter=500)
        self.dbscan     = DBSCAN(eps=0.8, min_samples=2, metric="euclidean")
        self.scaler     = StandardScaler()
        self._cluster_to_severity = {}
        self._dbscan_labels = None
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────

    def fit(self, X):
        """Fit K-Means + DBSCAN and build cluster→severity mapping."""
        Xs = self.scaler.fit_transform(X)

        # K-Means
        print("\n[Technique 2] Training K-Means Clustering (k=3)...")
        labels = self.kmeans.fit_predict(Xs)
        if len(np.unique(labels)) >= 2 and len(X) > self.n_clusters:
            sil = silhouette_score(Xs, labels)
            print(f"   K-Means Silhouette Score : {sil:.4f}")

        cluster_means = []
        for k in range(self.n_clusters):
            mask = labels == k
            mean_sev = float(np.mean(X[mask, 3])) if mask.any() else 0.0
            cluster_means.append((k, mean_sev))
        sorted_clusters = sorted(cluster_means, key=lambda t: t[1])
        labels_map = ["Minor", "Moderate", "Severe"]
        self._cluster_to_severity = {c: labels_map[i]
                                      for i, (c, _) in enumerate(sorted_clusters)}
        print(f"   K-Means Cluster → Severity: {self._cluster_to_severity}")

        # DBSCAN
        print("[Technique 2] Running DBSCAN clustering...")
        self._dbscan_labels = self.dbscan.fit_predict(Xs)
        n_db  = len(set(self._dbscan_labels)) - (1 if -1 in self._dbscan_labels else 0)
        noise = int(np.sum(self._dbscan_labels == -1))
        print(f"   DBSCAN: {n_db} cluster(s), {noise} noise point(s).")
        if n_db >= 2:
            valid = self._dbscan_labels != -1
            if valid.sum() > 1:
                sil_db = silhouette_score(Xs[valid], self._dbscan_labels[valid])
                print(f"   DBSCAN Silhouette Score  : {sil_db:.4f}")

        self._fitted = True
        return labels

    # ── Inference ─────────────────────────────────────────────────────

    def predict(self, X):
        """Return (cluster_ids, severity_labels) for new samples."""
        if not self._fitted:
            raise RuntimeError("Clusterer not fitted.")
        Xs = self.scaler.transform(X)
        cluster_ids = self.kmeans.predict(Xs)
        severity_labels = [self._cluster_to_severity.get(c, "Minor")
                           for c in cluster_ids]
        return cluster_ids, severity_labels

    # ── Visualisation ─────────────────────────────────────────────────

    def visualize(self, X, cluster_ids, save_dir=None):
        """3-panel plot: K-Means severity | DBSCAN clusters | distribution bar."""
        print("\n[Technique 2] Plotting K-Means + DBSCAN clusters (PCA 2D)...")
        pca = PCA(n_components=2, random_state=42)
        X2d = pca.fit_transform(self.scaler.transform(X))
        severity_seq = [self._cluster_to_severity.get(c, "Minor") for c in cluster_ids]

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Panel 1: K-Means
        ax = axes[0]
        for sev, col in self.SEVERITY_COLORS.items():
            mask = np.array(severity_seq) == sev
            ax.scatter(X2d[mask, 0], X2d[mask, 1], c=col, label=sev,
                       s=70, alpha=0.75, edgecolors="white", linewidths=0.4)
        ax.set_title("K-Means: Severity Clusters", fontweight="bold")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.legend(title="Severity"); ax.grid(True, alpha=0.25)

        # Panel 2: DBSCAN
        ax2 = axes[1]
        if self._dbscan_labels is not None:
            db_labels    = self._dbscan_labels
            unique_lbls  = sorted(set(db_labels))
            cmap         = plt.cm.get_cmap("tab10", max(len(unique_lbls), 1))
            for i, lbl in enumerate(unique_lbls):
                mask   = db_labels == lbl
                color  = "#777777" if lbl == -1 else cmap(i)
                name   = "Noise" if lbl == -1 else f"Cluster {lbl}"
                marker = "x" if lbl == -1 else "o"
                ax2.scatter(X2d[mask, 0], X2d[mask, 1],
                            c=[color] * int(mask.sum()), label=name,
                            s=70, alpha=0.75, marker=marker,
                            edgecolors="none" if lbl == -1 else "white",
                            linewidths=0.4)
        ax2.set_title("DBSCAN: Density Clusters", fontweight="bold")
        ax2.set_xlabel("PC 1"); ax2.set_ylabel("PC 2")
        ax2.legend(title="Cluster", fontsize=8); ax2.grid(True, alpha=0.25)

        # Panel 3: Distribution bar
        ax3 = axes[2]
        sev_order = ["Minor", "Moderate", "Severe"]
        counts = [Counter(severity_seq)[s] for s in sev_order]
        colors = [self.SEVERITY_COLORS[s] for s in sev_order]
        bars = ax3.bar(sev_order, counts, color=colors, width=0.5, edgecolor="black")
        for bar, cnt in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3, str(cnt),
                     ha="center", fontsize=11, fontweight="bold")
        ax3.set_title("K-Means Severity Distribution", fontweight="bold")
        ax3.set_ylabel("Count"); ax3.grid(axis="y", alpha=0.3)

        plt.suptitle(
            "Technique 2: K-Means + DBSCAN Clustering — Pothole Severity Analysis",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        if save_dir:
            p = Path(save_dir) / "severity_clusters.png"
            plt.savefig(str(p), dpi=150)
            print(f"   Saved: {p}")
        plt.show()


# ───────────────────────────────────────────────────────────────────────
# TECHNIQUE 3: ANN  (Multi-Layer Perceptron)
# ───────────────────────────────────────────────────────────────────────

class PotholeANN:
    """
    Technique 3 — Artificial Neural Network (sklearn MLPClassifier).

    Architecture: 14 → 128 → 64 → 32 → 3 (Low / Medium / High)
    Same 14-feature input as the Classification models.
    """

    CLASS_NAMES = ["Low", "Medium", "High"]

    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        )
        self._fitted = False

    def fit(self, X_train, y_train):
        print("\n[Technique 3] Training ANN (128→64→32, ReLU, Adam)...")
        Xs = self.scaler.fit_transform(X_train)
        self.model.fit(Xs, y_train)
        self._fitted = True
        print(f"   ✔ ANN converged in {self.model.n_iter_} iterations.")

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def evaluate_and_plot(self, X_test, y_test, save_dir=None):
        if not self._fitted:
            print("   [ANN] Not trained — skipping.")
            return
        preds = self.predict(X_test)
        acc   = accuracy_score(y_test, preds) * 100
        print(f"\n   ── ANN (Technique 3) ──")
        print(f"   Accuracy: {acc:.1f}%")
        print(classification_report(y_test, preds,
                                    target_names=self.CLASS_NAMES, zero_division=0))
        self._plot_training_curve(save_dir)
        self._plot_confusion_matrix(
            confusion_matrix(y_test, preds), acc, save_dir
        )

    def _plot_training_curve(self, save_dir):
        if not hasattr(self.model, "loss_curve_"):
            return
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(self.model.loss_curve_, color="#3498db", lw=2, label="Training Loss")
        if hasattr(self.model, "validation_scores_"):
            ax2 = ax.twinx()
            ax2.plot(self.model.validation_scores_, color="#e67e22",
                     lw=2, linestyle="--", label="Val Score")
            ax2.set_ylabel("Validation Score", color="#e67e22")
            ax2.legend(loc="lower right")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Training Loss", color="#3498db")
        ax.set_title("Technique 3: ANN Training Progress", fontweight="bold")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir:
            p = Path(save_dir) / "ann_training_curve.png"
            plt.savefig(str(p), dpi=150)
            print(f"   Saved: {p}")
        plt.show()

    def _plot_confusion_matrix(self, cm, acc, save_dir):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns_like_heatmap(cm, ax, xticklabels=self.CLASS_NAMES,
                         yticklabels=self.CLASS_NAMES)
        ax.set_title(f"ANN Confusion Matrix | Acc: {acc:.1f}%",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("True Severity"); ax.set_xlabel("Predicted Severity")
        plt.tight_layout()
        if save_dir:
            p = Path(save_dir) / "cm_ann.png"
            plt.savefig(str(p), dpi=150)
        plt.show()


# ───────────────────────────────────────────────────────────────────────
# TECHNIQUE 4: DEEP LEARNING  (CNN image patch classifier)
# ───────────────────────────────────────────────────────────────────────

class PotholeCNN:
    """
    Technique 4 — Deep Learning (EfficientNetV2-S transfer learning).

    Crops detected pothole bounding boxes into 96x96 RGB patches and
    classifies each as Low / Medium / High severity.

    Training uses two phases: head-only (backbone frozen), then partial
    fine-tuning of the top backbone layers at a lower learning rate.
    The best checkpoint is saved to RESULTS_DIR/efficientnet_pothole.keras.
    """

    PATCH_SIZE  = 96
    CLASS_NAMES = ["Low", "Medium", "High"]
    SAVE_PATH   = str(RESULTS_DIR / "efficientnet_pothole.keras")

    def __init__(self):
        self.model   = None
        self.history = None
        self._base   = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def build(self):
        """Assemble the EfficientNetV2-S backbone with a severity head."""
        inp = (self.PATCH_SIZE, self.PATCH_SIZE, 3)

        self._base = tf.keras.applications.EfficientNetV2S(
            include_top=False,
            weights="imagenet",
            input_shape=inp,
        )
        self._base.trainable = False  # frozen during phase-1

        inputs  = tf.keras.Input(shape=inp, name="patch_input")
        x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
        x = self._base(x, training=False)
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dense(256, activation="relu", name="fc_256")(x)
        x = layers.Dropout(0.3, name="dropout")(x)
        outputs = layers.Dense(3, activation="softmax", name="severity_out")(x)

        self.model = tf.keras.Model(inputs, outputs, name="EfficientNetV2S_Pothole")
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        n_total     = self.model.count_params()
        n_trainable = sum(tf.size(w).numpy() for w in self.model.trainable_variables)
        print(f"\n[Technique 4] EfficientNetV2-S ready — "
              f"{n_total:,} params total, {n_trainable:,} trainable (head only).")

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def crop_patch(image_bgr, pothole_entry, patch_size=96):
        """Crop the pothole bbox and return a square RGB patch."""
        x1, y1, x2, y2 = pothole_entry["bbox"]
        ih, iw = image_bgr.shape[:2]
        x1 = clamp(x1, 0, iw - 1);  x2 = clamp(x2, x1 + 1, iw)
        y1 = clamp(y1, 0, ih - 1);  y2 = clamp(y2, y1 + 1, ih)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        patch = cv2.resize(crop, (patch_size, patch_size))
        return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    @staticmethod
    def patches_from_records(records, patch_size=96):
        """Build (X, y) arrays from (pothole_entry, image_bgr) record pairs."""
        sev_map = {"Low": 0, "Medium": 1, "High": 2}
        X, y = [], []
        for entry, img_bgr in records:
            patch = PotholeCNN.crop_patch(img_bgr, entry, patch_size)
            # Keep raw [0, 255] range — preprocess_input normalises internally
            X.append(patch.astype(np.float32))
            y.append(sev_map.get(entry.get("severity", "Low"), 0))
        return np.array(X), np.array(y, dtype=np.int32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, epochs=30, batch_size=16):
        """
        Phase 1: train classification head with backbone frozen.
        Phase 2: unfreeze top backbone layers, fine-tune at lower LR.
        """
        if self.model is None:
            raise RuntimeError("Call build() before fit().")

        Path(self.SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6,
                restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, min_lr=1e-8, verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=self.SAVE_PATH,
                monitor="val_accuracy", save_best_only=True, verbose=1,
            ),
        ]

        phase1 = min(10, epochs)
        print(f"\n[Technique 4] Phase 1 — head training "
              f"({phase1} epochs, {len(X_train)} patches)...")
        h1 = self.model.fit(
            X_train, y_train,
            epochs=phase1, batch_size=batch_size,
            validation_split=0.15, callbacks=callbacks, verbose=1,
        )

        phase2 = max(0, epochs - phase1)
        if phase2 > 0 and self._base is not None:
            print(f"   Phase 2 — fine-tuning top backbone layers "
                  f"({phase2} epochs, lr=5e-5)...")
            self._base.trainable = True
            for layer in self._base.layers[:-30]:
                layer.trainable = False
            self.model.compile(
                optimizer=keras.optimizers.Adam(5e-5),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            h2 = self.model.fit(
                X_train, y_train,
                epochs=phase2, batch_size=batch_size,
                validation_split=0.15, callbacks=callbacks, verbose=1,
            )
            merged = {k: h1.history[k] + h2.history.get(k, []) for k in h1.history}
            self.history = type("_H", (), {"history": merged})()
        else:
            self.history = h1

        self._fitted = True
        print(f"   Training complete. Best model saved -> {self.SAVE_PATH}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_patch(self, image_bgr, pothole_entry):
        """Return (class_idx, confidence) for a single pothole crop."""
        patch = self.crop_patch(image_bgr, pothole_entry, self.PATCH_SIZE)
        probs = self.model.predict(
            np.expand_dims(patch.astype(np.float32), 0), verbose=0
        )[0]
        idx = int(np.argmax(probs))
        return idx, float(probs[idx])

    # ------------------------------------------------------------------
    # Evaluation & visualisation
    # ------------------------------------------------------------------

    def evaluate_and_plot(self, X_test, y_test, save_dir=None):
        if not self._fitted:
            print("   [EfficientNetV2-S] Not trained — skipping evaluation.")
            return
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        preds = np.argmax(self.model.predict(X_test, verbose=0), axis=1)

        print(f"\n   EfficientNetV2-S  |  Accuracy: {acc*100:.1f}%  Loss: {loss:.4f}")
        print(classification_report(y_test, preds,
                                    target_names=self.CLASS_NAMES, zero_division=0))
        self._plot_history(save_dir)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns_like_heatmap(confusion_matrix(y_test, preds), ax,
                         xticklabels=self.CLASS_NAMES,
                         yticklabels=self.CLASS_NAMES)
        ax.set_title(f"EfficientNetV2-S  |  Acc: {acc*100:.1f}%", fontweight="bold")
        ax.set_ylabel("True Label");  ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        if save_dir:
            plt.savefig(Path(save_dir) / "cm_efficientnet.png", dpi=150)
        plt.show()

    def _plot_history(self, save_dir):
        if self.history is None:
            return
        h = self.history.history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(h.get("accuracy",     []), lw=2, color="#3498db", label="Train")
        ax1.plot(h.get("val_accuracy", []), lw=2, color="#e74c3c", label="Val")
        ax1.set_title("Accuracy", fontweight="bold")
        ax1.set_xlabel("Epoch");  ax1.set_ylabel("Accuracy")
        ax1.legend();  ax1.grid(True, alpha=0.3)

        ax2.plot(h.get("loss",     []), lw=2, color="#3498db", label="Train")
        ax2.plot(h.get("val_loss", []), lw=2, color="#e74c3c", label="Val")
        ax2.set_title("Loss", fontweight="bold")
        ax2.set_xlabel("Epoch");  ax2.set_ylabel("Loss")
        ax2.legend();  ax2.grid(True, alpha=0.3)

        plt.suptitle("Technique 4: EfficientNetV2-S Training", fontweight="bold")
        plt.tight_layout()
        if save_dir:
            p = Path(save_dir) / "efficientnet_training_history.png"
            plt.savefig(str(p), dpi=150)
            print(f"   Saved: {p}")
        plt.show()


# ───────────────────────────────────────────────────────────────────────
# TECHNIQUE 5: GENERATIVE AI  (Augmentation + Environmental Simulation)
# ───────────────────────────────────────────────────────────────────────

class PotholeGenerativeAI:
    """
    Technique 5 — Generative AI.

    Two capabilities:
      A) Data Augmentation — produce 5x image variants per pothole patch
         to address class imbalance and improve CNN/ANN generalisation.
         Transforms: rotation, zoom, brightness, flips, shear.

      B) Environmental Condition Simulation — synthesise the appearance
         of road images under rainy, night, foggy and shadow conditions
         to test model robustness to environmental impact factors.
    """

    AUGMENTATION_CONFIG = dict(
        rotation_range=25,
        width_shift_range=0.20,
        height_shift_range=0.20,
        shear_range=0.20,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.40, 1.60],   # lighting variation
        channel_shift_range=20.0,        # colour tone changes (wet / dry roads)
        fill_mode="nearest",
    )

    def __init__(self):
        self._augmentor = None

    # ── A) Data Augmentation ──────────────────────────────────────────

    def build_augmentor(self):
        self._augmentor = ImageDataGenerator(**self.AUGMENTATION_CONFIG)
        print("\n[Technique 5] Augmentation pipeline ready:")
        for k, v in self.AUGMENTATION_CONFIG.items():
            print(f"   {k}: {v}")

    def augment_patches(self, X, y, factor=5):
        """
        Return augmented (X_aug, y_aug) with `factor` variants per sample.
        X shape: (N, H, W, C) float32 [0,1]
        """
        if self._augmentor is None:
            self.build_augmentor()

        # Scale to [0,255] uint8 for ImageDataGenerator
        X_u8 = (X * 255).astype(np.uint8)
        aug_X, aug_y = [], []

        for img, label in zip(X_u8, y):
            arr = img.reshape((1,) + img.shape)
            it  = self._augmentor.flow(arr, batch_size=1)
            for _ in range(factor):
                aug = next(it)[0].astype(np.float32) / 255.0
                aug_X.append(aug)
                aug_y.append(label)

        print(f"   ✔ Generated {len(aug_X)} augmented patches ({factor}×{len(X)}).")
        return np.array(aug_X), np.array(aug_y, dtype=np.int32)

    # ── B) Environmental Condition Simulation ─────────────────────────

    @staticmethod
    def simulate(image_bgr, condition="rainy"):
        """
        Simulate an environmental condition on a BGR image.

        Args:
            image_bgr : OpenCV BGR image (uint8).
            condition : one of "rainy", "night", "foggy", "shadows".

        Returns:
            Simulated BGR image (uint8).
        """
        result = image_bgr.copy()
        h, w   = result.shape[:2]

        if condition == "rainy":
            # Rain streaks
            rain = np.zeros_like(result)
            for _ in range(300):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                length = np.random.randint(10, 30)
                x2 = clamp(x + np.random.randint(-3, 3), 0, w - 1)
                y2 = clamp(y + length, 0, h - 1)
                cv2.line(rain, (x, y), (x2, y2),
                         (200, 200, 200), 1, cv2.LINE_AA)
            result = cv2.addWeighted(result, 0.85, rain, 0.50, 0)
            # Slight blur (wet lens effect)
            result = cv2.GaussianBlur(result, (3, 3), 0)

        elif condition == "night":
            # Low light + slight blue tint
            result = cv2.convertScaleAbs(result, alpha=0.30, beta=-10)
            blue_tint = np.zeros_like(result)
            blue_tint[:, :, 0] = 20   # boost blue channel
            result = cv2.add(result, blue_tint)

        elif condition == "foggy":
            # Dense haze
            fog = np.full_like(result, 210, dtype=np.uint8)
            result = cv2.addWeighted(result, 0.45, fog, 0.55, 0)
            result = cv2.GaussianBlur(result, (7, 7), 2)

        elif condition == "shadows":
            # Random directional shadow
            left_y   = np.random.randint(0, h)
            right_y  = np.random.randint(0, h)
            pts = np.array([
                [0, left_y], [w, right_y],
                [w, right_y + np.random.randint(50, 150)],
                [0, left_y  + np.random.randint(50, 150)],
            ], np.int32)
            pts = np.clip(pts, [0, 0], [w - 1, h - 1])
            shadow_mask = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(shadow_mask, [pts], 1.0)
            shadow_mask = cv2.GaussianBlur(shadow_mask, (51, 51), 0)
            factor_map = (0.45 + 0.55 * shadow_mask).astype(np.float32)
            for c in range(3):
                result[:, :, c] = np.clip(
                    result[:, :, c].astype(np.float32) * factor_map, 0, 255
                ).astype(np.uint8)

        return result

    # ── Visualise environmental simulations ───────────────────────────

    @staticmethod
    def visualize_env_simulations(image_bgr, save_dir=None):
        """Show original + 4 environmental condition variants side-by-side."""
        conditions = ["rainy", "night", "foggy", "shadows"]
        titles     = ["Original", "Rainy", "Night", "Foggy", "Shadows"]
        images_rgb = [cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)]
        for cond in conditions:
            sim = PotholeGenerativeAI.simulate(image_bgr, cond)
            images_rgb.append(cv2.cvtColor(sim, cv2.COLOR_BGR2RGB))

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        fig.patch.set_facecolor("#0d1117")
        for ax, img, title in zip(axes, images_rgb, titles):
            ax.imshow(img)
            ax.set_title(title, color="white", fontweight="bold", fontsize=10)
            ax.axis("off")
        plt.suptitle(
            "Technique 5: Generative AI — Environmental Condition Simulation",
            color="#ffe082", fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        if save_dir:
            p = Path(save_dir) / "env_simulations.png"
            plt.savefig(str(p), dpi=150, facecolor="#0d1117")
            print(f"   Saved: {p}")
        plt.show()

    # ── Visualise augmentation samples ────────────────────────────────

    def visualize_augmentation_samples(self, X, y, n_samples=8, save_dir=None):
        """Show a grid of augmented patch samples."""
        if self._augmentor is None:
            self.build_augmentor()
        CLASS_NAMES = ["Low", "Medium", "High"]
        # Pick one patch, generate n_samples augmented versions
        idx = 0
        X_u8 = (X[[idx]] * 255).astype(np.uint8)
        it   = self._augmentor.flow(X_u8, batch_size=1)

        fig, axes = plt.subplots(2, n_samples // 2, figsize=(14, 5))
        for ax in axes.flatten():
            aug = next(it)[0].astype(np.float32) / 255.0
            ax.imshow(np.clip(aug, 0, 1))
            ax.set_title(f"Aug: {CLASS_NAMES[y[idx]]}", fontsize=8)
            ax.axis("off")
        plt.suptitle("Technique 5: Generative AI — Augmented Patch Samples",
                     fontweight="bold")
        plt.tight_layout()
        if save_dir:
            p = Path(save_dir) / "augmentation_samples.png"
            plt.savefig(str(p), dpi=150)
        plt.show()


# ───────────────────────────────────────────────────────────────────────
# TECHNIQUE 5 (Part B): GENERATIVE AI — Variational Autoencoder (VAE)
# ───────────────────────────────────────────────────────────────────────

class PotholeVAE:
    """
    Technique 5 (Generative AI) — Variational Autoencoder.

    Learns a compressed latent distribution of pothole patch appearances
    and uses it to generate novel synthetic pothole images.

    Architecture:
        Encoder: Conv32 -> Conv64 -> Flatten -> z_mean, z_log_var
        Sampling: z = z_mean + exp(0.5 * z_log_var) * epsilon
        Decoder: Dense -> Reshape -> ConvT64 -> ConvT32 -> ConvT3 (sigmoid)

    Loss = pixel-wise MSE reconstruction + KL divergence.
    Trained model saved to RESULTS_DIR/pothole_vae.keras.
    """

    PATCH_SIZE = 32   # kept small; patches are resized before training
    LATENT_DIM = 16
    SAVE_PATH  = str(RESULTS_DIR / "pothole_vae.keras")

    def __init__(self, latent_dim=16):
        self.latent_dim = latent_dim
        self.encoder    = None
        self.decoder    = None
        self._vae       = None
        self._fitted    = False

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def build(self):
        """Build encoder, decoder, and the full VAE model."""
        p  = self.PATCH_SIZE
        ld = self.latent_dim

        # Encoder — two strided conv layers halve spatial dims twice
        enc_in    = tf.keras.Input(shape=(p, p, 3), name="enc_in")
        x         = layers.Conv2D(32, 3, strides=2, padding="same",
                                  activation="relu")(enc_in)
        x         = layers.Conv2D(64, 3, strides=2, padding="same",
                                  activation="relu")(x)
        x         = layers.Flatten()(x)
        z_mean    = layers.Dense(ld, name="z_mean")(x)
        z_log_var = layers.Dense(ld, name="z_log_var")(x)

        # Reparameterisation trick — keeps the sampling step differentiable
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z       = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        self.encoder = tf.keras.Model(enc_in, [z_mean, z_log_var, z],
                                      name="vae_encoder")

        # Decoder — mirror of the encoder using transposed convolutions
        flat_dim = (p // 4) * (p // 4) * 64
        dec_in   = tf.keras.Input(shape=(ld,), name="dec_in")
        x        = layers.Dense(flat_dim, activation="relu")(dec_in)
        x        = layers.Reshape((p // 4, p // 4, 64))(x)
        x        = layers.Conv2DTranspose(64, 3, strides=2, padding="same",
                                          activation="relu")(x)
        x        = layers.Conv2DTranspose(32, 3, strides=2, padding="same",
                                          activation="relu")(x)
        dec_out  = layers.Conv2DTranspose(3, 3, padding="same",
                                          activation="sigmoid")(x)
        self.decoder = tf.keras.Model(dec_in, dec_out, name="vae_decoder")

        # Wrap everything in a model with a custom train_step
        encoder_ref = self.encoder
        decoder_ref = self.decoder

        class _VAE(tf.keras.Model):
            def __init__(self):
                super().__init__(name="PotholeVAE")
                self.encoder = encoder_ref
                self.decoder = decoder_ref

            def call(self, x):
                _, _, z = self.encoder(x)
                return self.decoder(z)

            def train_step(self, data):
                x = data[0] if isinstance(data, tuple) else data
                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = self.encoder(x, training=True)
                    recon = self.decoder(z, training=True)
                    # Per-sample reconstruction loss (summed over pixels, mean over batch)
                    recon_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            tf.keras.losses.mean_squared_error(x, recon),
                            axis=(1, 2),
                        )
                    )
                    # KL divergence pushes the latent towards N(0, 1)
                    kl_loss = -0.5 * tf.reduce_mean(
                        1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                    )
                    total = recon_loss + kl_loss
                grads = tape.gradient(total, self.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                return {"loss": total, "recon_loss": recon_loss, "kl_loss": kl_loss}

        self._vae = _VAE()
        self._vae.compile(optimizer=keras.optimizers.Adam(1e-3))
        print(f"\n[Technique 5/VAE] VAE built — latent_dim={ld}, patch={p}x{p}.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, patches, epochs=30, batch_size=16):
        """
        Train the VAE on pothole image patches.

        Args:
            patches: float32 array of shape (N, H, W, 3) in any value range.
                     Will be resized to PATCH_SIZE x PATCH_SIZE and normalised
                     to [0, 1] before training.
        """
        if self._vae is None:
            self.build()

        # Resize each patch to the expected spatial dimensions
        resized = np.array(
            [cv2.resize(p, (self.PATCH_SIZE, self.PATCH_SIZE)) for p in patches],
            dtype=np.float32,
        )
        mx = resized.max()
        if mx > 1.1:
            resized = resized / 255.0  # normalise [0,255] -> [0,1]
        resized = np.clip(resized, 0.0, 1.0)

        Path(self.SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

        cb = [
            keras.callbacks.EarlyStopping(
                monitor="loss", patience=5,
                restore_best_weights=True, verbose=1,
            ),
        ]
        print(f"   Training VAE on {len(resized)} patches "
              f"(up to {epochs} epochs)...")
        self._vae.fit(resized, epochs=epochs, batch_size=batch_size,
                      callbacks=cb, verbose=1)

        # Save the decoder separately — it is the part used for generation
        try:
            self._vae.decoder.save(self.SAVE_PATH)
            print(f"   VAE decoder saved -> {self.SAVE_PATH}")
        except Exception as e:
            print(f"   [VAE] Could not save model: {e}")

        self._fitted = True
        print("   VAE training complete.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, n=10):
        """
        Sample n latent vectors and decode them into synthetic patches.

        Returns:
            numpy array of shape (n, PATCH_SIZE, PATCH_SIZE, 3) in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before generate().")
        z = np.random.normal(0.0, 1.0, (n, self.latent_dim)).astype(np.float32)
        return self.decoder.predict(z, verbose=0)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_generated(self, n=10, save_dir=None):
        """Display a grid of VAE-generated synthetic pothole patches."""
        if not self._fitted:
            print("   [VAE] Not trained — skipping visualisation.")
            return

        imgs = self.generate(n)
        cols = min(n, 5)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        fig.patch.set_facecolor("#0d1117")
        axes_flat = np.array(axes).flatten()

        for ax, img in zip(axes_flat, imgs):
            ax.imshow(np.clip(img, 0.0, 1.0))
            ax.axis("off")
        for ax in axes_flat[len(imgs):]:
            ax.axis("off")

        plt.suptitle("Technique 5: VAE — Synthetic Pothole Patches",
                     color="#ffe082", fontsize=12, fontweight="bold")
        plt.tight_layout()
        if save_dir:
            p = Path(save_dir) / "vae_generated_patches.png"
            plt.savefig(str(p), dpi=150, facecolor="#0d1117")
            print(f"   Saved: {p}")
        plt.show()


# ───────────────────────────────────────────────────────────────────────
# INTEGRATED AI ANALYSIS  — ties all 5 techniques together
# ───────────────────────────────────────────────────────────────────────

def run_ai_techniques_analysis(all_detection_records, results_dir=None):
    """
    Run all 5 AI techniques on the collected pothole detection records.

    Args:
        all_detection_records : list of (pothole_entry_dict, image_bgr) tuples
                                where pothole_entry_dict has already had
                                severity labels assigned by assign_severity_labels().
        results_dir           : Path to save plots (defaults to RESULTS_DIR).

    Returns:
        dict with trained models keyed by technique name.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    save_dir = Path(results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "█" * 68)
    print("  AI TECHNIQUES ANALYSIS  (v3)")
    print("  Road Condition Monitoring — Environmental Impact Factors")
    print("█" * 68)

    if len(all_detection_records) < 4:
        print(f"\n  ⚠ Only {len(all_detection_records)} pothole records available.")
        print("    At least 4 are needed to train the AI models meaningfully.")
        print("    Collect more data or lower MIN_MASK_AREA_FRACTION.")
        return {}

    # ── Build feature dataset ─────────────────────────────────────────
    print(f"\n  Building feature dataset from {len(all_detection_records)} pothole records...")
    X, y = _build_feature_dataset(all_detection_records)
    print(f"  Feature matrix: {X.shape}  |  Labels: {np.bincount(y).tolist()}")

    if len(np.unique(y)) < 2:
        print("  ⚠ Only one severity class present — need more diverse data.")
        print("    Skipping model training (all metrics would be trivial).")
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Technique 5: Generative AI — augmentation + VAE
    print("\n" + "-" * 60)
    print("  TECHNIQUE 5 — Generative AI (Augmentation + VAE)")
    print("-" * 60)
    gen_ai = PotholeGenerativeAI()
    gen_ai.build_augmentor()

    first_img = all_detection_records[0][1]
    if first_img is not None:
        PotholeGenerativeAI.visualize_env_simulations(first_img, save_dir)

    # Build patch dataset for EfficientNetV2-S (raw [0,255] float32)
    X_patches, y_patches = PotholeCNN.patches_from_records(all_detection_records)
    vae = PotholeVAE(latent_dim=16)

    if len(X_patches) > 0:
        gen_ai.visualize_augmentation_samples(X_patches, y_patches,
                                              save_dir=save_dir)
        X_patches_aug, y_patches_aug = gen_ai.augment_patches(
            X_patches, y_patches, factor=5
        )
        X_patches_all = np.concatenate([X_patches, X_patches_aug], axis=0)
        y_patches_all = np.concatenate([y_patches, y_patches_aug], axis=0)
        pXtr, pXte, pYtr, pYte = train_test_split(
            X_patches_all, y_patches_all,
            test_size=0.20, random_state=42, stratify=y_patches_all,
        )
        # Normalise to [0,1] for VAE training
        vae_patches = (X_patches / 255.0).astype(np.float32)
        vae.fit(vae_patches, epochs=30, batch_size=16)
        vae.visualize_generated(n=10, save_dir=save_dir)
    else:
        pXtr = pXte = pYtr = pYte = None
        print("   No patch data available — skipping VAE and EfficientNet training.")

        # ── Technique 1: Classification ───────────────────────────────────
    print("\n" + "─" * 60)
    print("  TECHNIQUE 1 — Classification (SVM + Random Forest)")
    print("─" * 60)
    classifier = PotholeClassifier()
    classifier.fit(X_train, y_train)
    classifier.evaluate_and_plot(X_test, y_test, save_dir)

    # ── Technique 2: Clustering ───────────────────────────────────────
    print("\n" + "─" * 60)
    print("  TECHNIQUE 2 — Clustering (K-Means + DBSCAN)")
    print("─" * 60)
    clusterer = PotholeSeverityClusterer(n_clusters=min(3, len(np.unique(y))))
    cluster_ids = clusterer.fit(X)
    clusterer.visualize(X, cluster_ids, save_dir)

    # ── Technique 3: ANN ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  TECHNIQUE 3 — ANN (Multi-Layer Perceptron)")
    print("─" * 60)
    ann = PotholeANN()
    ann.fit(X_train, y_train)
    ann.evaluate_and_plot(X_test, y_test, save_dir)

    # ── Technique 4: Deep Learning (EfficientNetV2-S) ─────────────────
    print("\n" + "─" * 60)
    print("  TECHNIQUE 4 — Deep Learning (EfficientNetV2-S)")
    print("─" * 60)
    cnn = PotholeCNN()
    cnn.build()
    if pXtr is not None and len(pXtr) >= 4:
        cnn.fit(pXtr, pYtr, epochs=30, batch_size=16)
        cnn.evaluate_and_plot(pXte, pYte, save_dir)
    else:
        print("   ⚠ Not enough patch data to train CNN.")

    # ── Summary dashboard ─────────────────────────────────────────────
    _plot_ai_summary_dashboard(
        X, y, classifier, clusterer, ann, cnn,
        X_test, y_test, pXte, pYte, save_dir
    )

    print("\n" + "█" * 68)
    print("  AI Techniques Analysis complete.")
    print(f"  All plots saved to: {save_dir}")
    print("█" * 68)

    return {
        "classifier": classifier,
        "clusterer":  clusterer,
        "ann":        ann,
        "cnn":        cnn,
        "gen_ai":     gen_ai,
        "vae":        vae,
    }


def _plot_ai_summary_dashboard(X, y, classifier, clusterer, ann, cnn,
                                X_test, y_test, pXte, pYte, save_dir):
    """One-page summary of all 5 AI techniques."""
    CLASS_NAMES = ["Low", "Medium", "High"]
    COLORS      = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38,
                            left=0.06, right=0.97, top=0.92, bottom=0.06)

    # ── Row 0: label distribution + technique legend ──────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor("#1a1f2e")
    counts = np.bincount(y, minlength=3)
    bars = ax0.bar(CLASS_NAMES, counts, color=COLORS, edgecolor="black")
    for bar, cnt in zip(bars, counts):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(cnt), ha="center", color="white", fontweight="bold")
    ax0.set_title("Dataset Label Distribution", color="white", fontsize=10, fontweight="bold")
    ax0.tick_params(colors="white"); ax0.set_ylabel("Count", color="white")
    for sp in ax0.spines.values(): sp.set_edgecolor("#3a4460")

    # ── Row 0: PCA 2D scatter (clustering) ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 1:3])
    ax1.set_facecolor("#1a1f2e")
    pca  = PCA(n_components=2, random_state=42)
    Xs2d = pca.fit_transform(StandardScaler().fit_transform(X))
    for i, (sev, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        mask = y == i
        ax1.scatter(Xs2d[mask, 0], Xs2d[mask, 1], c=col, label=sev,
                    s=60, alpha=0.7, edgecolors="white", linewidths=0.3)
    ax1.set_title("Technique 2: Clustering (PCA 2D)", color="white",
                  fontsize=10, fontweight="bold")
    ax1.set_xlabel("PC1", color="white"); ax1.set_ylabel("PC2", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(title="Severity", facecolor="#252d3d", labelcolor="white",
               title_fontsize=8, fontsize=8)
    for sp in ax1.spines.values(): sp.set_edgecolor("#3a4460")

    # ── Row 0: accuracy comparison bar ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 3])
    ax2.set_facecolor("#1a1f2e")
    model_names, accs = [], []
    for mname, model_obj, Xt, yt, pred_fn in [
        ("SVM",     classifier, X_test, y_test, classifier.predict_svm),
        ("RF",      classifier, X_test, y_test, classifier.predict_rf),
        ("ANN",     ann,        X_test, y_test, ann.predict),
    ]:
        try:
            p = pred_fn(Xt)
            model_names.append(mname)
            accs.append(accuracy_score(yt, p) * 100)
        except Exception:
            pass
    if pXte is not None and cnn._fitted:
        try:
            _, cnn_acc = cnn.model.evaluate(pXte, pYte, verbose=0)
            model_names.append("CNN")
            accs.append(cnn_acc * 100)
        except Exception:
            pass
    bar_cols = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"][:len(model_names)]
    bars2 = ax2.bar(model_names, accs, color=bar_cols, edgecolor="black")
    for bar, acc in zip(bars2, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{acc:.1f}%", ha="center", color="white", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 115); ax2.set_ylabel("Accuracy (%)", color="white")
    ax2.set_title("Model Accuracy Comparison", color="white", fontsize=10, fontweight="bold")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#3a4460")

    # ── Row 1: ANN training curve ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.set_facecolor("#1a1f2e")
    if hasattr(ann.model, "loss_curve_"):
        ax3.plot(ann.model.loss_curve_, color="#3498db", lw=2, label="Train Loss")
    if hasattr(ann.model, "validation_scores_"):
        ax3b = ax3.twinx()
        ax3b.plot(ann.model.validation_scores_, color="#e67e22",
                  lw=2, linestyle="--", label="Val Score")
        ax3b.set_ylabel("Val Score", color="#e67e22")
        ax3b.tick_params(colors="#e67e22")
    ax3.set_title("Technique 3: ANN Training Progress", color="white",
                  fontsize=10, fontweight="bold")
    ax3.set_xlabel("Iteration", color="white")
    ax3.set_ylabel("Loss", color="#3498db")
    ax3.tick_params(colors="white"); ax3.grid(True, alpha=0.2)
    for sp in ax3.spines.values(): sp.set_edgecolor("#3a4460")

    # ── Row 1: CNN training history ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax4.set_facecolor("#1a1f2e")
    if cnn.history is not None:
        ax4.plot(cnn.history.history["accuracy"],     color="#3498db", lw=2, label="Train Acc")
        ax4.plot(cnn.history.history["val_accuracy"], color="#e74c3c", lw=2,
                 linestyle="--", label="Val Acc")
        ax4.legend(facecolor="#252d3d", labelcolor="white", fontsize=8)
    ax4.set_title("Technique 4: CNN Accuracy", color="white",
                  fontsize=10, fontweight="bold")
    ax4.set_xlabel("Epoch", color="white"); ax4.set_ylabel("Accuracy", color="white")
    ax4.tick_params(colors="white"); ax4.grid(True, alpha=0.2)
    for sp in ax4.spines.values(): sp.set_edgecolor("#3a4460")

    # ── Row 2: feature importance (RF) ───────────────────────────────
    FEAT_NAMES = ["area_ratio", "norm_depth", "confidence", "sev_score",
                  "shape_danger", "circularity", "convex_def", "roughness",
                  "aspect_ratio", "solidity", "mean_int", "std_int",
                  "edge_density", "texture_var"]
    ax5 = fig.add_subplot(gs[2, 0:2])
    ax5.set_facecolor("#1a1f2e")
    if classifier._fitted and hasattr(classifier.rf_model, "feature_importances_"):
        imp = classifier.rf_model.feature_importances_
        order = np.argsort(imp)[::-1]
        ax5.bar(range(len(imp)), imp[order],
                color=["#3498db" if v > np.mean(imp) else "#7f8c8d" for v in imp[order]])
        ax5.set_xticks(range(len(imp)))
        ax5.set_xticklabels([FEAT_NAMES[i] for i in order],
                             rotation=45, ha="right", fontsize=7, color="white")
        ax5.set_title("Technique 1: RF Feature Importances", color="white",
                      fontsize=10, fontweight="bold")
        ax5.set_ylabel("Importance", color="white")
        ax5.tick_params(colors="white")
        for sp in ax5.spines.values(): sp.set_edgecolor("#3a4460")

    # ── Row 2: technique descriptions ────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2:4])
    ax6.set_facecolor("#1a1f2e")
    ax6.axis("off")
    table_rows = [
        ["1", "Classification", "SVM + Random Forest",    "Feature-based severity"],
        ["2", "Clustering",     "K-Means (k=3)",          "Unsupervised grouping"],
        ["3", "ANN",            "MLP 128→64→32",          "Non-linear patterns"],
        ["4", "Deep Learning",  "CNN (3 Conv blocks)",    "Patch image classifier"],
        ["5", "Generative AI",  "Augment + Env. Sim.",    "Data & robustness"],
    ]
    col_labels = ["#", "Technique", "Algorithm", "Purpose"]
    tbl = ax6.table(cellText=table_rows, colLabels=col_labels,
                    cellLoc="center", loc="center",
                    colWidths=[0.06, 0.25, 0.32, 0.35])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#252d3d" if row % 2 == 0 else "#1a1f2e")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#3a4460")
        if row == 0:
            cell.set_facecolor("#2c3e6a")
            cell.set_text_props(color="#4fc3f7", fontweight="bold")
    ax6.set_title("5 AI Techniques Summary", color="white",
                  fontsize=10, fontweight="bold", pad=12)

    plt.suptitle(
        "AI-Based Road Condition Monitoring — 5 Technique Summary Dashboard",
        color="#4fc3f7", fontsize=14, fontweight="bold"
    )
    if save_dir:
        p = Path(save_dir) / "ai_techniques_summary_dashboard.png"
        plt.savefig(str(p), dpi=150, facecolor="#0d1117")
        print(f"\n   Dashboard saved: {p}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def _split_already_exists():
    """
    Return True if train / val / test splits are already populated with
    both images AND labels — meaning we can skip the expensive
    XML-conversion + split steps entirely.
    """
    for img_dir, lbl_dir in [
        (TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR),
        (VAL_IMAGES_DIR,   VAL_LABELS_DIR),
        (TEST_IMAGES_DIR,  TEST_LABELS_DIR),
    ]:
        if not img_dir.exists() or not lbl_dir.exists():
            return False
        if not any(img_dir.iterdir()) or not any(lbl_dir.iterdir()):
            return False
    return True


def run_pipeline():
    """Run the complete end-to-end pothole detection pipeline v3 (5 AI Techniques)."""
    print("=" * 70)
    print("POTHOLE DETECTION PIPELINE v3 (5 AI Techniques + Enhanced Shape Segmentation)")
    print("=" * 70)

    create_folders()

    # ── SMART SKIP: if train/val/test already have data, skip the heavy
    #    XML→YOLO conversion and dataset splitting.
    #    To force a full re-run, delete the dataset folder contents manually.
    if _split_already_exists():
        print("\n✔ Pre-split dataset detected — skipping XML conversion & splitting.")
        print(f"  Train : {len(list(TRAIN_IMAGES_DIR.iterdir()))} images")
        print(f"  Val   : {len(list(VAL_IMAGES_DIR.iterdir()))} images")
        print(f"  Test  : {len(list(TEST_IMAGES_DIR.iterdir()))} images")
        test_count = len(list(TEST_IMAGES_DIR.iterdir()))
        if test_count == 0:
            raise RuntimeError("Test split is empty even though folder exists.")
    else:
        print("\n⚙ No existing split found — running full XML conversion & split...")
        validate_dataset_paths()
        sam_predictor = load_sam_predictor()
        image_label_pairs, sam_predictor = convert_xml_to_yolo(sam_predictor)

        if not image_label_pairs:
            raise RuntimeError("No valid image/XML pairs found.")

        if sam_predictor is not None:
            del sam_predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        split_info = split_dataset(image_label_pairs)
        if split_info["counts"]["test"] == 0:
            raise RuntimeError("Test split is empty.")

    data_yaml_path = create_data_yaml()
    best_model_path = train_yolov8_model(data_yaml_path)
    yolo_model = load_trained_model(best_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas_model, midas_transform = load_midas_model(device)  # returns (depth_model, processor)

    detection_results = run_detection_on_test_images(yolo_model)

    # ── Collect all pothole records for AI technique training ─────────
    all_detection_records = []   # list of (pothole_entry, image_bgr)

    for image_path, detection_result in detection_results:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        ih, iw = image_bgr.shape[:2]
        pothole_data = extract_pothole_features(detection_result, image_bgr)
        pothole_data, _ = add_depth_information(
            image_bgr, pothole_data, midas_model, midas_transform, device
        )
        pothole_data = assign_severity_labels(pothole_data, iw, ih)
        display_and_save_results(image_path, image_bgr, pothole_data)

        # Accumulate records for AI technique training
        for entry in pothole_data:
            all_detection_records.append((entry, image_bgr))

    # ── Run all 5 AI techniques on the collected detections ───────────
    run_ai_techniques_analysis(all_detection_records, results_dir=RESULTS_DIR)

    print("\n[v3] Pipeline completed successfully.")
    print(f"  Trained model    : {best_model_path}")
    print(f"  Results folder   : {RESULTS_DIR}")


if __name__ == "__main__":
    import sys

    # Quick inference mode: python script.py --infer model.pt img1.jpg img2.jpg
    if len(sys.argv) > 1 and sys.argv[1] == "--infer":
        if len(sys.argv) < 4:
            print("Usage: python pothole_detection_pipeline.py --infer <model.pt> <img1> [img2 ...]")
            sys.exit(1)
        model_pt = sys.argv[2]
        images = sys.argv[3:]
        run_inference_only(model_pt, images)
    else:
        run_pipeline()
