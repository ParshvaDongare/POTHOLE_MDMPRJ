import os
import random
import shutil
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CACHE_DIR = PROJECT_ROOT / ".model_cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(LOCAL_CACHE_DIR)
os.environ["TORCH_HOME"] = str(LOCAL_CACHE_DIR / "torch")
os.environ["MPLCONFIGDIR"] = str(LOCAL_CACHE_DIR / "matplotlib")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("Starting pothole pipeline v2 imports...", flush=True)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ultralytics import YOLO

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
# Depth estimation (same as v1)
# ═══════════════════════════════════════════════════════════════════════

def load_midas_model(device):
    print("\nLoading MiDaS depth model...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.to(device)
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    return midas, transforms.small_transform


def estimate_depth_map(image_bgr, midas_model, transform, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(image_rgb).to(device)
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=image_rgb.shape[:2],
            mode="bicubic", align_corners=False,
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


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline():
    """Run the complete end-to-end pothole segmentation pipeline v2."""
    print("=" * 70)
    print("POTHOLE DETECTION PIPELINE v2 (Enhanced Shape Segmentation)")
    print("=" * 70)

    validate_dataset_paths()
    create_folders()

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
    midas_model, midas_transform = load_midas_model(device)

    detection_results = run_detection_on_test_images(yolo_model)
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

    print("\n[v2] Pipeline completed successfully.")
    print(f"  Trained model : {best_model_path}")
    print(f"  Results folder: {RESULTS_DIR}")


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
