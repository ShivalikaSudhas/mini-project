import os
import sys
import cv2
import numpy as np

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Suppress verbose paddle logging
os.environ["GLOG_minloglevel"] = "2"
from paddleocr import PaddleOCR

# -------------------------------------------------------------------------
# OCR-DERIVED NUMERICAL FEATURE EXTRACTOR
# -------------------------------------------------------------------------
# Extracts an 11-dimensional numerical feature vector from an ID document.
#
# Feature Vector Components:
#   [Confidence Metrics (6)]
#     1. mean_confidence   : Average OCR confidence across all fields
#     2. min_confidence    : Lowest confidence score observed
#     3. max_confidence    : Highest confidence score observed
#     4. std_confidence    : Standard deviation of confidence
#     5. low_conf_count    : Count of boxes with confidence < 0.85
#     6. low_conf_ratio    : Fraction of boxes with low confidence
#   [Geometric & Layout Metrics (5)]
#     7. num_text_boxes    : Total count of detected text regions
#     8. text_area_ratio   : Sum of text bounding box areas / total image area
#     9. mean_aspect_ratio : Average width-to-height ratio of text boxes
#    10. box_height_std    : Variance in text box heights (font size inconsistency)
#    11. mean_char_density : Total character count / total text box area
# -------------------------------------------------------------------------

OCR_FEATURE_NAMES = [
    'mean_confidence', 'min_confidence', 'max_confidence', 'std_confidence',
    'low_conf_count', 'low_conf_ratio', 'num_text_boxes', 'text_area_ratio',
    'mean_aspect_ratio', 'box_height_std', 'mean_char_density'
]

def extract_ocr_features(image_path, ocr_engine=None):
    """
    Extracts an 11-dimensional numerical feature vector from an ID document image.
    Returns:
        feature_vector (np.ndarray): Shape (11,), dtype float32
        feature_dict (dict): Feature name-to-value mapping
    """
    if ocr_engine is None:
        ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at '{image_path}'")
        
    img_h, img_w = image.shape[:2]
    img_total_area = float(img_h * img_w)
    
    ocr_results = ocr_engine.ocr(image_path, cls=True)
    
    # Graceful fallback for unreadable / blank cards
    if not ocr_results or not ocr_results[0]:
        feature_dict = {col: 0.0 for col in OCR_FEATURE_NAMES}
        return np.zeros(11, dtype=np.float32), feature_dict

    confidences = []
    box_areas = []
    box_aspect_ratios = []
    box_heights = []
    total_characters = 0

    for item in ocr_results[0]:
        box = item[0]
        text, score = item[1]
        
        confidences.append(float(score))
        total_characters += len(text.replace(" ", ""))
        
        # Bounding box geometry
        pts = np.array(box, dtype=np.float32)
        box_width = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2.0
        box_height = max(1.0, (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2.0)
        
        box_areas.append(box_width * box_height)
        box_aspect_ratios.append(box_width / box_height)
        box_heights.append(box_height)

    # 1. Statistical Confidence Metrics
    conf_array = np.array(confidences)
    mean_conf = float(np.mean(conf_array))
    min_conf = float(np.min(conf_array))
    max_conf = float(np.max(conf_array))
    std_conf = float(np.std(conf_array))
    low_conf_count = int(np.sum(conf_array < 0.85))
    low_conf_ratio = float(low_conf_count / len(conf_array))
    
    # 2. Geometric & Density Metrics
    num_boxes = len(conf_array)
    total_box_area = float(np.sum(box_areas))
    text_area_ratio = float(total_box_area / img_total_area) if img_total_area > 0 else 0.0
    mean_aspect_ratio = float(np.mean(box_aspect_ratios))
    box_height_std = float(np.std(box_heights))
    mean_char_density = float(total_characters / total_box_area) if total_box_area > 0 else 0.0

    feature_dict = {
        "mean_confidence": round(mean_conf, 4),
        "min_confidence": round(min_conf, 4),
        "max_confidence": round(max_conf, 4),
        "std_confidence": round(std_conf, 4),
        "low_conf_count": float(low_conf_count),
        "low_conf_ratio": round(low_conf_ratio, 4),
        "num_text_boxes": float(num_boxes),
        "text_area_ratio": round(text_area_ratio, 4),
        "mean_aspect_ratio": round(mean_aspect_ratio, 4),
        "box_height_std": round(box_height_std, 4),
        "mean_char_density": round(mean_char_density, 6),
    }

    feature_vector = np.array(list(feature_dict.values()), dtype=np.float32)
    return feature_vector, feature_dict
