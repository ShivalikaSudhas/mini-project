import os
import sys
import glob
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR
from ocr_feature_extractor import extract_ocr_features

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Suppress verbose paddle logging
os.environ["GLOG_minloglevel"] = "2"

# -------------------------------------------------------------------------
# SYNTHETIC MULTI-DOCUMENT DATASET GENERATOR & OCR BATCH PROCESSOR
# -------------------------------------------------------------------------
# Generates 5 fictional, non-issuable document types (zero real PII):
#   1. national_id
#   2. driver_license
#   3. student_card
#   4. employee_badge
#   5. health_card
# Injects realistic digital tampering attacks into forged samples:
#   - Text Splicing (Font & background mismatch)
#   - Copy-Move (Digit duplication)
#   - Localized Gaussian Edge Blurring
# -------------------------------------------------------------------------

DOC_TEMPLATES = {
    "national_id": {
        "title": "NATIONAL IDENTITY CARD",
        "header_color": (30, 60, 120),
        "fields": ["FULL NAME", "CITIZEN ID", "DOB", "NATIONALITY", "EXPIRES"]
    },
    "driver_license": {
        "title": "MOTOR VEHICLE DRIVER LICENSE",
        "header_color": (40, 110, 70),
        "fields": ["NAME", "LICENSE NO", "CLASS", "ISSUE DATE", "RESTRICTIONS"]
    },
    "student_card": {
        "title": "UNIVERSITY STUDENT IDENTIFICATION",
        "header_color": (130, 30, 40),
        "fields": ["STUDENT NAME", "ROLL NO", "DEPARTMENT", "VALID UNTIL", "CAMPUS"]
    },
    "employee_badge": {
        "title": "CORPORATE ACCESS BADGE",
        "header_color": (70, 70, 80),
        "fields": ["EMPLOYEE", "BADGE ID", "DESIGNATION", "SECURITY LEVEL", "OFFICE"]
    },
    "health_card": {
        "title": "HEALTH INSURANCE BENEFICIARY CARD",
        "header_color": (0, 115, 130),
        "fields": ["POLICY HOLDER", "POLICY NO", "PLAN TYPE", "EFFECTIVE DATE", "GROUP ID"]
    }
}

FIRST_NAMES = ["ALEX", "JORDAN", "TAYLOR", "MORGAN", "CASEY", "RILEY", "CHRIS", "SAM"]
LAST_NAMES = ["SMITH", "PATEL", "KUMAR", "JOHNSON", "GARCIA", "LEE", "BROWN", "SINGH"]

def generate_base_card(doc_type, sample_idx):
    """Draws a clean, synthetic, fictional ID card."""
    template = DOC_TEMPLATES[doc_type]
    w, h = 640, 400
    
    card = Image.new("RGB", (w, h), color=(248, 250, 252))
    draw = ImageDraw.Draw(card)
    
    # Header banner
    draw.rectangle([(0, 0), (w, 60)], fill=template["header_color"])
    draw.text((25, 20), template["title"], fill=(255, 255, 255))
    
    # Photo box
    photo_box = [(40, 90), (170, 260)]
    draw.rectangle(photo_box, fill=(210, 220, 230), outline=(160, 170, 180), width=2)
    draw.text((65, 165), "[PHOTO]", fill=(100, 110, 120))
    
    fname = FIRST_NAMES[(sample_idx + len(doc_type)) % len(FIRST_NAMES)]
    lname = LAST_NAMES[(sample_idx * 3) % len(LAST_NAMES)]
    
    field_values = [
        f"{fname} {lname}",
        f"{doc_type[:3].upper()}-88294{sample_idx:02d}",
        f"0{sample_idx%9+1}/14/199{sample_idx%9}",
        "AUTHORIZED / REGULAR",
        "12/2029"
    ]
    
    text_x = 200
    start_y = 95
    for i, field_name in enumerate(template["fields"]):
        y = start_y + (i * 48)
        draw.text((text_x, y), f"{field_name}:", fill=(100, 100, 110))
        draw.text((text_x, y + 18), field_values[i], fill=(20, 25, 30))
        
    draw.line([(0, 360), (w, 360)], fill=(200, 205, 215), width=1)
    draw.text((25, 370), "SPECIMEN DOCUMENT - FOR EXPERIMENTAL CSE EVALUATION ONLY", fill=(140, 145, 155))
    
    return cv2.cvtColor(np.array(card), cv2.COLOR_RGB2BGR)


def apply_tampering(img, attack_type="splicing"):
    tampered = img.copy()
    if attack_type == "splicing":
        cv2.rectangle(tampered, (198, 158), (420, 185), (238, 240, 243), -1)
        cv2.putText(tampered, "FORGED-99401X", (200, 178),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.58, (5, 5, 5), 1, cv2.LINE_AA)
    elif attack_type == "copy_move":
        src_patch = tampered[20:50, 40:140].copy()
        tampered[158:188, 200:300] = src_patch
    elif attack_type == "blur_resample":
        roi = tampered[155:185, 195:430]
        blurred = cv2.GaussianBlur(roi, (7, 7), 3.0)
        tampered[155:185, 195:430] = blurred
    return tampered


def build_synthetic_benchmark(dataset_root="dataset", samples_per_class=3):
    print(f"[INFO] Generating balanced benchmark dataset in '{dataset_root}'...")
    attacks = ["splicing", "copy_move", "blur_resample"]
    total = 0
    
    for doc_type in DOC_TEMPLATES.keys():
        for status in ["genuine", "forged"]:
            out_dir = os.path.join(dataset_root, doc_type, status)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(1, samples_per_class + 1):
                img = generate_base_card(doc_type, sample_idx=i)
                if status == "forged":
                    img = apply_tampering(img, attack_type=attacks[i % len(attacks)])
                cv2.imwrite(os.path.join(out_dir, f"{doc_type}_{status}_{i}.png"), img)
                total += 1
    print(f"[SUCCESS] Created {total} synthetic documents across 5 document types.")


def extract_ocr_dataset_csv(dataset_root="dataset", output_csv="ocr_features_dataset.csv"):
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    image_paths = glob.glob(f"{dataset_root}/*/*/*.png") + glob.glob(f"{dataset_root}/*/*/*.jpg")
    print(f"[INFO] Processing OCR features for {len(image_paths)} images...")
    
    records = []
    for path in image_paths:
        parts = os.path.normpath(path).split(os.sep)
        doc_type = parts[-3]
        label_str = parts[-2].lower()
        label = 0 if label_str == "genuine" else 1
        
        _, f_dict = extract_ocr_features(path, ocr_engine)
        records.append({
            "filepath": path,
            "filename": parts[-1],
            "doc_type": doc_type,
            "label": label,
            "label_name": label_str,
            **f_dict
        })
        
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Saved OCR feature matrix to: '{output_csv}'")
    return df

if __name__ == "__main__":
    build_synthetic_benchmark("dataset", samples_per_class=3)
    extract_ocr_dataset_csv("dataset", "ocr_features_dataset.csv")
