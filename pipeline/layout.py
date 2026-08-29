#!/usr/bin/env python3
"""dawaIndia pipeline — Stage 1: Layout detection.

Finds the prescription paper's boundary inside a photographed image and
crops to it. Classic document-scanner CV, no model download: grayscale ->
blur -> Canny edges -> largest external contour -> bounding rect (+margin).

Public contract (must match eval.py's STAGE CONTRACTS docstring exactly,
since eval.py imports and calls this directly):

    def detect_prescription_box(image_path: str) -> dict:
        {"success": bool, "bbox": [x, y, w, h] | None,
         "cropped_image_path": str | None}

`confidence` (float, 0-1) is included as an extra field beyond that
contract: Stage 6 (confidence/scorer.py) combines per-stage confidence
signals, and layout is where that signal originates for this stage.

Three possible outcomes:
  - Hard error (image missing / unreadable / can't be written): success=False,
    confidence=0.0, bbox=None, cropped_image_path=None. Nothing downstream
    can run without an image, so this is a real failure, not an abstain.
  - Confident detection (a large, sufficiently rectangular contour found):
    success=True, confidence in [0.95, 0.99].
  - No confident boundary found (e.g. paper fills the whole frame already,
    or edges are too weak/noisy to isolate): success=True but confidence is
    low (0.5) and bbox is the full image. We still hand a usable image
    downstream rather than dead-ending the pipeline on an edge case; the
    low confidence is what lets a later stage (or the pharmacist-facing UI,
    which per CLAUDE.md only shows a medicine clearly above 80% confidence)
    treat the result as unverified.
"""

from pathlib import Path

import cv2
import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
DEBUG_CROPS_DIR = PIPELINE_DIR / "debug_crops"

# Tuning constants
MIN_AREA_RATIO = 0.15   # candidate contour's bbox must cover >=15% of the frame
MIN_EXTENT = 0.5        # candidate contour must fill >=50% of its own bounding rect
MARGIN_FRACTION = 0.03  # crop margin around a detected box, as a fraction of its size
FALLBACK_CONFIDENCE = 0.5


def detect_prescription_box(image_path: str) -> dict:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        return _error_result()

    img_h, img_w = image.shape[:2]
    bbox = _find_paper_bbox(image, img_w, img_h)

    if bbox is not None:
        x0, y0, x1, y1 = _add_margin(bbox, img_w, img_h)
        extent = bbox[4]
        confidence = round(min(0.99, 0.95 + 0.04 * extent), 2)
    else:
        x0, y0, x1, y1 = 0, 0, img_w, img_h
        confidence = FALLBACK_CONFIDENCE

    cropped = image[y0:y1, x0:x1]
    cropped_path = _save_crop(cropped, image_path)
    if cropped_path is None:
        return _error_result()

    return {
        "success": True,
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "cropped_image_path": str(cropped_path),
        "confidence": confidence,
    }


def _find_paper_bbox(image, img_w, img_h):
    """Return (x, y, w, h, extent) for the paper boundary, or None if not confident."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w == 0 or h == 0:
        return None

    area_ratio = (w * h) / (img_w * img_h)
    extent = cv2.contourArea(largest) / (w * h)
    if area_ratio < MIN_AREA_RATIO or extent < MIN_EXTENT:
        return None

    return (x, y, w, h, extent)


def _add_margin(bbox, img_w, img_h):
    x, y, w, h, _extent = bbox
    margin_x = int(w * MARGIN_FRACTION)
    margin_y = int(h * MARGIN_FRACTION)
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(img_w, x + w + margin_x)
    y1 = min(img_h, y + h + margin_y)
    return x0, y0, x1, y1


def _save_crop(cropped, source_image_path):
    DEBUG_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    image_id = source_image_path.stem
    out_path = DEBUG_CROPS_DIR / f"{image_id}_cropped.jpg"
    if not cv2.imwrite(str(out_path), cropped):
        return None
    return out_path


def _error_result():
    return {"success": False, "bbox": None, "cropped_image_path": None, "confidence": 0.0}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <image_path>")
        sys.exit(1)

    result = detect_prescription_box(sys.argv[1])
    print(result)
    if result["success"]:
        print(f"bbox={result['bbox']} confidence={result['confidence']} "
              f"-> {result['cropped_image_path']}")
    else:
        print("FAILED to detect/crop a prescription box.")
