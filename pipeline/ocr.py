#!/usr/bin/env python3
"""dawaIndia pipeline — Stage 2: OCR.

Extracts text from the cropped prescription image. Cascade strategy:
EasyOCR first (fast, general-purpose, decent on print and cursive), then
TrOCR (slower, fine-tuned on handwriting) for any individual line EasyOCR
was unsure about.

Public contract (must match eval.py's STAGE CONTRACTS docstring exactly,
since eval.py imports and calls this directly):

    def extract_text(image_path: str) -> dict:
        {"success": bool, "raw_text": str, "confidence": float}

Extra fields beyond that contract, for debugging and for later stages:
  - "status": "error" | "partial" | "success" -- human-readable band over
    the same `confidence` number eval.py already reads.
  - "lines": [{"text": str, "confidence": float, "bbox": [[x,y]x4],
    "engine": "easyocr" | "trocr"}, ...] in top-to-bottom reading order.

TROCR ESCALATION IS REAL BUT UNVERIFIABLE IN THIS ENVIRONMENT. TrOCR is a
transformer encoder-decoder (microsoft/trocr-base-handwritten via
`transformers`), loaded from Hugging Face Hub on first use. This dev
sandbox's egress policy blocks huggingface.co outright (403 at the proxy,
confirmed via the proxy's own status endpoint -- not a transient failure,
not something to retry around). So the escalation path below is
implemented and wrapped in try/except -- if the model can't be loaded (here,
always; in an environment with normal internet access, rarely), we fall
back to the EasyOCR result for that line rather than failing the whole
stage -- but its actual accuracy/latency on real handwriting is UNVERIFIED.
Don't trust it beyond "doesn't crash" until it's been run somewhere that
can reach Hugging Face. Escalation is capped at MAX_TROCR_LINES lines per
image (lowest-confidence first) for when it does run, since each line is a
full transformer forward pass. TrOCR's own confidence is the mean
per-token max softmax probability from generation, not a fixed number.

PER-IMAGE LATENCY DOES NOT MEET THE 3-SECOND TARGET. Measured on this
4-core CPU-only sandbox (no GPU) against a real 3024x4032 photo: EasyOCR's
CRAFT detector dominates the cost, ~28s at full resolution. Downscaling
the frame fed to detection (`EASYOCR_CANVAS_SIZE` below) cuts that
substantially -- ~4.4s at 900px, ~6-7s at 1200px -- with no measurable
recognition-quality change across that range (see below), but does not
get under 3s. That gap is a hardware/library ceiling, not a bug: CRAFT is
a real CNN detector and this box has no GPU. Meeting <3s would need GPU
inference or a lighter detector, out of scope for this phase.

WHAT ACTUALLY LIMITS ACCURACY: resolution isn't the bottleneck. Across
canvas_size 900-2560 on IMG_001, zero of the 6 ground-truth brand names
were recoverable (not even as substrings) at any resolution -- EasyOCR's
recognition network reads the printed letterhead/footer text well (high
confidence, correct) but cannot read this doctor's cursive handwriting at
all. That's a model-capability gap, and it's exactly the gap TrOCR (a
handwriting-tuned model) exists to close -- untestable here per above.

NO HALLUCINATION: both engines return only what they detect. A line
neither engine is confident about stays in the output at its low
confidence rather than being smoothed over -- that confidence is exactly
what lets a later stage (or the CLAUDE.md rule "confidence must be >80% to
show a medicine clearly") decide whether to trust it.
"""

from pathlib import Path

import cv2

LOW_CONFIDENCE_THRESHOLD = 0.80  # EasyOCR lines below this are escalation candidates
MAX_TROCR_LINES = 5              # hard cap on escalations per image, for the latency budget
SUCCESS_THRESHOLD = 0.85         # overall confidence >= this -> status "success"
EASYOCR_CANVAS_SIZE = 1000  # detector-only resize target; see docstring for the speed/quality data

_easyocr_reader = None
_trocr_processor = None
_trocr_model = None


def extract_text(image_path: str) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        return _error_result()

    try:
        detections = _run_easyocr(image)
    except Exception:
        return _error_result()

    if not detections:
        return {"success": True, "raw_text": "", "confidence": 0.0, "status": "partial", "lines": []}

    lines = [
        {"bbox": [[float(px), float(py)] for px, py in bbox], "text": text,
         "confidence": float(conf), "engine": "easyocr"}
        for bbox, text, conf in detections
    ]
    lines.sort(key=lambda ln: _line_center_y(ln["bbox"]))

    escalate = sorted(lines, key=lambda ln: ln["confidence"])[:MAX_TROCR_LINES]
    for line in escalate:
        if line["confidence"] >= LOW_CONFIDENCE_THRESHOLD:
            continue
        trocr_text, trocr_conf = _try_trocr(image, line["bbox"])
        if trocr_text is not None and trocr_conf > line["confidence"]:
            line["text"] = trocr_text
            line["confidence"] = trocr_conf
            line["engine"] = "trocr"

    raw_text = "\n".join(ln["text"] for ln in lines)
    confidence = round(sum(ln["confidence"] for ln in lines) / len(lines), 3)
    # "partial" (not "error") below SUCCESS_THRESHOLD too: the engine ran and
    # returned something for every case that reaches this line -- a hard
    # "error" is reserved for cases that couldn't produce output at all.
    status = "success" if confidence >= SUCCESS_THRESHOLD else "partial"

    return {"success": True, "raw_text": raw_text, "confidence": confidence,
            "status": status, "lines": lines}


def _run_easyocr(image):
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return _easyocr_reader.readtext(rgb, canvas_size=EASYOCR_CANVAS_SIZE)


def _line_center_y(bbox):
    return sum(p[1] for p in bbox) / len(bbox)


def _try_trocr(image, bbox):
    """Re-run one line's crop through TrOCR. Returns (text, confidence) or (None, 0.0)."""
    try:
        crop = _crop_bbox(image, bbox)
        if crop is None:
            return None, 0.0
        processor, model = _load_trocr()
        import torch
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
        with torch.no_grad():
            output = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True, max_new_tokens=64)
        text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
        confidence = _trocr_confidence(output)
        return text, confidence
    except Exception:
        return None, 0.0


def _load_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_processor is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        _trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        _trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        _trocr_model.eval()
    return _trocr_processor, _trocr_model


def _trocr_confidence(generate_output):
    import torch
    if not generate_output.scores:
        return 0.5
    step_probs = [torch.softmax(step, dim=-1).max().item() for step in generate_output.scores]
    return float(sum(step_probs) / len(step_probs)) if step_probs else 0.5


def _crop_bbox(image, bbox, padding=4):
    img_h, img_w = image.shape[:2]
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x0 = max(0, int(min(xs)) - padding)
    y0 = max(0, int(min(ys)) - padding)
    x1 = min(img_w, int(max(xs)) + padding)
    y1 = min(img_h, int(max(ys)) + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]


def _error_result():
    return {"success": False, "raw_text": "", "confidence": 0.0, "status": "error", "lines": []}


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <image_path>")
        sys.exit(1)

    t0 = time.time()
    result = extract_text(sys.argv[1])
    elapsed = time.time() - t0

    print(f"status={result['status']} confidence={result['confidence']} elapsed={elapsed:.2f}s")
    print("--- raw_text ---")
    print(result["raw_text"])
    print("--- lines ---")
    for ln in result.get("lines", []):
        print(f"  [{ln['engine']:7s} {ln['confidence']:.2f}] {ln['text']!r}")
