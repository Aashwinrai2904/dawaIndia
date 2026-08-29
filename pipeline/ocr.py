#!/usr/bin/env python3
"""dawaIndia pipeline — Stage 2: OCR (Phase 3.5 — 4-layer tiered cascade).

Tries progressively slower/heavier engines until one reports confidence
above its own threshold: Donut (handwriting-tuned encoder-decoder) ->
PaddleOCR (general detector+recognizer with angle correction) -> TrOCR
(handwriting-tuned transformer) -> EasyOCR (general, fast, last resort).

Public contract (must match eval.py's STAGE CONTRACTS docstring exactly,
since eval.py imports and calls this directly):

    def extract_text(image_path: str) -> dict:
        {"success": bool, "raw_text": str, "confidence": float}

`extract_text` is a thin adapter over `extract_text_from_image` (the
4-layer cascade itself), which returns a different, richer shape --
{"status", "layer", "text"/"error", "confidence", "model"} -- that doesn't
match eval.py's contract on its own (no "success" key, "text" instead of
"raw_text"). Without the adapter, eval.py would import this module fine
but silently never find a usable `extract_text`, and OCR would keep
reporting SKIPPED exactly as if this file didn't exist -- a wiring bug,
not a design choice, so it's fixed here rather than left as-is.

ENVIRONMENT CONSTRAINT CARRIED OVER FROM PHASE 3, STILL TRUE: this dev
sandbox's egress policy blocks huggingface.co (403 at the proxy, confirmed
via the proxy's own status endpoint). Layers 1 and 3 both load a
Hugging-Face-hosted checkpoint on first use, so both fail here every time
-- not flakily, not "sometimes" -- and the cascade falls through to
whichever of Layer 2 / Layer 4 can actually load. This is a property of
this sandbox's network policy, not a bug in the cascade logic; it will
behave differently in an environment with normal internet access. See the
Phase 3.5 commit message for what was actually measured here.

Layer 1's checkpoint (chinmays18/medical-prescription-ocr) is a
third-party community fine-tune, not an Anthropic- or Microsoft-published
one like the others. It could not be inspected before use (same HF
block), so its provenance/trustworthiness is unverified -- worth a closer
look before depending on it in a real deployment, independent of whether
it's currently reachable.

CONFIDENCE VALUES ARE PER-LAYER CONSTANTS, NOT MEASURED (mostly) -- worth
flagging against CLAUDE.md's "measure everything" / "no hallucination"
rules, which this file otherwise follows (each layer returns only text it
actually produced, nothing invented). Donut's 0.94 and TrOCR's 0.82 are
fixed regardless of output quality; PaddleOCR and EasyOCR at least average
their own per-detection confidences before applying a cap. A layer that
loads successfully but produces garbage still reports its fixed
confidence. Kept as specified rather than changed unilaterally, since it
was an explicit design choice -- flagging it here so it's a decision
someone revisits on purpose, not a surprise found later.
"""

import logging
from typing import Any, Dict

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_easyocr_reader = None
_paddle_ocr = None
_donut = None
_trocr = None


def layer_1_donut(image_path: str) -> Dict[str, Any]:
    try:
        global _donut
        logger.info("Layer 1: Loading Donut...")
        if _donut is None:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            processor = DonutProcessor.from_pretrained("chinmays18/medical-prescription-ocr")
            model = VisionEncoderDecoderModel.from_pretrained("chinmays18/medical-prescription-ocr")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            _donut = (processor, model, device)
        processor, model, device = _donut
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        task_prompt = "<s_ocr>"
        decoder_input_ids = processor.tokenizer(task_prompt, return_tensors="pt").input_ids.to(device)
        generated_ids = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512, num_beams=1, early_stopping=True)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"status": "success", "layer": 1, "text": text, "confidence": 0.94, "model": "Donut"}
    except Exception as e:
        logger.warning(f"Layer 1 failed: {str(e)}")
        return {"status": "failed", "layer": 1, "error": str(e)}


def layer_2_paddle_layoutlm(image_path: str) -> Dict[str, Any]:
    try:
        global _paddle_ocr
        logger.info("Layer 2: Loading PaddleOCR...")
        if _paddle_ocr is None:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = _paddle_ocr.ocr(image_path, cls=True)
        # PaddleOCR result shape: [ [ [box, (text, conf)], ... ] ] (one outer list per input image).
        page = result[0] if result else []
        paddle_text = "\n".join(line[1][0] for line in page if line)
        paddle_confidence = sum(line[1][1] for line in page if line) / len(page) if page else 0
        return {"status": "success", "layer": 2, "text": paddle_text, "confidence": min(paddle_confidence, 0.90), "model": "PaddleOCR"}
    except Exception as e:
        logger.warning(f"Layer 2 failed: {str(e)}")
        return {"status": "failed", "layer": 2, "error": str(e)}


def layer_3_trocr(image_path: str) -> Dict[str, Any]:
    try:
        global _trocr
        logger.info("Layer 3: Loading TrOCR...")
        if _trocr is None:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            _trocr = (processor, model, device)
        processor, model, device = _trocr
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_length=512)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"status": "success", "layer": 3, "text": text, "confidence": 0.82, "model": "TrOCR"}
    except Exception as e:
        logger.warning(f"Layer 3 failed: {str(e)}")
        return {"status": "failed", "layer": 3, "error": str(e)}


def layer_4_easyocr(image_path: str) -> Dict[str, Any]:
    try:
        global _easyocr_reader
        logger.info("Layer 4: Loading EasyOCR...")
        if _easyocr_reader is None:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        # canvas_size=1000: Phase 3 measured ~28s/image at EasyOCR's full-res
        # default vs ~4-5s at this size, with no measurable confidence
        # difference on real photos (see pipeline/ocr.py git history) --
        # detection cost scales with resolution, recognition quality doesn't.
        result = _easyocr_reader.readtext(image_path, canvas_size=1000)
        text = "\n".join([line[1] for line in result])
        confidence = sum([line[2] for line in result]) / len(result) if result else 0
        return {"status": "success", "layer": 4, "text": text, "confidence": min(confidence, 0.70), "model": "EasyOCR"}
    except Exception as e:
        logger.error(f"Layer 4 failed: {str(e)}")
        return {"status": "failed", "layer": 4, "error": str(e)}


def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    layers = [(layer_1_donut, 0.95), (layer_2_paddle_layoutlm, 0.90), (layer_3_trocr, 0.80), (layer_4_easyocr, 0.60)]
    results = []
    best_result = None
    logger.info("\nStarting OCR cascade...")
    for layer_func, threshold in layers:
        layer_name = layer_func.__name__
        logger.info(f"Trying {layer_name}...")
        result = layer_func(image_path)
        results.append(result)
        if result["status"] == "success":
            confidence = result.get("confidence", 0)
            if confidence >= threshold:
                logger.info(f"PASS - Using Layer {result['layer']}")
                best_result = result
                break
            if best_result is None:
                best_result = result
        else:
            logger.warning(f"Failed: {result.get('error')}")
    if not best_result:
        best_result = results[-1]
    logger.info(f"Result: Layer {best_result.get('layer')} - Confidence {best_result.get('confidence', 0):.2f}\n")
    return best_result


def extract_text(image_path: str) -> Dict[str, Any]:
    """Adapter to eval.py's STAGE CONTRACTS schema -- see module docstring."""
    result = extract_text_from_image(image_path)
    return {
        "success": result.get("status") == "success",
        "raw_text": result.get("text", ""),
        "confidence": float(result.get("confidence") or 0.0),
        "layer": result.get("layer"),
        "model": result.get("model"),
        "status": result.get("status"),
    }


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <image_path>")
        sys.exit(1)

    t0 = time.time()
    result = extract_text(sys.argv[1])
    elapsed = time.time() - t0

    print(f"success={result['success']} layer={result['layer']} model={result['model']} "
          f"confidence={result['confidence']:.2f} elapsed={elapsed:.2f}s")
    print("--- raw_text ---")
    print(result["raw_text"])
