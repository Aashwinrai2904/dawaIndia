#!/usr/bin/env python3
"""dawaIndia pipeline — Stage 3: Dosage parsing.

Parses OCR medicine lines into structured dosage fields: medicine name,
dosage value+unit, form, quantity, frequency (OD/BD/TDS/QID/QHS/SOS),
timing, duration.

Public contract (must match eval.py's STAGE CONTRACTS docstring exactly,
since eval.py imports and calls this directly):

    def parse_dosage(raw_text: str) -> dict:
        {"success": bool, "medicines": [
            {"brand_name": str, "strength": str, "dosage": str,
             "frequency": str, "duration": str}, ...]}

`parse_dosage` is a thin adapter over `parse_medicines_from_ocr` (the
actual parsing logic), which returns a richer, more granular shape --
separate dosage_value/dosage_unit/quantity/form fields, plus "medicine"
instead of "brand_name" -- that doesn't match eval.py's contract on its
own. Same pattern as pipeline/ocr.py's extract_text adapter: without it,
eval.py would import this module fine but never find a callable
parse_dosage, and Parse would keep reporting SKIPPED.

FREQUENCY FIELD DOESN'T MATCH THE EXISTING GROUND-TRUTH PHRASING, ON
PURPOSE. eval/test_cases/*.json's frequency values are verbose ("OD (once
daily)", "SOS (as needed)") because that's how they were transcribed by
hand from the photos. This parser outputs the bare clinical code ("OD",
"SOS") instead, because that's the standard, useful, structured form a
real prescription would be OCR'd into -- reconstructing an English
parenthetical to match one dataset's transcription style would be
overfitting the parser to this project's own test fixtures rather than to
real prescriptions. Net effect: eval.py's exact-string field comparison
scores every medicine's frequency as a mismatch even when the parser read
it correctly. This is a ground-truth-schema/parser-output mismatch worth
resolving on purpose (either loosen the schema's phrasing or loosen
eval.py's comparison), not a parsing bug.

FIXES OVER THE ORIGINAL SPEC (see git history for the request this
implements):
  - FREQUENCY_MAP/FORM_MAP/TIMING_MAP lookups now match whole words
    (regex \\b boundaries), not substrings. The original `if key in
    text_norm` check meant e.g. "od" matched inside "good"/"food"/
    "period"/"method" -- fixed. NOT fixed by this, and still a real gap:
    TIMING_MAP's "am"/"pm" keys are genuine standalone words, so a
    dosage_line that happens to contain a real clock time ("8.00 AM to
    10.00 AM", straight from this project's real OCR output) still
    produces a false timing match -- word-boundary checking can't
    disambiguate "AM" the dosing shorthand from "AM" the time-of-day
    suffix, they're the identical token. Only shows up when
    parse_medicines_from_ocr's line-pairing feeds it a non-dosage line as
    input (see that function's docstring); doesn't affect eval.py's
    score_parse, which doesn't compare the "timing" field.
  - extract_duration now also accepts the single-letter Indian shorthand
    ("x5d", "x2w", "x1m"), not just spelled-out "day(s)/week(s)/month(s)".
    The shorthand form is the common case on real Indian prescriptions
    (see eval/test_cases/*.json's transcription notes) and was previously
    unrecognized entirely.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FREQUENCY_MAP = {
    "od": "OD", "once": "OD",
    "bd": "BD", "twice": "BD",
    "tds": "TDS", "thrice": "TDS",
    "qid": "QID", "four": "QID",
    "qhs": "QHS", "bedtime": "QHS",
    "sos": "SOS", "as needed": "SOS",
}

FORM_MAP = {
    "tablet": "tablet", "tablets": "tablet", "tab": "tablet", "tabs": "tablet",
    "capsule": "capsule", "capsules": "capsule", "cap": "capsule", "caps": "capsule",
    "syrup": "syrup", "liquid": "syrup", "suspension": "suspension",
    "injection": "injection", "inj": "injection", "im": "injection", "iv": "injection",
    "cream": "cream", "ointment": "ointment", "gel": "gel",
    "drops": "drops", "drop": "drops",
}

TIMING_MAP = {
    "morning": "morning", "breakfast": "morning", "am": "morning",
    "afternoon": "afternoon", "noon": "afternoon", "lunch": "afternoon",
    "evening": "evening", "dinner": "evening", "pm": "evening",
    "night": "night", "bedtime": "night", "hs": "night",
}

DURATION_UNITS = {"day": "days", "days": "days", "d": "days",
                   "week": "weeks", "weeks": "weeks", "w": "weeks",
                   "month": "months", "months": "months", "m": "months"}


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


def _find_whole_word(mapping: Dict[str, str], text_norm: str) -> Optional[str]:
    """First mapping key that appears as a whole word/phrase in text_norm."""
    for key, value in mapping.items():
        if re.search(r'\b' + re.escape(key) + r'\b', text_norm):
            return value
    return None


def extract_dosage(text: str) -> Tuple[Optional[float], Optional[str]]:
    match = re.search(r'(\d+\.?\d*)\s*([a-z%]+)', normalize_text(text))
    if match:
        value = float(match.group(1))
        unit = match.group(2).strip()
        return (value if '.' in match.group(1) else int(value), unit)
    return (None, None)


def extract_frequency(text: str) -> Optional[str]:
    text_norm = normalize_text(text)
    found = _find_whole_word(FREQUENCY_MAP, text_norm)
    if found:
        return found
    if "+" in text:
        parts = text.split("+")
        if len(parts) == 3:
            total = sum(int(p) for p in parts if p.strip().isdigit())
            if total == 2:
                return "BD"
            elif total == 3:
                return "TDS"
    return None


def extract_quantity(text: str) -> Optional[int]:
    match = re.search(r'(\d+)\s*(tablet|capsule|drop|spoon)', normalize_text(text))
    if match:
        return int(match.group(1))
    return None


def extract_form(text: str) -> Optional[str]:
    return _find_whole_word(FORM_MAP, normalize_text(text))


def extract_timing(text: str) -> Optional[str]:
    return _find_whole_word(TIMING_MAP, normalize_text(text))


def extract_duration(text: str) -> Optional[str]:
    unit_pattern = "|".join(sorted(DURATION_UNITS, key=len, reverse=True))
    match = re.search(r'(\d+)\s*(' + unit_pattern + r')\b', normalize_text(text))
    if match:
        return f"{match.group(1)} {DURATION_UNITS[match.group(2)]}"
    return None


def parse_medicine_line(medicine_name: str, dosage_line: str) -> Dict[str, Any]:
    dosage_value, dosage_unit = extract_dosage(dosage_line)
    quantity = extract_quantity(dosage_line)
    form = extract_form(dosage_line)
    frequency = extract_frequency(dosage_line)
    timing = extract_timing(dosage_line)
    duration = extract_duration(dosage_line)

    fields_present = sum([
        dosage_value is not None,
        form is not None,
        frequency is not None,
        duration is not None,
    ])
    confidence = min(1.0, 0.5 + (fields_present * 0.15))

    return {
        "medicine": medicine_name.strip(),
        "dosage_value": dosage_value,
        "dosage_unit": dosage_unit,
        "form": form,
        "quantity": quantity,
        "frequency": frequency,
        "timing": timing,
        "duration": duration,
        "confidence": round(confidence, 2),
        "raw_input": dosage_line.strip(),
    }


def parse_medicines_from_ocr(ocr_text: str) -> List[Dict[str, Any]]:
    """Pairs each non-blank line with the line right after it as (name, dosage).

    Assumes clean, alternating OCR output: medicine name on one line, its
    dosage notation on the next. That holds for well-segmented input (a
    layout/OCR stage that isolates each medicine's own line) but not for
    this project's actual OCR output today, which is one undifferentiated
    blob of ~90 lines mixing letterhead, footer boilerplate, and medicine
    text with no reliable name/dosage alternation -- see pipeline/ocr.py's
    docstring. Fixing that is a line-segmentation problem upstream of
    dosage parsing, not something patched here.
    """
    medicines = []
    lines = ocr_text.strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if i + 1 < len(lines):
            medicine_name = line
            dosage_line = lines[i + 1]
            parsed = parse_medicine_line(medicine_name, dosage_line)
            medicines.append(parsed)
            i += 2
        else:
            i += 1
    logger.info(f"Parsed {len(medicines)} medicines from OCR")
    return medicines


def _format_strength(value, unit) -> str:
    if value is None:
        return ""
    value_str = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
    return f"{value_str}{unit}" if unit else value_str


def _format_dosage(quantity, form) -> str:
    parts = [str(p) for p in (quantity, form) if p is not None]
    return " ".join(parts)


def parse_dosage(raw_text: str) -> Dict[str, Any]:
    """Adapter to eval.py's STAGE CONTRACTS schema -- see module docstring.

    Drops entries where none of dosage/form/frequency/duration were
    extracted (confidence at its floor, 0.5): on real, messy OCR text
    parse_medicines_from_ocr's line-pairing pairs a lot of unrelated
    letterhead/footer lines together (see that function's docstring), and
    a pair that yields zero signal is noise, not a medicine -- reporting
    it as one would be exactly the "silent guess" CLAUDE.md's abstain-on-
    uncertainty rule warns against.
    """
    medicines = [m for m in parse_medicines_from_ocr(raw_text) if m["confidence"] > 0.5]
    return {
        "success": len(medicines) > 0,
        "medicines": [
            {
                "brand_name": m["medicine"],
                "strength": _format_strength(m["dosage_value"], m["dosage_unit"]),
                "dosage": _format_dosage(m["quantity"], m["form"]),
                "frequency": m["frequency"] or "",
                "duration": m["duration"] or "",
                "timing": m["timing"],
                "confidence": m["confidence"],
                "raw_input": m["raw_input"],
            }
            for m in medicines
        ],
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: python {__file__} <file_with_ocr_text>")
        sys.exit(1)

    text = open(sys.argv[1]).read()
    result = parse_dosage(text)
    print(json.dumps(result, indent=2))
