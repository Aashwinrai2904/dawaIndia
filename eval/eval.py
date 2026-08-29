#!/usr/bin/env python3
"""dawaIndia /eval accuracy harness — Phase 1.

Runs labeled prescription test cases through the pipeline (layout -> OCR ->
parse -> match -> confidence), scores each stage against the thresholds in
CLAUDE.md, and writes /eval/accuracy_report.md.

Per CLAUDE.md ("Boris Cherny Method"): this harness is built *before* the
pipeline stages it measures. Until pipeline/layout.py, pipeline/ocr.py,
parser/dosage.py, matcher/drug.py, and confidence/scorer.py exist, each
stage is reported as SKIPPED rather than failed — that is the expected,
correct state for Phase 1.

STAGE CONTRACTS (what Phase 2-6 must implement for eval.py to score them):

    pipeline/layout.py
        def detect_prescription_box(image_path: str) -> dict:
            {"success": bool, "bbox": [x, y, w, h] | None,
             "cropped_image_path": str | None}

    pipeline/ocr.py
        def extract_text(image_path: str) -> dict:
            {"success": bool, "raw_text": str, "confidence": float}

    parser/dosage.py
        def parse_dosage(raw_text: str) -> dict:
            {"success": bool, "medicines": [
                {"brand_name": str, "strength": str, "dosage": str,
                 "frequency": str, "duration": str}, ...]}

    matcher/drug.py
        def match_drugs(parsed_medicines: list) -> dict:
            {"success": bool, "matched_medicines": [
                {"brand_name": str, "generic_name": str,
                 "match_score": float}, ...]}

    confidence/scorer.py
        def compute_confidence(stage_results: dict) -> dict:
            {"success": bool, "overall_confidence": float,
             "per_medicine": [{"brand_name": str, "confidence": float}, ...]}

    stage_results passed to compute_confidence is {"layout": <dict>,
    "ocr": <dict>, "parse": <dict>, "match": <dict>} — whichever of those
    stages actually ran for this case.

Usage:
    python eval/eval.py                  # run all test cases, write report
    python eval/eval.py --case IMG_001   # run a single test case
    python eval/eval.py --quiet          # suppress per-case console output
"""

import argparse
import difflib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
TEST_CASES_DIR = EVAL_DIR / "test_cases"
REPORT_PATH = EVAL_DIR / "accuracy_report.md"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Stage order + thresholds, from CLAUDE.md "Success Metrics"
STAGES = [
    {"key": "layout", "label": "Layout detection", "module": "pipeline.layout",
     "function": "detect_prescription_box", "threshold": 0.98},
    {"key": "ocr", "label": "OCR accuracy", "module": "pipeline.ocr",
     "function": "extract_text", "threshold": 0.95},
    {"key": "parse", "label": "Dosage parsing", "module": "parser.dosage",
     "function": "parse_dosage", "threshold": 0.92},
    {"key": "match", "label": "Drug matching", "module": "matcher.drug",
     "function": "match_drugs", "threshold": 0.88},
    {"key": "confidence", "label": "Confidence scoring", "module": "confidence.scorer",
     "function": "compute_confidence", "threshold": 0.90},
]
STAGE_KEYS = [s["key"] for s in STAGES]
STAGE_BY_KEY = {s["key"]: s for s in STAGES}

PARSE_FIELDS = ["strength", "dosage", "frequency", "duration"]
CONFIDENCE_BUCKETS = {"high": (0.8, 1.01), "medium": (0.5, 0.8), "low": (0.0, 0.5)}
NAME_MATCH_MIN_SIMILARITY = 0.5  # below this we treat two brand names as "not the same medicine"


# ─────────────────────────── stage loading ───────────────────────────

def load_stage_function(module_path, function_name):
    """Import a pipeline stage module. Returns (function, None) or (None, reason)."""
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        return None, f"module '{module_path}' not implemented yet ({e})"
    fn = getattr(module, function_name, None)
    if fn is None:
        return None, f"module '{module_path}' exists but has no function '{function_name}'"
    return fn, None


def make_result(key, score, detail):
    threshold = STAGE_BY_KEY[key]["threshold"]
    status = "PASS" if score >= threshold else "FAIL"
    return {"status": status, "score": score, "threshold": threshold, "detail": detail}


def make_skipped(key, reason):
    return {"status": "SKIPPED", "score": None, "threshold": STAGE_BY_KEY[key]["threshold"], "reason": reason}


def make_error(key, exc):
    return {"status": "ERROR", "score": None, "threshold": STAGE_BY_KEY[key]["threshold"],
            "error": f"{type(exc).__name__}: {exc}"}


# ─────────────────────────── matching helpers ───────────────────────────

def normalize(value):
    return str(value or "").strip().lower()


def similarity(a, b):
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def find_best_match(target_name, candidates, name_key):
    """Fuzzy-match target_name against candidates[*][name_key]. Returns (candidate, score)."""
    best, best_score = None, 0.0
    for candidate in candidates:
        score = similarity(target_name, candidate.get(name_key, ""))
        if score > best_score:
            best, best_score = candidate, score
    if best_score < NAME_MATCH_MIN_SIMILARITY:
        return None, best_score
    return best, best_score


# ─────────────────────────── per-stage scoring ───────────────────────────
# Ground truth (see eval/test_cases/*.json) is medicine-level output, not a
# per-stage transcript. Layout and OCR therefore use weaker proxies until the
# label schema grows dedicated fields (see "Known Limitations" in the report).

def score_layout(layout_out):
    detail = {"success": bool(layout_out.get("success"))}
    score = 1.0 if layout_out.get("success") else 0.0
    return score, detail


def score_ocr(raw_text, ground_truth_medicines):
    if not ground_truth_medicines:
        return None, {"note": "no ground-truth medicines to check against"}
    text_norm = normalize(raw_text)
    per_medicine = []
    found = 0
    for gt in ground_truth_medicines:
        present = normalize(gt["brand_name"]) in text_norm
        found += present
        per_medicine.append({"brand_name": gt["brand_name"], "found_in_ocr_text": present})
    return found / len(ground_truth_medicines), per_medicine


def score_parse(parsed_medicines, ground_truth_medicines):
    if not ground_truth_medicines:
        return None, {"note": "no ground-truth medicines to check against"}
    per_medicine = []
    total = 0.0
    for gt in ground_truth_medicines:
        match, name_score = find_best_match(gt["brand_name"], parsed_medicines, "brand_name")
        if match is None:
            per_medicine.append({"brand_name": gt["brand_name"], "field_accuracy": 0.0,
                                  "note": "no matching parsed medicine found"})
            continue
        matched_fields = sum(1 for f in PARSE_FIELDS if normalize(match.get(f)) == normalize(gt.get(f)))
        accuracy = matched_fields / len(PARSE_FIELDS)
        total += accuracy
        per_medicine.append({"brand_name": gt["brand_name"], "field_accuracy": accuracy,
                              "name_match_confidence": round(name_score, 2)})
    return total / len(ground_truth_medicines), per_medicine


def score_match(matched_medicines, ground_truth_medicines):
    if not ground_truth_medicines:
        return None, {"note": "no ground-truth medicines to check against"}
    per_medicine = []
    correct = 0
    for gt in ground_truth_medicines:
        match, _ = find_best_match(gt["brand_name"], matched_medicines, "brand_name")
        if match is None:
            per_medicine.append({"brand_name": gt["brand_name"], "correct": False,
                                  "note": "no matched medicine found"})
            continue
        is_correct = normalize(match.get("generic_name")) == normalize(gt.get("generic_name"))
        correct += is_correct
        per_medicine.append({"brand_name": gt["brand_name"], "correct": is_correct,
                              "expected_generic": gt.get("generic_name"),
                              "matched_generic": match.get("generic_name")})
    return correct / len(ground_truth_medicines), per_medicine


def score_confidence(confidence_out, ground_truth_medicines):
    targets = [gt for gt in ground_truth_medicines if gt.get("confidence_target")]
    if not targets:
        return None, {"note": "no confidence_target labels to check against"}
    reported = {normalize(m.get("brand_name")): m.get("confidence")
                for m in confidence_out.get("per_medicine", [])}
    per_medicine = []
    correct = 0
    for gt in targets:
        conf = reported.get(normalize(gt["brand_name"]))
        target = gt["confidence_target"]
        if conf is None:
            per_medicine.append({"brand_name": gt["brand_name"], "correct": False,
                                  "note": "no confidence reported for this medicine"})
            continue
        lo, hi = CONFIDENCE_BUCKETS.get(target, (0.0, 1.01))
        is_correct = lo <= conf < hi
        correct += is_correct
        per_medicine.append({"brand_name": gt["brand_name"], "confidence": conf,
                              "target_bucket": target, "correct": is_correct})
    return correct / len(targets), per_medicine


# ─────────────────────────── pipeline cascade ───────────────────────────

def run_pipeline(test_case, image_path, verbose=True):
    """Run one labeled prescription through every implemented pipeline stage."""
    gt_medicines = test_case.get("ground_truth_medicines", [])
    stages = {}
    stage_outputs = {}

    # 1. Layout
    fn, reason = load_stage_function(*_stage_target("layout"))
    if fn is None:
        stages["layout"] = make_skipped("layout", reason)
        ocr_input_image = str(image_path)
    else:
        try:
            out = fn(str(image_path))
            stage_outputs["layout"] = out
            score, detail = score_layout(out)
            stages["layout"] = make_result("layout", score, detail)
            ocr_input_image = out.get("cropped_image_path") or str(image_path)
        except Exception as e:
            stages["layout"] = make_error("layout", e)
            ocr_input_image = str(image_path)

    # 2. OCR
    fn, reason = load_stage_function(*_stage_target("ocr"))
    raw_text = None
    if fn is None:
        stages["ocr"] = make_skipped("ocr", reason)
    else:
        try:
            out = fn(ocr_input_image)
            stage_outputs["ocr"] = out
            raw_text = out.get("raw_text", "")
            score, detail = score_ocr(raw_text, gt_medicines)
            stages["ocr"] = _skip_if_no_score("ocr", score, detail)
        except Exception as e:
            stages["ocr"] = make_error("ocr", e)

    # 3. Parse
    fn, reason = load_stage_function(*_stage_target("parse"))
    parsed_medicines = None
    if fn is None:
        stages["parse"] = make_skipped("parse", reason)
    elif raw_text is None:
        stages["parse"] = make_skipped("parse", "upstream stage 'ocr' produced no text")
    else:
        try:
            out = fn(raw_text)
            stage_outputs["parse"] = out
            parsed_medicines = out.get("medicines", [])
            score, detail = score_parse(parsed_medicines, gt_medicines)
            stages["parse"] = _skip_if_no_score("parse", score, detail)
        except Exception as e:
            stages["parse"] = make_error("parse", e)

    # 4. Match
    fn, reason = load_stage_function(*_stage_target("match"))
    matched_medicines = None
    if fn is None:
        stages["match"] = make_skipped("match", reason)
    elif not parsed_medicines:
        stages["match"] = make_skipped("match", "upstream stage 'parse' produced no medicines")
    else:
        try:
            out = fn(parsed_medicines)
            stage_outputs["match"] = out
            matched_medicines = out.get("matched_medicines", [])
            score, detail = score_match(matched_medicines, gt_medicines)
            stages["match"] = _skip_if_no_score("match", score, detail)
        except Exception as e:
            stages["match"] = make_error("match", e)

    # 5. Confidence
    fn, reason = load_stage_function(*_stage_target("confidence"))
    if fn is None:
        stages["confidence"] = make_skipped("confidence", reason)
    elif not matched_medicines:
        stages["confidence"] = make_skipped("confidence", "upstream stage 'match' produced no medicines")
    else:
        try:
            out = fn(stage_outputs)
            score, detail = score_confidence(out, gt_medicines)
            stages["confidence"] = _skip_if_no_score("confidence", score, detail)
        except Exception as e:
            stages["confidence"] = make_error("confidence", e)

    result = {"image_id": test_case.get("image_id"), "image_path": str(image_path), "stages": stages}
    if verbose:
        print_case_summary(result)
    return result


def _stage_target(key):
    stage = STAGE_BY_KEY[key]
    return stage["module"], stage["function"]


def _skip_if_no_score(key, score, detail):
    if score is None:
        return make_skipped(key, detail.get("note", "no ground truth available for this stage"))
    return make_result(key, score, detail)


# ─────────────────────────── public entrypoint ───────────────────────────

def evaluate(image_path, verbose=True, ground_truth=None):
    """Run one prescription image through the pipeline and score each stage.

    Returns a JSON-serializable dict: {"image_id", "image_path", "stages": {...}}.

    If `ground_truth` isn't given, looks for a sibling label file next to the
    image (e.g. test_cases/IMG_001.jpg -> test_cases/IMG_001.json).
    """
    image_path = Path(image_path)
    if ground_truth is None:
        label_path = image_path.with_suffix(".json")
        if not label_path.exists():
            raise FileNotFoundError(f"No ground-truth label file found at {label_path}")
        ground_truth = json.loads(label_path.read_text())
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")
    return run_pipeline(ground_truth, image_path, verbose=verbose)


# ─────────────────────────── test runner ───────────────────────────

def load_test_cases(test_cases_dir):
    cases = []
    for json_file in sorted(test_cases_dir.glob("*.json")):
        try:
            cases.append(json.loads(json_file.read_text()))
        except json.JSONDecodeError as e:
            print(f"WARNING: skipping malformed test case {json_file}: {e}", file=sys.stderr)
    return cases


def print_case_summary(result):
    print(f"\n[{result['image_id']}]")
    for key in STAGE_KEYS:
        s = result["stages"][key]
        label = STAGE_BY_KEY[key]["label"]
        if s["status"] == "SKIPPED":
            print(f"  {label:22s} SKIPPED  ({s['reason']})")
        elif s["status"] == "ERROR":
            print(f"  {label:22s} ERROR    ({s['error']})")
        else:
            pct = f"{s['score']*100:.1f}%"
            thr = f"{s['threshold']*100:.0f}%"
            print(f"  {label:22s} {s['status']:8s} {pct} (threshold {thr})")


# ─────────────────────────── report generation ───────────────────────────

def aggregate_stage(results, key):
    scores, skipped, errored = [], 0, 0
    for r in results:
        s = r["stages"][key]
        if s["status"] == "SKIPPED":
            skipped += 1
        elif s["status"] == "ERROR":
            errored += 1
        else:
            scores.append(s["score"])
    avg = sum(scores) / len(scores) if scores else None
    threshold = STAGE_BY_KEY[key]["threshold"]
    if errored:
        status = "ERROR"
    elif not scores:
        status = "SKIPPED"
    elif avg >= threshold:
        status = "PASS"
    else:
        status = "FAIL"
    return {"avg_score": avg, "cases_run": len(scores), "cases_skipped": skipped,
            "cases_errored": errored, "threshold": threshold, "status": status}


def overall_status(stage_summaries):
    statuses = {s["status"] for s in stage_summaries.values()}
    if "ERROR" in statuses:
        return "ERROR"
    if statuses - {"PASS"}:
        if statuses <= {"PASS", "SKIPPED"}:
            return "NOT READY"
        return "FAIL"
    return "PASS"


def summarize_detail(key, status, detail):
    if status in ("SKIPPED", "ERROR"):
        return None
    if key == "layout":
        return "captured" if detail.get("success") else "not captured"
    if key == "ocr":
        found = sum(1 for d in detail if d.get("found_in_ocr_text"))
        return f"{found}/{len(detail)} brand names recognized in OCR text (proxy)"
    if key == "parse":
        return f"avg field accuracy across {len(detail)} medicine(s)"
    if key == "match":
        correct = sum(1 for d in detail if d.get("correct"))
        return f"{correct}/{len(detail)} generics matched correctly"
    if key == "confidence":
        correct = sum(1 for d in detail if d.get("correct"))
        return f"{correct}/{len(detail)} confidence buckets matched target"
    return None


def generate_report(all_results, run_started_at):
    lines = []
    lines.append("# dawaIndia — Accuracy Report")
    lines.append("")
    lines.append(f"Generated: {run_started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Test cases evaluated: {len(all_results)}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Stage | Threshold | Avg Score | Status | Run | Skipped | Errored |")
    lines.append("|---|---|---|---|---|---|---|")

    stage_summaries = {}
    for s in STAGES:
        key = s["key"]
        agg = aggregate_stage(all_results, key)
        stage_summaries[key] = agg
        avg_str = f"{agg['avg_score']*100:.1f}%" if agg["avg_score"] is not None else "—"
        lines.append(f"| {s['label']} | {s['threshold']*100:.0f}% | {avg_str} | {agg['status']} | "
                     f"{agg['cases_run']} | {agg['cases_skipped']} | {agg['cases_errored']} |")

    overall = overall_status(stage_summaries)
    implemented = sum(1 for s in stage_summaries.values() if s["status"] != "SKIPPED")
    lines.append("")
    if overall == "PASS":
        lines.append(f"**Overall: PASS** — all {len(STAGES)} pipeline stages implemented and meeting threshold.")
    elif overall == "NOT READY":
        lines.append(f"**Overall: NOT READY** — {implemented}/{len(STAGES)} pipeline stages implemented. "
                      f"Per CLAUDE.md, nothing ships until /eval passes (all stages at or above threshold).")
    elif overall == "FAIL":
        lines.append(f"**Overall: FAIL** — one or more implemented stages are below threshold. Do not ship.")
    else:
        lines.append(f"**Overall: ERROR** — one or more stages raised an exception. Do not ship.")

    lines.append("")
    lines.append("## Per-Case Results")
    for result in all_results:
        lines.append("")
        lines.append(f"### {result['image_id']}")
        lines.append("")
        lines.append("| Stage | Status | Score | Threshold | Notes |")
        lines.append("|---|---|---|---|---|")
        for s in STAGES:
            key = s["key"]
            entry = result["stages"][key]
            score_str = f"{entry['score']*100:.1f}%" if entry.get("score") is not None else "—"
            if entry["status"] == "SKIPPED":
                note = entry["reason"]
            elif entry["status"] == "ERROR":
                note = entry["error"]
            else:
                note = summarize_detail(key, entry["status"], entry["detail"]) or ""
            lines.append(f"| {s['label']} | {entry['status']} | {score_str} | {s['threshold']*100:.0f}% | {note} |")

    lines.append("")
    lines.append("## Failed Cases")
    failed_any = False
    for result in all_results:
        bad = {k: v for k, v in result["stages"].items() if v["status"] in ("FAIL", "ERROR")}
        if not bad:
            continue
        failed_any = True
        lines.append("")
        lines.append(f"### {result['image_id']}")
        for key, entry in bad.items():
            label = STAGE_BY_KEY[key]["label"]
            if entry["status"] == "ERROR":
                lines.append(f"- **{label}**: ERROR — {entry['error']}")
            else:
                lines.append(f"- **{label}**: {entry['score']*100:.1f}% (below {entry['threshold']*100:.0f}% threshold) — {json.dumps(entry['detail'])}")
    if not failed_any:
        lines.append("")
        lines.append("(none)")

    lines.append("")
    lines.append("## Known Limitations")
    lines.append("")
    lines.append("- Layout stage has no ground-truth bounding box in the current test case schema; "
                 "it is scored only on the module's self-reported `success` flag.")
    lines.append("- OCR stage is scored via a brand-name substring recall proxy, not true word-error-rate, "
                 "because test cases don't yet carry a full ground-truth transcription.")
    lines.append("- Parse/Match/Confidence scoring aligns ground-truth medicines to pipeline output by "
                 "fuzzy brand-name matching (difflib), since output order isn't guaranteed.")
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────── CLI ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="dawaIndia /eval accuracy harness")
    parser.add_argument("--case", help="Run only the test case with this image_id")
    parser.add_argument("--test-cases-dir", default=str(TEST_CASES_DIR))
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case console output")
    args = parser.parse_args()

    test_cases_dir = Path(args.test_cases_dir)
    cases = load_test_cases(test_cases_dir)
    if args.case:
        cases = [c for c in cases if c.get("image_id") == args.case]
        if not cases:
            print(f"No test case found with image_id={args.case!r} in {test_cases_dir}", file=sys.stderr)
            sys.exit(1)
    if not cases:
        print(f"No test cases found in {test_cases_dir}", file=sys.stderr)
        sys.exit(1)

    run_started = datetime.now(timezone.utc)
    all_results = []
    for case in cases:
        image_path = EVAL_DIR / case["image_path"]
        try:
            result = run_pipeline(case, image_path, verbose=not args.quiet)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        all_results.append(result)

    report_md = generate_report(all_results, run_started)
    REPORT_PATH.write_text(report_md)
    print(f"\nReport written to {REPORT_PATH.relative_to(REPO_ROOT)}")

    stage_summaries = {s["key"]: aggregate_stage(all_results, s["key"]) for s in STAGES}
    status = overall_status(stage_summaries)
    print(f"Overall: {status}")
    sys.exit(1 if status in ("FAIL", "ERROR") else 0)


if __name__ == "__main__":
    main()
