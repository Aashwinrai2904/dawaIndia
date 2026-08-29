# dawaIndia — Accuracy Report

Generated: 2026-08-29 18:15:23 UTC
Test cases evaluated: 2

## Summary

| Stage | Threshold | Avg Score | Status | Run | Skipped | Errored |
|---|---|---|---|---|---|---|
| Layout detection | 98% | 100.0% | PASS | 2 | 0 | 0 |
| OCR accuracy | 95% | 0.0% | FAIL | 2 | 0 | 0 |
| Dosage parsing | 92% | — | SKIPPED | 0 | 2 | 0 |
| Drug matching | 88% | — | SKIPPED | 0 | 2 | 0 |
| Confidence scoring | 90% | — | SKIPPED | 0 | 2 | 0 |

**Overall: FAIL** — one or more implemented stages are below threshold. Do not ship.

## Per-Case Results

### IMG_001

| Stage | Status | Score | Threshold | Notes |
|---|---|---|---|---|
| Layout detection | PASS | 100.0% | 98% | captured |
| OCR accuracy | FAIL | 0.0% | 95% | 0/6 brand names recognized in OCR text (proxy) |
| Dosage parsing | SKIPPED | — | 92% | module 'parser.dosage' not implemented yet (No module named 'parser') |
| Drug matching | SKIPPED | — | 88% | module 'matcher.drug' not implemented yet (No module named 'matcher') |
| Confidence scoring | SKIPPED | — | 90% | module 'confidence.scorer' not implemented yet (No module named 'confidence') |

### IMG_002

| Stage | Status | Score | Threshold | Notes |
|---|---|---|---|---|
| Layout detection | PASS | 100.0% | 98% | captured |
| OCR accuracy | FAIL | 0.0% | 95% | 0/3 brand names recognized in OCR text (proxy) |
| Dosage parsing | SKIPPED | — | 92% | module 'parser.dosage' not implemented yet (No module named 'parser') |
| Drug matching | SKIPPED | — | 88% | module 'matcher.drug' not implemented yet (No module named 'matcher') |
| Confidence scoring | SKIPPED | — | 90% | module 'confidence.scorer' not implemented yet (No module named 'confidence') |

## Failed Cases

### IMG_001
- **OCR accuracy**: 0.0% (below 95% threshold) — [{"brand_name": "Calpol", "found_in_ocr_text": false}, {"brand_name": "Azilide", "found_in_ocr_text": false}, {"brand_name": "Hicet-Ax", "found_in_ocr_text": false}, {"brand_name": "Planokuf", "found_in_ocr_text": false}, {"brand_name": "Nexpro-RD", "found_in_ocr_text": false}, {"brand_name": "Senzicool", "found_in_ocr_text": false}]

### IMG_002
- **OCR accuracy**: 0.0% (below 95% threshold) — [{"brand_name": "Grandcef-CV (325)", "found_in_ocr_text": false}, {"brand_name": "Hicet-DC", "found_in_ocr_text": false}, {"brand_name": "Aquilinctus", "found_in_ocr_text": false}]

## Known Limitations

- Layout stage has no ground-truth bounding box in the current test case schema; it is scored only on the module's self-reported `success` flag.
- OCR stage is scored via a brand-name substring recall proxy, not true word-error-rate, because test cases don't yet carry a full ground-truth transcription.
- Parse/Match/Confidence scoring aligns ground-truth medicines to pipeline output by fuzzy brand-name matching (difflib), since output order isn't guaranteed.
