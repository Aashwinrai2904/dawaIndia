# dawaIndia — Accuracy Report

Generated: 2026-08-29 17:27:08 UTC
Test cases evaluated: 1

## Summary

| Stage | Threshold | Avg Score | Status | Run | Skipped | Errored |
|---|---|---|---|---|---|---|
| Layout detection | 98% | 100.0% | PASS | 1 | 0 | 0 |
| OCR accuracy | 95% | — | SKIPPED | 0 | 1 | 0 |
| Dosage parsing | 92% | — | SKIPPED | 0 | 1 | 0 |
| Drug matching | 88% | — | SKIPPED | 0 | 1 | 0 |
| Confidence scoring | 90% | — | SKIPPED | 0 | 1 | 0 |

**Overall: NOT READY** — 1/5 pipeline stages implemented. Per CLAUDE.md, nothing ships until /eval passes (all stages at or above threshold).

## Per-Case Results

### IMG_001

| Stage | Status | Score | Threshold | Notes |
|---|---|---|---|---|
| Layout detection | PASS | 100.0% | 98% | captured |
| OCR accuracy | SKIPPED | — | 95% | module 'pipeline.ocr' not implemented yet (No module named 'pipeline.ocr') |
| Dosage parsing | SKIPPED | — | 92% | module 'parser.dosage' not implemented yet (No module named 'parser') |
| Drug matching | SKIPPED | — | 88% | module 'matcher.drug' not implemented yet (No module named 'matcher') |
| Confidence scoring | SKIPPED | — | 90% | module 'confidence.scorer' not implemented yet (No module named 'confidence') |

## Failed Cases

(none)

## Known Limitations

- Layout stage has no ground-truth bounding box in the current test case schema; it is scored only on the module's self-reported `success` flag.
- OCR stage is scored via a brand-name substring recall proxy, not true word-error-rate, because test cases don't yet carry a full ground-truth transcription.
- Parse/Match/Confidence scoring aligns ground-truth medicines to pipeline output by fuzzy brand-name matching (difflib), since output order isn't guaranteed.
