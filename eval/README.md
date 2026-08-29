# /eval — dawaIndia Accuracy Harness

Runs labeled prescription images through the pipeline and scores each stage
against the thresholds in `CLAUDE.md`. See `eval.py`'s module docstring for
the exact function contract each pipeline stage must implement.

## Running it

```bash
python eval/eval.py                  # run every test case, write accuracy_report.md
python eval/eval.py --case IMG_001   # run a single test case by image_id
python eval/eval.py --quiet          # suppress per-case console output
```

Exit code is `0` when no implemented stage is below threshold (this includes
the current all-SKIPPED state, since stages that don't exist yet aren't a
regression), and `1` when a stage FAILs its threshold or ERRORs.

## Adding a test case

Drop two files into `eval/test_cases/`, sharing the same basename:

- `IMG_00N.jpg` — the prescription photo
- `IMG_00N.json` — its ground-truth label

Label schema:

```json
{
  "image_id": "IMG_001",
  "image_path": "test_cases/IMG_001.jpg",
  "doctor_name": "Dr. S. Chakravorty",
  "patient_name": "Mr. Siddharth",
  "patient_age": 23,
  "date": "2023-04-21",
  "ground_truth_medicines": [
    {
      "medicine_number": 1,
      "brand_name": "Calpol",
      "generic_name": "Paracetamol",
      "strength": "650mg",
      "dosage": "1 tablet",
      "frequency": "SOS (as needed)",
      "condition": "fever >100°F",
      "duration": "as needed",
      "confidence_target": "high"
    }
  ]
}
```

`confidence_target` is one of `"high"` (>=80%), `"medium"` (50-80%), or
`"low"` (<50%) — it's what the confidence-scoring stage is expected to land
in for that medicine.

`IMG_001.jpg` and `IMG_002.jpg` are real photographed prescriptions (from
the same clinic, one week apart). Real prescription photos are **not**
committed by default (see `.gitignore`) due to size and patient privacy —
each committed image is an explicit allowlist entry, and must be redacted
first:

1. Black out the patient's name, any phone numbers/emails, and any
   signatures before saving the image into `eval/test_cases/`.
2. Add `!/eval/test_cases/<filename>.jpg` to `.gitignore` for that specific
   file (deny-by-default: an unlisted image never gets committed even by
   accident).
3. In the label JSON, set `"patient_name": "REDACTED"` and don't put a real
   name anywhere else in the file either.
4. Only transcribe `generic_name` when the brand is confidently recognized;
   otherwise use `"UNVERIFIED"` with `"confidence_target": "low"` and a
   `"note"` explaining why. Don't invent a plausible-looking generic name —
   an incorrect one silently corrupts Match-stage (Phase 5) scoring later,
   and CLAUDE.md's "abstain on uncertainty" applies to labeling the data,
   not just to the pipeline that reads it.

For images that don't need this (synthetic/placeholder fixtures with no
real patient involved), skip straight to step 2.

## Known limitations (see also `accuracy_report.md`)

- **Layout**: no ground-truth bounding box in the schema yet, so this stage
  is scored only on the module's self-reported `success` flag.
- **OCR**: scored via a brand-name substring recall proxy against the raw
  extracted text, not true word-error-rate — the schema has no full
  ground-truth transcription to diff against yet.
- **Parse / Match / Confidence**: ground-truth medicines are aligned to
  pipeline output by fuzzy brand-name matching (`difflib`), since pipeline
  output order isn't guaranteed to match the label order.
