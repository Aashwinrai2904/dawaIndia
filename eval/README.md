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

`IMG_001.jpg` currently checked in is a 1x1 synthetic placeholder, not a real
prescription — it exists so the harness is runnable end-to-end before any
real labeled photos are collected. Real prescription photos are **not**
committed (see `.gitignore`) due to size and patient privacy; store them
locally or in a private data store, and point `image_path` at wherever you
keep them.

## Known limitations (see also `accuracy_report.md`)

- **Layout**: no ground-truth bounding box in the schema yet, so this stage
  is scored only on the module's self-reported `success` flag.
- **OCR**: scored via a brand-name substring recall proxy against the raw
  extracted text, not true word-error-rate — the schema has no full
  ground-truth transcription to diff against yet.
- **Parse / Match / Confidence**: ground-truth medicines are aligned to
  pipeline output by fuzzy brand-name matching (`difflib`), since pipeline
  output order isn't guaranteed to match the label order.
