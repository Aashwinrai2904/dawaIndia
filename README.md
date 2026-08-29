# dawaIndia

**Read Prescriptions. Save Money.**

A prescription OCR + generic medicine affordability platform for Indian pharmacists.

## What It Does

1. **Read** handwritten prescriptions in <5 seconds
2. **Display** each medicine clearly with correct spelling, dosage, timing
3. **Show** government-approved generic alternatives (80% cheaper)
4. **Locate** nearest generic medicine stores

## Features

- ✓ Fast OCR (<5 seconds, no 9-minute processing)
- ✓ Accurate (>95% accuracy, each stage measured independently)
- ✓ Pharmacist-first (one-screen output, one-click print)
- ✓ Affordability layer (generic alternatives + store locator)
- ✓ No silent errors (flags uncertainty instead of guessing)

## Tech Stack

- **Frontend:** Next.js + TypeScript (PWA)
- **Backend:** Python + FastAPI
- **OCR:** EasyOCR + TrOCR + Claude Vision
- **Layout:** YOLO / Contour detection
- **Matching:** Fuzzywuzzy (300k Indian drugs)
- **Database:** PostgreSQL (Supabase)
- **Deploy:** Vercel (frontend) + simple server (backend)

## Quick Start

```bash
# Clone
git clone https://github.com/Aashwinrai2904/dawaIndia.git
cd dawaIndia

# Install
npm install
pip install -r requirements.txt

# Run eval (measure accuracy)
python eval/eval.py

# Start frontend
npm run dev

# Start backend (separate terminal)
uvicorn pipeline.api:app --reload
```

## Project Structure

```
dawaIndia/
├── eval/              — Accuracy measurement harness
├── pipeline/          — Main OCR + parsing logic
├── database/          — Drug databases
├── frontend/          — Next.js PWA
├── config/            — Configuration files
├── docs/              — Documentation
├── CLAUDE.md          — Project context (read this)
└── README.md          — This file
```

## Methodology

We follow **Boris Cherny's approach**:
- Plan before code
- Measure everything (no guessing)
- Each stage tested independently
- Abstain on uncertainty

See CLAUDE.md for detailed context.

## Status

**Phase 0:** Project setup ✓
**Phase 1:** Building /eval harness (coming next)

## License

MIT License

## Questions?

See CLAUDE.md or open an issue.
