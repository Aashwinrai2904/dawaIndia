# dawaIndia Project Context

## Project Overview

**Name:** dawaIndia
**Domain:** dawaIndia.online
**Type:** Progressive Web App (PWA)
**Primary User:** Pharmacists in Indian retail shops
**Secondary User:** Patients/families seeking affordable generics

## Problem

Doctors write prescriptions by hand. Handwriting is illegible. Pharmacists can't read 1-2 out of 8 medicines. Patients don't know generic medicines exist (80% cheaper, government-approved). No tool connects accurate OCR + affordability.

## Solution

dawaIndia reads prescriptions in <5 seconds, shows each medicine clearly with correct spelling/dosage, and displays government-approved generic alternatives with nearest store locations.

## Competitive Analysis

| Tool | Speed | Accuracy | Pharmacist-Focused | Affordability |
|------|-------|----------|-------------------|--------------|
| doctordocs.in | 9 min ✗ | Bad layout ✗ | Yes | No |
| lekhak.app | 2 min ✗ | Hallucinates ✗ | Yes | No |
| GoDavaii | 5 sec ✓ | Good ✓ | No (patient-focused) | No |
| **dawaIndia** | **<5 sec** | **>95% accurate** | **Yes** | **80% savings ✓** |

## Architecture (Boris Cherny Method)

```
Image → [Layout] → [OCR] → [Parse] → [Match] → [Score] → Output
         ↓         ↓        ↓        ↓        ↓
       Measure   Measure  Measure  Measure  Measure
       >98%      >95%     >92%     >88%     >90%
```

Each stage:
- Tested independently
- Measured against accuracy metric
- Abstains on uncertainty (doesn't guess)

## Databases Needed

1. **master_drugs.json** — 300k Indian drugs (brand, generic, strength, schedule)
2. **generic_equivalents.json** — Generics + pricing + store availability
3. **drug_interactions.json** — Top 500 dangerous medicine combinations
4. **drug_schedules.json** — OTC/H/H1/H2/X classifications
5. **generic_store_chains.json** — Jan Aushadhi, Dava India, Cipla, Mankind, etc.
6. **store_locator.json** — Store coordinates + phone numbers
7. **salt_composition.json** — Medicine composition reference

## Data Sources

- Jan Aushadhi: https://janaushadhi.gov.in/ (11,261+ stores, government)
- CDSCO: https://cdsco.gov.in/ (official drug approvals)
- 1mg.com: Web scraping for pricing + availability
- Dava India: Physical stores + app
- Cipla, Mankind, Aurobindo: Generic divisions
- Indian Pharmacopoeia: Drug properties reference

## Success Metrics (/eval harness measures)

- Layout detection: >98% (full prescription box captured)
- OCR accuracy: >95% (word error rate <5%)
- Dosage parsing: >92% (notation correctly structured)
- Drug matching: >88% (fuzzy match quality)
- Confidence scoring: >90% (overall system accuracy)
- Pharmacist workflow: <30 seconds (upload → verify → print)

## Assumptions

- Pharmacist has smartphone/tablet for counter use
- Prescriptions photographed in good lighting
- User has internet (PWA loads offline after first visit)
- Generics available in user's area (show store chains as fallback)

## Constraints

- No cloud APIs (privacy + speed)
- Local processing only
- 5-second response maximum
- Mobile-first UI
- Confidence must be >80% to show medicine clearly

## Boris Cherny Method (Standing Instructions)

1. **Plan before code** — Write architecture first, get approval before building
2. **Measure everything** — Build /eval harness first, run after every change
3. **Each stage independent** — Layout, OCR, parse, match, score all tested separately
4. **Abstain on uncertainty** — Never silent guess; flag with confidence score
5. **Give Claude a way to check its work** — /eval shows accuracy dashboard
6. **More context = smarter decisions** — Document assumptions, data sources, success criteria
7. **Commit only after /eval passes** — No bad code in repo

## Current Phase

**Phase 0: Setup** — Folder structure + CLAUDE.md + project configuration

## Next Phases (After Phase 0)

- Phase 1: Build /eval harness (labeled data + accuracy metrics)
- Phase 2: Build /pipeline/layout.py (prescription box detection)
- Phase 3: Build /pipeline/ocr.py (fast OCR + TrOCR fallback)
- Phase 4: Build /parser/dosage.py (notation parsing)
- Phase 5: Build /matcher/drug.py (fuzzy drug matching)
- Phase 6: Build /confidence/scorer.py (combine stages → confidence)
- Phase 7: Build /frontend/ (UI + generic alternatives)
- Phase 8: Integration testing (end-to-end)

## Team

- **Developer:** You (building the product)
- **Methodology:** Boris Cherny's approach (plan, measure, iterate)
- **AI Partner:** Claude (in Claude Code)

---

Last updated: 2026-08-29
Status: Setting up Phase 0 (project structure)
