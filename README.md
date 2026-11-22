# planparse

Extracts structured data (building, floors, wings, units, shared spaces, dimensions)
from architectural plans and stores it in Postgres.

Status: **Phase 0** — an accuracy and cost test of the extraction pipeline. There is no app or database schema yet.

## Setup

```bash
pnpm install
cp .env.example .env          # then put your OPENROUTER_API_KEY in .env
docker compose up -d          # Postgres (PostGIS) + Redis — not needed for Phase 0 yet
```

## Phase 0 runner

```bash
# Mall plan, scored against the hand-labelled answer key
pnpm phase0 samples/mall/ground-floor.png --truth samples/mall/ground-truth.json

# Compare models (only anthropic/, openai/, google/ models are allowed)
pnpm phase0 samples/mall/ground-floor.png --truth samples/mall/ground-truth.json --model anthropic/claude-sonnet-5.5

# PDFs: list pages, then pick one
pnpm phase0 samples/downloaded/lake-point.pdf --list-pages
pnpm phase0 samples/downloaded/lake-point.pdf --page 6

# Speed options: one big AI call instead of 6 parallel ones, or a different reasoning effort
pnpm phase0 samples/mall/ground-floor.png --single
pnpm phase0 samples/mall/ground-floor.png --effort medium   # minimal | low (default) | medium | high | none

# Without AI (OCR + rules only, free)
pnpm phase0 samples/mall/ground-floor.png --truth samples/mall/ground-truth.json --no-ai
```

Results are written to `out/<plan>/`:

| File | Contents |
|---|---|
| `report.md` | Scores, cost, validation issues, table of every extracted space |
| `tokens.json` | Every text snippet read (OCR or PDF text layer), with its position |
| `rules.json` | OCR + rules result (no AI) |
| `ai.json` | AI result + tokens used + cost |
| `issues.json` | Validation issues for the AI result |

## Pipeline

1. **Load** (`ingest.ts`) — images, or PDF pages rendered with pdf.js. Each PDF page is classified:
   - vector: CAD export with real text, so no OCR is needed
   - mixed: has an embedded picture, so OCR runs as well
   - raster: a scan, so OCR only
2. **Text** (`ocr.ts`) — Tesseract.js steps:
   - upscale and boost contrast
   - sparse-text mode
   - extra passes at 90° and 270° for vertical labels
   - split glued neighbouring labels apart
3. **Rules** (`parse.ts`) — sort each snippet into a kind:
   - unit code, mm dimension, ft dimension, category, or amenity
   - also builds a rules-only baseline result
4. **AI** (`ai.ts`) — OpenAI direct or OpenRouter (`AI_PROVIDER` in `.env`):
   - 6 calls run at the same time: overview (building, floor, wings), units in the left half, units in the right half, indoor circulation, entrances and outdoor, toilets/service/house rooms
   - the model writes a compact format, then code expands it and assigns wings by position
   - reasoning effort low; strict JSON checked against Zod; vendors are allow-listed
5. **Validate** (`validate.ts`):
   - every space must point back to its source snippets
   - printed dimensions must appear in OCR text
   - mm and ft values must agree
   - no duplicate codes or gaps in numbered series
6. **Score** (`evaluate.ts`) — compare with a hand-labelled answer key.

Shared Zod schemas, the space-type list and the dimension parsing live in `packages/shared`.

## Samples

Plan files are **not committed**: they may be confidential client drawings or belong to other publishers. `samples/mall/` and `samples/downloaded/` are git-ignored.

```bash
pnpm samples   # downloads the public sample plans into samples/downloaded/
```

- `samples/downloaded/` — public sample plans, fetched from their original sources:
  - Chief Architect "Lake Point" CAD PDF (demo set; page 6 is the main floor)
  - Maricopa County sample (PDF with a plan picture inside)
  - Wikimedia `Sample_Floorplan.jpg`
  - White House main floor, 1900 (public domain scan)
- `samples/mall/` — local only: your own plans and their hand-labelled answer keys (`ground-truth.json`)
