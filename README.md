# Lens

Ask questions about your PDFs. Every answer shows the page it came from, and when
the documents do not hold the answer, Lens says so instead of inventing one.

Full specification: [docs/LENS.md](docs/LENS.md).

## What it is

Upload PDFs, ask questions in a chat, and get an answer with clickable citations —
each one opens the actual page with the passage highlighted. Ask something the
documents do not cover and you get a plain refusal, not a plausible paragraph.

The refusal is the point. It is a numeric threshold in code, checked before any
answer model is called, backed by a second layer in the prompt for the questions a
threshold cannot catch.

Two processes:

| Part | What it is | Port |
|---|---|---|
| [backend/](backend/) | FastAPI. Reads, stores, searches and answers | 8000 |
| [frontend/](frontend/) | Streamlit. The screen. Talks to the backend over HTTP only | 8501 |

Both have their own README with the detail.

## Setup

Python 3.11 or 3.12. Nothing else has to be installed first — no database server,
no Tesseract, no Homebrew packages. The vector store and the registry are both
local files.

```bash
git clone https://github.com/VinitP31/Lens.git
cd Lens

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Then open `.env` and put your OpenAI key in it:

```
OPENAI_API_KEY=sk-...
```

That is the only secret. Every other value — chunk sizes, the gate threshold,
model names, upload limits, timeouts — lives in
[config/settings.py](config/settings.py), which is committed, so there is exactly
one place to look and one place to change.

## Run

Two terminals, both with the virtual environment active.

```bash
uvicorn backend.main:app --reload      # terminal 1
```

```bash
streamlit run frontend/app.py          # terminal 2
```

The backend reads `.env` itself on startup, so the key needs nothing else. The
screen needs no key at all — it makes no model calls.

Then open `http://localhost:8501` and upload a PDF. There are six to try in
[samples/](samples/).

Two things to expect on a first run:

- **The first PDF is slow to start.** Docling downloads its layout models, about
  500 MB, and prints nothing while it does. It happens once. Afterwards,
  extraction runs at roughly three seconds a page.
- **Indexing a 30-page document takes a couple of minutes.** The screen shows each
  stage as it happens. The upload is refused immediately if the file is too big,
  password protected, corrupt or already in the library — those never wait for
  indexing.

`data/` appears on first start and holds everything the app produces: the SQLite
database, the vector store, the uploaded PDFs, and the trace logs. It is not
committed. Deleting the whole directory resets the app to empty.

**Only one process may open the vector store.** Milvus Lite is a local file with a
single-process lock, so the backend and the evaluation script cannot run at the
same time. Stop the backend before running the evaluation.

## Checks

```bash
python scripts/check.py            # lint, format, 528 tests. About 3 minutes
python scripts/check.py --corpus   # also re-audits the six sample PDFs. Add a couple more
```

When a check fails, the console shows the tail and the whole output is written to
`data/last-check.log`, which the failure line points at. That exists because a test
failed once, could not be reproduced afterwards, and the lines explaining it had
already scrolled away.

Run this after every change. The suite makes no network calls — the embedding
function is injected, so it costs nothing and needs no key. The three minutes are
spent on real work: a handful of tests extract an actual PDF through the worker,
which loads Docling each time, and every test touching storage opens a fresh local
vector store.

Run `--corpus` before committing anything that touches extraction or chunking. It
re-extracts the real documents and asserts what a generated fixture cannot see: no
text lost, no chunk filed under the wrong page, every chunk carrying coordinates,
no contents page indexed. Each of those invariants has caught a real defect at
least once.

## Evaluation

The confidence threshold was measured, not chosen. So was the answer model.

```bash
python evaluation/run_eval.py --ingest     # index the samples, then measure
python evaluation/run_eval.py              # retrieval and the gate. Costs 41 embeddings
python evaluation/run_eval.py --answers    # also answers and refusals. One model call per question
```

Two question sets, both written before any tuning, so the system could not be
quietly fitted to questions it already passed:
[golden_set.csv](evaluation/golden_set.csv) holds 24 answerable questions with the
document and page the answer lives on;
[out_of_scope.csv](evaluation/out_of_scope.csv) holds 17 questions that sound
plausible for this corpus and have no answer in it.

Measured on the six sample documents:

| | Result |
|---|---|
| Answerable questions where the expected page was retrieved | 24/24, mean rank 1.25 |
| Answerable questions answered | 24/24 |
| Citations landing on the expected page | 24/24 |
| Answerable questions wrongly refused | 0/24 |
| Unanswerable questions refused | 17/17 |
| — stopped by the gate, with no model call | 4/17 |
| — refused by the model after reading the passages | 13/13 of those that reached it |
| Answers citing a passage that was never supplied | 0 |

Both refusal layers are needed. The gate cannot tell that an on-topic question's
answer is absent, and the four questions it stopped for free would never have
reached the prompt.

## How it works

Reading a document:

```
validate → extract → (OCR only if the file has no text layer) → chunk → embed → store
```

Answering a question:

```
condense → analyze → retrieve → gate → generate → validate citations
```

The rule the whole design follows: **the model handles language, and code makes
every decision that has to be right.**

- **Refusal is arithmetic.** `if top_similarity < 0.45: refuse`, evaluated before
  any answer model is called. Whether there is enough to answer from is never a
  question put to a model.
- **The model never writes a citation.** It cites by number. Code checks the
  number was among the passages supplied, discards any that was not, and resolves
  the survivors to a real document, page, section and set of coordinates.
- **No document-specific logic anywhere.** No filenames, no titles, no page
  offsets in code. Structure detection is generic and every stage degrades rather
  than failing, so an unseen PDF works with no code changes.

## Repository layout

```
Lens/
├── README.md               This file
├── backend/                FastAPI: ingestion, retrieval, storage, rendering
│   └── README.md
├── frontend/               Streamlit: the screen
│   └── README.md
├── config/settings.py      Every tunable value. No literals elsewhere
├── docs/LENS.md            The specification
├── samples/                Six real PDFs, plus two stress files never indexed
├── evaluation/             The two question sets and the measurement script
├── tests/                  528 tests, no network
├── scripts/                Profiler, checks, reset, stress-PDF generator
└── data/                   Runtime state. Created on first start, not committed
```

## Tools

```bash
python scripts/profile_pdf.py samples/*.pdf          # what the pipeline sees in a PDF
python scripts/profile_pdf.py somefile.pdf --dump    # also write the extracted text out
python scripts/reset_store.py                        # list what a reset would remove
python scripts/reset_store.py --yes                  # remove it
python scripts/make_stress_pdfs.py                   # rebuild the two stress PDFs
```

`profile_pdf.py` is the fastest way to understand why a document behaved oddly. It
reports pages, text density, whether OCR would trigger, headings, tables, figures,
what was dropped as page furniture, and whether every element carries coordinates.

`reset_store.py` empties the library. It prints what it is about to remove and
removes nothing without `--yes`, and it refuses to run at all while the backend is
serving — the vector store is one file held open by one process, and deleting it
underneath a live server leaves the backend answering from a store that is gone.

## Known limits

Stated rather than hidden — a system with its weaknesses written down is easier to
trust than one claiming to have none. Section 20 of
[docs/LENS.md](docs/LENS.md) has the full list with what each one costs.

- One shared library and no login. Anyone who can reach an instance can read every
  document in it.
- 50 pages and 25 MB a file, English only, PDFs only.
- Five passages are sent to the model, so a question needing wide synthesis across
  a document can come back partly answered.
- A borderless table — a grid drawn with spacing rather than lines — can lose the
  pairing between a row and its values.
- An answer can be right, cited, and citing the wrong page. No code can check that
  a sentence follows from a passage. This is why the cited page is one click away.
