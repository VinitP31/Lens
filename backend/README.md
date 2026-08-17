# Lens backend

The API and everything behind it: reading PDFs, storing them, searching them, and
answering from them. A FastAPI app on port 8000. It has no user interface of its
own — the screen lives in [frontend/](../frontend/) and reaches this over HTTP.

Full specification: [docs/LENS.md](../docs/LENS.md).

## What it does

Two pipelines that never mix, because they have opposite shapes. Ingestion is
slow, write-heavy and runs in the background. Answering a question is fast,
read-only and interactive.

**Ingestion** — `validate → extract → (OCR only if needed) → chunk → embed → store`

Validation runs before the upload is acknowledged, so a file that is too big,
password protected, corrupt or already in the library is refused while the user is
still looking at the dialog. Everything after that runs in a background task and
the caller polls for progress. Any stage that fails rolls the document back
completely, so the library never holds a half-indexed document.

**Answering** — `condense → analyze → retrieve → gate → generate → validate citations`

A very long message is shortened before it is searched with. One model call then
decides whether the message is a greeting, a question about the app, or a real
question, and rewrites a follow-up into something that stands on its own. Only a
real question is searched. The result passes a numeric gate before any answer
model is called, and the citations that come back are checked against what was
actually supplied.

## Setup

Python 3.11 or 3.12. Run these from the project root, not from this directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then put your OPENAI_API_KEY in it
```

`OPENAI_API_KEY` is the only secret. Everything else is a setting, and every
setting lives in [config/settings.py](../config/settings.py) — chunk sizes, the
gate threshold, model names, limits, timeouts. There are no tunable literals
anywhere else in the code.

The first run downloads Docling's layout models, about 500 MB, and prints nothing
while it does. It happens once.

## Run

```bash
uvicorn backend.main:app --reload
```

`main.py` reads `.env` into the environment on import, by absolute path, so this
works from any working directory. A variable already exported wins over the file:
an export is a deliberate choice for one run, and a file on disk should not beat
it.

- API: `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`

`data/` is created on startup and holds everything the app produces: the SQLite
database, the vector store, the uploaded PDFs, and the trace logs. It is not
committed. Deleting it resets the app.

**Only one process may open the vector store.** Milvus Lite is a local file with a
single-process lock, so the backend and `evaluation/run_eval.py` cannot run at the
same time. Stop the backend before running the evaluation.

## Startup fails rather than degrades

The app refuses to start on any of these, because each one produces answers that
are quietly wrong rather than obviously broken:

- no `OPENAI_API_KEY` in the environment
- the vector store cannot be opened, or holds vectors of the wrong width
- the configured embedding model is not the one the index was built with
- the registry lists documents whose chunks are gone from the vector store

The last two are the ones worth having. Vectors from two different models sit in
different spaces, so mixing them rots retrieval while every part of the system
reports success. A registry that expects chunks against an empty store answers
"not found in your documents" to a library that visibly contains documents.

A document left mid-ingest by a killed process is discarded at startup, before the
first request.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/documents` | Upload a PDF. Validates synchronously, indexes in the background, returns `202` with a document id and a job id |
| `GET` | `/documents` | The library. `?ready_only=true` for only what is searchable. Deleted documents are never listed |
| `GET` | `/documents/{doc_id}/status` | Which stage the ingest reached, a fraction, and whether it finished |
| `DELETE` | `/documents/{doc_id}` | Soft delete. The row and the file stay so old citations keep rendering |
| `GET` | `/documents/{doc_id}/pages/{page}` | The page as a PNG. With `?chunk_id=…` the cited region is highlighted |
| `POST` | `/conversations` | Start a chat, with a scope of the whole library or a list of documents |
| `GET` | `/conversations` | The chat list, most recently used first |
| `GET` | `/conversations/{conv_id}` | One chat with its messages and their stored citations |
| `PATCH` | `/conversations/{conv_id}` | Rename, or change which documents it searches |
| `DELETE` | `/conversations/{conv_id}` | Remove a chat and its messages |
| `POST` | `/conversations/{conv_id}/messages` | Ask a question. Streams the answer (see below) |
| `GET` | `/health` | Store reachability, document and chunk counts, model names, whether the embedding model matches, the gate threshold |

### The answer stream

`POST /conversations/{conv_id}/messages` replies with server-sent events, in this
order:

| Event | Payload |
|---|---|
| `token` | A piece of the answer text. Many of these |
| `citations` | The validated citations, once generation has finished. They cannot be resolved before the model has stopped citing |
| `done` | The message id, the full text, whether it abstained, the citations again, and the diagnostics |
| `error` | Something failed after the stream opened, so there was no status code left to send |

A refusal sends no `token` at all, only `done` with `abstained: true`. That lets
the screen show it as its own calm state rather than as an answer that says no.

`done` always carries diagnostics: the top similarity score, the gate threshold,
how many chunks were retrieved and how many were used, how many citations were
discarded, and the latency. Every turn is stored with those numbers and with the
scope it was actually searched against, so "why did it say that?" never requires
asking again.

## Errors

Every failure raises a typed exception from [errors.py](errors.py) carrying a
stable `code`. One handler in [main.py](main.py) turns it into
`{"code": ..., "message": ...}` with an HTTP status. Callers switch on the code
and never on the message, so wording can change freely.

| Code | Status | Cause |
|---|---|---|
| `duplicate_document` | 409 | These exact bytes are already in the library |
| `file_too_large` | 413 | Over `MAX_FILE_BYTES` |
| `too_many_pages` | 413 | Over `MAX_PAGES` |
| `encrypted_pdf` | 415 | Password protected, so the text cannot be read |
| `corrupt_file` | 415 | Could not be opened as a PDF at all |
| `empty_document` | 422 | Opened, but there is nothing in it |
| `unreadable_document` | 422 | Pages, but no readable text even after OCR |
| `extraction_failed` | 422 | The extraction worker could not process the file |
| `empty_scope` | 422 | A chat given no documents to search |
| `render_failed` | 422 | The page could not be turned into an image |
| `document_not_found` | 404 | No such document, or it was deleted |
| `conversation_not_found` | 404 | No such chat |
| `page_not_found` | 404 | That page is outside the document |
| `embedding_failed` | 502 | The embedding provider failed, after retries |
| `generation_failed` | 502 | The answer model failed. Never reported as an abstention |
| `missing_api_key` | 503 | No key in the environment |
| `vector_store_error` | 503 | The vector store could not be opened or written |
| `store_mismatch` | 503 | The registry and the vector store disagree about what exists |

A provider that cannot be reached raises rather than abstaining. Telling somebody
their documents do not cover a question when the truth is that a network call
failed would be a lie in the one place this system exists not to tell one.

## Ingest stages

`/documents/{doc_id}/status` reports one of these:

```
queued → validating → extracting → [ocr] → chunking → embedding → indexing → ready
```

`ocr` appears only when the first read found almost no text. `ready` and `deleted`
are the only terminal states. A failed ingest is discarded, not stored, so there
is never a broken entry in the library — `failed` exists for the trace log alone.

## Layout

```
backend/
├── main.py                 App setup, startup checks, error translation
├── errors.py               Typed exceptions with stable codes
│
├── api/
│   ├── schemas.py          Pydantic request and response models
│   ├── routes_documents.py Upload, list, status, delete, page image
│   ├── routes_conversations.py
│   └── routes_chat.py      The answer stream, and /health
│
├── ingestion/
│   ├── pipeline.py         Stage order, status updates, rollback, recovery
│   ├── validator.py        Hash, size, pages, encryption, corruption
│   ├── prepare.py          Launches the worker and reads its result
│   ├── worker.py           Extraction as its own program
│   ├── extractor.py        Docling, with a page and coordinates on every element
│   ├── ocr.py              The density check and the OCR fallback
│   ├── chunker.py          Structure-first chunking, context headers
│   ├── chunk.py            The Chunk type, free of Docling imports
│   └── embedder.py         Batched, retry-capped
│
├── retrieval/
│   ├── condenser.py        Shortens a very long message before searching
│   ├── analyzer.py         Intent and follow-up rewrite, in one call
│   ├── retriever.py        Scoped search, over-fetch, dedupe
│   ├── gate.py             The threshold check. No model call
│   ├── prompt.py           Prompt assembly, stable prefix first
│   ├── generator.py        Grounded generation, streaming
│   └── citations.py        Validate cited numbers, resolve them to pages
│
├── storage/
│   ├── vector_store.py     Milvus Lite schema, upsert, filtered search, delete
│   ├── registry.py         Document records and status
│   ├── conversations.py    Chats, messages, the scope each was searched with
│   ├── files.py            Original PDFs on disk, under their content hash
│   └── schema.sql          documents, ingestion_jobs, conversations, messages
│
├── rendering/
│   └── page_renderer.py    PyMuPDF page image with the cited box highlighted
│
└── logging/
    └── trace.py            One JSON line per question and per indexed document
```

Imports go one way. Nothing in `ingestion/`, `retrieval/`, `storage/` or
`rendering/` imports from `api/`, and nothing in the backend imports from
`frontend/`.

## Four things worth knowing before changing this

**The Milvus score is already a similarity.** With the cosine metric, Milvus puts
the cosine *similarity* in the field it calls `distance`: identical is `+1.0`,
unrelated is `0.0`, opposite is `-1.0`. Measured against a live collection, not
assumed. `1 - distance` is right for L2 and **inverts the gate here**, giving a
system that answers out-of-scope questions and refuses real ones. The conversion
lives in `vector_store._similarity` and nowhere else; use `Hit.similarity`, which
is already the right way up. `Hit.raw_distance` sits beside it so both are logged.

**Extraction runs in a separate process.** Docling and Milvus Lite each bundle
their own copy of the OpenMP runtime, and a process that initialises both dies.
`worker.py` is launched as a plain subprocess with its file descriptors closed —
not a `multiprocessing` child, which inherits the parent's live connection to the
vector store and dies with an empty stderr. `prepare.py` retries once if the
worker is killed by a signal.

**The gate runs before the answer model, and refusal is arithmetic.** Whether
there is enough to answer from is never a question put to a model:

```python
if hits[0].similarity < GATE_THRESHOLD:
    return abstention          # no model call at all
```

`GATE_THRESHOLD` is 0.45, measured from 24 answerable and 17 unanswerable
questions rather than chosen by taste. The reasoning, including what every other
candidate value would have cost, is written out in `settings.py` beside it.
Raising it is the right answer to off-topic questions being answered. It is the
wrong answer to on-topic questions whose answer is simply absent — no threshold
separates those, and the prompt's abstention rule is what catches them.

**The model never writes a citation.** It cites by number. Code checks the number
was among the passages supplied, discards any that was not, and resolves the
survivors to a real document, page, section and set of coordinates. Those resolved
values are stored on the message, never re-resolved at render time, so deleting a
document cannot break an answer given last week.

## Tests

```bash
python -m pytest tests/ -q
```

495 tests, and no network at all: the embedding function is injected, so the suite
costs nothing to run and works with no key set.

It takes about three minutes on this machine, and almost all of that is two things
rather than the tests themselves. A handful of tests extract a real PDF, which
launches a worker that loads Docling — around 20 seconds each. And every test
touching storage opens a fresh vector store, which starts a local Milvus process,
roughly half a second a time. Everything else runs in milliseconds:

```bash
python -m pytest tests/ -q --ignore=tests/test_prepare.py --ignore=tests/test_ocr.py
```

Nothing asserts on the wording of a generated answer — that varies between runs
and a test on it is a test that will eventually fail for no reason. What is tested
is the deterministic half: page attribution, chunk bounds, the distance-to-
similarity conversion, which citations survive validation, which uploads are
refused and with what code, and that a rejection leaves the library untouched.

The most important single test is page attribution. An off-by-one page map makes
every citation quietly wrong.

Some invariants a fixture cannot see, because every test PDF is generated:

```bash
python scripts/check.py --corpus
```

That re-extracts the six real sample documents and asserts no text was lost, no
chunk was filed under the wrong page, every chunk carries coordinates, and no
contents page was indexed. Run it before committing anything that touches
extraction or chunking.

Whichever check fails, its full output goes to `data/last-check.log` and the console
prints that path. The tail on screen is a summary; the file is what a failure that
will not reproduce has to be diagnosed from.

## Tools

```bash
python scripts/profile_pdf.py samples/*.pdf          # what the pipeline sees
python scripts/profile_pdf.py somefile.pdf --dump    # also write the text out
python evaluation/run_eval.py                        # retrieval and the gate
python evaluation/run_eval.py --answers              # also answers and refusals
```

`profile_pdf.py` is the fastest way to understand why a document behaved oddly: it
reports pages, text density, whether OCR would trigger, headings, tables, figures,
what was dropped as furniture, and whether every element carries coordinates.

The evaluation needs the backend stopped, for the single-process reason above.
