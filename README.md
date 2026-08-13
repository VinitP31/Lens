# Lens

Ask questions about your documents. Get answers that show their source.

Full specification: [docs/LENS.md](docs/LENS.md)

## Setup

Python 3.11 or 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then add your OPENAI_API_KEY
```

First run downloads Docling models (~500 MB) and pauses with no output. Once only.

## Run

```bash
uvicorn backend.main:app --reload      # backend, port 8000
streamlit run frontend/app.py          # UI, port 8501
```

## Checks

One command runs everything:

```bash
python scripts/check.py            # lint, format, tests
python scripts/check.py --corpus   # also re-audits the sample PDFs (slow)
```

Run the fast checks after every change. Run `--corpus` before committing anything that
touches extraction or chunking: it re-extracts the real documents and asserts what a
fixture-based test cannot see — no text lost, no chunk filed under the wrong page, every
chunk carrying coordinates, no contents page indexed.

## Evaluation

`GATE_THRESHOLD` must be measured, never guessed.

```bash
python evaluation/run_eval.py
```
