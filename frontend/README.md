# Lens frontend

The screen. A Streamlit app on port 8501 that reaches the backend over HTTP and
holds no knowledge of its own about documents, chats or answers.

Full specification: [docs/LENS.md](../docs/LENS.md).

## Setup

Python 3.11 or 3.12. Run these from the project root, not from this directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then put your OPENAI_API_KEY in it
```

The key is used by the backend, not by this. The screen makes no model calls.

## Run

The backend has to be up first. In one terminal:

```bash
uvicorn backend.main:app --reload
```

In another:

```bash
streamlit run frontend/app.py
```

Then open `http://localhost:8501`.

If the backend is not running, the first paint says so plainly and stops, rather
than letting every control on the page fail one at a time. The address it uses is
`API_BASE_URL` in [config/settings.py](../config/settings.py) — the only place it
is written down.

## What the screen is

Chat-first. The app opens straight into a conversation. There is no dashboard and
no home page, and an empty library is this same screen with the input switched off
and a short welcome, so nobody has to learn a page they will see once.

Three surfaces, and no more:

| Surface | Holds |
|---|---|
| Sidebar | New chat, and the chat history. Never documents |
| Main area | The context indicator, the message thread, the input, and two buttons |
| Documents drawer | A dialog, opened on demand: the document list, upload, delete |

Documents live in a drawer rather than the sidebar on purpose. A permanent list of
files turns the app into a file manager, and the thing being used is the
conversation.

## How a turn appears

Text streams in as the backend produces it. The sources appear underneath once
generation has finished — they cannot exist before the model has stopped citing.
Each source shows the document name and page, expands to the passage it came from,
and clicking it opens the page itself as an image with the cited region
highlighted, rendered by the backend.

A refusal is its own calm state, not an error. The backend sends no answer text at
all in that case, and the wording belongs here because only the screen knows what
is currently being searched: suggesting a wider selection is helpful when the chat
is scoped to two documents and misleading when it is already searching everything.

A citation into a document that has since been removed still renders, from the
values stored with the answer at the time, and is marked as removed from the
library.

## Context

Every chat searches either the whole library or a chosen subset, shown above the
thread at all times and editable mid-conversation in both directions. Three
display states: `Entire knowledge base`, one document's name, or the first name
and a count of the rest.

Uploading a document while a chat is open is keyed on one question — has this
conversation started?

| Situation | What happens |
|---|---|
| No messages yet | The context silently becomes the new document |
| Messages, searching the whole library | Included silently, no prompt |
| Messages, searching a subset | Asks: switch to the new document, or add it |
| Uploaded from the drawer | Goes to the library only, no change to any chat |

An empty selection is refused with an explanation rather than saved. Every
question against an empty scope would be refused for having nothing to search, and
a user would read that as the app being broken rather than as the setting they
chose.

## Layout

```
frontend/
├── app.py              Entry point, layout, and the health check on first paint
├── api_client.py       The only thing that talks to the backend
├── state.py            Session keys, the upload guard, and one-shot notices
└── components/
    ├── sidebar.py           New chat, history, delete-with-confirm
    ├── chat.py              The thread, streaming, and the refusal state
    ├── context_indicator.py What this chat searches, and changing it
    ├── documents_drawer.py  Document list, upload, delete
    ├── upload.py            Sending a file and watching it index
    ├── citations.py         The source list, and the cited page shown beside the answer
    └── empty_state.py       First run, and the backend-is-down screen
```

Nothing here imports a backend module. Everything goes through `api_client`, which
means the two halves can be run, restarted and reasoned about separately, and no
screen can reach into the database by accident.

## Three things that will bite you

**Streamlit re-runs this whole file on every interaction.** Every click, every
keystroke that submits, every rerun triggered by code — the script runs again from
the top. So nothing may be remembered in a module-level variable, no heavy work
may sit at module level, and anything that must survive goes either in
`st.session_state` or in the backend.

**The uploader re-returns the same file on every rerun.** Without a guard, one
upload is sent four or five times, burning embedding calls in a loop. `state.py`
keeps a set of hashes this session has already sent and checks it before every
upload. This is separate from the backend's own duplicate detection by content
hash, and both are needed: the backend stops the same file being indexed twice,
and this stops the same file being *sent* dozens of times.

**Session state is wiped on refresh.** So chat history, the document registry and
every answer live in the backend's SQLite database and are read back on each run.
Session state holds only what is genuinely transient: which chat is open, which
uploads this session has sent, a scope edit in flight, and which delete button is
armed. No browser storage is used anywhere — `localStorage` and `sessionStorage`
are not supported in this environment and fail silently.

## Errors

`api_client` raises `LensApiError` carrying a `code` and a `message`. The code is
stable and is what the screen decides with; the message is what it shows. Nothing
matches on message text, so rewording one can never change which case the UI
thinks it is in.

A backend that cannot be reached at all arrives as code `unreachable`, so it reads
as "the backend is not running" rather than as an unexplained failure.

## Tests

```bash
python -m pytest tests/test_frontend.py -q
```

27 tests, a couple of seconds, with no Streamlit server and no backend running.
They cover the parts that
are decisions rather than rendering: the upload guard, the three context display
states, refusing an empty scope, which refusal wording is honest for which scope,
the confirm-once chat delete, and that a stream of events is turned into the right
`Answer`.

Rendering itself is not tested. A test asserting that a button exists tells you
nothing a glance at the running app does not tell you faster.
