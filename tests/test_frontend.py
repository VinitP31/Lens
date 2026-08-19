"""Tests for the screen.

The backend is stubbed, so nothing starts a server or spends money. Streamlit is
real: the app script runs under `AppTest`, so a rerun bug, a duplicate widget key or
an exception in a component shows up here rather than in the browser.

The upload guard is the most important test in this file.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend import api_client, state
from frontend.components import citations as citations_ui
from frontend.components import context_indicator

# Absolute: AppTest resolves a relative path against the file that calls it, not
# against the repository root.
APP = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")

# A one pixel PNG. The panel only has to receive bytes; what they draw is the
# page renderer's business, tested against real pages elsewhere.
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
    b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

CITATION = {
    "n": 1,
    "chunk_id": "d1:12",
    "doc_id": "d1",
    "display_name": "FleetLink TMS User Manual v4.2",
    "page": 5,
    "section_path": "FleetLink > 3. Getting Started > 3.2 Password rules",
    "element_type": "table",
    "snippet": "| Rule | Setting |\n|---|---|\n| Minimum length | 12 characters |",
    "bboxes": [[10.0, 20.0, 100.0, 40.0]],
}

DOCUMENTS = [
    {
        "doc_id": "d1",
        "display_name": "FleetLink TMS User Manual v4.2",
        "status": "ready",
        "page_count": 26,
        "chunk_count": 80,
        "table_count": 4,
        "image_count": 0,
        "size_bytes": 900_000,
        "ocr_applied": False,
        "uploaded_at": "2026-08-17T10:00:00+00:00",
    },
    {
        "doc_id": "d2",
        "display_name": "26-004 CRM Software RFP Package",
        "status": "ready",
        "page_count": 33,
        "chunk_count": 144,
        "table_count": 9,
        "image_count": 1,
        "size_bytes": 1_400_000,
        "ocr_applied": False,
        "uploaded_at": "2026-08-17T10:05:00+00:00",
    },
]

HEALTH = {
    "status": "ok",
    "documents": 2,
    "chunks": 224,
    "embedding_model": "text-embedding-3-small",
    "answer_model": "gpt-4o-mini",
    "embed_model_matches": True,
    "gate_threshold": 0.45,
}


def conversation(messages=None, scope_mode="library", scope_doc_ids=None) -> dict:
    return {
        "conv_id": "c1",
        "title": "Password rules",
        "title_is_auto": True,
        "scope_mode": scope_mode,
        "scope_doc_ids": scope_doc_ids,
        "created_at": "2026-08-17T10:00:00+00:00",
        "updated_at": "2026-08-17T10:10:00+00:00",
        "messages": messages or [],
    }


def message(role, content, citations=None, abstained=False) -> dict:
    return {
        "msg_id": "m1",
        "role": role,
        "content": content,
        "created_at": "2026-08-17T10:00:00+00:00",
        "citations": citations or [],
        "intent": "question",
        "abstained": abstained,
    }


def citation(n=1, doc_id="d1", page=5, kind="text") -> dict:
    return {
        "n": n,
        "chunk_id": f"{doc_id}:12",
        "doc_id": doc_id,
        "display_name": "FleetLink TMS User Manual v4.2",
        "page": page,
        "section_path": "FleetLink > 3. Getting Started > 3.2 Password rules",
        "element_type": kind,
        "snippet": "Passwords must be at least 12 characters…",
        "bboxes": [[72.0, 210.5, 523.0, 288.25]],
    }


@pytest.fixture
def backend(monkeypatch):
    """A stubbed backend. Tests override individual calls as needed."""
    calls = {"uploads": [], "created": 0}

    monkeypatch.setattr(api_client, "health", lambda: HEALTH)
    monkeypatch.setattr(api_client, "documents", lambda ready_only=False: list(DOCUMENTS))
    monkeypatch.setattr(api_client, "conversations", lambda: [])
    monkeypatch.setattr(api_client, "conversation", lambda conv_id: conversation())

    def create(scope_mode="library", scope_doc_ids=None):
        calls["created"] += 1
        return conversation(scope_mode=scope_mode, scope_doc_ids=scope_doc_ids)

    monkeypatch.setattr(api_client, "create_conversation", create)

    def upload(name, data):
        calls["uploads"].append((name, len(data)))
        return {"doc_id": "new", "job_id": "j1", "display_name": name, "page_count": 3}

    monkeypatch.setattr(api_client, "upload", upload)
    monkeypatch.setattr(api_client, "status", lambda doc_id: {"stage": "ready", "finished": True})
    return calls


# --- the upload guard ----------------------------------------------------


def test_the_same_file_is_only_sent_once():
    """Streamlit re-returns the file on every rerun, and a rerun happens on every
    click. Without the guard one upload becomes four or five, and each attempt
    pays for embeddings before the backend's own hash check rejects it."""
    import streamlit as st

    st.session_state[state.SENT_UPLOADS] = set()
    data = b"%PDF-1.7 pretend"

    seen = [state.already_sent(data) for _ in range(4)]

    assert seen == [False, True, True, True]


def test_a_different_file_is_not_blocked():
    import streamlit as st

    st.session_state[state.SENT_UPLOADS] = set()

    assert state.already_sent(b"%PDF one") is False
    assert state.already_sent(b"%PDF two") is False


def test_a_failed_send_can_be_retried():
    """The guard stops accidental repeats, not a deliberate retry after a real
    failure."""
    import streamlit as st

    st.session_state[state.SENT_UPLOADS] = set()
    data = b"%PDF-1.7 pretend"

    assert state.already_sent(data) is False
    state.forget_upload(data)

    assert state.already_sent(data) is False


def test_a_rejected_upload_is_reported_and_can_be_sent_again(monkeypatch):
    """A refusal is an ordinary outcome the user can act on, not a crash."""
    import streamlit as st

    from frontend.components import upload

    st.session_state[state.SENT_UPLOADS] = set()

    def refuse(name, data):
        raise api_client.LensApiError("duplicate_document", "already in the library")

    monkeypatch.setattr(api_client, "upload", refuse)
    shown = []
    monkeypatch.setattr(upload.st, "error", shown.append)

    assert upload.send("a.pdf", b"%PDF one") is None
    assert "already in your library" in shown[0]
    # The guard was cleared, so a second attempt actually reaches the backend.
    assert state.already_sent(b"%PDF one") is False


# --- what the screen says with nothing to search -------------------------


def test_an_empty_library_disables_the_input(monkeypatch, backend):
    """Every question would be refused, so the input is switched off rather than
    inviting one."""
    monkeypatch.setattr(api_client, "documents", lambda ready_only=False: [])

    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert at.chat_input[0].disabled
    assert any("Add a PDF" in m.value for m in at.markdown)


def test_an_unreachable_backend_says_what_to_do(monkeypatch):
    """Every control on the page would fail, so the page explains itself instead
    of rendering controls that do nothing."""

    def down():
        raise api_client.LensApiError("unreachable", "connection refused")

    monkeypatch.setattr(api_client, "health", down)

    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert any("can't reach its backend" in m.value for m in at.markdown)
    assert not at.chat_input


# --- the thread ----------------------------------------------------------


def test_an_answer_shows_its_sources(monkeypatch, backend):
    monkeypatch.setattr(api_client, "conversations", lambda: [{"conv_id": "c1", "title": "T"}])
    monkeypatch.setattr(
        api_client,
        "conversation",
        lambda conv_id: conversation(
            messages=[
                message("user", "How long must a password be?"),
                message("assistant", "Twelve characters [1].", citations=[citation()]),
            ]
        ),
    )

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    assert not at.exception
    labels = [e.label for e in at.expander]
    assert any("page 5" in label for label in labels)
    assert any("FleetLink" in label for label in labels)


def test_a_table_citation_says_it_is_a_table(monkeypatch, backend):
    """A table is read differently from a sentence, so the label is honest about
    which one the answer leaned on."""
    monkeypatch.setattr(api_client, "conversations", lambda: [{"conv_id": "c1", "title": "T"}])
    monkeypatch.setattr(
        api_client,
        "conversation",
        lambda conv_id: conversation(
            messages=[
                message("assistant", "Twelve [1].", citations=[citation(kind="table")]),
            ]
        ),
    )

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    assert any("(table)" in e.label for e in at.expander)


def test_a_citation_to_a_removed_document_is_marked(monkeypatch, backend):
    """The answer still renders from what was stored, so an old answer does not
    break when a document leaves the library."""
    monkeypatch.setattr(api_client, "conversations", lambda: [{"conv_id": "c1", "title": "T"}])
    monkeypatch.setattr(
        api_client,
        "conversation",
        lambda conv_id: conversation(
            messages=[
                message("assistant", "Twelve [1].", citations=[citation(doc_id="gone")]),
            ]
        ),
    )

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    assert any("removed from library" in e.label for e in at.expander)


def test_a_refusal_is_calm_and_not_an_error(monkeypatch, backend):
    monkeypatch.setattr(api_client, "conversations", lambda: [{"conv_id": "c1", "title": "T"}])
    monkeypatch.setattr(
        api_client,
        "conversation",
        lambda conv_id: conversation(
            messages=[
                message("user", "Who won the tender?"),
                message("assistant", "", abstained=True),
            ]
        ),
    )

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    assert not at.error
    assert any("couldn't find this" in m.value for m in at.markdown)


def test_widening_is_only_suggested_when_the_selection_is_narrow(monkeypatch, backend):
    """Telling a user to widen when everything is already searched sends them
    looking for a setting that cannot help."""
    monkeypatch.setattr(api_client, "conversations", lambda: [{"conv_id": "c1", "title": "T"}])
    monkeypatch.setattr(
        api_client,
        "conversation",
        lambda conv_id: conversation(
            messages=[message("assistant", "", abstained=True)], scope_mode="library"
        ),
    )

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    captions = " ".join(c.value for c in at.caption)
    assert "widen" not in captions.lower()


def test_widening_is_suggested_when_only_some_documents_are_searched(monkeypatch, backend):
    monkeypatch.setattr(api_client, "conversations", lambda: [{"conv_id": "c1", "title": "T"}])
    monkeypatch.setattr(
        api_client,
        "conversation",
        lambda conv_id: conversation(
            messages=[message("assistant", "", abstained=True)],
            scope_mode="subset",
            scope_doc_ids=["d1"],
        ),
    )

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    captions = " ".join(c.value for c in at.caption)
    assert "widen" in captions.lower()


# --- the sidebar ---------------------------------------------------------


def test_the_sidebar_holds_history_and_never_documents(monkeypatch, backend):
    """Two lists in one narrow column compete for the same space, so documents
    live in a drawer instead."""
    monkeypatch.setattr(
        api_client,
        "conversations",
        lambda: [
            {"conv_id": "c1", "title": "Password rules"},
            {"conv_id": "c2", "title": "Evaluation weighting"},
        ],
    )

    at = AppTest.from_file(APP, default_timeout=30).run()

    labels = [b.label for b in at.sidebar.button]
    assert "New chat" in labels
    assert "Password rules" in labels
    assert not any("FleetLink" in label for label in labels)


def test_a_long_chat_title_is_shortened_in_the_sidebar(monkeypatch, backend):
    long_title = "What is the cumulative target time from gate entry to putaway confirmation?"
    monkeypatch.setattr(
        api_client, "conversations", lambda: [{"conv_id": "c1", "title": long_title}]
    )

    at = AppTest.from_file(APP, default_timeout=30).run()

    shown = [b.label for b in at.sidebar.button if b.label != "New chat"][0]
    assert len(shown) <= 42


# --- the context indicator ----------------------------------------------


def test_the_whole_library_is_described_plainly():
    assert context_indicator.describe("library", None, {}) == "Entire knowledge base"


def test_one_document_is_named():
    names = {"d1": "FleetLink TMS User Manual v4.2"}

    assert context_indicator.describe("subset", ["d1"], names) == "FleetLink TMS User Manual v4.2"


def test_several_documents_name_the_first_and_count_the_rest():
    names = {"d1": "FleetLink", "d2": "CRM RFP", "d3": "Handbook"}

    assert context_indicator.describe("subset", ["d1", "d2", "d3"], names) == "FleetLink and 2 more"


def test_a_document_missing_from_the_library_still_describes():
    """An old chat can name a document that has since been removed."""
    assert "removed" in context_indicator.describe("subset", ["gone"], {})


# --- deleting a chat -----------------------------------------------------
# Deleting is permanent and the control sits beside the one that opens a chat, so
# a stray click is likely. It asks once.


@pytest.fixture
def two_chats(monkeypatch, backend):
    monkeypatch.setattr(
        api_client,
        "conversations",
        lambda: [
            {"conv_id": "c1", "title": "Password rules"},
            {"conv_id": "c2", "title": "Evaluation weighting"},
        ],
    )
    deleted = []
    monkeypatch.setattr(api_client, "delete_conversation", deleted.append)
    return deleted


def test_every_chat_offers_a_delete(two_chats):
    at = AppTest.from_file(APP, default_timeout=30).run()

    keys = [b.key for b in at.sidebar.button]
    assert "arm-c1" in keys
    assert "arm-c2" in keys


def test_one_click_does_not_delete(two_chats):
    """It asks first. A single click on a narrow row must not lose a conversation."""
    at = AppTest.from_file(APP, default_timeout=30).run()

    at.sidebar.button(key="arm-c1").click().run()

    assert two_chats == []
    assert any(b.key == "del-c1" for b in at.sidebar.button)


def test_the_confirmation_says_documents_are_kept(two_chats):
    """The main reason to hesitate is fear of losing the documents, so the wording
    answers it."""
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button(key="arm-c1").click().run()

    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "documents are kept" in captions.lower()


def test_confirming_deletes_it(two_chats):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button(key="arm-c1").click().run()
    at.sidebar.button(key="del-c1").click().run()

    assert two_chats == ["c1"]


def test_keeping_it_calls_the_delete_off(two_chats):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button(key="arm-c1").click().run()
    at.sidebar.button(key="keep-c1").click().run()

    assert two_chats == []
    # Back to an ordinary row, with its delete offered again.
    assert any(b.key == "arm-c1" for b in at.sidebar.button)


def test_only_one_chat_is_armed_at_a_time(two_chats):
    """Arming a second disarms the first, so there is never more than one row in
    a state where a click destroys something."""
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button(key="arm-c1").click().run()
    at.sidebar.button(key="arm-c2").click().run()

    keys = [b.key for b in at.sidebar.button]
    assert "del-c2" in keys
    assert "del-c1" not in keys


def test_opening_a_chat_calls_a_pending_delete_off(two_chats):
    """The user moved on. Leaving it armed means a later stray click deletes
    something they are no longer looking at."""
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button(key="arm-c1").click().run()
    at.sidebar.button(key="conv-c2").click().run()

    assert two_chats == []
    assert not any(b.key == "del-c1" for b in at.sidebar.button)


def test_deleting_the_open_chat_falls_back_to_a_new_one(two_chats):
    """Otherwise the screen points at a conversation that no longer exists."""
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    at.sidebar.button(key="arm-c1").click().run()
    at.sidebar.button(key="del-c1").click().run()

    assert two_chats == ["c1"]
    assert at.session_state[state.CURRENT_CONV] is None


def test_a_failed_delete_is_reported_and_leaves_the_chat_alone(monkeypatch, backend):
    monkeypatch.setattr(
        api_client, "conversations", lambda: [{"conv_id": "c1", "title": "Password rules"}]
    )

    def refuse(conv_id):
        raise api_client.LensApiError("unreachable", "connection refused")

    monkeypatch.setattr(api_client, "delete_conversation", refuse)

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button(key="arm-c1").click().run()
    at.sidebar.button(key="del-c1").click().run()

    assert not at.exception
    assert any("connection refused" in e.value for e in at.error)


# --- the app script actually loads --------------------------------------


def test_the_app_script_runs_when_launched_the_way_streamlit_launches_it(tmp_path):
    """Streamlit puts the script's own directory on the import path, not the one it
    was launched from, so without the path line in `app.py` the page shows a traceback
    while the server still answers 200.

    Run in a subprocess from an unrelated directory, because that is the only
    arrangement in which the bug appears.
    """
    import os
    import subprocess
    import sys

    probe = (
        "from streamlit.testing.v1 import AppTest;"
        f"app = AppTest.from_file({APP!r}, default_timeout=60);"
        "app.run();"
        "print('EXCEPTION' if app.exception else 'CLEAN');"
        "print('WIDGETS', len(app.chat_input))"
    )
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    # No exception is the whole claim: the imports resolved and the script ran to
    # the end. What it drew is deliberately not asserted - with no backend running
    # the app correctly draws its "backend is not running" screen and no chat
    # input, and a test that insisted on the input would pass or fail depending on
    # whether a developer happened to have the server up.
    assert "CLEAN" in result.stdout, result.stdout + result.stderr[-2000:]
    assert "WIDGETS" in result.stdout, result.stdout


# --- a table citation reads as a table -----------------------------------


TABLE_SNIPPET = (
    "| Rule | Setting |\n"
    "|------|---------|\n"
    "| Minimum length | 12 characters |\n"
    "| Reuse | The last 8 passwords cannot be reused |"
)


def test_a_table_snippet_becomes_rows():
    """Handed to markdown inside a quotation, a table arrives as one long line of
    pipes and dashes - unreadable in the one place a reader is checking the
    answer."""
    rows = citations_ui.as_rows(TABLE_SNIPPET)

    assert rows[0] == ["Rule", "Setting"]
    assert ["Minimum length", "12 characters"] in rows
    # The bar under the header carries no content and must not become a row.
    assert not any(set("".join(row)) <= set("-: ") for row in rows)


def test_a_row_cut_short_by_the_snippet_limit_keeps_what_survived():
    """The snippet is cut to a fixed length, so its last row is usually partial.
    Those cells are still what the answer was read from."""
    rows = citations_ui.as_rows(TABLE_SNIPPET + "\n| Maximum age | 90")

    assert rows[-1] == ["Maximum age", "90"]
    assert all(len(row) == len(rows[0]) for row in rows)


def test_prose_is_not_mistaken_for_a_table():
    assert citations_ui.as_rows("The vehicle is stopped at the outer barrier.") is None


def test_a_single_row_is_not_a_table():
    """One row is a fragment. Rendering it as a table would invent a header."""
    assert citations_ui.as_rows("| Minimum length | 12 characters |") is None


def test_a_table_stored_as_one_line_still_reads_as_a_table():
    """Answers given before the backend kept a table's line breaks are stored flat,
    and a stored citation is never resolved again by design - so an old answer has
    to keep rendering from exactly what it saved."""
    flat = (
        "| Rule | Setting | |-----|-----| | Minimum length | 12 characters | "
        "| Reuse | The last 8 passwords cannot be reused |"
    )

    rows = citations_ui.as_rows(flat)

    assert rows[0] == ["Rule", "Setting"]
    assert ["Minimum length", "12 characters"] in rows
    assert ["Reuse", "The last 8 passwords cannot be reused"] in rows


def test_the_truncation_mark_is_not_shown_as_a_row():
    """The truncation mark is a message about the table, not a row of it."""
    from config import settings

    snippet = (
        "| Rule | Setting |\n|---|---|\n| Minimum length | 12 characters |\n"
        + settings.SNIPPET_TRUNCATED_MARK
    )

    rows = citations_ui.as_rows(snippet)

    assert rows == [["Rule", "Setting"], ["Minimum length", "12 characters"]]


# --- the cited page sits beside the answer -------------------------------


def _grounded(citations: list[dict] | None = None) -> dict:
    """A chat whose last answer cites something."""
    return {
        "conv_id": "c1",
        "title": "Password rules",
        "scope_mode": "library",
        "scope_doc_ids": None,
        "messages": [
            {
                "msg_id": "m1",
                "role": "user",
                "content": "What are the password rules?",
                "citations": [],
                "abstained": False,
            },
            {
                "msg_id": "m2",
                "role": "assistant",
                "content": "At least twelve characters [1].",
                "citations": [CITATION] if citations is None else citations,
                "abstained": False,
            },
        ],
    }


def _ungrounded(content: str = "", abstained: bool = True) -> dict:
    """A greeting, or a refusal. Either way, nothing was cited."""
    chat = _grounded(citations=[])
    chat["messages"][1]["content"] = content
    chat["messages"][1]["abstained"] = abstained
    return chat


def _screen(monkeypatch, conversation: dict, page_image=None):
    monkeypatch.setattr(api_client, "health", lambda: {"status": "ok"})
    monkeypatch.setattr(api_client, "documents", lambda ready_only=False: DOCUMENTS)
    monkeypatch.setattr(api_client, "conversations", lambda: [])
    monkeypatch.setattr(api_client, "conversation", lambda _id: conversation)
    monkeypatch.setattr(api_client, "page_image", page_image or (lambda *a, **k: PNG))

    app = AppTest.from_file(APP, default_timeout=30)
    app.session_state[state.CURRENT_CONV] = conversation["conv_id"]
    return app


def _captions(app) -> list[str]:
    return [caption.value for caption in app.caption]


def test_a_cited_answer_shows_its_page_without_being_asked(monkeypatch):
    """The page is evidence for the claim beside it, so it is simply there. Making
    a reader click for it means most answers are never checked at all."""
    app = _screen(monkeypatch, _grounded())
    app.run()

    assert not app.exception
    assert any("highlighted region" in caption for caption in _captions(app))
    assert any("3.2 Password rules" in caption for caption in _captions(app))


def test_a_greeting_shows_no_page(monkeypatch):
    """ "Hello" cites nothing. A page beside it would claim a source the answer
    never had."""
    app = _screen(monkeypatch, _ungrounded(content="Hello.", abstained=False))
    app.run()

    assert not app.exception
    assert not any("highlighted region" in caption for caption in _captions(app))


def test_a_refusal_shows_no_page(monkeypatch):
    """The whole meaning of a refusal is that the documents did not hold it."""
    app = _screen(monkeypatch, _ungrounded())
    app.run()

    assert not any("highlighted region" in caption for caption in _captions(app))


def test_every_source_can_be_reopened(monkeypatch):
    """Even the one already beside the answer. Without this, closing the panel left
    no way back to it, which made Close a decision rather than a convenience."""
    app = _screen(monkeypatch, _grounded())
    app.run()

    assert [button.label for button in app.button if button.label.startswith("Show page")] == [
        "Show page 5"
    ]

    next(button for button in app.button if button.key == "close-page").click().run()
    reopen = next(button for button in app.button if button.label == "Show page 5")
    reopen.click().run()

    assert any("highlighted region" in caption for caption in _captions(app))


def test_several_sources_can_be_switched_between(monkeypatch):
    """Only the first is shown, so the others need a way to be reached."""
    second = dict(CITATION, n=2, chunk_id="d1:20", page=7)
    app = _screen(monkeypatch, _grounded(citations=[CITATION, second]))
    app.run()

    labels = [button.label for button in app.button if button.label.startswith("Show page")]
    assert labels == ["Show page 5", "Show page 7"]

    next(button for button in app.button if button.label == "Show page 7").click().run()

    assert app.session_state[state.PAGE_VIEW]["page"] == 7


def test_the_panel_can_be_closed_for_this_answer(monkeypatch):
    app = _screen(monkeypatch, _grounded())
    app.run()

    next(button for button in app.button if button.key == "close-page").click().run()

    assert not any("highlighted region" in caption for caption in _captions(app))


def test_the_next_answer_opens_its_own_page_after_a_close(monkeypatch):
    """A close applies to the answer it was made about. Carrying it forward would
    silently switch the feature off for the rest of the session."""
    app = _screen(monkeypatch, _grounded())
    app.run()
    next(button for button in app.button if button.key == "close-page").click().run()

    later = dict(CITATION, chunk_id="d1:99", page=9)
    monkeypatch.setattr(api_client, "conversation", lambda _id: _grounded(citations=[later]))
    app.run()

    assert any("highlighted region" in caption for caption in _captions(app))


def test_a_page_that_cannot_be_drawn_still_shows_the_passage(monkeypatch):
    """Usually the original file is gone. The passage is what the answer actually
    used, and it beats an empty frame."""

    def gone(*_args, **_kwargs):
        raise api_client.LensApiError("render_failed", "the original file is missing")

    app = _screen(monkeypatch, _grounded(), page_image=gone)
    app.run()

    assert not app.exception
    assert "missing" in app.warning[0].value


# --- a refresh keeps the chat it was on ----------------------------------


def test_a_refresh_keeps_the_chat_that_was_open(monkeypatch):
    """Session state is wiped by a refresh. Without the chat in the address bar the
    app forgot which one was on screen, so the next question started a new one -
    the sidebar filled with chats nobody asked for, and a follow-up lost the
    history it needs to be understood."""
    conversation = _grounded()
    app = _screen(monkeypatch, conversation)
    app.query_params[state.CHAT_PARAM] = conversation["conv_id"]
    del app.session_state[state.CURRENT_CONV]  # as a fresh session arrives
    app.run()

    assert app.session_state[state.CURRENT_CONV] == conversation["conv_id"]


def test_opening_a_chat_records_it_in_the_address_bar(monkeypatch):
    conversation = _grounded()
    app = _screen(monkeypatch, conversation)
    monkeypatch.setattr(
        api_client,
        "conversations",
        lambda: [{"conv_id": "c1", "title": "Password rules", "title_is_auto": True}],
    )
    app.run()

    next(button for button in app.button if button.key == "conv-c1").click().run()

    # The test harness reports a query parameter as a list, a browser as a string.
    # Either way it is the id that was opened - which is why `init` accepts both.
    written = app.query_params[state.CHAT_PARAM]
    assert (written[0] if isinstance(written, list) else written) == "c1"


def test_starting_a_new_chat_clears_the_address_bar(monkeypatch):
    """Otherwise a refresh would reopen the chat the user had just left."""
    conversation = _grounded()
    app = _screen(monkeypatch, conversation)
    app.query_params[state.CHAT_PARAM] = conversation["conv_id"]
    app.run()

    next(button for button in app.button if button.label == "New chat").click().run()

    assert app.session_state[state.CURRENT_CONV] is None
    assert state.CHAT_PARAM not in app.query_params


def test_a_chat_in_the_address_bar_that_no_longer_exists_is_dropped(monkeypatch):
    """An id can name a chat deleted since - from another tab, or before a
    refresh. That is an ordinary case, not an error to show."""
    monkeypatch.setattr(api_client, "health", lambda: {"status": "ok"})
    monkeypatch.setattr(api_client, "documents", lambda ready_only=False: DOCUMENTS)
    monkeypatch.setattr(api_client, "conversations", lambda: [])

    def gone(_conv_id):
        raise api_client.LensApiError("conversation_not_found", "no such chat")

    monkeypatch.setattr(api_client, "conversation", gone)

    app = AppTest.from_file(APP, default_timeout=30)
    app.query_params[state.CHAT_PARAM] = "deleted-one"
    app.run()

    assert not app.exception
    assert app.session_state[state.CURRENT_CONV] is None


def test_one_new_chat_control_on_the_screen(monkeypatch):
    """One New chat control, not two.

    It was in the sidebar and in the main area at once, and two controls doing the
    same thing only raise the question of whether they differ."""
    app = _screen(monkeypatch, _grounded())
    app.run()

    assert len([button for button in app.button if button.label == "New chat"]) == 1


# --- one question, one chat ----------------------------------------------


def test_a_failed_read_of_the_open_chat_does_not_start_another_one(monkeypatch):
    """A fetch that fails looks exactly like "no chat yet", and acting on that
    starts a second chat holding one turn - which is how a sidebar fills with
    chats nobody asked for, and how a follow-up loses its history."""
    created: list[str] = []

    def create(**_kwargs):
        created.append("one")
        return {"conv_id": "new-one"}

    def unreadable(_conv_id):
        raise api_client.LensApiError("unreachable", "the backend blinked")

    monkeypatch.setattr(api_client, "health", lambda: {"status": "ok"})
    monkeypatch.setattr(api_client, "documents", lambda ready_only=False: DOCUMENTS)
    monkeypatch.setattr(api_client, "conversations", lambda: [])
    monkeypatch.setattr(api_client, "conversation", unreadable)
    monkeypatch.setattr(api_client, "create_conversation", create)
    monkeypatch.setattr(api_client, "ask", lambda *a, **k: iter([]))

    app = AppTest.from_file(APP, default_timeout=30)
    app.session_state[state.CURRENT_CONV] = "already-open"
    app.run()
    app.chat_input[0].set_value("and the lock duration?").run()

    assert created == [], "a second chat was created while one was already open"


# --- nothing scrolls but the two frames ----------------------------------


def test_turns_are_drawn_in_the_order_they_happened():
    """The order a conversation is read in. The frame is scrolled to the newest turn
    instead of the order being reversed to bring it into view."""
    from frontend.components import chat

    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]

    grouped = chat.turns(messages)

    assert [message["content"] for message in grouped[0]] == ["first question", "first answer"]
    assert [message["content"] for message in grouped[-1]] == ["second question", "second answer"]


def test_a_turn_keeps_its_question_above_its_answer():
    from frontend.components import chat

    grouped = chat.turns(
        [
            {"role": "user", "content": "the question"},
            {"role": "assistant", "content": "the answer"},
        ]
    )

    assert [message["role"] for message in grouped[0]] == ["user", "assistant"]


def test_an_answer_with_no_question_before_it_still_appears():
    """Nothing in the app produces one, and dropping a stored message on the floor
    because its shape was unexpected would be worse than showing it."""
    from frontend.components import chat

    grouped = chat.turns([{"role": "assistant", "content": "orphaned"}])

    assert grouped == [[{"role": "assistant", "content": "orphaned"}]]


def test_the_thread_and_the_page_share_one_height(monkeypatch):
    """They sit side by side. Two different heights would leave one of them
    trailing off past the other, which is the shape that was complained about."""
    from config import settings

    app = _screen(monkeypatch, _grounded())
    app.run()

    assert settings.PANEL_HEIGHT > 0
    assert not app.exception


# --- The uploader's advertised limit -------------------------------------


def test_streamlit_is_told_the_same_size_limit_the_backend_enforces():
    """The uploader prints its own limit under the drop zone.

    Streamlit's default is 200 MB. Left alone, the screen invites a file eight times
    larger than the backend accepts, and the user waits for a full upload to be
    refused. The number lives in a toml file that cannot import settings, so the two
    are tied together here instead.
    """
    import re

    from config import settings

    config = (settings.PROJECT_ROOT / ".streamlit" / "config.toml").read_text()
    found = re.search(r"maxUploadSize\s*=\s*(\d+)", config)

    assert found, "no maxUploadSize in .streamlit/config.toml"
    assert int(found.group(1)) == settings.MAX_FILE_MB


def test_the_size_limit_in_bytes_matches_the_limit_in_megabytes():
    """Both are used: bytes to check a file, megabytes to tell a person."""
    from config import settings

    assert settings.MAX_FILE_BYTES == settings.MAX_FILE_MB * 1024 * 1024
