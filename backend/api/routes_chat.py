"""The chat endpoint, and the health check.

Where the query pipeline is assembled:

    condense -> analyze -> retrieve -> gate -> generate -> validate -> store

A greeting and a question about the app skip search entirely. Answers stream as
`token` events, then `citations` once generation has finished, then `done` with the
diagnostics; a refusal sends no `token` at all.
"""

import json
import logging
import time

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from backend.api.schemas import ChatDone, CitationOut, Diagnostics, Health, SendMessage
from backend.logging import trace
from backend.retrieval import analyzer, condenser, gate, generator, retriever
from backend.storage import conversations, registry, vector_store
from config import settings

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

GREETING_REPLY = "Hello. Ask me anything about the documents in your library."

NO_DOCUMENTS_REPLY = (
    "There are no documents in your library yet. Upload a PDF and I can answer questions about it."
)


def _citation_dicts(citations) -> list[dict]:
    """Citations as they are stored on the message and returned to the UI.

    Plain dictionaries, because this is what goes into the message row as JSON
    and is read back verbatim years later without being resolved again.
    """
    return [
        {
            "n": citation.number,
            "chunk_id": citation.chunk_id,
            "doc_id": citation.doc_id,
            "display_name": citation.document_name,
            "page": citation.page,
            "section_path": citation.section_path,
            "element_type": citation.element_type,
            "snippet": citation.snippet,
            "bboxes": [list(box) for box in citation.bboxes],
        }
        for citation in citations
    ]


def _meta_reply(db) -> str:
    """Answer a question about the app itself, from the registry.

    No search, because the answer is not in the documents - it is about them.
    """
    documents = registry.list_documents(db, ready_only=True)
    if not documents:
        return NO_DOCUMENTS_REPLY
    names = "\n".join(f"- {document.display_name}" for document in documents)
    plural = "document" if len(documents) == 1 else "documents"
    return f"There are {len(documents)} {plural} in your library:\n{names}"


def _scope_for(conversation, db) -> list[str] | None:
    """Which documents this chat searches, as ids, or None for the whole library.

    A subset is filtered against what is still present, because a document
    selected last week may have been deleted since. A subset that empties out
    entirely stays empty rather than silently widening to the whole library -
    the user chose those documents, and answering from others would be answering
    a different question.
    """
    if conversation.is_library_wide:
        return None
    live = {document.doc_id for document in registry.list_documents(db, ready_only=True)}
    return [doc_id for doc_id in (conversation.scope_doc_ids or []) if doc_id in live]


def _event(name: str, payload) -> dict:
    return {"event": name, "data": json.dumps(payload)}


@router.post("/conversations/{conv_id}/messages")
async def send(request: Request, conv_id: str, body: SendMessage):
    """Answer one turn, streaming the text and then the citations.

    The conversation is read before the stream opens so a missing chat fails as
    an ordinary error response rather than as a stream that ends immediately.
    """
    db = request.app.state.db
    store = request.app.state.store
    conversation = conversations.get(db, conv_id)
    message = body.message.strip()

    async def events():
        started = time.perf_counter()
        record = trace.QueryTrace(conv_id=conv_id, message=message)

        def elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        # 1. A very long message is reduced before it is searched with. What the
        #    user typed is still what is stored and shown.
        condensed = condenser.condense(message)
        record.condensed = condensed.text if condensed.was_condensed else None
        record.stage_ms["condense"] = elapsed()

        # 2. What kind of message is it, and what does it mean on its own.
        history = conversations.recent_turns(db, conv_id)
        analysis = analyzer.analyze(condensed.text, history=history)
        record.intent = analysis.intent
        record.analysis_degraded = analysis.degraded
        record.rewritten = analysis.standalone if analysis.was_rewritten else None
        record.stage_ms["analyze"] = elapsed() - record.stage_ms["condense"]

        scope = _scope_for(conversation, db)
        record.scope_mode = conversation.scope_mode
        record.scope_doc_ids = scope
        conversations.add_message(
            db,
            conv_id,
            conversations.ROLE_USER,
            message,
            intent=analysis.intent,
            scope_snapshot=scope,
        )

        diagnostics = Diagnostics(
            gate_threshold=settings.GATE_THRESHOLD,
            intent=analysis.intent,
            searched_as=analysis.standalone if analysis.standalone != message else None,
            condensed=condensed.was_condensed,
        )

        def finish(text: str, *, abstained: bool, reason: str | None = None, citations=()) -> dict:
            """Store the answer and build the final event.

            Storing happens here rather than in each branch so that every path -
            greeting, refusal, answer - records the same fields, and a later
            branch cannot forget one.
            """
            diagnostics.latency_ms = elapsed()
            record.total_ms = diagnostics.latency_ms
            trace.write_query(record)
            stored = conversations.add_message(
                db,
                conv_id,
                conversations.ROLE_ASSISTANT,
                text,
                citations=_citation_dicts(citations),
                scope_snapshot=scope,
                intent=analysis.intent,
                gate_passed=None if diagnostics.top_score is None else not abstained,
                top_score=diagnostics.top_score,
                latency_ms=diagnostics.latency_ms,
            )
            return _event(
                "done",
                ChatDone(
                    message_id=stored.msg_id,
                    answer=text,
                    abstained=abstained,
                    reason=reason,
                    citations=[CitationOut(**c) for c in _citation_dicts(citations)],
                    diagnostics=diagnostics,
                ).model_dump(),
            )

        # 3. Neither of these searches anything.
        if analysis.intent == conversations.INTENT_GREETING:
            yield finish(GREETING_REPLY, abstained=False)
            return
        if analysis.intent == conversations.INTENT_META:
            yield finish(_meta_reply(db), abstained=False)
            return

        # A real question names the chat, if it has not been named already. A
        # greeting never does: a chat called "hi" tells nobody anything.
        conversations.set_auto_title(db, conv_id, message)

        # 4. Search, then decide whether there is enough to answer from at all.
        #
        # Wrapped, because searching embeds the question and that is a network
        # call. Once the stream has opened there is no status code left to send,
        # so a failure here has to arrive as an error event rather than as a
        # half-written response - and it must never be stored as an answer.
        try:
            found = retriever.retrieve(db, store, analysis.standalone, doc_ids=scope)
        except Exception as error:  # noqa: BLE001 - reported, never stored
            log.exception("retrieval failed for conversation %s", conv_id)
            record.error = str(error)
            record.total_ms = elapsed()
            trace.write_query(record)
            yield _event(
                "error", {"code": getattr(error, "code", "retrieval_failed"), "message": str(error)}
            )
            return

        record.record_retrieval(found)
        record.stage_ms["retrieve"] = elapsed() - sum(record.stage_ms.values())

        decision = gate.evaluate(found)
        record.record_gate(decision)
        diagnostics.top_score = decision.top_similarity
        diagnostics.retrieved = found.candidates_fetched
        diagnostics.used = len(found.hits)

        if not decision.passed:
            # No model call. This is the whole point of the gate sitting here.
            yield finish("", abstained=True, reason=decision.reason)
            return

        # 5. Generate, streaming text as it arrives, then validate the citations.
        names = {document.doc_id: document.display_name for document in registry.list_documents(db)}
        answer = None
        try:
            for piece in generator.stream(analysis.standalone, found.hits, names):
                if isinstance(piece, generator.Answer):
                    answer = piece
                else:
                    yield _event("token", piece)
        except Exception as error:  # noqa: BLE001 - reported, never stored as an answer
            log.exception("generation failed for conversation %s", conv_id)
            record.error = str(error)
            record.total_ms = elapsed()
            trace.write_query(record)
            yield _event("error", {"code": "generation_failed", "message": str(error)})
            return

        record.stage_ms["generate"] = elapsed() - sum(record.stage_ms.values())
        if answer is not None:
            record.record_answer(answer)

        if answer is None or answer.abstained:
            reason = answer.reason if answer else generator.REASON_EMPTY_ANSWER
            yield finish("", abstained=True, reason=reason)
            return

        diagnostics.rejected_citations = len(answer.fabricated)
        diagnostics.partly_absent = answer.partly_absent
        citations = _citation_dicts(answer.citations)
        yield _event("citations", citations)
        yield finish(answer.text, abstained=False, citations=answer.citations)

    return EventSourceResponse(events())


@router.get("/health", response_model=Health)
async def health(request: Request) -> Health:
    """Enough to tell a working backend from one that will fail on first use."""
    db = request.app.state.db
    store = request.app.state.store

    matches = True
    try:
        registry.assert_embed_model(db)
    except Exception:  # noqa: BLE001 - reported as a field, not as a failure
        matches = False

    return Health(
        status="ok" if matches else "degraded",
        documents=len(registry.list_documents(db, ready_only=True)),
        chunks=vector_store.count(store),
        embedding_model=settings.EMBEDDING_MODEL,
        answer_model=settings.MODEL_ANSWER,
        embed_model_matches=matches,
        gate_threshold=settings.GATE_THRESHOLD,
    )
