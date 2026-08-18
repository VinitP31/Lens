"""The only part of the frontend that talks to the backend.

Everything the UI needs comes through here over HTTP, so no screen can reach into
the database and the two halves can be run and restarted separately.

Errors arrive as a code and a message: callers switch on `LensApiError.code`, never
on its text. Answers arrive as a stream of events - text, then the validated
citations once generation has finished, then the diagnostics - so `ask` yields the
text as it comes and returns the rest at the end, which is the shape
`st.write_stream` wants.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx

from config import settings

# The backend is a separate process on the same machine. Read from settings so
# there is no literal here and no second place to change it.
BASE_URL = settings.API_BASE_URL

# Long enough for an answer, which waits on a model. Uploads get their own,
# larger, because validation reads the whole file.
TIMEOUT = httpx.Timeout(settings.API_TIMEOUT_SECONDS, connect=5.0)
UPLOAD_TIMEOUT = httpx.Timeout(settings.API_UPLOAD_TIMEOUT_SECONDS, connect=5.0)


class LensApiError(Exception):
    """A failure the backend named.

    `code` is what the UI decides with. `message` is what it shows.
    """

    def __init__(self, code: str, message: str, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class Answer:
    """One finished turn, as the screen needs it."""

    message_id: str
    text: str
    abstained: bool
    reason: str | None = None
    citations: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _client(timeout: httpx.Timeout = TIMEOUT) -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=timeout)


def _raise(response: httpx.Response) -> None:
    """Turn a failed response into a typed error carrying the backend's code."""
    if response.is_success:
        return
    try:
        body = response.json()
        code = body.get("code", "unknown_error")
        message = body.get("message", response.text)
    except ValueError:
        # Not JSON: a proxy error page, or the backend died mid-response.
        code = "unreachable" if response.status_code >= 500 else "unknown_error"
        message = f"the backend returned {response.status_code}"
    raise LensApiError(code, message, response.status_code)


def _get(path: str, **params):
    try:
        with _client() as client:
            response = client.get(path, params=params or None)
    except httpx.RequestError as error:
        raise LensApiError("unreachable", f"could not reach the backend: {error}") from error
    _raise(response)
    return response.json()


# --- documents -----------------------------------------------------------


def health() -> dict:
    """Whether the backend is up and its stores agree.

    Called on first paint. A backend that will not answer anything should be
    reported once, plainly, rather than as a failure on every later action.
    """
    return _get("/health")


def documents(ready_only: bool = False) -> list[dict]:
    return _get("/documents", ready_only=ready_only)


def upload(name: str, data: bytes) -> dict:
    """Send a PDF. Returns the document and job ids, or raises.

    Validation happens before the response, so a rejection - too large,
    encrypted, corrupt, already present - arrives here immediately rather than
    later through the status endpoint.
    """
    try:
        with _client(UPLOAD_TIMEOUT) as client:
            response = client.post("/documents", files={"file": (name, data, "application/pdf")})
    except httpx.RequestError as error:
        raise LensApiError("unreachable", f"could not reach the backend: {error}") from error
    _raise(response)
    return response.json()


def status(doc_id: str) -> dict:
    """How far an upload has got. Polled while indexing runs."""
    return _get(f"/documents/{doc_id}/status")


def page_image(doc_id: str, page: int, chunk_id: str | None = None) -> bytes:
    """A PNG of one page, with the cited region highlighted.

    Returns the bytes rather than a URL so the image travels through the same
    client as everything else - the screen never builds a backend URL of its own,
    which is what keeps this module the only thing that knows where the backend
    is.
    """
    try:
        with _client() as client:
            response = client.get(
                f"/documents/{doc_id}/pages/{page}",
                params={"chunk_id": chunk_id} if chunk_id else None,
            )
    except httpx.RequestError as error:
        raise LensApiError("unreachable", f"could not reach the backend: {error}") from error
    _raise(response)
    return response.content


def delete_document(doc_id: str) -> None:
    try:
        with _client() as client:
            response = client.delete(f"/documents/{doc_id}")
    except httpx.RequestError as error:
        raise LensApiError("unreachable", f"could not reach the backend: {error}") from error
    _raise(response)


# --- conversations -------------------------------------------------------


def conversations() -> list[dict]:
    return _get("/conversations")


def conversation(conv_id: str) -> dict:
    """One chat with its messages and the scope it was left searching."""
    return _get(f"/conversations/{conv_id}")


def create_conversation(
    scope_mode: str = "library", scope_doc_ids: list[str] | None = None
) -> dict:
    return _post_json("/conversations", {"scope_mode": scope_mode, "scope_doc_ids": scope_doc_ids})


def update_conversation(
    conv_id: str,
    title: str | None = None,
    scope_mode: str | None = None,
    scope_doc_ids: list[str] | None = None,
) -> dict:
    return _patch_json(
        f"/conversations/{conv_id}",
        {"title": title, "scope_mode": scope_mode, "scope_doc_ids": scope_doc_ids},
    )


def delete_conversation(conv_id: str) -> None:
    try:
        with _client() as client:
            response = client.delete(f"/conversations/{conv_id}")
    except httpx.RequestError as error:
        raise LensApiError("unreachable", f"could not reach the backend: {error}") from error
    _raise(response)


def _post_json(path: str, payload: dict) -> dict:
    try:
        with _client() as client:
            response = client.post(path, json=payload)
    except httpx.RequestError as error:
        raise LensApiError("unreachable", f"could not reach the backend: {error}") from error
    _raise(response)
    return response.json()


def _patch_json(path: str, payload: dict) -> dict:
    try:
        with _client() as client:
            response = client.patch(path, json=payload)
    except httpx.RequestError as error:
        raise LensApiError("unreachable", f"could not reach the backend: {error}") from error
    _raise(response)
    return response.json()


# --- asking a question ---------------------------------------------------


def ask(conv_id: str, message: str) -> Iterator[str | Answer]:
    """Send a turn. Yields the answer text in pieces, then one `Answer`.

    The mixed yield type is deliberate: `st.write_stream` consumes the strings as
    they arrive, and the final `Answer` carries the citations and diagnostics,
    which cannot exist until generation has finished.

    A refusal yields no text at all, only the `Answer`. That lets the screen
    render it as its own calm state rather than as an answer that happens to say
    no.
    """
    import json

    try:
        with (
            _client() as client,
            client.stream(
                "POST", f"/conversations/{conv_id}/messages", json={"message": message}
            ) as response,
        ):
            if not response.is_success:
                response.read()
                _raise(response)

            event = None
            citations: list[dict] = []

            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                    continue
                if not line.startswith("data:"):
                    continue

                payload = json.loads(line.split(":", 1)[1].strip())

                if event == "token":
                    yield payload
                elif event == "citations":
                    citations = payload
                elif event == "error":
                    raise LensApiError(
                        payload.get("code", "generation_failed"),
                        payload.get("message", "the answer could not be produced"),
                    )
                elif event == "done":
                    yield Answer(
                        message_id=payload["message_id"],
                        text=payload["answer"],
                        abstained=payload["abstained"],
                        reason=payload.get("reason"),
                        citations=payload.get("citations") or citations,
                        diagnostics=payload.get("diagnostics") or {},
                    )
    except httpx.RequestError as error:
        raise LensApiError("unreachable", f"could not reach the backend: {error}") from error
