"""Building the prompt.

Assembly lives here so it can be read and tested without a network.

The fixed order matters twice: providers discount a repeated prefix only if it is
byte-identical, and the model is told what it may use before it is shown what it
has. Reversed, the passages read as an invitation to answer however possible.

The model never writes a document name or page number, and abstains with an exact
string rather than prose.
"""

from collections.abc import Sequence

from backend.storage.vector_store import Hit
from config import settings

# The four instruction blocks, in the order the specification fixes: role and
# scope, the grounding rule, the citation contract, then the abstention contract.
#
# Written as one constant rather than assembled from parts. The prefix has to be
# byte-identical between calls for a provider to discount it, and a prompt built
# by joining fragments invites a later change that reorders them invisibly.
SYSTEM_PROMPT = f"""You answer questions about a set of documents that have been \
provided to you as numbered passages.

Answer only from the numbered passages below. Never use knowledge from outside \
them, and never fill a gap with something you know to be generally true. If the \
passages disagree with what you believe, the passages win.

Cite every passage you used by its number, like [1] or [2], placed where the \
information from it appears. Never write a document name, a section name or a \
page number yourself - the numbers are resolved to real sources after you \
reply, and a name you write cannot be checked.

If the passages do not contain the answer, reply with exactly this and nothing \
else:

{settings.ABSTENTION_MARKER}

Do this even when the passages are clearly about the right subject. A passage \
discussing a topic is not the same as a passage stating the fact that was asked \
for. Do not infer it, do not estimate it, and do not assemble it from parts that \
each say something else. Answering "I am not sure, but" is not an option: either \
the passages state it, or you reply with the exact words above."""


def label(hit: Hit, document_name: str) -> str:
    """The source line shown above a passage: id, document, section, page.

    The model is given this so it can tell the passages apart and see that they
    come from different places, which is what makes a citation meaningful rather
    than decorative. It is never asked to repeat any of it.

    Built from the same separator and page prefix as a chunk's context header, so
    the model sees one format everywhere instead of two that nearly match.
    """
    parts = [document_name]
    if hit.section_path:
        parts.append(hit.section_path)
    parts.append(f"{settings.CONTEXT_PAGE_PREFIX}{hit.page}")
    return f"(id: {hit.chunk_id} | {settings.CONTEXT_SEPARATOR.join(parts)})"


def passages(hits: Sequence[Hit], names: dict[str, str]) -> str:
    """The retrieved passages, numbered from 1.

    Numbering starts at 1 because that is what the model is asked to cite and
    what a reader sees. The offset between this and a list index is the whole
    reason citation resolution is written once, in one place.

    `names` maps a document id to its display name. Passed in rather than looked
    up here, so this module needs no database and stays testable on its own.
    """
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(f"[{number}] {label(hit, names.get(hit.doc_id, hit.doc_id))}\n{hit.text}")
    return "\n\n".join(blocks)


def assemble(question: str, hits: Sequence[Hit], names: dict[str, str]) -> list[dict[str, str]]:
    """The full message list for one answer.

    The passages and the question travel together in the user message, with the
    question last. A question placed before the evidence gets answered from
    memory of it; placed after, it is answered from evidence in front of it.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Passages:\n\n{passages(hits, names)}\n\nQuestion: {question}",
        },
    ]
