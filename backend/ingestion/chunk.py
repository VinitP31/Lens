"""What a chunk is.

Deliberately its own module, importing nothing but configuration.

`Chunk` is the boundary object between ingestion and retrieval: the chunker
produces them, the embedder and the vector store consume them. Defining it inside
the chunker would mean anything touching a chunk also imports the extractor, and
through it Docling and PyTorch - so answering a question would load a document
layout model it never uses, and the two libraries' threading runtimes conflict at
import time on some platforms.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """One unit of retrieval.

    `text` is the body alone, because that is what a user is shown when they open
    the source of an answer, and a header they did not write would read as if the
    document contained it.

    `embed_text` is what actually gets embedded: the context header followed by
    the body. The header is cheap and does real work, because a question asked in
    a heading's words then matches a chunk whose body uses different words.
    """

    index: int
    text: str
    page: int
    section_path: str
    element_type: str
    token_count: int
    # Boxes for the highlight. All of them are on `page`, because a chunk never
    # holds text from more than one page: a citation resolves to one page and one
    # set of coordinates on it, so text from the next page filed under this chunk
    # would open a page that text is not on.
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    # Set once the document has a name. Held on the chunk rather than rebuilt
    # later so that what was embedded is exactly what is stored.
    context_header: str = ""

    @property
    def embed_text(self) -> str:
        return f"{self.context_header}\n{self.text}" if self.context_header else self.text
