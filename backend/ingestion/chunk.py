"""The `Chunk` type: one passage of a document, ready to embed and to cite.

In its own module, importing nothing but configuration. Were it defined in the chunker,
anything handling a chunk would import the extractor too, and with it Docling - which
must never load in the process that answers questions.
"""

from dataclasses import dataclass, field

# Element types Lens stores. Here rather than in the extractor because retrieval
# needs to tell a table from a sentence, and importing the extractor for that would
# pull Docling into the process that answers questions.
TYPE_TEXT = "text"
TYPE_TABLE = "table"
TYPE_FIGURE_CAPTION = "figure_caption"


@dataclass(frozen=True)
class Chunk:
    """One unit of retrieval.

    `text` is the body alone, because that is what a user is shown as the source of
    an answer. `embed_text` is what gets embedded - the context header then the body
    - so a question asked in a heading's words matches a body that uses different
    ones.
    """

    index: int
    text: str
    page: int
    section_path: str
    element_type: str
    token_count: int
    # Boxes for the highlight, all on `page`: a citation resolves to one page, so
    # text from the next page filed here would open a page it is not on.
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    # Set once the document has a name. Held on the chunk rather than rebuilt
    # later so that what was embedded is exactly what is stored.
    context_header: str = ""

    @property
    def embed_text(self) -> str:
        return f"{self.context_header}\n{self.text}" if self.context_header else self.text
