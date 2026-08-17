"""Shared test fixtures.

PDFs used by tests are generated here with PyMuPDF rather than committed as
binary files. A generated PDF has known content on a known page, so a test can
assert exactly where text ended up. A real-world PDF can only be asserted
against whatever it happens to contain, which tests nothing in particular.

The sample PDFs in samples/ are for reading by eye and for evaluation. They are
never used in unit tests.
"""

from pathlib import Path

import pymupdf
import pytest

from config import settings


@pytest.fixture(autouse=True)
def _traces_go_to_a_temporary_directory(tmp_path, monkeypatch):
    """Keep the trace log out of the real data directory.

    Every test that ingests or answers writes a trace, and without this the
    suite appends hundreds of lines of fixture noise to the file a person reads
    when they want to know why a real answer looked the way it did.
    """
    monkeypatch.setattr(settings, "TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(settings, "QUERY_TRACE_PATH", tmp_path / "traces" / "queries.jsonl")
    monkeypatch.setattr(settings, "DOCUMENT_TRACE_PATH", tmp_path / "traces" / "documents.jsonl")


# Text placed on each page of the simple fixture. Index 0 is page 1.
#
# Each body is deliberately several hundred characters long. A real document runs
# well over a thousand characters per page, and the conditional OCR decision is
# made on that density, so a sparse fixture would look like a scanned page.
PAGE_BODIES = [
    "Annual leave is granted at eighteen days per completed year of service. "
    "Entitlement accrues monthly and appears on the payslip issued at the end of "
    "each calendar month. Employees who join partway through a year receive a "
    "pro rata entitlement calculated from their confirmed start date.",
    "Requests for planned absence require approval fourteen days in advance. "
    "Approval is recorded by the reporting manager in the leave register, and an "
    "absence taken without a recorded approval is treated as unplanned. Urgent "
    "medical absence is exempt from the notice period described in this section.",
    "Unused leave lapses at the end of the calendar year and is not paid out. "
    "A maximum of five days may be carried forward with written approval from "
    "the head of department, and any carried balance must be consumed within the "
    "first quarter of the following year or it is forfeited.",
]

# One word that appears on exactly one page, used to check page attribution.
# Single words are used rather than phrases because wrapped text can break a
# phrase across lines, and a broken phrase would fail for the wrong reason.
PAGE_MARKERS = ["eighteen", "fourteen", "forfeited"]

RUNNING_HEADER = "Northbridge Logistics - Confidential"
RUNNING_FOOTER = "Page {page} of 3"

HEADING_TEXT = "Leave Entitlement"

# Used by the hierarchy fixture: a large chapter title over a smaller section
# heading, so heading depth can be checked.
CHAPTER_TITLE = "Expenses And Reimbursement"
SECTION_TITLE = "Claim Deadlines"

# A borderless grid. Each date sits level with the event it refers to, so the
# pairing only survives if the page is read row by row.
GRID_ROWS = [
    ("27 February 2026", "Issue request for proposal"),
    ("10 March 2026", "Supplier questions close"),
    ("31 March 2026", "Proposal submission review begins"),
    ("14 April 2026", "Award selection decision"),
]

# Two independent flows of prose, side by side. Reading these row by row would
# interleave them, so they must be left in column order.
LEFT_COLUMN = [
    "Every inbound vehicle is checked at the gate before the seal is broken.",
    "Drivers surrender keys at the gatehouse and collect them on release.",
    "Dock plates are inspected each morning and any defect is reported.",
]
RIGHT_COLUMN = [
    "Quality inspection samples one carton in every twenty received.",
    "Rejected stock is quarantined in the bonded cage pending a decision.",
    "Discrepancy reports are raised the same day the shortage is found.",
]


def _write_pdf(path: Path, build) -> Path:
    doc = pymupdf.open()
    build(doc)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(scope="session")
def simple_pdf(tmp_path_factory) -> Path:
    """Three pages of plain prose, each with a running header and footer.

    Page N contains PAGE_BODIES[N - 1] and nothing else of substance, so page
    attribution can be checked exactly.
    """

    def build(doc):
        for index, body in enumerate(PAGE_BODIES, start=1):
            page = doc.new_page()
            page.insert_text((72, 40), RUNNING_HEADER, fontsize=8)
            if index == 1:
                page.insert_text((72, 100), HEADING_TEXT, fontsize=20)
            # A textbox wraps the body, so it stays on the page instead of
            # running off the right edge as one very long line.
            page.insert_textbox(pymupdf.Rect(72, 140, 520, 400), body, fontsize=11)
            page.insert_text((72, 780), RUNNING_FOOTER.format(page=index), fontsize=8)

    return _write_pdf(tmp_path_factory.mktemp("pdfs") / "simple.pdf", build)


@pytest.fixture(scope="session")
def table_pdf(tmp_path_factory) -> Path:
    """One page holding a small grid of text laid out as a table."""

    def build(doc):
        page = doc.new_page()
        page.insert_text((72, 80), "Expense Approval Limits", fontsize=16)
        rows = [
            ("Role", "Limit"),
            ("Team lead", "5000"),
            ("Manager", "25000"),
            ("Director", "100000"),
        ]
        top = 140
        for row_index, (left, right) in enumerate(rows):
            y = top + row_index * 24
            page.insert_text((80, y), left, fontsize=11)
            page.insert_text((300, y), right, fontsize=11)
            page.draw_line(pymupdf.Point(72, y + 6), pymupdf.Point(420, y + 6))
        page.draw_line(pymupdf.Point(72, top - 18), pymupdf.Point(420, top - 18))
        page.draw_line(pymupdf.Point(72, top - 18), pymupdf.Point(72, top + len(rows) * 24 - 18))
        page.draw_line(pymupdf.Point(420, top - 18), pymupdf.Point(420, top + len(rows) * 24 - 18))
        page.draw_line(pymupdf.Point(280, top - 18), pymupdf.Point(280, top + len(rows) * 24 - 18))

    return _write_pdf(tmp_path_factory.mktemp("pdfs") / "table.pdf", build)


@pytest.fixture(scope="session")
def hierarchy_pdf(tmp_path_factory) -> Path:
    """One page with a large chapter title above a smaller section heading.

    Docling reports level 1 for every heading it finds, so depth has to come from
    type size. This fixture is the smallest case that proves it: the body text
    must end up beneath both headings, in the right order.
    """

    def build(doc):
        page = doc.new_page()
        # Three distinct type sizes: chapter, section, body. A heading set at the
        # same size as body text is not detectable as a heading by any generic
        # rule, so the fixture gives each level its own size, as real documents do.
        page.insert_text((72, 90), CHAPTER_TITLE, fontsize=24)
        page.insert_text((72, 150), SECTION_TITLE, fontsize=16)
        page.insert_textbox(
            pymupdf.Rect(72, 175, 520, 420),
            "Reimbursement claims must be submitted within thirty days of the "
            "expense being incurred. Claims submitted after that window require "
            "written approval from the head of department, and approval is "
            "recorded against the claim in the finance system before payment.",
            fontsize=11,
        )

    return _write_pdf(tmp_path_factory.mktemp("pdfs") / "hierarchy.pdf", build)


@pytest.fixture(scope="session")
def grid_pdf(tmp_path_factory) -> Path:
    """A borderless grid: a date on the left, the event it refers to on the right.

    No ruling lines, so layout analysis does not see a table and each cell
    becomes its own element. Read row by row the pairing survives; read any other
    way a date can be attached to the wrong event.
    """

    def build(doc):
        page = doc.new_page()
        page.insert_text((72, 80), "Calendar Of Events", fontsize=18)
        top = 140
        for index, (date, event) in enumerate(GRID_ROWS):
            y = top + index * 40
            page.insert_text((80, y), date, fontsize=11)
            page.insert_text((300, y), event, fontsize=11)

    return _write_pdf(tmp_path_factory.mktemp("pdfs") / "grid.pdf", build)


@pytest.fixture(scope="session")
def two_column_pdf(tmp_path_factory) -> Path:
    """A two-column article, the case where reading row by row would be wrong.

    Each column is a separate flow of prose. Reading across the page would
    interleave them into nonsense, so this proves the re-ordering leaves genuine
    columns alone.
    """

    def build(doc):
        page = doc.new_page()
        page.insert_text((72, 70), "Warehouse Safety Review", fontsize=18)
        left = pymupdf.Rect(72, 100, 290, 700)
        right = pymupdf.Rect(320, 100, 540, 700)
        page.insert_textbox(left, "\n\n".join(LEFT_COLUMN), fontsize=10)
        page.insert_textbox(right, "\n\n".join(RIGHT_COLUMN), fontsize=10)

    return _write_pdf(tmp_path_factory.mktemp("pdfs") / "twocolumn.pdf", build)


@pytest.fixture(scope="session")
def imageonly_pdf(tmp_path_factory) -> Path:
    """Two pages with no text layer at all, standing in for a scanned document."""

    def build(doc):
        for _ in range(2):
            page = doc.new_page()
            page.draw_rect(pymupdf.Rect(72, 72, 400, 300), fill=(0.85, 0.85, 0.85))

    return _write_pdf(tmp_path_factory.mktemp("pdfs") / "imageonly.pdf", build)


@pytest.fixture(scope="session")
def corrupt_pdf(tmp_path_factory) -> Path:
    """A file that is not a PDF at all, despite the extension."""
    path = tmp_path_factory.mktemp("pdfs") / "corrupt.pdf"
    path.write_bytes(b"this is definitely not a PDF")
    return path
