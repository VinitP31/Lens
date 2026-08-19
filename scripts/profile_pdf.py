"""Report what the pipeline finds in a PDF.

    python scripts/profile_pdf.py samples/*.pdf
    python scripts/profile_pdf.py samples/*.pdf --dump

Pages, text density, whether OCR would trigger, headings, tables, figures, what was
dropped as furniture, and whether every element carries a box.

`--dump` writes the extracted text by page to data/profiles/, to read beside the real
PDF and check reading order.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.errors import LensError  # noqa: E402
from backend.ingestion import extractor  # noqa: E402
from backend.ingestion.extractor import ExtractedDocument  # noqa: E402
from config import settings  # noqa: E402


def _report(path: Path, doc: ExtractedDocument) -> None:
    types = Counter(element.element_type for element in doc.elements)
    with_bbox = sum(1 for element in doc.elements if element.bboxes)
    total = len(doc.elements)
    coverage = f"{with_bbox / total:.1%}" if total else "n/a"
    pages_seen = sorted({element.page for element in doc.elements})

    print(f"\n{path.name}")
    print("-" * len(path.name))
    print(f"  pages                {doc.page_count}")
    print(f"  extraction time      {doc.seconds:.1f}s  ({doc.seconds / doc.page_count:.2f}s/page)")
    print(f"  elements             {total}")
    print(f"    text               {types[extractor.TYPE_TEXT]}")
    print(f"    tables             {types[extractor.TYPE_TABLE]}")
    print(f"    figure captions    {types[extractor.TYPE_FIGURE_CAPTION]}")
    print(f"  headings found       {doc.heading_count}")
    print(f"  figures on page      {doc.picture_count}")
    print(f"  dropped as furniture {doc.dropped_count}   (headers, footers, contents)")
    print(f"  characters           {doc.char_count}  ({doc.chars_per_page}/page)")
    print(
        f"  OCR would trigger    {doc.needs_ocr}"
        f"   (threshold {settings.OCR_TRIGGER_CHARS_PER_PAGE}/page)"
    )
    print(f"  bbox coverage        {coverage}  ({with_bbox}/{total})")

    if doc.contents_pages:
        print(f"  contents pages       {sorted(doc.contents_pages)}  excluded from indexing")

    # An empty page is either excluded on purpose, genuinely blank, or one the
    # pipeline lost. Saying which is the difference between a report and a riddle.
    empty = [page for page in range(1, doc.page_count + 1) if page not in pages_seen]
    excluded = [page for page in empty if page in doc.contents_pages]
    blank = [page for page in empty if page not in doc.contents_pages]
    if excluded:
        print(f"  pages excluded       {excluded}  (contents)")
    if blank:
        print(f"  pages with no text   {blank}  (blank, or image-only)")


def _dump(path: Path, doc: ExtractedDocument) -> Path:
    """Write the extracted text grouped by page, for reading by eye."""
    settings.ensure_dirs()
    out = settings.PROFILE_DIR / f"{path.stem}.txt"

    lines: list[str] = [f"{path.name} - {doc.page_count} pages, {len(doc.elements)} elements", ""]
    for page in range(1, doc.page_count + 1):
        on_page = [element for element in doc.elements if element.page == page]
        lines.append("=" * 70)
        note = ""
        if page in doc.contents_pages:
            note = "  - CONTENTS PAGE, excluded from indexing"
        lines.append(f"PAGE {page}  ({len(on_page)} elements){note}")
        lines.append("=" * 70)
        if not on_page:
            if page in doc.contents_pages:
                lines.append(
                    "(this is a table of contents. It is deliberately not indexed: it holds "
                    "the vocabulary of every topic and the answer to none, and a citation to "
                    "it would point at a list of headings rather than at content.)"
                )
            else:
                lines.append("(no text on this page - blank, or an image with no text layer)")
        for element in on_page:
            label = element.element_type
            if element.section_path:
                label = f"{label} | {element.section_path}"
            lines.append(f"[{label}]")
            lines.append(element.text)
            lines.append("")
        lines.append("")

    out.write_text("\n".join(lines))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Report what the pipeline finds in a PDF.")
    parser.add_argument("pdfs", nargs="+", type=Path, help="one or more PDF files")
    parser.add_argument(
        "--dump",
        action="store_true",
        help="also write extracted text per page to data/profiles/",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="read the file the way ingestion would, applying OCR if the text layer is too thin",
    )
    args = parser.parse_args()

    failures = 0
    for path in args.pdfs:
        try:
            if args.ocr:
                # The same call ingestion makes, so the report shows what would
                # actually be indexed rather than only what the first pass found.
                from backend.ingestion import ocr

                doc, applied = ocr.read(path)
            else:
                doc, applied = extractor.extract(path), False
        except LensError as error:
            print(f"\n{path.name}\n  FAILED [{error.code}] {error.detail}")
            failures += 1
            continue

        _report(path, doc)
        if args.ocr:
            print(f"  OCR applied          {applied}")
        if args.dump:
            print(f"  text written to      {_dump(path, doc)}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
