"""Build the two stress PDFs, which are checked in and never indexed.

    python scripts/make_stress_pdfs.py

`stress_scanned.pdf` has no text layer at all, so it exercises OCR and the
rejection that follows if OCR finds nothing either. `stress_two_column.pdf` sets
prose in two columns, which reads as nonsense row by row, so it exercises the
reading-order rule that leaves genuine columns alone.

Generated rather than downloaded so a test can assert what should come back out.
"""

import sys
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent.parent / "samples"

SCANNED_PAGES = [
    [
        "WAREHOUSE SAFETY NOTICE",
        "",
        "All personnel entering the cold store must wear",
        "insulated gloves and a high-visibility jacket.",
        "",
        "The maximum continuous working period inside",
        "the cold store is forty-five minutes, after which",
        "a fifteen minute warming break is mandatory.",
        "",
        "Report any door seal damage to the shift lead",
        "before the next inbound consignment arrives.",
    ],
    [
        "EQUIPMENT CHECK RECORD",
        "",
        "Forklift batteries are charged overnight and",
        "checked each morning before the first shift.",
        "",
        "A unit showing less than twenty per cent charge",
        "is taken out of service until fully charged.",
        "",
        "Charging bays must never be blocked by pallets.",
    ],
]

LEFT_COLUMN = [
    "Inbound receiving begins when a vehicle books in",
    "at the gate. The gate clerk records the vehicle",
    "registration, the carrier name and the seal number",
    "shown on the trailer door. A consignment arriving",
    "without an intact seal is held for inspection and",
    "the shift lead is notified before unloading starts.",
    "",
    "Unloading is scheduled against a dock appointment.",
    "Where a vehicle arrives outside its window the",
    "clerk offers the next free slot rather than turning",
    "the vehicle away, and the delay is recorded so that",
    "carrier performance can be reviewed each month.",
]

RIGHT_COLUMN = [
    "Quality disposition happens after the physical",
    "count is agreed. A sample is drawn according to",
    "the lot size table and inspected against the",
    "specification held for that product. Anything",
    "outside specification is quarantined in the hold",
    "area and never placed into pickable stock.",
    "",
    "Putaway follows disposition. The system proposes",
    "a location by product velocity, and the operator",
    "confirms the location by scanning it. A confirmed",
    "putaway closes the receipt and makes the stock",
    "available to picking on the next wave.",
]


def scanned() -> Path:
    """A PDF whose pages are images of text, with no text layer."""
    document = pymupdf.open()

    for lines in SCANNED_PAGES:
        # The text is drawn on a temporary page, rendered to a bitmap, and that
        # bitmap becomes the real page. What survives is only pixels.
        scratch = pymupdf.open()
        temporary = scratch.new_page(width=612, height=792)
        y = 96
        for line in lines:
            if line:
                size = 16 if line.isupper() and len(line) < 40 else 12
                temporary.insert_text((72, y), line, fontsize=size)
            y += 22
        # JPEG at a modest density: a checked-in fixture has to be small, and
        # 110 DPI is still comfortably readable by an OCR engine.
        image = temporary.get_pixmap(dpi=110).tobytes("jpeg", jpg_quality=70)
        scratch.close()

        page = document.new_page(width=612, height=792)
        page.insert_image(page.rect, stream=image)

    path = OUT / "stress_scanned.pdf"
    document.save(path)
    document.close()
    return path


def two_column() -> Path:
    """A PDF of prose set in two columns, which must not be read row by row."""
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)

    page.insert_text((72, 80), "INBOUND RECEIVING SUMMARY", fontsize=15)

    y = 120
    for line in LEFT_COLUMN:
        if line:
            page.insert_text((72, y), line, fontsize=10)
        y += 16

    y = 120
    for line in RIGHT_COLUMN:
        if line:
            page.insert_text((320, y), line, fontsize=10)
        y += 16

    path = OUT / "stress_two_column.pdf"
    document.save(path)
    document.close()
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (scanned(), two_column()):
        print(f"  wrote {path.relative_to(OUT.parent)}  ({path.stat().st_size:,} bytes)")
    print("\nThese are never indexed. Run them through scripts/profile_pdf.py by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
