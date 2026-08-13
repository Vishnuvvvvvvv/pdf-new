"""
extract_to_markdown.py
======================
Standalone PDF → Markdown extractor.

Pipeline:
  1. Load the PDF with PyMuPDF.
  2. Run the PP-DocLayoutV2 neural model on every page to get bounding boxes
     and their reading-order indices (the hard part – already solved).
  3. For each page, process boxes *in reading-order*:
       title / paragraph_title / abstract / doc_title  →  Markdown heading / text
       text / list / index / ref_text                  →  Markdown paragraph
       table                                           →  Markdown table (via PyMuPDF find_tables)
       image / chart / figure                          →  saved JPG + ![](path) tag
       header / footer / number / vertical_text        →  skipped
  4. Write one combined <stem>.md file next to the PDF.

Usage:
    uv run extract_to_markdown.py "path/to/doc.pdf" \
        --weights "weights/PP-DocLayoutV2/models/Layout/PP-DocLayoutV2"

    # Single page (fast test):
    uv run extract_to_markdown.py "path/to/doc.pdf" \
        --weights "weights/PP-DocLayoutV2/models/Layout/PP-DocLayoutV2" \
        --page 0
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

# ── dependency guard ────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    import pymupdf
    import numpy as np
    from PIL import Image
except ImportError as exc:
    print(f"\n[CRITICAL ERROR] Missing Dependencies!\nDetails: {exc}")
    print("\nInstall with:\n    uv pip install pymupdf numpy Pillow torch torchvision transformers")
    sys.exit(1)

from pp_doclayoutv2_standalone import PPDocLayoutV2LayoutModel

# ── constants ────────────────────────────────────────────────────────────────
# Labels that map to Markdown headings (H1 → H3)
HEADING_LABELS = {"doc_title", "title", "paragraph_title"}

# Labels that produce a plain paragraph of text
TEXT_LABELS = {"text", "list", "abstract", "index", "ref_text"}

# Labels for tables
TABLE_LABELS = {"table"}

# Labels for visual content (images / charts)
IMAGE_LABELS = {"image", "chart", "figure"}

# Labels to silently skip (headers, footers, page numbers, watermarks …)
SKIP_LABELS = {
    "header", "footer", "number", "formula_number",
    "seal", "vertical_text", "header_image", "footer_image",
}

# Render resolution for layout prediction  (higher = more accurate, slower)
LAYOUT_DPI = 144

# Render resolution for image crops that go into the Markdown
IMAGE_DPI = 150

# Heading depth per label
HEADING_DEPTH: Dict[str, int] = {
    "doc_title":       1,
    "title":           2,
    "paragraph_title": 3,
}


# ── helpers ──────────────────────────────────────────────────────────────────

def render_page_to_pil(page: fitz.Page, dpi: int = LAYOUT_DPI) -> Image.Image:
    """Render a PDF page to a PIL Image at the given DPI."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def scale_bbox(bbox: List[float], page: fitz.Page, img_w: int, img_h: int):
    """
    Convert bounding box from image-pixel space back to PDF user-space points.
    The model receives a PIL image rendered at LAYOUT_DPI; we must invert that
    scaling to get the correct fitz.Rect for text / table extraction.
    """
    pw = page.rect.width   # points
    ph = page.rect.height  # points
    sx = pw / img_w
    sy = ph / img_h
    x0, y0, x1, y1 = bbox
    return fitz.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)


def extract_text_from_rect(page: fitz.Page, rect: fitz.Rect) -> str:
    """Extract and clean plain text from a PDF page rectangle."""
    text = page.get_text("text", clip=rect).strip()
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_table_from_rect(page: fitz.Page, rect: fitz.Rect) -> Optional[str]:
    """
    Try to extract a table as Markdown using PyMuPDF's native find_tables().
    Falls back to plain text extraction if no table is detected.
    """
    try:
        tabs = page.find_tables(clip=rect)
        if tabs and tabs.tables:
            tab = tabs.tables[0]
            rows = tab.extract()
            if not rows:
                return None
            # Build a Markdown table
            lines = []
            header = rows[0]
            lines.append("| " + " | ".join(str(c or "").replace("\n", " ") for c in header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for row in rows[1:]:
                lines.append("| " + " | ".join(str(c or "").replace("\n", " ") for c in row) + " |")
            return "\n".join(lines)
    except Exception:
        pass
    return None


def crop_image_from_page(
    page: fitz.Page,
    rect: fitz.Rect,
    output_path: Path,
    dpi: int = IMAGE_DPI,
) -> str:
    """Crop the given rectangle from a page, save as JPEG, return relative path."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(output_path))
    return str(output_path)


# ── visual debug helpers ──────────────────────────────────────────────────────

COLORS = {
    "title": (1, 0, 0),              # Red
    "paragraph_title": (1, 0, 0),    # Red
    "text": (0, 0, 1),               # Blue
    "table": (0, 1, 0),              # Green
    "image": (1, 0.5, 0),            # Orange
    "chart": (1, 0.5, 0),            # Orange
    "figure": (1, 0.5, 0),           # Orange
    "equation": (0.5, 0, 0.5),       # Purple
    "interline_equation": (0.5, 0, 0.5),
    "header": (0.5, 0.5, 0.5),       # Gray
    "footer": (0.5, 0.5, 0.5),       # Gray
}

def draw_layout_on_page(page: fitz.Page, boxes: List[Dict], img_w: int, img_h: int):
    """Draw bounding boxes and reading order indices directly onto the PDF page."""
    for box in boxes:
        label = box.get("label", "text")
        idx = box.get("index", "?")
        rect = scale_bbox(box.get("bbox", [0,0,0,0]), page, img_w, img_h)
        color = COLORS.get(label, (0.2, 0.2, 0.2)) # default dark gray
        
        # Draw rectangle
        page.draw_rect(rect, color=color, width=1.5)
        
        # Draw label & index tag
        tag_text = f"[{idx}] {label}"
        text_rect = fitz.Rect(rect.x0, max(0, rect.y0 - 10), rect.x1, rect.y0)
        page.draw_rect(text_rect, color=color, fill=color)
        page.insert_text(
            (rect.x0 + 2, rect.y0 - 2),
            tag_text,
            fontsize=8,
            color=(1, 1, 1)
        )


def boxes_to_markdown(
    page: fitz.Page,
    boxes: List[Dict],
    img_w: int,
    img_h: int,
    images_dir: Path,
    image_counter: List[int],  # mutable int wrapper so we can share across pages
) -> str:
    """Convert one page's layout boxes (already sorted by reading order) to Markdown."""
    md_parts: List[str] = []

    # Sort by reading-order index (ascending); ties broken by top-to-bottom position
    ordered = sorted(
        boxes,
        key=lambda b: (b.get("index", 10_000), b.get("bbox", [0, 0, 0, 0])[1]),
    )

    for box in ordered:
        label = box.get("label", "text")
        bbox = box.get("bbox", [0, 0, 0, 0])
        rect = scale_bbox(bbox, page, img_w, img_h)

        # ── SKIP ─────────────────────────────────────────────────────
        if label in SKIP_LABELS:
            continue

        # ── HEADINGS ─────────────────────────────────────────────────
        if label in HEADING_LABELS:
            text = extract_text_from_rect(page, rect).replace("\n", " ").strip()
            if text:
                depth = HEADING_DEPTH.get(label, 2)
                md_parts.append(f"\n{'#' * depth} {text}\n")

        # ── TEXT / LIST / ABSTRACT / etc. ─────────────────────────────
        elif label in TEXT_LABELS:
            text = extract_text_from_rect(page, rect)
            if text:
                md_parts.append(f"\n{text}\n")

        # ── TABLE ────────────────────────────────────────────────────
        elif label in TABLE_LABELS:
            md_table = extract_table_from_rect(page, rect)
            if md_table:
                md_parts.append(f"\n{md_table}\n")
            else:
                # Fallback: plain text
                text = extract_text_from_rect(page, rect)
                if text:
                    md_parts.append(f"\n{text}\n")

        # ── IMAGE / CHART / FIGURE ───────────────────────────────────
        elif label in IMAGE_LABELS:
            img_idx = image_counter[0]
            image_counter[0] += 1
            img_filename = f"img_{img_idx:04d}.jpg"
            img_path = images_dir / img_filename
            try:
                saved_path = crop_image_from_page(page, rect, img_path)
                rel_path = os.path.relpath(saved_path, images_dir.parent)
                md_parts.append(f"\n![{label} {img_idx}]({rel_path})\n")
            except Exception as e:
                md_parts.append(f"\n<!-- image extraction failed: {e} -->\n")

        # ── FORMULA (interline_equation) ─────────────────────────────
        elif label in {"interline_equation", "equation"}:
            text = extract_text_from_rect(page, rect)
            if text:
                md_parts.append(f"\n$$\n{text}\n$$\n")

        # ── CAPTION ──────────────────────────────────────────────────
        elif "caption" in label:
            text = extract_text_from_rect(page, rect).replace("\n", " ").strip()
            if text:
                md_parts.append(f"\n*{text}*\n")

        # ── ANY OTHER unknown label → plain text ─────────────────────
        else:
            text = extract_text_from_rect(page, rect)
            if text:
                md_parts.append(f"\n{text}\n")

    return "".join(md_parts)


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_extraction(
    pdf_path: str,
    weights_dir: str,
    page_num: Optional[int] = None,
) -> None:
    """Full extraction pipeline: PDF → layout detection → Markdown."""

    pdf_path = os.path.abspath(pdf_path)
    weights_dir = os.path.abspath(weights_dir)

    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF not found: {pdf_path}")
        sys.exit(1)
    if not os.path.exists(weights_dir):
        print(f"[ERROR] Weights directory not found: {weights_dir}")
        sys.exit(1)

    output_dir = Path(__file__).parent / "output_markdown"
    output_dir.mkdir(exist_ok=True)
    images_dir = output_dir / "images"

    stem = Path(pdf_path).stem
    output_md_path = output_dir / f"{stem}.md"
    output_pdf_path = output_dir / f"{stem}_debug_layout.pdf"

    print("=" * 60)
    print("  Standalone PDF -> Markdown Extractor")
    print("  Model: PP-DocLayoutV2 (Layout + Reading Order)")
    print("=" * 60)

    # Step 1: Load model
    print(f"\n[1/3] Loading PP-DocLayoutV2 model from:\n      {weights_dir}")
    layout_model = PPDocLayoutV2LayoutModel(weight=weights_dir, device="cpu")
    print("      Model loaded successfully.")

    # Step 2: Open PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if page_num is not None:
        if page_num < 0 or page_num >= total_pages:
            print(f"[ERROR] Page {page_num} out of range (document has {total_pages} pages).")
            sys.exit(1)
        pages_to_process = [page_num]
        mode_label = f"Single Page: {page_num + 1} / {total_pages}"
    else:
        pages_to_process = list(range(total_pages))
        mode_label = f"Entire Document: {total_pages} pages"

    print(f"\n[2/3] Processing {mode_label}...")

    all_markdown_parts: List[str] = []
    image_counter = [0]  # shared mutable counter across pages

    for idx, pg_num in enumerate(pages_to_process):
        page = doc[pg_num]
        print(f"  -> Page {pg_num + 1} / {total_pages}: rendering + layout inference...")

        # Render to PIL for model input
        pil_image = render_page_to_pil(page, dpi=LAYOUT_DPI)
        img_w, img_h = pil_image.size

        # Run neural layout + reading order prediction
        boxes = layout_model.predict(pil_image)

        if not boxes:
            print(f"     (no layout detected on page {pg_num + 1}, skipping)")
            continue

        # Draw on PDF for debugging
        draw_layout_on_page(page, boxes, img_w, img_h)

        # Convert to Markdown
        page_md = boxes_to_markdown(page, boxes, img_w, img_h, images_dir, image_counter)

        if page_md.strip():
            all_markdown_parts.append(f"\n\n<!-- Page {pg_num + 1} -->\n")
            all_markdown_parts.append(page_md)

    # Save debug PDF
    doc.save(str(output_pdf_path))
    print(f"\n[3/3] Saving Outputs...")
    print(f"  Debug PDF saved to: {output_pdf_path}")
    doc.close()

    final_md = "".join(all_markdown_parts).strip()
    output_md_path.write_text(final_md, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  SUCCESS!")
    print(f"  Markdown saved to: {output_md_path}")
    if image_counter[0] > 0:
        print(f"  Images saved to:   {images_dir}  ({image_counter[0]} file(s))")
    print(f"{'=' * 60}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract PDF content to Markdown using PP-DocLayoutV2 neural layout detection."
    )
    parser.add_argument("pdf", help="Path to the PDF file to process.")
    parser.add_argument(
        "--weights",
        required=True,
        help='Path to the PP-DocLayoutV2 weights directory. '
             'Example: "weights/PP-DocLayoutV2/models/Layout/PP-DocLayoutV2"',
    )
    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="0-indexed page number to process. If omitted, all pages are processed.",
    )
    args = parser.parse_args()
    run_extraction(args.pdf, args.weights, args.page)


if __name__ == "__main__":
    main()
