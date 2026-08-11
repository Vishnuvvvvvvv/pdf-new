"""
Step 1 - Layout Detection & Reading-Order Visualisation
========================================================

This is the first and most foundational step of any PDF extraction pipeline.

What it does
------------
1. Renders each PDF page to an image.
2. Detects layout blocks (text, title, table, figure, formula, etc.)
   using one of two backends chosen automatically:

   Backend A - PyMuPDF native (ZERO network, ZERO ML, already installed)
   - Works perfectly for native-text PDFs (PDFs with embedded fonts/text).
   - Extracts blocks directly from the PDF's internal structure.
   - Instantaneous. No model download needed.

   Backend B - DocLayout-YOLO (ultralytics, ~33 MB one-time HuggingFace download)
   - Used automatically when a page is detected as scanned/image-only.
   - The same model family as MinerU's PP-DocLayoutV2.
   - Correct model path: hf://juliozhao/DocLayout-YOLO-DocStructBench/...

3. Sorts blocks into reading order using the XY-Cut algorithm.
4. Annotates the original PDF with:
   - Coloured bounding boxes per block type.
   - Reading-order sequence number on each block.
   - A legend in the top-right corner.
5. Saves the annotated PDF alongside the original.

Why this Column-Aware Sort?
----------------------------
MinerU's PP-DocLayoutV2 uses a dedicated transformer model to predict reading
order. Standard geometric algorithms (like XY-Cut) often fail on complex layouts 
where a full-width header overlaps the y-coordinates of the columns below it.

This script uses a robust Column-Aware heuristic:
  1. Finds vertical gutters by projecting only narrow, non-header blocks.
  2. Classifies every block as belonging to a specific column, or marks it
     as 'SPANNING' if it crosses a gutter.
  3. Uses spanning blocks as horizontal barriers to divide the page into sections.
  4. Reads each section column-by-column, left-to-right, top-to-bottom.

This perfectly handles double-column papers with centered headers and spanning 
abstracts without the brittle failure cases of recursive XY-Cut.

Why this step first?
--------------------
This mirrors MinerU's pipeline exactly:
  classify_pdf -> render_pages -> LAYOUT DETECTION -> (OCR | table | formula)

Every downstream step (OCR, table parsing, formula extraction) depends on
knowing WHERE each block is on the page and in what order to process them.

Usage
-----
    python layout_detector.py paper.pdf
    python layout_detector.py paper.pdf --output-dir ./results --scale 2.5
    python layout_detector.py paper.pdf --backend pymupdf   # force offline mode
    python layout_detector.py paper.pdf --backend yolo      # force YOLO
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Block taxonomy
# ---------------------------------------------------------------------------
class BlockLabel(str, Enum):
    TEXT    = "Text"
    TITLE   = "Title"
    TABLE   = "Table"
    FIGURE  = "Figure"
    FORMULA = "Formula"
    HEADER  = "Header"
    FOOTER  = "Footer"
    CAPTION = "Caption"
    LIST    = "List-item"
    OTHER   = "Other"


# Colours per label (RGB tuples)
_LABEL_COLOR: dict[str, tuple[int, int, int]] = {
    BlockLabel.TEXT:    (52,  168, 83),   # green
    BlockLabel.TITLE:   (66,  133, 244),  # blue
    BlockLabel.TABLE:   (251, 188, 4),    # yellow
    BlockLabel.FIGURE:  (234, 67,  53),   # red
    BlockLabel.FORMULA: (156, 39,  176),  # purple
    BlockLabel.HEADER:  (0,   172, 193),  # cyan
    BlockLabel.FOOTER:  (0,   172, 193),  # cyan
    BlockLabel.CAPTION: (255, 112, 67),   # orange
    BlockLabel.LIST:    (0,   150, 136),  # teal
    BlockLabel.OTHER:   (158, 158, 158),  # grey
}

# DocLayout-YOLO label -> BlockLabel
_YOLO_LABEL_MAP: dict[str, str] = {
    "Caption":        BlockLabel.CAPTION,
    "Footnote":       BlockLabel.FOOTER,
    "Formula":        BlockLabel.FORMULA,
    "List-item":      BlockLabel.LIST,
    "Page-footer":    BlockLabel.FOOTER,
    "Page-header":    BlockLabel.HEADER,
    "Picture":        BlockLabel.FIGURE,
    "Section-header": BlockLabel.TITLE,
    "Table":          BlockLabel.TABLE,
    "Text":           BlockLabel.TEXT,
    "Title":          BlockLabel.TITLE,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Block:
    """One detected layout block on a page."""
    label: str           # BlockLabel constant
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in image pixels
    score: float = 1.0
    reading_order: int = -1
    text_preview: str = ""   # first ~80 chars of text (for native blocks)


@dataclass
class PageLayout:
    page_idx: int
    image: Image.Image           # rendered PIL image (used for visualisation)
    page_size: tuple[int, int]   # (width_px, height_px) at render scale
    blocks: list[Block] = field(default_factory=list)
    backend_used: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _classify_page(fitz_page: fitz.Page) -> str:
    """
    Return 'text' if the page has embedded text, 'scanned' otherwise.
    Mirrors MinerU's pdf_classify logic.
    """
    text = fitz_page.get_text("text").strip()
    return "text" if len(text) > 30 else "scanned"


def _render_page(page: fitz.Page, scale: float = 2.0) -> Image.Image:
    """Render a PyMuPDF page to a PIL Image with memory fallbacks."""
    for current_scale in [scale, 1.0, 0.5]:
        try:
            mat = fitz.Matrix(current_scale, current_scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            # Try efficient memoryview approach first
            try:
                return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            except Exception:
                # Fallback to PNG encoding if buffer access fails
                from io import BytesIO
                png_data = pix.tobytes("png")
                return Image.open(BytesIO(png_data)).convert("RGB")
        except Exception as e:
            if current_scale == 0.5:
                print(f"  [Warning] Extremely low memory. Could not render page background: {e}")
                
    # If all rendering fails due to memory, return a blank white canvas so execution continues
    w = int(page.rect.width * scale)
    h = int(page.rect.height * scale)
    return Image.new("RGB", (w, h), "white")


# ---------------------------------------------------------------------------
# XY-Cut reading order algorithm
# ---------------------------------------------------------------------------

def _block_cx(b: Block) -> float:
    return (b.bbox[0] + b.bbox[2]) / 2.0


def _block_cy(b: Block) -> float:
    return (b.bbox[1] + b.bbox[3]) / 2.0





def _reading_order_sort(
    blocks: list[Block],
    page_width: int,
    page_height: int = 0,
) -> list[Block]:
    """
    Robust Column-Aware Reading Order.
    
    1. Detect main column gutters by projecting blocks (excluding headers/footers).
    2. Assign each block to a column, or mark it as 'SPANNING' if it crosses a gutter.
    3. Group blocks into horizontal 'sections' divided by SPANNING blocks.
    4. Within each section, read Column 1 top-to-bottom, then Column 2, etc.
    """
    if not blocks:
        return blocks

    pw = float(page_width) if page_width else max(b.bbox[2] for b in blocks)
    ph = float(page_height) if page_height else max(b.bbox[3] for b in blocks)

    # 1. Find gutters
    # Ignore headers and footers to avoid them masking columns
    valid_blocks = [b for b in blocks if b.label not in (BlockLabel.HEADER, BlockLabel.FOOTER)]
    
    # First try: Ignore blocks that look like centered titles or spanning authors
    # (i.e. they are in the top 30% of the page and cross the horizontal center).
    # This prevents them from masking the real gutters below them, while preserving
    # short columns that might sit at the top of a page.
    cx = pw / 2.0
    gutter_blocks = []
    for b in valid_blocks:
        is_top = b.bbox[1] < ph * 0.30
        crosses_center = b.bbox[0] < cx and b.bbox[2] > cx
        if not (is_top and crosses_center):
            gutter_blocks.append(b)
    
    # Fallback just in case
    if not gutter_blocks:
        gutter_blocks = valid_blocks

    # Project x-intervals of our selected blocks
    x_intervals = sorted((b.bbox[0], b.bbox[2]) for b in gutter_blocks)
    covered = x_intervals[0][0] if x_intervals else 0.0
    gaps = []
    for a, b_edge in x_intervals:
        if a > covered + 2.0:
            gaps.append((covered, a))
        covered = max(covered, b_edge)
    
    # Filter for significant gaps (>= 2% of page width)
    min_gutter = pw * 0.02
    gutters = [(start, end) for start, end in gaps if (end - start) >= min_gutter]

    # 2. Assign blocks to columns or SPANNING
    # A block is spanning if it intersects any gutter by more than 10% of its width
    def get_col(block: Block) -> int:
        bx1, _, bx2, _ = block.bbox
        bw = bx2 - bx1
        for i, (g_start, g_end) in enumerate(gutters):
            overlap = max(0.0, min(bx2, g_end) - max(bx1, g_start))
            if overlap > bw * 0.1 or (bx1 < g_start + 5 and bx2 > g_end - 5):
                return -1  # SPANNING
            if bx2 <= g_start + 5:
                return i
        return len(gutters)
        
    for b in blocks:
        b._col = get_col(b)

    # 3. Sort all blocks purely by Y first
    sorted_by_y = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    
    # 4. Group into sections separated by spanning blocks
    sections = []
    current_section = []
    
    for b in sorted_by_y:
        if b._col == -1:
            if current_section:
                sections.append(current_section)
                current_section = []
            sections.append([b])
        else:
            current_section.append(b)
            
    if current_section:
        sections.append(current_section)
        
    # 5. Assemble final reading order
    result = []
    for sec in sections:
        # If this section is just a spanning block, add it
        if len(sec) == 1 and sec[0]._col == -1:
            result.append(sec[0])
            continue
            
        # Otherwise, group by column and sort each column top-down
        cols = {}
        for b in sec:
            cols.setdefault(b._col, []).append(b)
            
        for c_idx in sorted(cols.keys()):
            col_blocks = cols[c_idx]
            # Ensure strict top-to-bottom within the column
            col_blocks.sort(key=lambda b: b.bbox[1])
            result.extend(col_blocks)
            
    # Clean up internal attr and set reading order
    for i, b in enumerate(result):
        b.reading_order = i + 1
        if hasattr(b, '_col'):
            del b._col
            
    return result


# ---------------------------------------------------------------------------
# Backend A - PyMuPDF native text-block extraction
# ---------------------------------------------------------------------------
def _detect_pymupdf(fitz_page: fitz.Page, scale: float) -> list[Block]:
    """
    Extract layout blocks directly from PDF internal structure.
    No ML, no network. Works perfectly for native-text PDFs.

    PyMuPDF's get_text("dict") returns blocks with:
      - type 0 = text block
      - type 1 = image block
    We also check page header/footer zones heuristically.
    """
    page_rect = fitz_page.rect
    page_h = page_rect.height

    raw = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks: list[Block] = []

    # 1. Detect Tables natively
    table_bboxes = []
    tabs = fitz_page.find_tables()
    if tabs and len(tabs.tables) > 0:
        for tab in tabs.tables:
            x0, y0, x1, y1 = tab.bbox
            sx0, sy0, sx1, sy1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
            blocks.append(Block(
                label=BlockLabel.TABLE,
                bbox=(sx0, sy0, sx1, sy1),
                score=1.0,
                text_preview="[table]"
            ))
            table_bboxes.append((sx0, sy0, sx1, sy1))

    # 2. Process text/image blocks
    for b in raw.get("blocks", []):
        btype = b.get("type", 0)
        
        if btype == 1:  # image block
            x0, y0, x1, y1 = b["bbox"]
            label = BlockLabel.FIGURE
            preview = "[image]"
        else:
            # Recalculate bbox from non-empty spans to avoid invisible space inflations
            lines = b.get("lines", [])
            valid_spans = [
                sp for ln in lines for sp in ln.get("spans", [])
                if sp.get("text", "").strip()
            ]
            if not valid_spans:
                continue
                
            x0 = min(sp["bbox"][0] for sp in valid_spans)
            y0 = min(sp["bbox"][1] for sp in valid_spans)
            x1 = max(sp["bbox"][2] for sp in valid_spans)
            y1 = max(sp["bbox"][3] for sp in valid_spans)

            # Gather text
            text = " ".join(sp.get("text", "") for sp in valid_spans).strip()
            preview = text[:80]

            # Heuristic label assignment
            # Header/footer zone: top or bottom 8% of page height
            if y0 < page_h * 0.08:
                label = BlockLabel.HEADER
            elif y1 > page_h * 0.92:
                label = BlockLabel.FOOTER
            else:
                # Font-size based title detection
                font_sizes = [sp.get("size", 0) for sp in valid_spans]
                avg_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                is_bold = any(
                    "Bold" in sp.get("font", "") or sp.get("flags", 0) & 16
                    for sp in valid_spans
                )
                word_count = len(text.split())

                if avg_size > 14 and word_count < 20:
                    label = BlockLabel.TITLE
                elif is_bold and word_count < 15:
                    label = BlockLabel.TITLE
                else:
                    label = BlockLabel.TEXT

        # Scale bbox to image coordinates
        sx0, sy0, sx1, sy1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
        bbox = (sx0, sy0, sx1, sy1)
        
        # Suppress text/images that are already inside a detected table
        is_in_table = False
        for tx0, ty0, tx1, ty1 in table_bboxes:
            ix0 = max(sx0, tx0)
            iy0 = max(sy0, ty0)
            ix1 = min(sx1, tx1)
            iy1 = min(sy1, ty1)
            inter_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if inter_area > 0:
                area = (sx1 - sx0) * (sy1 - sy0)
                if inter_area / area > 0.8:
                    is_in_table = True
                    break
        if is_in_table:
            continue

        blocks.append(Block(label=label, bbox=bbox, score=1.0, text_preview=preview))

    return blocks


# ---------------------------------------------------------------------------
# Backend B - DocLayout-YOLO via ultralytics
# ---------------------------------------------------------------------------
_YOLO_MODEL = None   # module-level singleton


def _load_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    from rapid_layout import RapidLayout
    print(f"  [AI] Loading RapidLayout model...")
    _YOLO_MODEL = RapidLayout(conf_thres=0.3)
    return _YOLO_MODEL


def _detect_yolo(pil_image: Image.Image, conf: float = 0.25) -> list[Block]:
    model = _load_yolo()
    
    # RapidLayout expects an OpenCV BGR image
    import numpy as np
    img_array = np.array(pil_image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = img_array[:, :, ::-1]
    else:
        img_bgr = img_array
        
    res = model(img_bgr)
    
    blocks = []
    
    # Mapping RapidLayout classes to our BlockLabel enum
    class_map = {
        "text": BlockLabel.TEXT,
        "title": BlockLabel.TITLE,
        "figure": BlockLabel.FIGURE,
        "figure_caption": BlockLabel.TEXT,
        "table": BlockLabel.TABLE,
        "table_caption": BlockLabel.TEXT,
        "header": BlockLabel.HEADER,
        "footer": BlockLabel.FOOTER,
        "equation": BlockLabel.FORMULA,
    }
    
    raw_blocks = []
    if res.boxes is not None:
        for box, label_str, score in zip(res.boxes, res.class_names, res.scores):
            if score < conf:
                continue
            x1, y1, x2, y2 = box
            label = class_map.get(label_str.lower(), BlockLabel.TEXT)
            raw_blocks.append((x1, y1, x2, y2, label, score))
            
    # Apply Non-Maximum Suppression (NMS) using Intersection over Area (IoA)
    # This removes smaller duplicate boxes that are completely enclosed, or larger
    # generic boxes that overlap with highly confident granular boxes.
    blocks = []
    raw_blocks.sort(key=lambda b: b[5], reverse=True)  # Sort by confidence score
    
    for i, b1 in enumerate(raw_blocks):
        keep = True
        for j in range(i):
            b2 = raw_blocks[j]
            # Calculate Intersection
            xA = max(b1[0], b2[0])
            yA = max(b1[1], b2[1])
            xB = min(b1[2], b2[2])
            yB = min(b1[3], b2[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            if interArea > 0:
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                # If the smaller box is >80% enclosed in the larger one, suppress the lower confidence one
                ioa = interArea / min(area1, area2)
                if ioa > 0.8:
                    keep = False
                    break
        if keep:
            blocks.append(Block(bbox=(b1[0], b1[1], b1[2], b1[3]), label=b1[4]))
            
    return blocks


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------
def detect_layout(
    pdf_path: str,
    scale: float = 2.0,
    backend: str = "auto",   # "auto" | "pymupdf" | "yolo"
    yolo_conf: float = 0.25,
) -> Iterator[PageLayout]:
    """
    Yield one PageLayout per page, with XY-Cut reading-order-sorted blocks.

    backend="auto"    -> pymupdf for text pages, yolo for scanned pages.
    backend="pymupdf" -> always use pymupdf (offline, zero ML).
    backend="yolo"    -> always use YOLO (requires ultralytics + download).
    """
    doc = fitz.open(pdf_path)
    try:
        for page_idx in range(len(doc)):
            fitz_page = doc[page_idx]
            pil_image = _render_page(fitz_page, scale)
            w, h = pil_image.size

            page_type = _classify_page(fitz_page)

            # Choose backend
            if backend == "pymupdf" or (backend == "auto" and page_type == "text"):
                used = "pymupdf"
                raw_blocks = _detect_pymupdf(fitz_page, scale)
            else:
                used = "yolo"
                try:
                    raw_blocks = _detect_yolo(pil_image, conf=yolo_conf)
                except Exception as exc:
                    print(f"  [YOLO] failed ({exc}), falling back to pymupdf")
                    used = "pymupdf"
                    raw_blocks = _detect_pymupdf(fitz_page, scale)

            # PyMuPDF natively extracts text in near-perfect reading order.
            # XY-Cut is only needed for AI vision models that return random un-ordered boxes.
            if used == "pymupdf":
                ordered = raw_blocks
                # Ensure reading_order attribute is set
                for i, b in enumerate(ordered):
                    b.reading_order = i + 1
            else:
                ordered = _reading_order_sort(raw_blocks, page_width=w, page_height=h)

            yield PageLayout(
                page_idx=page_idx,
                image=pil_image,
                page_size=(w, h),
                blocks=ordered,
                backend_used=used,
            )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Visualiser - annotate image with boxes + reading order numbers
# ---------------------------------------------------------------------------
def _try_load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a decent font, fall back to default."""
    for font_name in [
        "arial.ttf", "Arial.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "FreeSans.ttf",
    ]:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def visualise_page(page_layout: PageLayout) -> Image.Image:
    """
    Return an annotated RGB image with coloured bboxes, reading-order numbers
    and a legend.
    """
    img = page_layout.image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    num_font  = _try_load_font(max(14, W // 80))
    leg_font  = _try_load_font(max(12, W // 100))

    seen_labels: dict[str, tuple] = {}

    for block in page_layout.blocks:
        color = _LABEL_COLOR.get(block.label, (158, 158, 158))
        x1, y1, x2, y2 = block.bbox

        # Semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=(*color, 35), outline=(*color, 230), width=2)

        # Reading-order badge
        badge_r = max(14, W // 70)
        cx, cy = x1 + badge_r + 2, y1 + badge_r + 2
        draw.ellipse([cx - badge_r, cy - badge_r, cx + badge_r, cy + badge_r],
                     fill=(*color, 220), outline=(255, 255, 255, 200), width=1)
        txt = str(block.reading_order)
        draw.text((cx, cy), txt, fill=(255, 255, 255), font=num_font, anchor="mm")

        # Label text
        label_y = y1 - badge_r * 2 - 4
        if label_y < 0:
            label_y = y2 + 4
        draw.text((x1 + 4, label_y), block.label, fill=color, font=leg_font)

        seen_labels[block.label] = color

    # Legend box (top-right)
    leg_x = W - max(140, W // 6)
    leg_y = 10
    leg_pad = 6
    leg_row_h = max(18, W // 60)
    leg_w = W - leg_x - 10
    leg_h = leg_row_h * len(seen_labels) + leg_pad * 2

    draw.rectangle([leg_x - leg_pad, leg_y, W - 10, leg_y + leg_h],
                   fill=(255, 255, 255, 200), outline=(100, 100, 100, 200), width=1)
    for i, (lbl, col) in enumerate(sorted(seen_labels.items())):
        ry = leg_y + leg_pad + i * leg_row_h
        draw.rectangle([leg_x, ry, leg_x + leg_row_h - 4, ry + leg_row_h - 4],
                       fill=(*col, 200))
        draw.text((leg_x + leg_row_h + 4, ry), lbl, fill=(40, 40, 40), font=leg_font)

    return img


# ---------------------------------------------------------------------------
# Main: build annotated PDF
# ---------------------------------------------------------------------------
def run(
    pdf_path: str,
    output_dir: str = ".",
    scale: float = 2.0,
    backend: str = "auto",
    yolo_conf: float = 0.25,
):
    """Run layout detection and write an annotated PDF."""
    pdf_path = str(Path(pdf_path).resolve())
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    stem = Path(pdf_path).stem
    out_pdf_path = output_dir_path / f"{stem}_layout.pdf"

    print(f"\nLayout Detection - Step 1")
    print(f"  Input  : {pdf_path}")
    print(f"  Output : {out_pdf_path}")
    print(f"  Scale  : {scale}x  Backend: {backend}  Sort: XY-Cut")
    print()

    annotated_pages: list[Image.Image] = []
    all_stats: list[dict] = []

    for page_layout in detect_layout(pdf_path, scale=scale, backend=backend, yolo_conf=yolo_conf):
        n = page_layout.page_idx + 1
        print(f"  Page {n:>3}  |  {len(page_layout.blocks):>2} blocks  |  backend: {page_layout.backend_used}")

        for b in page_layout.blocks:
            safe_preview = b.text_preview[:40].encode('ascii', errors='ignore').decode('ascii')
            preview = f'"{safe_preview}"' if b.text_preview else ""
            print(f"           {b.reading_order:>2}. [{b.label:<10}] {preview}")

        annotated = visualise_page(page_layout)
        annotated_pages.append(annotated)

        all_stats.append({
            "page": n,
            "blocks": len(page_layout.blocks),
            "backend": page_layout.backend_used,
            "block_types": {
                lbl: sum(1 for b in page_layout.blocks if b.label == lbl)
                for lbl in set(b.label for b in page_layout.blocks)
            },
        })

    if not annotated_pages:
        print("No pages found.")
        return

    # Save as PDF (first page saves, rest are appended)
    annotated_pages[0].save(
        str(out_pdf_path),
        save_all=True,
        append_images=annotated_pages[1:],
        resolution=72,
    )

    print(f"\nDone! Annotated PDF saved: {out_pdf_path}")
    print(f"  Total pages  : {len(annotated_pages)}")
    print(f"  Total blocks : {sum(s['blocks'] for s in all_stats)}")
    print(f"  Block types  :")
    combined: dict[str, int] = {}
    for s in all_stats:
        for lbl, cnt in s["block_types"].items():
            combined[lbl] = combined.get(lbl, 0) + cnt
    for lbl, cnt in sorted(combined.items(), key=lambda x: -x[1]):
        bar = "#" * cnt
        print(f"    {lbl:<12} {cnt:>3}  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1: Detect layout blocks and produce an annotated PDF with reading order."
    )
    parser.add_argument("pdf", help="Path to input PDF file")
    parser.add_argument("--output-dir", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--scale", type=float, default=2.0, help="Render scale (default: 2.0)")
    parser.add_argument(
        "--backend", choices=["auto", "pymupdf", "yolo"], default="auto",
        help=(
            "auto    = pymupdf for text PDFs, yolo for scanned (default)\n"
            "pymupdf = always use native PDF text extraction (zero ML, offline)\n"
            "yolo    = always use DocLayout-YOLO model"
        ),
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")

    args = parser.parse_args()
    run(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        scale=args.scale,
        backend=args.backend,
        yolo_conf=args.conf,
    )
