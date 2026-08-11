import os
import fitz
from PIL import Image
from pathlib import Path
import easyocr
import numpy as np

from layout_detector import detect_layout, BlockLabel, Block, PageLayout

# Initialize EasyOCR reader (only loaded once in memory when needed)
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        print("  [Init] Loading EasyOCR model (this happens only once)...")
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader

# Initialize RapidTable engines for intelligent table reconstruction
_table_engine = None

def get_table_engine():
    global _table_engine
    if _table_engine is None:
        print("  [Init] Loading RapidTable AI model (this happens only once)...")
        from rapid_table import RapidTable
        _table_engine = RapidTable()
    return _table_engine


def extract_image(page: fitz.Page, bbox: tuple, output_dir: Path, img_name: str) -> str:
    """Crop the bounding box from the page and save as an image."""
    mat = fitz.Matrix(3.0, 3.0)  # High resolution for saved figures
    clip = fitz.Rect(bbox)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    
    img_path = output_dir / img_name
    pix.save(str(img_path))
    return f"![{img_name}]({img_name})"


def extract_text(page: fitz.Page, bbox: tuple) -> str:
    """
    MinerU approach: Try native extraction first. If it's empty or garbage (scanned),
    fallback to a Vision OCR model.
    """
    clip = fitz.Rect(bbox)
    # 1. Native Extraction
    raw_text = page.get_text("text", clip=clip).strip()
    
    # Post-process list formatting: PyMuPDF often extracts bullet/number characters on their own line.
    import re
    # Merge hanging bullets (•, , ○, ▪, -) with the following line
    raw_text = re.sub(r'^[•○▪\-]\s*\n\s*', '- ', raw_text, flags=re.MULTILINE)
    # Merge hanging numbered lists (1., 2., etc.) with the following line
    raw_text = re.sub(r'^(\d+\.)\s*\n\s*', r'\1 ', raw_text, flags=re.MULTILINE)
    # Remove Wingdings envelope icons extracted as solitary 'p'
    raw_text = re.sub(r'^p\s*\n(?=E-mail)', '', raw_text, flags=re.MULTILINE)
    
    # If native extraction yielded nothing, it's likely a scanned region or an image pretending to be text.
    if not raw_text:
        # 2. OCR Fallback
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=clip, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            reader = get_ocr_reader()
            ocr_result = reader.readtext(np.array(img), detail=0, paragraph=True)
            return "\n".join(ocr_result)
        except Exception as e:
            return f"*(Failed to OCR scanned region: {e})*"
        
    return raw_text


def extract_table(page: fitz.Page, bbox: tuple, output_dir: Path, table_id: str) -> str:
    """
    MinerU uses a dedicated Table Structure Recognition (TSR) model. 
    We use PyMuPDF's built-in TSR first (great for digital PDFs). 
    If it fails, we fall back to exporting the table as an image.
    """
    clip = fitz.Rect(bbox)
    
    # 1. Try PyMuPDF's native table recognition (fast, high fidelity for digital lines)
    tabs = page.find_tables(clip=clip)
    if tabs and len(tabs.tables) > 0:
        table = tabs.tables[0]
        # Convert to Markdown table
        md = table.to_markdown()
        if md and len(md) > 10:
            return md
            
    # 2. TSR Fallback (MinerU AI approach for complex/scanned tables)
    try:
        # Convert the cropped table region to an OpenCV format image
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # Convert RGB to BGR for RapidOCR/OpenCV
        img_bgr = img_array[:, :, ::-1]
        
        table_engine = get_table_engine()
        
        # RapidTable internally uses RapidOCR to extract text and SLANet to reconstruct the grid
        res = table_engine(img_bgr)
        if res and hasattr(res, 'html') and res.html:
            return f"\n{res.html}\n"
    except Exception as e:
        print(f"  [Warning] RapidTable failed: {e}")
        
    # 3. Absolute Fallback (Save as Image)
    img_name = f"table_{table_id}.png"
    img_md = extract_image(page, bbox, output_dir, img_name)
    return f"*[Table could not be parsed. Saved as image:]*\n{img_md}"


def run_pipeline(pdf_path: str, output_dir: str, backend: str = "pymupdf"):
    """
    End-to-End Extraction Pipeline (Following MinerU architecture):
    1. Layout Detection & Reading Order Sort
    2. Modular Content Extraction based on Label
    3. Markdown Assembly
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(output_dir)
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    markdown_lines = []
    
    print(f"\nStarting Extraction Pipeline: {pdf_path.name} (Backend: {backend})")
    
    # Step 1: Layout Detection (Yields sorted blocks page-by-page)
    # Run at scale=1.0 so bounding boxes perfectly match native PDF coordinates for text extraction
    for page_idx, layout in enumerate(detect_layout(pdf_path, backend=backend, scale=1.0)):
        page = doc[page_idx]
        markdown_lines.append(f"\n\n<!-- PAGE {page_idx + 1} -->\n")
        
        print(f"  Processing Page {page_idx + 1} ({len(layout.blocks)} blocks)...")
        
        # Step 2: Content Extraction
        for block in layout.blocks:
            if block.label in (BlockLabel.TITLE, BlockLabel.HEADER):
                text = extract_text(page, block.bbox)
                markdown_lines.append(f"## {text}\n")
                
            elif block.label in (BlockLabel.TEXT, BlockLabel.FOOTER):
                text = extract_text(page, block.bbox)
                markdown_lines.append(f"{text}\n")
                
            elif block.label == BlockLabel.FIGURE:
                img_name = f"page{page_idx+1}_fig{block.reading_order}.png"
                img_md = extract_image(page, block.bbox, media_dir, img_name)
                markdown_lines.append(f"\n{img_md}\n")
                
            elif block.label == BlockLabel.TABLE:
                table_id = f"p{page_idx+1}_b{block.reading_order}"
                table_md = extract_table(page, block.bbox, media_dir, table_id)
                markdown_lines.append(f"\n{table_md}\n")
                
    # Step 3: Save Markdown
    md_path = out_dir / f"{pdf_path.stem}.md"
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"\nPipeline Complete! Output saved to: {md_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Path to PDF")
    parser.add_argument("--out", default="pipeline_output", help="Output directory")
    parser.add_argument("--backend", default="pymupdf", choices=["pymupdf", "yolo", "auto"], 
                        help="Layout detection backend (default: pymupdf for fully offline use)")
    args = parser.parse_args()
    
    run_pipeline(args.pdf, args.out, args.backend)
