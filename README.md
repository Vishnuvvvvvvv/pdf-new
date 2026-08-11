# Advanced PDF-to-Markdown Extraction Architecture

This document explains the technical architecture of our PDF-to-Markdown extraction pipeline. It is modeled after production-grade systems like **MinerU (Magic-PDF)** and **Docling**, designed to handle highly complex layouts (multi-column, spanning headers, nested tables) while outputting clean, structurally accurate Markdown.

---

## 1. The Core Problem: Why is PDF Extraction Hard?
PDFs do not store logical structure. They do not know what a "paragraph", a "column", or a "table" is. A PDF is essentially a digital canvas that only stores absolute X,Y coordinates for individual characters or lines of text. 

If you use a naive extractor (like standard PyPDF2), it simply reads the text layer from top-to-bottom, left-to-right. On a 2-column or 3-column document, this results in the left and right columns being blindly mashed together, completely destroying the reading order and mangling tables into vertical lists.

To solve this, a production-grade system must separate the process into two strict phases: **Layout Detection (Vision)** and **Content Extraction**.

---

## 2. The Extraction Flow (How Our Code Works)

Our pipeline (`pipeline.py`) processes documents sequentially through the following stages:

### Phase 1: Layout Detection & Sorting (The "Vision" Phase)
Before we extract any text, we must first figure out the geometric structure of the page. This is handled by `layout_detector.py`.

1. **Block Identification**:
   - **Native Method (`pymupdf` backend)**: For clean digital PDFs, we scan the native PDF grid and font metadata. We accurately locate images (`FIGURE`), grid lines (`TABLE`), and text clusters (`TEXT`/`TITLE`).
   - **AI Vision Method (`yolo` backend)**: For scanned documents, we convert the page to an image and run it through **RapidLayout** (an ONNX-based AI vision model). The AI acts like human eyes, visually drawing bounding boxes around tables, figures, and text paragraphs.
2. **Non-Maximum Suppression (NMS)**: AI models often draw overlapping boxes (e.g., a huge box for a column, and smaller boxes for the paragraphs inside it). We mathematically calculate the Intersection-over-Area (IoA) and delete redundant overlaps to prevent duplicate text extraction.
3. **XY-Cut Sorting**: We take all the bounding boxes and run an XY-Cut algorithm. This algorithm detects vertical "gutters" (white space between columns) and horizontal breaks, mathematically sorting the blocks into perfect human reading order (e.g., Column 1 top-to-bottom, then Column 2).

> **Is Layout Detection Necessary?**
> **Absolutely.** Without it, multi-column parsing is impossible. By identifying blocks first, we know exactly *where* a table is, allowing us to send that specific region to specialized table-extraction models instead of extracting it as raw text.

### Phase 2: Targeted Content Extraction
Once we have a sorted list of bounding boxes (each tagged with a label like `TEXT`, `TITLE`, or `TABLE`), we iterate through them and apply targeted extraction strategies:

- **For `TEXT` / `TITLE`**: We pass the exact coordinates to PyMuPDF, which clips the text layer and extracts the perfect text. We then apply Regex post-processing to fix PDF rendering quirks (like merging hanging bullet points and numbered lists).
- **For `TABLE`**: We extract the exact bounding box of the table. If it is a clean digital table, we parse the native grid into a Markdown table. If it is complex, borderless, or scanned, we pass the cropped image to **RapidTable AI**, which uses deep learning to visually reconstruct the merged cells and HTML structure.
- **For `FIGURE`**: We crop the image from the PDF at high resolution (3.0x scale) and save it to a local `media/` folder, injecting a standard Markdown image link `![figure](path.png)` in its place.

---

## 3. How Production-Grade Systems (MinerU) Work

Systems like MinerU utilize this exact same architecture but scale it massively using cloud infrastructure:

1. **Massive AI Checkpoints**: MinerU relies heavily on custom PyTorch models like `DocLayout-YOLO` and `TableMaster`. These are gigabytes in size and require powerful GPUs. Our pipeline swaps these out for lightweight, CPU-friendly **ONNX** equivalents (`RapidLayout`, `RapidTable`) that achieve the same result instantly on local machines.
2. **Classification First**: Before processing, production systems run a classifier to check if a PDF is "Scanned" or "Digital". 
   - *If Scanned*: They route it through heavy OCR and Vision AI pipelines.
   - *If Digital*: They route it through native PDF parsing because it is 100x faster and mathematically accurate (zero AI hallucinations). 
   - *Our Implementation*: We allow you to toggle this explicitly via `--backend pymupdf` (Digital) vs `--backend yolo` (Scanned/Complex).
3. **Formula Recognition**: Production systems include a dedicated model (like `LaTeX-OCR`) just for mathematical equations. When a `FORMULA` block is detected, it is cropped and sent to this model to generate raw LaTeX.

## 4. Summary: The "Why"
By keeping **Layout (Where things are)** entirely separated from **Extraction (What things say)**, the pipeline remains entirely modular. If an OCR model fails, it doesn't break the column sorting. If a Table model is updated, you just swap out the `extract_table` function. This separation of concerns is the secret to flawlessly parsing the world's most complex documents.
