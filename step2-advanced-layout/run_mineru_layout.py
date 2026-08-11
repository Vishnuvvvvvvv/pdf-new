import os
import sys
import argparse

def check_dependencies():
    """Check if the PyTorch/Transformers dependencies are installed."""
    try:
        import torch
        import transformers
        import numpy
        import fitz  # PyMuPDF
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        print("\n[CRITICAL ERROR] Missing Dependencies!")
        print(f"Details: {e}")
        print("\nTo run the standalone PP-DocLayoutV2 model, you MUST install:")
        print("    pip install torch torchvision transformers numpy Pillow pymupdf")
        sys.exit(1)

def run_advanced_layout(pdf_path: str, page_number: int = None, weights_dir: str = None):
    """
    Run the standalone PP-DocLayoutV2 Neural Layout Model on a PDF.
    If page_number is None, processes the entire PDF.
    """
    check_dependencies()
    import fitz
    from PIL import Image
    
    # Import our standalone layout model
    try:
        from pp_doclayoutv2_standalone import PPDocLayoutV2LayoutModel
    except ImportError as e:
        print(f"Failed to import standalone model: {e}")
        sys.exit(1)
    
    if not weights_dir or not os.path.exists(weights_dir):
        print(f"\n[CRITICAL ERROR] You must provide the path to the downloaded PP-DocLayoutV2 weights!")
        print("Because this is a standalone folder, we cannot use MinerU's auto-downloader.")
        print("Please pass: --weights \"C:\\path\\to\\weights\"")
        sys.exit(1)

    print("=" * 60)
    print("Starting Standalone Layout Detection (PP-DocLayoutV2)")
    print("=" * 60)

    # 1. Initialize the PyTorch Model
    print(f"\n[1/3] Loading PyTorch Model from {weights_dir} (Device: CPU)...")
    layout_model = PPDocLayoutV2LayoutModel(weight=weights_dir, device="cpu")
    
    # 2. Process the PDF
    doc = fitz.open(pdf_path)
    
    # Determine pages to process
    if page_number is not None:
        if page_number < 0 or page_number >= len(doc):
            print(f"Page number {page_number} is out of range (0 to {len(doc)-1}).")
            sys.exit(1)
        pages_to_process = [page_number]
        print(f"\n[2/3] Processing Single Page: {page_number + 1} / {len(doc)}...")
    else:
        pages_to_process = list(range(len(doc)))
        print(f"\n[2/3] Processing Entire Document: {len(doc)} Pages...")
        
    annotated_images = []
    
    for idx, p_num in enumerate(pages_to_process):
        print(f"  -> Inferring layout for page {p_num + 1}...")
        page = doc[p_num]
        pix = page.get_pixmap(dpi=150)
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Predict layout and reading order
        results = layout_model.predict(pil_image)
        
        # Visualize the output natively
        annotated_image = layout_model.visualize(pil_image, results)
        annotated_images.append(annotated_image)
        
    # 3. Save as Multi-page PDF
    print(f"\n[3/3] Exporting to PDF...")
    if page_number is not None:
        out_filename = f"marked_layout_page_{page_number + 1}.pdf"
    else:
        out_filename = f"marked_layout_full_document.pdf"
        
    out_path = os.path.join(os.path.dirname(__file__), out_filename)
    
    # Save the PIL Images as a PDF (supports multiple pages)
    if len(annotated_images) == 1:
        annotated_images[0].save(out_path, "PDF", resolution=150.0)
    else:
        annotated_images[0].save(
            out_path, 
            "PDF", 
            resolution=150.0,
            save_all=True, 
            append_images=annotated_images[1:]
        )
    
    print(f"\nSuccess! Annotated PDF saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Test of PP-DocLayoutV2")
    parser.add_argument("pdf", help="Path to PDF file to analyze")
    parser.add_argument("--page", type=int, default=None, help="0-indexed page number to test. If omitted, processes ALL pages.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the downloaded PP-DocLayoutV2 PyTorch weights folder")
    args = parser.parse_args()
    
    run_advanced_layout(args.pdf, args.page, args.weights)
