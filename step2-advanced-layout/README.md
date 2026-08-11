# Standalone Neural Layout & Reading Order (PP-DocLayoutV2)

This folder contains a 100% standalone extraction of MinerU's advanced Layout and Reading Order vision model (`PP-DocLayoutV2`). It takes a PDF, uses a PyTorch Transformer to predict the bounding boxes and reading order, and outputs a natively annotated, multi-page PDF without requiring the rest of the MinerU codebase.

## Prerequisites

Because this script uses a massive Deep Learning Vision Model, it requires PyTorch and HuggingFace Transformers. 

### 1. Install Dependencies (using `uv`)

Create a fast virtual environment and install the required deep learning stack:

```bash
# 1. Create a virtual environment
uv venv

# 2. Activate it (Windows)
.venv\Scripts\activate

# 3. Install the required deep learning ecosystem
uv pip install torch torchvision transformers numpy Pillow pymupdf
```

### 2. Download Model Weights
You must download the **PP-DocLayoutV2** model weights from ModelScope or HuggingFace. 
If you previously ran MinerU, they are likely already downloaded on your machine (usually under `C:\Users\<YourUsername>\.cache\modelscope\hub\Opendatalab\PP-DocLayoutV2`).

## Usage

Run the script by pointing it to your PDF and the directory where your model weights are saved:

### Process the Entire Document
This will iterate through every page and output a single `marked_layout_full_document.pdf`.
```bash
uv run run_mineru_layout.py "..\6838_THREE COLUMN CASHBOOK.pdf" --weights "C:\path\to\weights\PP-DocLayoutV2"
```

### Process a Single Page (Fast Testing)
Add the `--page` argument (0-indexed) to process just one page and output `marked_layout_page_1.pdf`.
```bash
uv run run_mineru_layout.py "..\6838_THREE COLUMN CASHBOOK.pdf" --page 0 --weights "C:\path\to\weights\PP-DocLayoutV2"
```
