# Standalone Neural Layout & Reading Order (PP-DocLayoutV2)

This folder contains a 100% standalone extraction of MinerU's advanced Layout and Reading Order vision model (`PP-DocLayoutV2`). It takes a PDF, uses a PyTorch Transformer to predict the bounding boxes and reading order, and outputs a natively annotated, multi-page PDF without requiring the rest of the massive MinerU codebase.

⚠️ **Note on Model Weights:** Due to GitHub's 100MB file limit, the 205MB AI model weights are **not** included in this repository. You must download them before running the code.

---

## 1. Install Dependencies

It is highly recommended to use `uv` for blazing-fast installations.

```bash
# 1. Create a virtual environment
uv venv

# 2. Activate it (Windows)
.venv\Scripts\activate

# 3. Install the required deep learning ecosystem
uv pip install torch torchvision transformers numpy Pillow pymupdf huggingface_hub modelscope
```

---

## 2. Download the Model Weights

You cannot run the layout pipeline without the neural network weights. Choose one of the methods below:

### Method A: Automated Download (Recommended)
Run the included Python script. It will automatically connect to Hugging Face and download the weights into a local `weights/` folder. If your corporate firewall blocks Hugging Face, the script will automatically bypass the block and download from **ModelScope** (Alibaba's cloud) instead!

```bash
uv run download_weights.py
```

### Method B: Force ModelScope (Bypass HuggingFace)
If you know Hugging Face is blocked and you don't want to wait for the automatic script to time out, you can run this Python one-liner to download directly from Alibaba's ModelScope:
```bash
uv run python -c "from modelscope import snapshot_download; snapshot_download('OpenDataLab/PDF-Extract-Kit-1.0', local_dir='weights/PP-DocLayoutV2', allow_patterns=['models/Layout/PP-DocLayoutV2/*'])"
```

### Method C: Manual ZIP Transfer
If your company blocks *both* Hugging Face and ModelScope, you must transfer the weights manually:
1. On an unrestricted computer, compress the `weights/` folder into a `weights.zip` file.
2. Transfer that ZIP file to your office computer via Google Drive, OneDrive, or USB.
3. Extract `weights.zip` directly inside this folder.

---

## 3. Run the Application

Once the weights are securely inside the `weights/` directory, you can run the PDF layout extractor.

### Process a Single Page (Fast Testing)
Add the `--page` argument (0-indexed) to process just the first page and output `marked_layout_page_1.pdf`.
```bash
uv run run_mineru_layout.py "..\6838_THREE COLUMN CASHBOOK.pdf" --weights "weights\PP-DocLayoutV2\models\Layout\PP-DocLayoutV2" --page 0
```

### Generate Layout PDF Only
If you just want to visualize the bounding boxes without generating Markdown:
```bash
uv run run_mineru_layout.py "..\6838_THREE COLUMN CASHBOOK.pdf" --weights "weights\PP-DocLayoutV2\models\Layout\PP-DocLayoutV2"
```

### Generate Full Markdown & Layout PDF (Recommended)
This uses PyMuPDF to extract the text from the layout boxes based on their exact reading order, generating a final Markdown file alongside a layout debug PDF.

```bash
uv run extract_to_markdown.py "..\6838_THREE COLUMN CASHBOOK.pdf" --weights "weights\PP-DocLayoutV2\models\Layout\PP-DocLayoutV2"
```

*Output will be saved to `output_markdown/6838_THREE COLUMN CASHBOOK.md`.*
