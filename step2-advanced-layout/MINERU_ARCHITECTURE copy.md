# MinerU Document Flow: Step-by-Step

When you input a PDF into the MinerU pipeline, it goes through a highly orchestrated, step-by-step journey to become a structured Markdown file. 

Here is exactly what happens to the document at each stage:

### Step 1: Ingestion & Rasterization
* **Input:** `document.pdf`
* **Action:** The system first checks if the PDF has selectable text (native) or is just images (scanned). Regardless of type, the system uses PyMuPDF (`fitz`) to convert every single page of the PDF into a high-resolution, flat image (a picture of the page).
* **Output:** A collection of high-res page images.

### Step 2: Layout Detection (The "Brain")
* **Input:** The high-res page images.
* **Action:** The images are fed into the **PP-DocLayoutV2** neural network. This Vision Transformer looks at the page and draws bounding boxes around every visual element. 
* **Output:** A list of coordinates (boxes) for each element, tagged with:
  1. **A semantic label** (e.g., this box is a `title`, this box is a `table`, this box is an `equation`).
  2. **A reading order index** (e.g., this is block #1, this is block #2).

### Step 3: Cropping & Slicing
* **Input:** The original page images + the bounding box coordinates.
* **Action:** The pipeline acts like a pair of scissors. It literally cuts the high-res page image into dozens of smaller sub-images based on the bounding boxes.
* **Output:** A folder of tiny cropped images (one image for a paragraph, one image for a table, one image for a math formula).

### Step 4: Specialized Extraction (Parallel Routing)
* **Input:** The cropped sub-images.
* **Action:** The pipeline routes each cropped image to a specific AI model designed just for that type of content:
  * **If it's Text/Title:** It is sent to PyMuPDF (or **PaddleOCR** if scanned) to extract the text strings.
  * **If it's a Table:** It is sent to the **SlaNet-Plus** model, which looks at the grid lines and converts the image into structured HTML code (`<table><tr><td>...`).
  * **If it's a Formula:** It is sent to the **UniMERNet** Vision-Language model, which looks at the math symbols and translates them into raw LaTeX code.
  * **If it's an Image:** It is simply saved to the hard drive as a `.jpg`.
* **Output:** Raw extracted Text, HTML code, LaTeX code, and saved JPGs.

### Step 5: Markdown Assembly & Filtering
* **Input:** The raw extracted data from Step 4 + the Reading Order Index from Step 2.
* **Action:** The pipeline looks at the mathematical index predicted back in Step 2. It grabs the extracted text for Box #1, then Box #2, then Box #3, merging them together in perfect reading order. 
* **Filtering:** Any boxes that were labeled as `header`, `footer`, or `page_number` in Step 2 are deliberately thrown in the trash so they don't interrupt the flow of the text.
* **Output:** The final, cohesive `document.md` file!
