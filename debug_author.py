import fitz
import sys
sys.path.append(r'f:\Other\pdf-extraction\step1-layout-detection')
from layout_detector import _detect_pymupdf

def run_sim(blocks, pw, ph):
    valid_blocks = [b for b in blocks if b.label not in ('HEADER', 'FOOTER')]
    cx = pw / 2.0
    gutter_blocks = []
    for b in valid_blocks:
        is_top = b.bbox[1] < ph * 0.30
        crosses_center = b.bbox[0] < cx and b.bbox[2] > cx
        if not (is_top and crosses_center):
            gutter_blocks.append(b)
    
    x_intervals = sorted((b.bbox[0], b.bbox[2]) for b in gutter_blocks)
    covered = x_intervals[0][0] if x_intervals else 0.0
    gaps = []
    for a, b_edge in x_intervals:
        if a > covered + 2.0:
            gaps.append((covered, a))
        covered = max(covered, b_edge)
    
    min_gutter = pw * 0.02
    gutters = [(start, end) for start, end in gaps if (end - start) >= min_gutter]
    print("Gutters:", gutters)

    def get_col(block):
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
        if 'Author' in b.text_preview or 'E-mail' in b.text_preview:
            c = get_col(b)
            print(f"Col {c}: {b.text_preview[:20]}... Y={b.bbox[1]:.2f}")

doc = fitz.open(r'f:\Other\pdf-extraction\23092015_Double Column Research Paper Format.pdf')
blocks = _detect_pymupdf(doc[0], scale=1.0)
run_sim(blocks, doc[0].rect.width, doc[0].rect.height)
