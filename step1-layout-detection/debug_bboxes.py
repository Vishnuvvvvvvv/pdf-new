import fitz

doc = fitz.open(r"f:\Other\pdf-extraction\23092015_Double Column Research Paper Format.pdf")
page = doc[0]
raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
scale = 2.0
pw = page.rect.width * scale
ph = page.rect.height * scale
print(f"Page image size: {pw:.0f} x {ph:.0f}")
print()

for b in raw.get("blocks", []):
    x0, y0, x1, y1 = b["bbox"]
    lines = b.get("lines", [])
    text = " ".join(
        " ".join(sp.get("text","") for sp in ln.get("spans",[]))
        for ln in lines
    ).strip()[:35]
    sx0, sy0, sx1, sy1 = x0*scale, y0*scale, x1*scale, y1*scale
    cx = (sx0 + sx1) / 2
    cy = (sy0 + sy1) / 2
    print(f"  x:[{sx0:6.1f} - {sx1:6.1f}]  cx={cx:6.1f}  y:[{sy0:6.1f} - {sy1:6.1f}]  cy={cy:6.1f}  \"{text}\"")
