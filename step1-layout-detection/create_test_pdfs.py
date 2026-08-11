from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_multi_column_pdf(filename, num_columns):
    doc = BaseDocTemplate(filename, pagesize=letter)
    
    # Calculate column dimensions
    margin = 0.5 * inch
    page_width, page_height = letter
    usable_width = page_width - 2 * margin
    gutter = 0.25 * inch
    col_width = (usable_width - (num_columns - 1) * gutter) / num_columns
    
    # Create frames
    frames = []
    for i in range(num_columns):
        x = margin + i * (col_width + gutter)
        # Create a frame for the column. Leave room at the top for a header.
        frame = Frame(x, margin, col_width, page_height - 2 * margin - 1 * inch, id=f'col{i}')
        frames.append(frame)
        
    # We want a header frame as well, which spans the whole width
    header_frame = Frame(margin, page_height - margin - 0.9 * inch, usable_width, 0.9 * inch, id='header')
    
    # Actually, Platypus page templates process frames in order. We can have a header frame, then the columns.
    template = PageTemplate(id='multi_col', frames=[header_frame] + frames)
    doc.addPageTemplates([template])
    
    styles = getSampleStyleSheet()
    header_style = styles['Heading1']
    header_style.alignment = 1 # Center
    
    body_style = styles['Normal']
    
    story = []
    
    # Header
    story.append(Paragraph(f"This is a {num_columns}-Column Test Document", header_style))
    story.append(Paragraph("This spanning header is placed in the first frame. Then the text flows into the columns.", styles['Normal']))
    
    # Next frame (the first column)
    from reportlab.platypus.doctemplate import FrameBreak
    story.append(FrameBreak())
    
    # Generate some dummy text for each column
    lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
    
    for i in range(1, 15):
        story.append(Paragraph(f"<b>Block {i}</b>", styles['Heading3']))
        story.append(Paragraph(f"Text block {i}: " + lorem, body_style))
        story.append(Spacer(1, 12))
        
    doc.build(story)

create_multi_column_pdf("f:/Other/pdf-extraction/test_3_column.pdf", 3)
create_multi_column_pdf("f:/Other/pdf-extraction/test_4_column.pdf", 4)
print("PDFs created successfully.")
