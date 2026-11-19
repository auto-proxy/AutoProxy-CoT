#!/usr/bin/env python3
"""
Custom Academic PDF Generator for Research Proposals
Outputs standard PDF 1.4 without external dependencies.
"""

import sys
import os
import re
import zlib
from typing import List, Dict, Tuple, Any

# Standard Helvetica character widths (scaled to 1000 units)
HELVETICA_WIDTHS = {
    ' ': 278, '!': 278, '"': 355, '#': 556, '$': 556, '%': 889, '&': 667, "'": 191,
    '(': 333, ')': 333, '*': 389, '+': 584, ',': 278, '-': 333, '.': 278, '/': 278,
    '0': 556, '1': 556, '2': 556, '3': 556, '4': 556, '5': 556, '6': 556, '7': 556,
    '8': 556, '9': 556, ':': 278, ';': 278, '<': 584, '=': 584, '>': 584, '?': 556,
    '@': 1015, 'A': 667, 'B': 667, 'C': 722, 'D': 722, 'E': 667, 'F': 611, 'G': 778,
    'H': 722, 'I': 278, 'J': 500, 'K': 667, 'L': 556, 'M': 833, 'N': 722, 'O': 778,
    'P': 667, 'Q': 778, 'R': 722, 'S': 667, 'T': 611, 'U': 722, 'V': 667, 'W': 944,
    'X': 667, 'Y': 667, 'Z': 611, '[': 278, '\\': 278, ']': 278, '^': 469, '_': 556,
    '`': 333, 'a': 556, 'b': 556, 'c': 500, 'd': 556, 'e': 556, 'f': 278, 'g': 556,
    'h': 556, 'i': 222, 'j': 222, 'k': 500, 'l': 222, 'm': 833, 'n': 556, 'o': 556,
    'p': 556, 'q': 556, 'r': 333, 's': 500, 't': 278, 'u': 556, 'v': 500, 'w': 722,
    'x': 500, 'y': 500, 'z': 500, '{': 334, '|': 260, '}': 334, '~': 584
}

HELVETICA_BOLD_WIDTHS = {k: min(1000, int(v * 1.05 + 20)) for k, v in HELVETICA_WIDTHS.items()}
COURIER_WIDTH = 600

def get_text_width(text: str, font: str, size: float) -> float:
    if font.startswith('/F4') or font.startswith('/F5'): # Courier
        return len(text) * (COURIER_WIDTH * size / 1000.0)
    elif font.startswith('/F2'): # Helvetica-Bold
        w = sum(HELVETICA_BOLD_WIDTHS.get(c, 556) for c in text)
        return w * size / 1000.0
    else: # Helvetica / Oblique
        w = sum(HELVETICA_WIDTHS.get(c, 556) for c in text)
        return w * size / 1000.0

def escape_pdf_str(text: str) -> str:
    # Clean non-ASCII characters to safe PDF WinAnsi / Latin-1 representations
    text = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
    # Clean smart quotes and dashes
    replacements = {
        '\u2014': '---', '\u2013': '--', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2022': '*', '\u2192': '->',
        '\u2264': '<=', '\u2265': '>=', '\u2248': '~=', '\u2260': '!=',
        '\u03b1': 'alpha', '\u03b2': 'beta', '\u03b8': 'theta', '\u03c4': 'tau',
        '\u03c0': 'pi', '\u03bb': 'lambda', '\u2208': 'in', '\u222a': 'U',
        '\u2282': 'subset', '\u2211': 'sum', '\u2260': '!=', '\u00b1': '+/-',
        '\u2265': '>=', '\u2264': '<='
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Filter any remaining non-ascii to ascii equivalents
    res = []
    for c in text:
        if ord(c) < 128:
            res.append(c)
        else:
            res.append(' ')
    return ''.join(res)

class PDFDocument:
    def __init__(self, page_width=612.0, page_height=792.0):
        self.width = page_width
        self.height = page_height
        self.margin_left = 54.0   # 0.75 in
        self.margin_right = 54.0
        self.margin_top = 54.0
        self.margin_bottom = 54.0
        self.content_width = self.width - self.margin_left - self.margin_right
        
        self.pages = []
        self.current_page_stream = []
        self.y = self.height - self.margin_top - 20.0
        self.page_number = 1
        
    def new_page(self):
        if self.current_page_stream:
            self.pages.append(self.current_page_stream)
            self.current_page_stream = []
            self.page_number += 1
        self.y = self.height - self.margin_top - 20.0

    def add_command(self, cmd: str):
        self.current_page_stream.append(cmd)

    def draw_rect(self, x, y, w, h, fill_rgb=None, stroke_rgb=None, line_width=0.5):
        self.add_command("q")
        if stroke_rgb:
            self.add_command(f"{stroke_rgb[0]:.3f} {stroke_rgb[1]:.3f} {stroke_rgb[2]:.3f} RG")
            self.add_command(f"{line_width:.2f} w")
        if fill_rgb:
            self.add_command(f"{fill_rgb[0]:.3f} {fill_rgb[1]:.3f} {fill_rgb[2]:.3f} rg")
        self.add_command(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re")
        if fill_rgb and stroke_rgb:
            self.add_command("B")
        elif fill_rgb:
            self.add_command("f")
        elif stroke_rgb:
            self.add_command("S")
        self.add_command("Q")

    def draw_line(self, x1, y1, x2, y2, stroke_rgb=(0.7, 0.7, 0.7), line_width=0.5):
        self.add_command("q")
        self.add_command(f"{stroke_rgb[0]:.3f} {stroke_rgb[1]:.3f} {stroke_rgb[2]:.3f} RG")
        self.add_command(f"{line_width:.2f} w")
        self.add_command(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")
        self.add_command("Q")

    def draw_text(self, x, y, text, font="/F1", size=10.0, rgb=(0.15, 0.15, 0.15)):
        escaped = escape_pdf_str(text)
        self.add_command("BT")
        self.add_command(f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} rg")
        self.add_command(f"{font} {size:.2f} Tf")
        self.add_command(f"{x:.2f} {y:.2f} Td")
        self.add_command(f"({escaped}) Tj")
        self.add_command("ET")

    def check_space(self, needed_height: float):
        if self.y - needed_height < self.margin_bottom + 15.0:
            self.new_page()

    def build_pdf_bytes(self) -> bytes:
        if self.current_page_stream:
            self.pages.append(self.current_page_stream)
            self.current_page_stream = []

        total_pages = len(self.pages)
        
        # Add running header and footer to each page
        final_page_streams = []
        for idx, pstream in enumerate(self.pages, 1):
            ps = []
            # Header (pages > 1)
            if idx > 1:
                # Running title header
                hdr_text = escape_pdf_str("AutoProxy-CoT: Research Paper Proposal")
                ps.append("BT")
                ps.append("0.45 0.50 0.55 rg")
                ps.append("/F3 8.0 Tf")
                ps.append(f"{self.margin_left:.2f} 752.0 Td")
                ps.append(f"({hdr_text}) Tj")
                ps.append("ET")
                # Thin divider rule
                ps.append("q 0.80 0.84 0.88 RG 0.5 w")
                ps.append(f"{self.margin_left:.2f} 746.0 m {self.width - self.margin_right:.2f} 746.0 l S Q")

            # Page content
            ps.extend(pstream)

            # Footer on all pages
            ftr_rule_y = self.margin_bottom - 10.0
            ps.append("q 0.80 0.84 0.88 RG 0.5 w")
            ps.append(f"{self.margin_left:.2f} {ftr_rule_y:.2f} m {self.width - self.margin_right:.2f} {ftr_rule_y:.2f} l S Q")

            # Running footer text
            ftr_text = escape_pdf_str("Confidential - Academic Research Proposal")
            ps.append("BT")
            ps.append("0.45 0.50 0.55 rg")
            ps.append("/F1 8.0 Tf")
            ps.append(f"{self.margin_left:.2f} {ftr_rule_y - 12.0:.2f} Td")
            ps.append(f"({ftr_text}) Tj")
            ps.append("ET")

            # Page number
            pnum_str = f"Page {idx} of {total_pages}"
            pnum_escaped = escape_pdf_str(pnum_str)
            pnum_w = get_text_width(pnum_str, "/F1", 8.0)
            ps.append("BT")
            ps.append("0.45 0.50 0.55 rg")
            ps.append("/F1 8.0 Tf")
            ps.append(f"{self.width - self.margin_right - pnum_w:.2f} {ftr_rule_y - 12.0:.2f} Td")
            ps.append(f"({pnum_escaped}) Tj")
            ps.append("ET")

            final_page_streams.append("\n".join(ps).encode('latin-1', 'replace'))

        # Construct standard PDF structure
        # Objects:
        # 1: Catalog
        # 2: Pages
        # 3..3+total_pages-1: Page objects
        # 3+total_pages..3+2*total_pages-1: Content stream objects
        # Fonts: F1 (Helv), F2 (Helv-Bold), F3 (Helv-Oblique), F4 (Courier), F5 (Courier-Bold)
        objects = {}
        next_obj_id = 1

        catalog_id = next_obj_id; next_obj_id += 1
        pages_id = next_obj_id; next_obj_id += 1

        # Font objects
        fonts = [
            ("F1", "Helvetica"),
            ("F2", "Helvetica-Bold"),
            ("F3", "Helvetica-Oblique"),
            ("F4", "Courier"),
            ("F5", "Courier-Bold")
        ]
        font_obj_ids = {}
        for ftag, fname in fonts:
            fid = next_obj_id; next_obj_id += 1
            font_obj_ids[ftag] = fid
            objects[fid] = f"<< /Type /Font /Subtype /Type1 /BaseFont /{fname} /Encoding /WinAnsiEncoding >>".encode('latin-1')

        # ProcSet & Font dict
        font_dict_entries = " ".join([f"/{tag} {fid} 0 R" for tag, fid in font_obj_ids.items()])
        res_dict = f"<< /ProcSet [/PDF /Text /ImageB /ImageC /ImageI] /Font << {font_dict_entries} >> >>"

        page_obj_ids = []
        content_obj_ids = []
        for i in range(total_pages):
            p_id = next_obj_id; next_obj_id += 1
            c_id = next_obj_id; next_obj_id += 1
            page_obj_ids.append(p_id)
            content_obj_ids.append(c_id)

        # Build Page and Content objects
        for i in range(total_pages):
            p_id = page_obj_ids[i]
            c_id = content_obj_ids[i]
            c_stream = final_page_streams[i]
            comp_stream = zlib.compress(c_stream)

            objects[c_id] = (
                f"<< /Length {len(comp_stream)} /Filter /FlateDecode >>\nstream\n".encode('latin-1')
                + comp_stream
                + b"\nendstream"
            )

            objects[p_id] = (
                f"<< /Type /Page /Parent {pages_id} 0 R "
                f"/MediaBox [0 0 {self.width:.2f} {self.height:.2f}] "
                f"/Contents {c_id} 0 R "
                f"/Resources {res_dict} >>".encode('latin-1')
            )

        # Pages object
        kids_str = " ".join([f"{pid} 0 R" for pid in page_obj_ids])
        objects[pages_id] = f"<< /Type /Pages /Kids [{kids_str}] /Count {total_pages} >>".encode('latin-1')

        # Catalog object
        objects[catalog_id] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode('latin-1')

        # Write PDF binary
        out = bytearray()
        out.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

        offsets = {}
        for oid in range(1, next_obj_id):
            offsets[oid] = len(out)
            out.extend(f"{oid} 0 obj\n".encode('latin-1'))
            out.extend(objects[oid])
            out.extend(b"\nendobj\n")

        xref_offset = len(out)
        out.extend(f"xref\n0 {next_obj_id}\n0000000000 65535 f \n".encode('latin-1'))
        for oid in range(1, next_obj_id):
            out.extend(f"{offsets[oid]:010d} 00000 n \n".encode('latin-1'))

        out.extend(
            f"trailer\n<< /Size {next_obj_id} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n".encode('latin-1')
        )
        return bytes(out)


def render_markdown_proposal(doc: PDFDocument, md_text: str):
    lines = md_text.splitlines()
    
    in_code_block = False
    code_lines = []
    
    in_table = False
    table_rows = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check code block
        if line.strip().startswith('```'):
            if in_code_block:
                # End of code block: render code box
                in_code_block = False
                render_code_box(doc, code_lines)
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue
            
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
            
        # Check table line
        if line.strip().startswith('|') and line.strip().endswith('|'):
            # Collect full table
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            render_markdown_table(doc, table_lines)
            continue
            
        # Empty line
        if not line.strip():
            doc.y -= 6.0
            i += 1
            continue
            
        # Horizontal rule
        if line.strip() in ['---', '***', '___']:
            doc.check_space(15.0)
            doc.draw_line(doc.margin_left, doc.y, doc.width - doc.margin_right, doc.y, stroke_rgb=(0.78, 0.82, 0.88), line_width=0.75)
            doc.y -= 12.0
            i += 1
            continue
            
        # Metadata / blockquote / callout
        if line.startswith('>'):
            callout_lines = []
            while i < len(lines) and lines[i].startswith('>'):
                callout_lines.append(lines[i].lstrip('>').strip())
                i += 1
            render_callout_box(doc, callout_lines)
            continue
            
        # Headers
        if line.startswith('# '):
            # Document Title
            title = line[2:].strip()
            doc.check_space(60.0)
            doc.y -= 10.0
            render_text_block(doc, title, font="/F2", size=18.0, line_height=22.0, rgb=(0.10, 0.21, 0.36), space_after=8.0)
            i += 1
            continue
            
        if line.startswith('## '):
            # Subtitle or major unit
            subtitle = line[3:].strip()
            doc.check_space(45.0)
            render_text_block(doc, subtitle, font="/F2", size=12.5, line_height=16.0, rgb=(0.17, 0.42, 0.69), space_after=10.0)
            i += 1
            continue
            
        if line.startswith('### '):
            # Section Header (H1)
            h1 = line[4:].strip()
            doc.check_space(35.0)
            doc.y -= 8.0
            # Accent pill or bar
            doc.draw_rect(doc.margin_left, doc.y - 1.0, 3.5, 14.0, fill_rgb=(0.10, 0.25, 0.45))
            render_text_block(doc, h1, font="/F2", size=12.0, line_height=15.0, rgb=(0.10, 0.21, 0.36), space_after=6.0, indent=8.0)
            doc.draw_line(doc.margin_left + 8.0, doc.y + 3.0, doc.width - doc.margin_right, doc.y + 3.0, stroke_rgb=(0.85, 0.88, 0.92), line_width=0.5)
            doc.y -= 4.0
            i += 1
            continue
            
        if line.startswith('#### '):
            # Subsection Header (H2)
            h2 = line[5:].strip()
            doc.check_space(28.0)
            doc.y -= 4.0
            render_text_block(doc, h2, font="/F2", size=10.5, line_height=13.5, rgb=(0.18, 0.28, 0.42), space_after=5.0)
            i += 1
            continue
            
        if line.startswith('##### '):
            h3 = line[6:].strip()
            doc.check_space(22.0)
            render_text_block(doc, h3, font="/F2", size=9.5, line_height=12.5, rgb=(0.25, 0.35, 0.45), space_after=4.0)
            i += 1
            continue
            
        # Bullet list items
        if line.strip().startswith(('- ', '* ')):
            bullet_text = line.strip()[2:].strip()
            render_list_item(doc, bullet_text, is_numbered=False)
            i += 1
            continue
            
        # Numbered list items
        m_num = re.match(r'^(\d+)\.\s+(.*)$', line.strip())
        if m_num:
            num_str = m_num.group(1) + "."
            num_text = m_num.group(2)
            render_list_item(doc, num_text, is_numbered=True, num_prefix=num_str)
            i += 1
            continue
            
        # Metadata block (lines with **Key:** Value)
        if line.startswith('**') and ':**' in line:
            render_metadata_line(doc, line)
            i += 1
            continue

        # Regular paragraph
        render_text_block(doc, line, font="/F1", size=9.5, line_height=13.0, rgb=(0.18, 0.22, 0.26), space_after=6.0)
        i += 1


def wrap_text_spans(text: str, max_width: float, default_font: str, default_size: float) -> List[List[Tuple[str, str, float]]]:
    """
    Parses **bold** and regular segments, wrapping them into lines fitted to max_width.
    Returns list of lines, where each line is list of (token, font, size).
    """
    # Parse markdown formatting tokens
    # We support **bold** and `code`
    tokens = []
    pattern = re.compile(r'(\*\*[^*]+\*\*|`[^`]+`|[^\s*`]+|\s+)')
    raw_tokens = pattern.findall(text)
    
    for raw in raw_tokens:
        if not raw:
            continue
        if raw.startswith('**') and raw.endswith('**') and len(raw) >= 4:
            clean = raw[2:-2]
            tokens.append((clean, "/F2", default_size))
        elif raw.startswith('`') and raw.endswith('`') and len(raw) >= 2:
            clean = raw[1:-1]
            tokens.append((clean, "/F4", default_size * 0.9))
        else:
            tokens.append((raw, default_font, default_size))
            
    # Now wrap tokens into lines
    lines = []
    current_line = []
    current_w = 0.0
    
    for tok, fnt, sz in tokens:
        tok_w = get_text_width(tok, fnt, sz)
        if tok.isspace():
            if current_line:
                current_line.append((tok, fnt, sz))
                current_w += tok_w
        else:
            if current_w + tok_w > max_width and current_line:
                # Wrap to new line
                lines.append(current_line)
                current_line = [(tok, fnt, sz)]
                current_w = tok_w
            else:
                current_line.append((tok, fnt, sz))
                current_w += tok_w
                
    if current_line:
        lines.append(current_line)
        
    return lines


def render_text_block(doc: PDFDocument, text: str, font="/F1", size=9.5, line_height=13.0, rgb=(0.18, 0.22, 0.26), space_after=6.0, indent=0.0):
    avail_w = doc.content_width - indent
    lines = wrap_text_spans(text, avail_w, font, size)
    
    doc.check_space(len(lines) * line_height + space_after)
    
    for line in lines:
        cur_x = doc.margin_left + indent
        for tok, tok_font, tok_size in line:
            if tok.isspace():
                cur_x += get_text_width(tok, tok_font, tok_size)
                continue
            doc.draw_text(cur_x, doc.y, tok, font=tok_font, size=tok_size, rgb=rgb)
            cur_x += get_text_width(tok, tok_font, tok_size)
        doc.y -= line_height
        
    doc.y -= space_after


def render_metadata_line(doc: PDFDocument, line: str):
    doc.check_space(14.0)
    # Highlight metadata with subtle icon / tag styling
    doc.draw_rect(doc.margin_left, doc.y - 2.0, doc.content_width, 15.0, fill_rgb=(0.96, 0.97, 0.99))
    render_text_block(doc, line, font="/F1", size=9.0, line_height=12.0, rgb=(0.15, 0.25, 0.35), space_after=2.0, indent=6.0)


def render_list_item(doc: PDFDocument, text: str, is_numbered=False, num_prefix="1."):
    avail_w = doc.content_width - 16.0
    lines = wrap_text_spans(text, avail_w, "/F1", 9.5)
    line_height = 13.0
    space_after = 3.0
    
    doc.check_space(len(lines) * line_height + space_after)
    
    # Draw bullet or number
    if is_numbered:
        doc.draw_text(doc.margin_left + 2.0, doc.y, num_prefix, font="/F2", size=9.0, rgb=(0.15, 0.35, 0.60))
    else:
        # Draw clean vector bullet circle
        doc.draw_rect(doc.margin_left + 5.0, doc.y + 3.0, 3.0, 3.0, fill_rgb=(0.20, 0.40, 0.65))
        
    for line in lines:
        cur_x = doc.margin_left + 16.0
        for tok, tok_font, tok_size in line:
            if tok.isspace():
                cur_x += get_text_width(tok, tok_font, tok_size)
                continue
            doc.draw_text(cur_x, doc.y, tok, font=tok_font, size=tok_size, rgb=(0.18, 0.22, 0.26))
            cur_x += get_text_width(tok, tok_font, tok_size)
        doc.y -= line_height
        
    doc.y -= space_after


def render_callout_box(doc: PDFDocument, lines: List[str]):
    full_text = " ".join(lines)
    avail_w = doc.content_width - 24.0
    wrapped = wrap_text_spans(full_text, avail_w, "/F3", 9.0) # Helvetica-Oblique
    box_height = len(wrapped) * 12.5 + 12.0
    
    doc.check_space(box_height + 8.0)
    box_y = doc.y - box_height + 8.0
    
    # Background rectangle
    doc.draw_rect(doc.margin_left, box_y, doc.content_width, box_height, fill_rgb=(0.96, 0.98, 1.00), stroke_rgb=(0.82, 0.88, 0.94), line_width=0.5)
    # Accent left border
    doc.draw_rect(doc.margin_left, box_y, 3.5, box_height, fill_rgb=(0.20, 0.45, 0.75))
    
    doc.y -= 8.0
    for wline in wrapped:
        cur_x = doc.margin_left + 14.0
        for tok, tok_font, tok_size in wline:
            if tok.isspace():
                cur_x += get_text_width(tok, tok_font, tok_size)
                continue
            doc.draw_text(cur_x, doc.y, tok, font=tok_font, size=tok_size, rgb=(0.15, 0.25, 0.35))
            cur_x += get_text_width(tok, tok_font, tok_size)
        doc.y -= 12.5
    doc.y -= 8.0


def render_code_box(doc: PDFDocument, lines: List[str]):
    line_height = 11.0
    box_height = len(lines) * line_height + 14.0
    
    doc.check_space(min(box_height + 8.0, 150.0))
    box_y = doc.y - box_height + 8.0
    
    # Draw background
    doc.draw_rect(doc.margin_left, box_y, doc.content_width, box_height, fill_rgb=(0.95, 0.96, 0.98), stroke_rgb=(0.82, 0.85, 0.90), line_width=0.5)
    
    doc.y -= 8.0
    for line in lines:
        doc.check_space(line_height)
        safe_line = line.replace('\t', '    ')
        # Truncate if overly long
        if get_text_width(safe_line, "/F4", 8.0) > doc.content_width - 16.0:
            while safe_line and get_text_width(safe_line + '...', "/F4", 8.0) > doc.content_width - 16.0:
                safe_line = safe_line[:-1]
            safe_line += '...'
        doc.draw_text(doc.margin_left + 8.0, doc.y, safe_line, font="/F4", size=8.0, rgb=(0.20, 0.22, 0.28))
        doc.y -= line_height
    doc.y -= 8.0


def render_markdown_table(doc: PDFDocument, raw_table_lines: List[str]):
    if len(raw_table_lines) < 2:
        return
        
    # Split into cells
    rows = []
    for line in raw_table_lines:
        cells = [c.strip() for c in line.strip('|').split('|')]
        # Skip separator row like |:---|:---|
        if all(re.match(r'^:?-+:?$', c) for c in cells if c):
            continue
        rows.append(cells)
        
    if not rows:
        return
        
    num_cols = max(len(r) for r in rows)
    # Normalize row lengths
    for r in rows:
        while len(r) < num_cols:
            r.append('')
            
    # Calculate column widths
    # Base allocation on text length proportion
    col_max_lens = [0] * num_cols
    for r in rows:
        for idx, c in enumerate(r):
            col_max_lens[idx] = max(col_max_lens[idx], len(c))
            
    total_len = sum(col_max_lens) or 1
    # Minimum width 50pt per column
    col_widths = [max(50.0, (l / total_len) * doc.content_width) for l in col_max_lens]
    # Normalize to exact content_width
    norm_factor = doc.content_width / sum(col_widths)
    col_widths = [w * norm_factor for w in col_widths]
    
    doc.y -= 6.0
    doc.check_space(40.0)
    
    for row_idx, row in enumerate(rows):
        is_header = (row_idx == 0)
        # Wrap each cell text to column width
        cell_wrapped_lines = []
        max_cell_lines = 1
        for c_idx, cell_text in enumerate(row):
            w = col_widths[c_idx] - 8.0 # 4pt padding each side
            font = "/F2" if is_header else "/F1"
            size = 8.5 if is_header else 8.0
            wrapped = wrap_text_spans(cell_text, w, font, size)
            cell_wrapped_lines.append(wrapped)
            max_cell_lines = max(max_cell_lines, len(wrapped))
            
        row_height = max_cell_lines * 11.0 + 8.0
        doc.check_space(row_height)
        
        # Row background
        if is_header:
            doc.draw_rect(doc.margin_left, doc.y - row_height + 4.0, doc.content_width, row_height, fill_rgb=(0.90, 0.93, 0.97), stroke_rgb=(0.75, 0.80, 0.88), line_width=0.75)
        elif row_idx % 2 == 1:
            doc.draw_rect(doc.margin_left, doc.y - row_height + 4.0, doc.content_width, row_height, fill_rgb=(0.98, 0.99, 1.00), stroke_rgb=(0.88, 0.90, 0.93), line_width=0.25)
        else:
            doc.draw_rect(doc.margin_left, doc.y - row_height + 4.0, doc.content_width, row_height, fill_rgb=(1.00, 1.00, 1.00), stroke_rgb=(0.88, 0.90, 0.93), line_width=0.25)
            
        # Draw cells
        cur_col_x = doc.margin_left
        for c_idx, cell_text in enumerate(row):
            cw = col_widths[c_idx]
            wrapped = cell_wrapped_lines[c_idx]
            font = "/F2" if is_header else "/F1"
            size = 8.5 if is_header else 8.0
            rgb = (0.10, 0.20, 0.35) if is_header else (0.18, 0.22, 0.26)
            
            cell_y = doc.y - 2.0
            for line_parts in wrapped:
                cur_x = cur_col_x + 4.0
                for tok, tok_font, tok_size in line_parts:
                    if tok.isspace():
                        cur_x += get_text_width(tok, tok_font, tok_size)
                        continue
                    doc.draw_text(cur_x, cell_y, tok, font=tok_font, size=tok_size, rgb=rgb)
                    cur_x += get_text_width(tok, tok_font, tok_size)
                cell_y -= 11.0
                
            cur_col_x += cw
            
        doc.y -= row_height
        
    doc.y -= 8.0


def main():
    md_path = r"C:\Users\danwi\.gemini\antigravity\scratch\research_proposal_autoproxy_cot.md"
    pdf_path = r"C:\Users\danwi\.gemini\antigravity\scratch\research_proposal_autoproxy_cot.pdf"
    
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        sys.exit(1)
        
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    doc = PDFDocument(page_width=612.0, page_height=792.0)
    render_markdown_proposal(doc, md_text)
    pdf_bytes = doc.build_pdf_bytes()
    
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
        
    print(f"Successfully generated {pdf_path}")
    print(f"Total size: {len(pdf_bytes):,} bytes, Total pages: {len(doc.pages)}")

if __name__ == "__main__":
    main()
