#!/usr/bin/env python3
"""
Academic PDF Generator matching IEEE / University Research Proposal Format
Outputs standard PDF 1.4 using Times-Roman and Helvetica fonts.
"""

import sys
import os
import re
import zlib
from typing import List, Dict, Tuple, Any

# Times-Roman character widths (scaled to 1000 units)
TIMES_WIDTHS = {
    ' ': 250, '!': 333, '"': 408, '#': 500, '$': 500, '%': 833, '&': 778, "'": 180,
    '(': 333, ')': 333, '*': 500, '+': 564, ',': 250, '-': 333, '.': 250, '/': 278,
    '0': 500, '1': 500, '2': 500, '3': 500, '4': 500, '5': 500, '6': 500, '7': 500,
    '8': 500, '9': 500, ':': 278, ';': 278, '<': 564, '=': 564, '>': 564, '?': 444,
    '@': 921, 'A': 722, 'B': 667, 'C': 667, 'D': 722, 'E': 611, 'F': 556, 'G': 722,
    'H': 722, 'I': 333, 'J': 389, 'K': 722, 'L': 611, 'M': 889, 'N': 722, 'O': 722,
    'P': 556, 'Q': 722, 'R': 667, 'S': 556, 'T': 611, 'U': 722, 'V': 667, 'W': 944,
    'X': 667, 'Y': 667, 'Z': 611, '[': 333, '\\': 278, ']': 333, '^': 422, '_': 500,
    '`': 333, 'a': 444, 'b': 500, 'c': 444, 'd': 500, 'e': 444, 'f': 278, 'g': 500,
    'h': 500, 'i': 278, 'j': 278, 'k': 444, 'l': 278, 'm': 778, 'n': 500, 'o': 500,
    'p': 500, 'q': 500, 'r': 333, 's': 389, 't': 278, 'u': 500, 'v': 444, 'w': 667,
    'x': 444, 'y': 444, 'z': 389, '{': 380, '|': 275, '}': 380, '~': 500
}

TIMES_BOLD_WIDTHS = {k: min(1000, int(v * 1.05 + 15)) for k, v in TIMES_WIDTHS.items()}
COURIER_WIDTH = 600

def get_text_width(text: str, font: str, size: float) -> float:
    if font.startswith('/F4') or font.startswith('/F5'): # Courier
        return len(text) * (COURIER_WIDTH * size / 1000.0)
    elif font.startswith('/F2'): # Times-Bold
        w = sum(TIMES_BOLD_WIDTHS.get(c, 500) for c in text)
        return w * size / 1000.0
    else: # Times-Roman / Italic
        w = sum(TIMES_WIDTHS.get(c, 500) for c in text)
        return w * size / 1000.0

def escape_pdf_str(text: str) -> str:
    text = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
    replacements = {
        '\u2014': '---', '\u2013': '--', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2022': '*', '\u2192': '->',
        '\u2264': '<=', '\u2265': '>=', '\u2248': '~=', '\u2260': '!=',
        '\u03b1': 'alpha', '\u03b2': 'beta', '\u03b8': 'theta', '\u03c4': 'tau',
        '\u03c0': 'pi', '\u03bb': 'lambda', '\u2208': 'in', '\u222a': 'U',
        '\u2282': 'subset', '\u2211': 'sum', '\u00b1': '+/-'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    res = []
    for c in text:
        if ord(c) < 128:
            res.append(c)
        else:
            res.append(' ')
    return ''.join(res)

class IEEEPDFDocument:
    def __init__(self, page_width=612.0, page_height=792.0):
        self.width = page_width
        self.height = page_height
        self.margin_left = 54.0   # 0.75 in
        self.margin_right = 54.0
        self.margin_top = 45.0
        self.margin_bottom = 45.0
        self.content_width = self.width - self.margin_left - self.margin_right
        
        self.pages = []
        self.current_page_stream = []
        self.y = self.height - self.margin_top - 10.0
        self.page_number = 1
        
    def new_page(self):
        if self.current_page_stream:
            self.pages.append(self.current_page_stream)
            self.current_page_stream = []
            self.page_number += 1
        self.y = self.height - self.margin_top - 10.0

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

    def draw_line(self, x1, y1, x2, y2, stroke_rgb=(0.0, 0.0, 0.0), line_width=0.5):
        self.add_command("q")
        self.add_command(f"{stroke_rgb[0]:.3f} {stroke_rgb[1]:.3f} {stroke_rgb[2]:.3f} RG")
        self.add_command(f"{line_width:.2f} w")
        self.add_command(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")
        self.add_command("Q")

    def draw_text(self, x, y, text, font="/F1", size=10.0, rgb=(0.0, 0.0, 0.0)):
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
        final_page_streams = []
        for idx, pstream in enumerate(self.pages, 1):
            ps = []
            # Footer on each page (centered page number in Times-Roman)
            ps.extend(pstream)
            pnum_str = f"- {idx} -"
            pnum_escaped = escape_pdf_str(pnum_str)
            pnum_w = get_text_width(pnum_str, "/F1", 9.0)
            ps.append("BT")
            ps.append("0.2 0.2 0.2 rg")
            ps.append("/F1 9.0 Tf")
            ps.append(f"{(self.width - pnum_w)/2.0:.2f} 32.0 Td")
            ps.append(f"({pnum_escaped}) Tj")
            ps.append("ET")
            final_page_streams.append("\n".join(ps).encode('latin-1', 'replace'))

        objects = {}
        next_obj_id = 1
        catalog_id = next_obj_id; next_obj_id += 1
        pages_id = next_obj_id; next_obj_id += 1

        fonts = [
            ("F1", "Times-Roman"),
            ("F2", "Times-Bold"),
            ("F3", "Times-Italic"),
            ("F4", "Courier"),
            ("F5", "Courier-Bold")
        ]
        font_obj_ids = {}
        for ftag, fname in fonts:
            fid = next_obj_id; next_obj_id += 1
            font_obj_ids[ftag] = fid
            objects[fid] = f"<< /Type /Font /Subtype /Type1 /BaseFont /{fname} /Encoding /WinAnsiEncoding >>".encode('latin-1')

        font_dict_entries = " ".join([f"/{tag} {fid} 0 R" for tag, fid in font_obj_ids.items()])
        res_dict = f"<< /ProcSet [/PDF /Text /ImageB /ImageC /ImageI] /Font << {font_dict_entries} >> >>"

        page_obj_ids = []
        content_obj_ids = []
        for i in range(total_pages):
            p_id = next_obj_id; next_obj_id += 1
            c_id = next_obj_id; next_obj_id += 1
            page_obj_ids.append(p_id)
            content_obj_ids.append(c_id)

        for i in range(total_pages):
            p_id = page_obj_ids[i]
            c_id = content_obj_ids[i]
            comp_stream = zlib.compress(final_page_streams[i])

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

        kids_str = " ".join([f"{pid} 0 R" for pid in page_obj_ids])
        objects[pages_id] = f"<< /Type /Pages /Kids [{kids_str}] /Count {total_pages} >>".encode('latin-1')
        objects[catalog_id] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode('latin-1')

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


def wrap_text(text: str, max_w: float, font: str, size: float) -> List[str]:
    words = text.split()
    if not words:
        return []
    lines = []
    curr = []
    curr_w = 0.0
    space_w = get_text_width(" ", font, size)
    
    for w in words:
        w_w = get_text_width(w, font, size)
        if curr and (curr_w + space_w + w_w > max_w):
            lines.append(" ".join(curr))
            curr = [w]
            curr_w = w_w
        else:
            if curr:
                curr_w += space_w
            curr.append(w)
            curr_w += w_w
    if curr:
        lines.append(" ".join(curr))
    return lines


def render_ieee_header(doc: IEEEPDFDocument):
    # Title: Bold, centered, 14pt Times-Bold
    title = "AutoProxy-CoT: Automated Proxy Synthesis and Evidence-Grounded Chain-of-Thought Tuning for Long-Context Reasoning in Compact Language Models"
    t_lines = wrap_text(title, doc.content_width - 20.0, "/F2", 14.0)
    for tl in t_lines:
        tl_w = get_text_width(tl, "/F2", 14.0)
        doc.draw_text((doc.width - tl_w)/2.0, doc.y, tl, font="/F2", size=14.0)
        doc.y -= 18.0
        
    doc.y -= 12.0
    
    # 4-column authors block matching the image
    authors = [
        {"name": "ASHIK RAJAN", "dept": "Department of AIML", "inst": "Jain University", "city": "Bengaluru, India", "role": ""},
        {"name": "ARIGHNA NANDI", "dept": "Department of AIML", "inst": "Jain University", "city": "Bengaluru, India", "role": ""},
        {"name": "DILKUSH M", "dept": "Department of AIML", "inst": "Jain University", "city": "Bengaluru, India", "role": ""},
        {"name": "Prof. Divyanshu Prakash", "dept": "Department of AIML", "inst": "Jain University", "city": "Bengaluru, India", "role": "(Faculty Supervisor)"}
    ]
    
    col_w = doc.content_width / 4.0
    start_y = doc.y
    
    for idx, a in enumerate(authors):
        cx = doc.margin_left + idx * col_w
        center_x = cx + col_w / 2.0
        
        # Name in small caps / bold
        name_w = get_text_width(a["name"], "/F2", 9.5)
        doc.draw_text(center_x - name_w/2.0, start_y, a["name"], font="/F2", size=9.5)
        
        # Dept in italic
        dept_w = get_text_width(a["dept"], "/F3", 9.0)
        doc.draw_text(center_x - dept_w/2.0, start_y - 12.0, a["dept"], font="/F3", size=9.0)
        
        # Inst in italic
        inst_w = get_text_width(a["inst"], "/F3", 9.0)
        doc.draw_text(center_x - inst_w/2.0, start_y - 23.0, a["inst"], font="/F3", size=9.0)
        
        # City in roman
        city_w = get_text_width(a["city"], "/F1", 9.0)
        doc.draw_text(center_x - city_w/2.0, start_y - 34.0, a["city"], font="/F1", size=9.0)
        
        # Role if present
        if a["role"]:
            role_w = get_text_width(a["role"], "/F3", 8.5)
            doc.draw_text(center_x - role_w/2.0, start_y - 45.0, a["role"], font="/F3", size=8.5)
            
    doc.y = start_y - 58.0


def render_ieee_section_header(doc: IEEEPDFDocument, heading: str):
    doc.check_space(32.0)
    doc.y -= 10.0
    # Clean IEEE Roman Section Heading
    doc.draw_text(doc.margin_left, doc.y, heading, font="/F2", size=10.5)
    doc.y -= 14.0


def render_ieee_paragraph(doc: IEEEPDFDocument, text: str):
    # Detect bold run-in prefixes like "Research Gap Statement:", "Research Aim:", "RQ1:", "Evaluation Protocol:", "Metrics:"
    runin_prefixes = [
        "Research Gap Statement:", "Research Aim:", "Objectives:",
        "Research Questions:", "RQ1:", "RQ2:", "RQ3:", "RQ4:",
        "Evaluation Protocol:", "Metrics:", "1) HotpotQA:", "2) SciTrek:",
        "Chunk Extraction:", "Schema Structuring:", "Sufficiency Verification:", "Grounded SFT Loss:"
    ]
    
    prefix = ""
    rest = text
    for p in runin_prefixes:
        if text.startswith(p):
            prefix = p
            rest = text[len(p):].strip()
            break
            
    line_h = 12.5
    if prefix:
        doc.check_space(20.0)
        # Draw prefix in bold
        pref_w = get_text_width(prefix + " ", "/F2", 10.0)
        # Wrap remaining text with indent on first line
        words = rest.split()
        lines = []
        curr = []
        curr_w = pref_w
        space_w = get_text_width(" ", "/F1", 10.0)
        
        for w in words:
            w_w = get_text_width(w, "/F1", 10.0)
            if curr and (curr_w + space_w + w_w > doc.content_width):
                lines.append(" ".join(curr))
                curr = [w]
                curr_w = w_w
            else:
                if curr:
                    curr_w += space_w
                curr.append(w)
                curr_w += w_w
        if curr:
            lines.append(" ".join(curr))
            
        doc.check_space(len(lines) * line_h + 6.0)
        # First line
        doc.draw_text(doc.margin_left, doc.y, prefix, font="/F2", size=10.0)
        if lines:
            doc.draw_text(doc.margin_left + pref_w, doc.y, lines[0], font="/F1", size=10.0)
            doc.y -= line_h
            for l in lines[1:]:
                doc.draw_text(doc.margin_left, doc.y, l, font="/F1", size=10.0)
                doc.y -= line_h
        else:
            doc.y -= line_h
        doc.y -= 4.0
    else:
        # Standard paragraph
        lines = wrap_text(text, doc.content_width, "/F1", 10.0)
        doc.check_space(len(lines) * line_h + 6.0)
        for l in lines:
            doc.draw_text(doc.margin_left, doc.y, l, font="/F1", size=10.0)
            doc.y -= line_h
        doc.y -= 4.0


def render_ieee_formula_box(doc: IEEEPDFDocument, formula_lines: List[str]):
    line_h = 13.0
    box_h = len(formula_lines) * line_h + 12.0
    doc.check_space(box_h)
    
    # Light shaded box for formula clarity
    box_y = doc.y - box_h + 8.0
    doc.draw_rect(doc.margin_left, box_y, doc.content_width, box_h, fill_rgb=(0.97, 0.98, 0.99), stroke_rgb=(0.85, 0.88, 0.92), line_width=0.5)
    
    doc.y -= 6.0
    for fl in formula_lines:
        fl_w = get_text_width(fl, "/F4", 9.0)
        # Center the formula
        cx = doc.margin_left + (doc.content_width - fl_w)/2.0
        doc.draw_text(cx, doc.y, fl, font="/F4", size=9.0, rgb=(0.1, 0.1, 0.15))
        doc.y -= line_h
    doc.y -= 6.0


def render_ieee_table(doc: IEEEPDFDocument, raw_table_lines: List[str]):
    if len(raw_table_lines) < 2:
        return
        
    rows = []
    for line in raw_table_lines:
        cells = [c.strip() for c in line.strip('|').split('|')]
        if all(re.match(r'^:?-+:?$', c) for c in cells if c):
            continue
        rows.append(cells)
        
    if not rows:
        return
        
    num_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < num_cols:
            r.append('')
            
    # Proportional column widths
    if num_cols == 4:
        # Literature Review Table: Paper (18%), Approach (32%), Key Contribution (25%), Relevance (25%)
        col_widths = [doc.content_width * 0.18, doc.content_width * 0.32, doc.content_width * 0.25, doc.content_width * 0.25]
    elif num_cols == 2:
        # Method / Description Table
        col_widths = [doc.content_width * 0.32, doc.content_width * 0.68]
    else:
        col_widths = [doc.content_width / num_cols] * num_cols
        
    doc.check_space(35.0)
    doc.y -= 4.0
    
    # Top horizontal rule (thick IEEE style)
    doc.draw_line(doc.margin_left, doc.y, doc.width - doc.margin_right, doc.y, line_width=1.0)
    doc.y -= 2.0
    
    for row_idx, row in enumerate(rows):
        is_header = (row_idx == 0)
        cell_wrapped = []
        max_lines = 1
        for c_idx, text in enumerate(row):
            font = "/F2" if is_header else "/F1"
            size = 8.5 if is_header else 8.0
            avail = col_widths[c_idx] - 8.0
            w_lines = wrap_text(text, avail, font, size)
            cell_wrapped.append(w_lines)
            max_lines = max(max_lines, len(w_lines))
            
        row_h = max_lines * 10.5 + 6.0
        doc.check_space(row_h)
        
        # Draw cells
        cur_x = doc.margin_left
        for c_idx, w_lines in enumerate(cell_wrapped):
            font = "/F2" if is_header else "/F1"
            size = 8.5 if is_header else 8.0
            cy = doc.y - 8.0
            for wl in w_lines:
                doc.draw_text(cur_x + 4.0, cy, wl, font=font, size=size)
                cy -= 10.5
            cur_x += col_widths[c_idx]
            
        doc.y -= row_h
        
        # Rule under header
        if is_header:
            doc.draw_line(doc.margin_left, doc.y + 2.0, doc.width - doc.margin_right, doc.y + 2.0, line_width=0.75)
            doc.y -= 2.0
        else:
            # Thin rule between rows
            doc.draw_line(doc.margin_left, doc.y + 2.0, doc.width - doc.margin_right, doc.y + 2.0, stroke_rgb=(0.85, 0.85, 0.85), line_width=0.25)
            
    # Bottom rule (thick IEEE style)
    doc.draw_line(doc.margin_left, doc.y + 2.0, doc.width - doc.margin_right, doc.y + 2.0, line_width=1.0)
    doc.y -= 8.0


def render_markdown_ieee(doc: IEEEPDFDocument, text: str):
    lines = text.splitlines()
    
    # 1. Header (Title + Authors)
    render_ieee_header(doc)
    
    i = 0
    # Skip title & author lines from markdown since header rendered them
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('I. INTRODUCTION'):
            break
        i += 1
        
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            doc.y -= 3.0
            i += 1
            continue
            
        # Section headers
        if re.match(r'^(I|II|III|IV|V|VI|VII|VIII)\.\s+[A-Z\s,:]+$', line):
            render_ieee_section_header(doc, line)
            i += 1
            continue
            
        # Tables
        if line.startswith('|') and line.endswith('|'):
            t_lines = []
            while i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                t_lines.append(lines[i].strip())
                i += 1
            render_ieee_table(doc, t_lines)
            continue
            
        # Math formulas in section VI
        if any(line.startswith(prefix) for prefix in ["S_i = Extract", "P* = Structure", "Gate(P*, q)", "L_Grounded(theta)", "tau = <grounding>"]):
            f_lines = []
            while i < len(lines) and (lines[i].strip().startswith(('S_i', 'P*', 'Gate', 'L_Grounded', 'where:', 'tau =', 'log')) or not lines[i].strip()):
                if lines[i].strip():
                    f_lines.append(lines[i].strip())
                i += 1
            render_ieee_formula_box(doc, f_lines)
            continue
            
        # Standard paragraph
        render_ieee_paragraph(doc, lines[i].strip())
        i += 1


def main():
    md_path = r"C:\Users\danwi\.gemini\antigravity\scratch\research_proposal_ieee_format.md"
    pdf_path = r"C:\Users\danwi\.gemini\antigravity\scratch\research_proposal_ieee_format.pdf"
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    doc = IEEEPDFDocument(page_width=612.0, page_height=792.0)
    render_markdown_ieee(doc, md_text)
    pdf_bytes = doc.build_pdf_bytes()
    
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
        
    print(f"Successfully generated {pdf_path}")
    print(f"Total size: {len(pdf_bytes):,} bytes, Total pages: {len(doc.pages)}")

if __name__ == "__main__":
    main()
