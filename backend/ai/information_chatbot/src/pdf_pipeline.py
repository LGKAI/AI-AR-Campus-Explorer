from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
import easyocr
from config import settings

# Regex helpers

# Detect vietnamese heading : Chương I, Điều 15, Mục 3, Phần II, Article 5...
# Supports both accented (Điều) and OCR-stripped (Dieu) variants for robustness.
SECTION_HEADING_REGEX = re.compile(
    r"^("
    r"Ch[uư]ơng|Chuong"
    r"|Đi[eề]u|Dieu"
    r"|M[uụ]c|Muc"
    r"|Ph[aầ]n|Phan"
    r"|Ti[eế]t|Tiet"
    r"|Article|Chapter|Section"
    r")\s+[\dIVXivx]+[\.:\-\s]?",
    re.IGNORECASE,
)

MATH_HINT_REGEX = re.compile(r"[\=\+\-\*/\^]|\\frac|\\sum|\\int")

# Dataclass

@dataclass
class Chunk:
    id: str
    source: str          #  file PDF
    doc_type: str        # quy_che | tuyen_sinh | chuong_trinh | lich_hoc | hoc_phi | thong_bao | general
    page: int
    section: str         # nearest heading , "" if None
    text: str
    kind: str            # native_text | ocr_text | formula

# Pipeline

class PdfPipeline:
    def __init__(self) -> None:
        self.ocr = easyocr.Reader(["en", "vi"], gpu=settings.use_gpu)

    def ingest_folder(self, input_dir: Path) -> list[Chunk]:
        pdfs = sorted(input_dir.glob("*.pdf"))
        all_chunks: list[Chunk] = []
        for pdf_path in pdfs:
            all_chunks.extend(self._parse_pdf(pdf_path))
        return all_chunks

    # Parse 1 file PDF

    def _parse_pdf(self, pdf_path: Path) -> list[Chunk]:
        doc_type = settings.resolve_doc_type(pdf_path.name)
        doc = fitz.open(pdf_path)
        chunks: list[Chunk] = []

        # Track the current section heading in file 
        current_section: str = ""

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            native_text = page.get_text("text").strip()

            if native_text:
                # Update section
                current_section = self._update_section(native_text, current_section)
                chunks.extend(
                    self._split_text(
                        source=pdf_path.name,
                        doc_type=doc_type,
                        page=page_idx + 1,
                        section=current_section,
                        raw_text=native_text,
                        kind="native_text",
                    )
                )

            # OCR
            pix = page.get_pixmap(dpi=settings.pdf_dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_text = self._ocr_image(img)
            if ocr_text:
                current_section = self._update_section(ocr_text, current_section)
                chunks.extend(
                    self._split_text(
                        source=pdf_path.name,
                        doc_type=doc_type,
                        page=page_idx + 1,
                        section=current_section,
                        raw_text=ocr_text,
                        kind="ocr_text",
                    )
                )

            # Formula candidates
            combined = native_text + "\n" + ocr_text
            formulas = self._extract_formula_candidates(combined)
            for i, formula in enumerate(formulas):
                chunks.append(
                    Chunk(
                        id=f"{pdf_path.stem}-p{page_idx + 1}-f{i}",
                        source=pdf_path.name,
                        doc_type=doc_type,
                        page=page_idx + 1,
                        section=current_section,
                        text=formula,
                        kind="formula",
                    )
                )

        doc.close()
        return chunks

    # Section detection

    def _update_section(self, text: str, current: str) -> str:
        """
        Scan each line in text, return the newest heading
        """
        for line in text.splitlines():
            line = line.strip()
            if SECTION_HEADING_REGEX.match(line):
                # Truncate to avoid overflow
                current = line[:120]
        return current

    # OCR — layout-aware reconstruction

    def _ocr_image(self, image: Image.Image) -> str:
        """
        Bounding-box aware OCR to correctly reconstruct table rows.

        Problem with detail=0 / paragraph=True:
          EasyOCR groups nearby text into paragraphs, but for multi-column
          tables it reads column-by-column, destroying row semantics.
          e.g. grade table → "9.0-10 8.0-9 ..." then "A+ A ..." then "4.0 3.5 ..."

        Solution:
          1. detail=1 → get individual boxes with (bbox, text, confidence)
          2. Group boxes into rows by y-centre proximity
          3. Sort cells left-to-right within each row
          4. If ≥40% of rows have ≥2 cells → table mode: join cells with " | "
          5. Otherwise → paragraph mode: join cells with " "

        Result for grade table:
          "9.0-10.0 | A+ | 4.0"
          "8.0-<9.0 | A  | 3.5"
          ...
        """
        arr = np.array(image)
        results = self.ocr.readtext(arr, detail=1, paragraph=False)
        if not results:
            return ""

        MIN_CONF = 0.25  # discard very-low-confidence detections

        # Build item list with spatial metadata
        items: list[dict] = []
        for bbox, text, conf in results:
            text = text.strip()
            if not text or conf < MIN_CONF:
                continue
            ys = [pt[1] for pt in bbox]
            xs = [pt[0] for pt in bbox]
            items.append(
                {
                    "text": text,
                    "cy": (min(ys) + max(ys)) / 2.0,
                    "height": max(ys) - min(ys),
                    "x_min": min(xs),
                }
            )

        if not items:
            return ""

        # Sort by vertical centre (top → bottom)
        items.sort(key=lambda x: x["cy"])

        # Adaptive row-merge tolerance = 60% of median line height (min 8 px)
        heights = sorted(i["height"] for i in items)
        median_h = heights[len(heights) // 2]
        row_tol = max(median_h * 0.6, 8.0)

        # Cluster into rows: compare each item to the first item in current row
        rows: list[list[dict]] = []
        current_row: list[dict] = [items[0]]
        for item in items[1:]:
            if abs(item["cy"] - current_row[0]["cy"]) <= row_tol:
                current_row.append(item)
            else:
                rows.append(sorted(current_row, key=lambda x: x["x_min"]))
                current_row = [item]
        rows.append(sorted(current_row, key=lambda x: x["x_min"]))

        # Table detection: ≥40% of rows have ≥2 columns (minimum 2 such rows)
        n_multi = sum(1 for r in rows if len(r) >= 2)
        table_mode = n_multi >= 2 and n_multi >= len(rows) * 0.4

        lines: list[str] = []
        for row in rows:
            if len(row) == 1:
                lines.append(row[0]["text"])
            elif table_mode:
                lines.append(" | ".join(c["text"] for c in row))
            else:
                lines.append(" ".join(c["text"] for c in row))

        return "\n".join(line for line in lines if line.strip())

    # Chunking

    def _split_text(
        self,
        source: str,
        doc_type: str,
        page: int,
        section: str,
        raw_text: str,
        kind: str,
    ) -> list[Chunk]:
        """
        Split raw text into overlapping chunks.

        Table-aware mode: if ≥35% of lines contain the ' | ' separator
        (produced by _ocr_image table detection), the text is treated as a
        structured table.  Newlines are preserved (not collapsed), the table
        is kept as a single chunk when it fits within 3× chunk_size, and large
        tables are split at row boundaries with a 1-row header overlap.

        Non-table text uses the original sliding-window approach with all
        whitespace collapsed to single spaces.
        """
        raw_lines = raw_text.strip().splitlines()
        pipe_lines = [l for l in raw_lines if " | " in l]
        is_table   = len(pipe_lines) >= 2 and len(raw_lines) > 0 and (
            len(pipe_lines) >= len(raw_lines) * 0.35
        )

        if is_table:
            return self._split_table(source, doc_type, page, section, raw_lines, kind)

        #  Non-table: original sliding-window 
        text = re.sub(r"\s+", " ", raw_text).strip()
        if not text:
            return []

        size    = settings.chunk_size
        overlap = settings.chunk_overlap
        chunks: list[Chunk] = []
        start = 0
        idx   = 0

        while start < len(text):
            part = text[start : start + size]
            chunks.append(
                Chunk(
                    id=f"{Path(source).stem}-p{page}-{kind}-{idx}",
                    source=source,
                    doc_type=doc_type,
                    page=page,
                    section=section,
                    text=part,
                    kind=kind,
                )
            )
            idx   += 1
            start += max(1, size - overlap)

        return chunks

    def _split_table(
        self,
        source:   str,
        doc_type: str,
        page:     int,
        section:  str,
        raw_lines: list[str],
        kind:     str,
    ) -> list[Chunk]:
        """
        Chunk a table by row boundaries, preserving structure.

        Strategy:
        - Normalise horizontal whitespace only (keep newlines).
        - If the whole table ≤ 3× chunk_size → single chunk.
        - Otherwise: split at row boundaries, prepend the header row to each
          continuation chunk so every chunk is self-contained for retrieval.
        """
        # Normalise spacing within each row but keep row separators
        rows = [re.sub(r"[ \t]+", " ", l).strip() for l in raw_lines if l.strip()]
        if not rows:
            return []

        full_text  = "\n".join(rows)
        max_single = settings.chunk_size * 3

        stem = Path(source).stem

        if len(full_text) <= max_single:
            return [Chunk(
                id=f"{stem}-p{page}-{kind}-tbl0",
                source=source, doc_type=doc_type,
                page=page, section=section,
                text=full_text, kind=kind,
            )]

        # Large table: split at row boundaries
        size        = settings.chunk_size
        header_row  = rows[0]   # repeated in each continuation for context
        chunks: list[Chunk] = []
        current: list[str]  = []
        current_len = 0
        idx         = 0

        for i, row in enumerate(rows):
            row_len = len(row) + 1  # +1 for newline
            if current_len + row_len > size and current:
                chunks.append(Chunk(
                    id=f"{stem}-p{page}-{kind}-tbl{idx}",
                    source=source, doc_type=doc_type,
                    page=page, section=section,
                    text="\n".join(current), kind=kind,
                ))
                idx += 1
                # Start next chunk with header row (context) + current row
                current     = [header_row, row] if header_row != row else [row]
                current_len = len(header_row) + row_len + 1
            else:
                current.append(row)
                current_len += row_len

        if current:
            chunks.append(Chunk(
                id=f"{stem}-p{page}-{kind}-tbl{idx}",
                source=source, doc_type=doc_type,
                page=page, section=section,
                text="\n".join(current), kind=kind,
            ))

        return chunks

    # Formula extraction

    def _extract_formula_candidates(self, text: str) -> list[str]:
        """
        Return up to 15 lines that look like mathematical expressions.
        Uses MATH_HINT_REGEX heuristic (backslash, Greek letters, operators).
        No external LaTeX parser — any line matching the heuristic is kept.
        """
        valid: list[str] = []
        for line in text.splitlines():
            s = line.strip()
            if s and MATH_HINT_REGEX.search(s):
                valid.append(s)
                if len(valid) == 15:
                    break
        return valid

# Utility used by graphrag.py

def quick_entities(text: str) -> list[str]:
    """
    Remain to backward-compatible..
    """
    entity_regex = re.compile(r"\b([A-Z][a-zA-Z0-9_\-]{2,}|[A-Z]{2,})\b")
    return sorted(set(entity_regex.findall(text)))