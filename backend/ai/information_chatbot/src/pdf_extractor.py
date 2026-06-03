"""PDF Extractor v4 - production chunking for Vietnamese legal documents."""
from __future__ import annotations
import logging, re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TARGET_CHARS   = 600
MAX_CHARS      = 850
MIN_CHARS      = 300
OVERLAP_SENTS  = 2
_MIN_CHARS_PER_PAGE = 20
_TABLE_RE = re.compile(r"Tiet \d+:|^\d{1,2}[gh]\d{2}", re.MULTILINE)
_DIEU_RE  = re.compile(
    r"(?:^|\n)\s*(?:Dieu|DIEU|Điều|ĐIỀU|Ðiều)\s+(\d+)[.:]\s*([^\n]{0,120})",
    re.MULTILINE,
)
_NOISE_PATTERNS = [
    re.compile(r"KT\.\s*HIE[^U]?U TR", re.IGNORECASE),
    re.compile(r"PHO HIEU TR", re.IGNORECASE),
    re.compile(r"Noi nhan:", re.IGNORECASE),
    re.compile(r"TRUONG PHONG", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^[-=\s]+$"),
]

@dataclass
class Chunk:
    text:           str
    source:         str
    page:           int
    article_number: str  = ""
    article_name:   str  = ""
    section:        str  = ""
    doc_type:       str  = "general"
    chunk_index:    int  = 0
    metadata:       dict = field(default_factory=dict)

class PDFExtractor:
    def __init__(self, ocr_langs=None, target_chars=TARGET_CHARS,
                 max_chars=MAX_CHARS, min_chars=MIN_CHARS, overlap_sents=OVERLAP_SENTS):
        self.ocr_langs     = ocr_langs or ["vi", "en"]
        self.target_chars  = target_chars
        self.max_chars     = max_chars
        self.min_chars     = min_chars
        self.overlap_sents = overlap_sents
        self._ocr_reader   = None

    def extract(self, pdf_path: Path) -> list[Chunk]:
        import fitz
        doc  = fitz.open(str(pdf_path))
        name = pdf_path.name
        pages_text, page_map = self._extract_pages(doc, name)
        doc.close()
        full_text = "\n".join(pages_text)
        doc_type  = self._detect_doc_type(full_text, name)
        full_text = self.normalize(full_text)
        chunks    = self._chunk_document(full_text, name, doc_type, page_map)
        chunks    = self._post_process(chunks)
        logger.info(f"[PDFExtractor] {name}: {len(pages_text)}p -> {len(chunks)} chunks "
                    f"(avg {sum(len(c.text) for c in chunks)//max(len(chunks),1)}c)")
        return chunks

    def _extract_pages(self, doc, source):
        pages, scanned = [], []
        for i, page in enumerate(doc):
            txt = page.get_text("text").strip()
            if len(txt) >= _MIN_CHARS_PER_PAGE:
                pages.append(txt)
            else:
                pages.append("")
                scanned.append(i)
        if scanned:
            logger.info(f"[PDFExtractor] {source}: {len(scanned)} scanned -> EasyOCR")
            ocr = self._ocr_pages(doc, scanned)
            for i, t in zip(scanned, ocr):
                pages[i] = t
        page_map, offset = {}, 0
        for pg, txt in enumerate(pages):
            page_map[offset] = pg + 1
            offset += len(txt) + 1
        return pages, page_map

    def _ocr_pages(self, doc, page_nums):
        reader = self._get_ocr_reader()
        texts  = []
        for pn in page_nums:
            page = doc[pn]
            mat  = __import__("fitz").Matrix(150/72, 150/72)
            pix  = page.get_pixmap(matrix=mat, colorspace="rgb")
            import numpy as np
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            results = reader.readtext(arr, detail=0, paragraph=True)
            texts.append(" ".join(results))
        return texts

    def _get_ocr_reader(self):
        if self._ocr_reader is None:
            import easyocr
            logger.info("[PDFExtractor] Loading EasyOCR ...")
            self._ocr_reader = easyocr.Reader(self.ocr_langs, gpu=False, verbose=False)
        return self._ocr_reader

    @staticmethod
    def normalize(text: str) -> str:
        _VI = r"[a-zà-ǿ]"
        text = re.sub(rf"({_VI})-?\n[ \t]*({_VI})", r"\1 \2", text)
        text = re.sub(r"([,\(])\n[ \t]*", r"\1 ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(ln.rstrip() for ln in text.splitlines())
        # Fix common OCR errors
        OCR_FIXES = [
            (r"\btối da\b", "tối đa"),
            (r"\btín chi\b", "tín chỉ"),
            (r"\bnam hoc\b", "năm học"),
            (r"\bhoc ky\b", "học kỳ"),
            (r"\bsinh vien\b", "sinh viên"),
        ]
        for pat, rep in OCR_FIXES:
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)
        return text.strip()

    def _chunk_document(self, full_text, source, doc_type, page_map):
        matches = list(_DIEU_RE.finditer(full_text))
        if len(matches) >= 2:
            return self._chunk_by_articles(full_text, source, doc_type, page_map, matches)
        logger.info(f"[PDFExtractor] No article boundaries - merge-forward: {source}")
        return self._merge_forward(full_text, source, doc_type, "", page_map, 0)

    def _chunk_by_articles(self, full_text, source, doc_type, page_map, matches):
        chunks = []
        for i, m in enumerate(matches):
            art_num  = m.group(1).strip()
            art_name = (m.group(2) or "").strip().rstrip(".")
            start    = m.start()
            end      = matches[i+1].start() if i+1 < len(matches) else len(full_text)
            art_text = full_text[start:end].strip()
            if len(art_text) < 60:
                continue
            header = f"Điều {art_num}. {art_name}".strip(". ")
            page   = self._offset_to_page(start, page_map)
            if len(art_text) <= self.target_chars:
                chunks.append(Chunk(text=art_text, source=source, page=page,
                    article_number=art_num, article_name=art_name,
                    section=header, doc_type=doc_type, chunk_index=0))
            else:
                chunks.extend(self._merge_forward(art_text, source, doc_type,
                    header, page_map, start, art_num, art_name))
        return chunks

    def _merge_forward(self, text, source, doc_type, header, page_map,
                       offset, article_number="", article_name=""):
        sents = self._split_sentences(text)
        if not sents:
            return []

        header_overhead = len(header) + 1 if header else 0
        eff_target = max(200, self.target_chars - header_overhead)
        eff_max    = max(300, self.max_chars    - header_overhead)

        expanded = []
        for s in sents:
            if len(s) > eff_max:
                expanded.extend(self._hard_split(s, eff_max))
            else:
                expanded.append(s)
        sents = expanded

        is_table    = bool(_TABLE_RE.search(text))
        eff_overlap = 0 if is_table else self.overlap_sents

        raw_chunks, current, current_len = [], [], 0
        for sent in sents:
            slen = len(sent) + 1
            if current_len + slen > eff_target and current:
                raw_chunks.append(current)
                overlap     = current[-eff_overlap:] if eff_overlap else []
                current     = overlap + [sent]
                current_len = sum(len(s)+1 for s in current)
            else:
                current.append(sent)
                current_len += slen
        if current:
            raw_chunks.append(current)

        # Merge tiny chunks up into previous
        merged = []
        for rc in raw_chunks:
            body_len = sum(len(s)+1 for s in rc)
            if merged and body_len < self.min_chars:
                prev = merged[-1]
                prev_tail = set(prev[-self.overlap_sents:])
                for s in rc:
                    if s not in prev_tail:
                        prev.append(s)
            else:
                merged.append(rc)

        chunks, global_idx = [], 0
        for rc in merged:
            body       = " ".join(rc).strip()
            chunk_text = f"{header}\n{body}".strip() if header else body
            if len(chunk_text) < 60:
                continue
            # Safety net: hard-split if still oversized after header prepend
            if len(chunk_text) > self.max_chars:
                body_parts = self._hard_split(body, eff_max)
                sub_texts  = ([f"{header}\n{body_parts[0]}".strip()] +
                              [f"{header}\n{p}".strip() for p in body_parts[1:]]
                             ) if header else body_parts
            else:
                sub_texts = [chunk_text]

            for sub_text in sub_texts:
                if len(sub_text) < 60:
                    continue
                pos  = text.find(rc[0][:40])
                page = self._offset_to_page(max(0, offset + (pos if pos >= 0 else 0)), page_map)
                chunks.append(Chunk(text=sub_text, source=source, page=page,
                    article_number=article_number, article_name=article_name,
                    section=header, doc_type=doc_type, chunk_index=global_idx))
                global_idx += 1
        return chunks

    def _post_process(self, chunks):
        clean = [c for c in chunks if not self._is_noise(c.text)]
        seen, deduped = set(), []
        for c in clean:
            t   = re.sub(r"^\s*(?:\d+[.)]\s+|[a-z][)]\s+|[-]\s+)", "", c.text, flags=re.MULTILINE)
            key = re.sub(r"\s+", " ", t[:180]).strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(c)
        removed = len(chunks) - len(deduped)
        if removed:
            logger.info(f"[PDFExtractor] removed {removed} noise/dup chunks ({len(chunks)}->{len(deduped)})")
        return deduped

    @staticmethod
    def _is_noise(text):
        t = text.strip()
        if len(t) < 60:
            return True
        for pat in _NOISE_PATTERNS:
            if pat.search(t):
                return True
        alnum = sum(1 for c in t if c.isalnum())
        return alnum / max(len(t), 1) < 0.25

    @staticmethod
    def _split_sentences(text):
        text = re.sub(r"\n([-]\s)", r"\n\n\1", text)
        text = re.sub(r"\n(\d+[.)]\s)", r"\n\n\1", text)
        _SPLIT = re.compile(
            r"\n{2,}"
            r"|(?<=[.!?])\s+(?=[A-ZĐÀ-ỹ\-\d])"
            r"|(?<=\n)(?=[-]\s)"
        )
        parts = [p.strip() for p in _SPLIT.split(text) if p.strip()]
        return parts if parts else [text.strip()]

    @staticmethod
    def _hard_split(text, max_chars):
        parts = []
        while len(text) > max_chars:
            cut = max_chars
            while cut > max_chars // 2 and text[cut] not in (" ", "\n"):
                cut -= 1
            parts.append(text[:cut].strip())
            text = text[cut:].strip()
        if text:
            parts.append(text)
        return parts

    @staticmethod
    def _offset_to_page(offset, page_map):
        page = 1
        for pos, pg in sorted(page_map.items()):
            if pos <= offset:
                page = pg
            else:
                break
        return page

    def _detect_doc_type(self, text, name):
        nl, tl = name.lower(), text.lower()[:2000]
        if any(k in nl for k in ["quy-che","quy_che","quyche","qd-"]):
            return "quy_che"
        if any(k in nl for k in ["thong-bao","thongbao","tb_","_tb"]):
            return "thong_bao"
        if any(k in tl for k in ["quy chế","quyết định","điều 1.","điều 2."]):
            return "quy_che"
        if any(k in tl for k in ["thông báo","kính gửi","trân trọng"]):
            return "thong_bao"
        return "general"
