"""
STELLAR-RAG v4 Agent — EHRAG + HybGRAG hybrid pipeline.

All v3 features retained:
  FastQueryProcessor  — heuristic entity extraction, zero LLM overhead.
  HyDE                — hypothetical document embedding for complex queries.
  LRU query cache     — thread-safe, 256-entry.
  MMR diversity       — Jaccard-based redundancy removal.
  Graph relation paths — PPR paths displayed without duplicate text.
  Cross-encoder reranker — optional CE rescoring.
  Streaming generation — answer_stream() yields tokens.
  Self-RAG confidence  — quality-based retrieval expansion.
  Proportional budget  — score-weighted char allocation.

New in v4:
  HybGRAG Critic loop  — validate context sufficiency, generate corrective
                         feedback, enrich query for up to 3 refinement iters.
  Verbalized paths     — entity-relation paths passed to validator for
                         richer context evaluation.
  EHRAG integration    — graphrag.query() now runs hypergraph diffusion
                         topic-aware re-scoring automatically.
  Extended debug_info  — critic_iterations, critic_feedback fields.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generator

from ollama import Client
from llm_client import LLMClient

from config import settings
from critic import Critic
from graphrag import GraphRAG
from guardrail import InputGuardrail, LLMSafetyClassifier, OutputGuardrail
from memory import Memory
from query_expander import QueryExpander
from reranker import Reranker
from router import QueryRouter

# Vietnamese unaccent

def _build_unaccent_table() -> dict:
    _groups = [
        ("áàảãạăắằẳẵặâấầẩẫậ", "a"),
        ("éèẻẽẹêếềểễệ",         "e"),
        ("íìỉĩị",                "i"),
        ("óòỏõọôốồổỗộơớờởỡợ",   "o"),
        ("úùủũụưứừửữự",          "u"),
        ("ýỳỷỹỵ",                "y"),
        ("đ",                    "d"),
    ]
    table: dict[int, int] = {}
    for chars, rep in _groups:
        for c in chars:
            table[ord(c)]        = ord(rep)
            table[ord(c.upper())] = ord(rep)
    return table

_UNACCENT_MAP = str.maketrans(_build_unaccent_table())

def _unaccent(text: str) -> str:
    """Strip Vietnamese diacritics -> ASCII lowercase for fuzzy matching."""
    return text.lower().translate(_UNACCENT_MAP)

# Self-RAG: context quality estimator

def _estimate_context_quality(context: str, query: str) -> float:
    """
    Lightweight token-overlap heuristic (no LLM required).

    Returns 0.0–1.0; values below settings.self_rag_threshold trigger
    a retrieval expansion pass.
    """
    q_norm   = _unaccent(query)
    q_tokens = {t for t in re.findall(r"\w{3,}", q_norm)}
    if not q_tokens:
        return 1.0
    ctx_norm   = _unaccent(context[:3000])
    ctx_tokens = set(re.findall(r"\w+", ctx_norm))
    return len(q_tokens & ctx_tokens) / len(q_tokens)

# Prompts

SYSTEM_PROMPT = """Bạn là trợ lý AI hỗ trợ sinh viên tra cứu thông tin quy chế đào tạo đại học.

NGÔN NGỮ: CHỈ trả lời bằng TIẾNG VIỆT. Tuyệt đối không dùng tiếng Trung, tiếng Anh hay ngôn ngữ khác.

Nguyên tắc bắt buộc:
1. CHỈ dùng thông tin có trong TÀI LIỆU NGỮ CẢNH được cung cấp bên dưới. Không dùng kiến thức ngoài.
2. Nếu tài liệu ngữ cảnh KHÔNG có thông tin cần thiết → nói rõ: "Tài liệu không đề cập đến [X]." Không suy đoán, không bịa đặt.
3. Trả lời trực tiếp, ngắn gọn. Không lặp lại câu hỏi. Không dùng markdown header (###) cho câu hỏi đơn giản.
4. Trích dẫn trang/điều khoản khi trả lời về quy định cụ thể.
5. Ưu tiên tài liệu: quy_che > thong_bao > general.

Quy tắc ĐẶC BIỆT QUAN TRỌNG — phải tuân thủ tuyệt đối:

[PHỦĐỊNH] "KHÔNG tính", "không được tính", "không tính vào điểm trung bình", "không ảnh hưởng đến GPA"
→ trả lời ĐÚNG nghĩa phủ định, TUYỆT ĐỐI không đảo ngược.
Ví dụ: "Không tính kết quả thi vào điểm trung bình đối với: Tin học cơ sở"
→ "Tin học cơ sở KHÔNG được tính vào điểm trung bình."

[BẢNG ĐIỂM HỌC PHẦN] ← CHỈ dùng để quy đổi điểm số → điểm chữ cho TỪNG MÔN HỌC.
KHÔNG dùng bảng này cho câu hỏi về xếp loại tốt nghiệp hay học lực toàn khóa.
Bảng quy đổi chính thức (Điều 11, QD-1175):
  9.0 – 10.0  →  A+  (4.0)   |   5.0 – <6.0  →  C   (2.0)
  8.0 – <9.0  →  A   (3.5)   |   4.0 – <5.0  →  D+  (1.5)
  7.0 – <8.0  →  B+  (3.0)   |   3.0 – <4.0  →  D   (1.0)
  6.0 – <7.0  →  B   (2.5)   |   < 3.0       →  F   (0.0)
Học phần ĐẠT khi điểm ≥ 5.0 (từ C trở lên). KHÔNG ĐẠT khi < 5.0 (D+, D, F).
Lưu ý: D và D+ đều KHÔNG ĐẠT. Tra bảng trực tiếp, không suy đoán range.

[XẾP LOẠI TỐT NGHIỆP / HỌC LỰC TOÀN KHÓA] ← Dùng khi hỏi về "rank", "xếp loại",
"hạng tốt nghiệp", "học lực", "loại bằng". KHÔNG dùng bảng điểm học phần ở trên.
  Áp dụng khóa 2020 trở về trước (tính trên điểm trung bình tích lũy):
    9.0 – 10.0  →  Xuất sắc
    8.0 – <9.0  →  Giỏi
    7.0 – <8.0  →  Khá
    6.0 – <7.0  →  Trung bình khá
    5.0 – <6.0  →  Trung bình
    4.0 – <5.0  →  Yếu  (không đạt — không được công nhận tốt nghiệp)
    Dưới 4.0    →  Kém  (không đạt)
  Khóa sau 2020: dùng bảng từ tài liệu trong ngữ cảnh tra cứu.
  Ví dụ: điểm TB 6,5 (khóa 2018) → 6,5 nằm trong [6.0, 7.0) → Trung bình khá.

[SỐ TÍN CHỈ] Khi hỏi về số tín chỉ tối thiểu/tối đa trong học kỳ:
  Tối thiểu: tiên tiến/liên kết/chất lượng cao = 10; đại trà = 14 (trừ GDQP, GDTC, NNTQ).
  Tối đa: tiên tiến/liên kết/chất lượng cao = 22; đại trà = 25; học kỳ hè = 12.
  Vượt tối đa hoặc dưới tối thiểu đều cần đơn đề nghị + Khoa + Hiệu trưởng chấp thuận.

[SỐ LIỆU] Trích dẫn số liệu chính xác từ tài liệu, không làm tròn hay ước lượng.
"""

QUERY_PROCESS_PROMPT = """Phân tích câu hỏi sau của sinh viên về quy chế đào tạo đại học.

Câu hỏi: {question}

Hướng dẫn tạo sub_queries:
- Câu hỏi LIỆT KÊ / ĐẾM ("có mấy loại", "các loại", "bao nhiêu", "thể loại"):
  → Tạo 2-3 sub_queries bao gồm: câu hỏi tổng quát + câu hỏi về từng loại/nhóm chính
  Ví dụ: "Có mấy loại học phần?" → ["Các loại học phần là gì?", "Học phần bắt buộc và tự chọn là gì?"]
- Câu hỏi SO SÁNH ("khác nhau", "giống nhau", "phân biệt"):
  → Tạo 2 sub_queries, mỗi câu hỏi về 1 đối tượng
- Câu hỏi ĐIỀU KIỆN / QUY TRÌNH ("làm thế nào", "điều kiện", "khi nào"):
  → Tạo 2-3 sub_queries bao gồm điều kiện cần, bước thực hiện
- Câu hỏi ĐƠN GIẢN (1 thực thể, 1 sự kiện):
  → 1 sub_query là bản dịch formal của câu hỏi sang ngôn ngữ văn bản quy chế

Trả về JSON thuần túy, không giải thích, không markdown:
{{
  "entities": ["các thực thể chính trong câu hỏi"],
  "sub_queries": ["các câu hỏi con theo hướng dẫn trên"],
  "expanded_terms": ["thuật ngữ quy chế/văn bản tương đương với từ trong câu hỏi"]
}}
"""

HYDE_PROMPT = """Hãy viết 2-3 câu ngắn như thể đây là đoạn văn trong tài liệu đại học \
chứa câu trả lời cho câu hỏi sau. Chỉ trả về đoạn văn, không giải thích, không tiêu đề.

Câu hỏi: {question}
Đoạn văn:"""

# Domain keywords for fast entity extraction

_DOMAIN_ENTITIES_RAW: list[str] = [
    "học phí", "tín chỉ", "học kỳ", "tốt nghiệp", "điều kiện tốt nghiệp",
    "tiên quyết", "học bổng", "chuẩn đầu ra", "gpa", "điểm trung bình",
    "luận văn", "khóa luận", "đồ án", "thực tập", "thực hành",
    "điểm chữ", "thang điểm", "điểm học phần", "điểm thi", "điểm quá trình",
    "điểm trung bình tích lũy", "xếp loại", "học lại", "cải thiện điểm",
    "học phần đạt", "học phần không đạt", "hoãn thi",
    "tin học cơ sở", "giáo dục quốc phòng", "giáo dục thể chất",
    "ngoại ngữ tổng quát", "không tính điểm", "không tính vào điểm trung bình",
    "lịch thi", "lịch học", "thời khóa biểu", "tkb", "lịch thi cuối kỳ",
    "ký túc xá", "học vụ", "đăng ký", "xét tuyển", "nhập học",
    "miễn giảm học phí", "hoãn thi", "xin nghỉ", "bảo lưu",
    "thôi học", "buộc thôi học", "nghỉ học tạm thời", "cảnh báo học tập",
    "khoa", "trường", "bộ môn", "phòng đào tạo", "phòng công tác sinh viên",
    "sinh viên", "giảng viên", "giáo sư", "tiến sĩ",
    "chương trình đào tạo", "ngành", "chuyên ngành", "môn học",
    "học phần", "loại học phần", "quy chế", "điều khoản",
]

_DOMAIN_ENTITIES: list[tuple[str, str]] = sorted(
    [(_unaccent(kw), kw) for kw in _DOMAIN_ENTITIES_RAW],
    key=lambda x: len(x[0]),
    reverse=True,
)

_CONJUNCTIONS = frozenset({
    "va ", "hay ", "hoac ", "dong thoi", "cung voi", "ngoai ra", "cung nhu",
    " and ", " or ", "ngoai ra",
})
_COMPLEX_KW_NORM = frozenset({
    "tai sao", "vi sao", "so sanh", "tong hop", "giai thich",
    "phan tich", "liet ke", "tat ca", "toan bo",
    "why", "how", "explain", "compare", "summarize", "analyze",
})

# Keywords that signal analytical intent — HyDE is beneficial here.
# Factual lookups (tables, numbers, dates) are NOT in this set.
_HYDE_ANALYTICAL_KW = frozenset({
    "tai sao", "vi sao", "giai thich", "so sanh", "tong hop", "phan tich",
    "mo ta", "tong ket", "anh huong", "tac dong", "nguyen nhan", "ket qua",
    "why", "how", "explain", "compare", "summarize", "analyze", "describe",
    "impact", "cause", "effect",
})

def _should_hyde(question: str, complexity: str) -> bool:
    """
    Return True only when HyDE is likely to improve dense retrieval.

    HyDE (hypothetical document embedding) helps when the query is analytical
    and a synthesised passage would resemble relevant document sections.

    For factual lookups (what is the rule, what is the number, which table row)
    HyDE can *hurt* by generating text that doesn't match the document's exact
    table/list format, pushing dense scores toward the wrong chunks.

    Rules:
      - Must be classified as 'complex' by router.
      - Must contain analytical intent keywords OR be very long (≥ 25 words,
        implying multi-faceted reasoning rather than a simple lookup).
    """
    if complexity != "complex":
        return False
    q_norm = _unaccent(question) + " "
    if any(kw in q_norm for kw in _HYDE_ANALYTICAL_KW):
        return True
    return len(question.split()) >= 25

# ProcessedQuery dataclass

@dataclass
class ProcessedQuery:
    original:      str
    entities:      list[str]
    sub_queries:   list[str]
    expanded_terms: list[str]

# Fast heuristic query processor (zero LLM overhead)

def _fast_process(question: str) -> ProcessedQuery:
    """Extract domain entities using normalised keyword matching."""
    q_norm = _unaccent(question)
    entities = [orig for (key, orig) in _DOMAIN_ENTITIES if key in q_norm]
    return ProcessedQuery(
        original=question,
        entities=entities[:6],
        sub_queries=[question],
        expanded_terms=[],
    )

def _needs_llm_processing(question: str) -> bool:
    """
    Decide if the extra LLM query-processing latency is worth it.

    LLM processing generates proper sub_queries and entities, enabling
    higher-quality retrieval routing. We trigger it whenever the query:
    - is long (> 22 words)
    - uses conjunctions / complex keywords
    - OR contains a known domain entity — even for short queries like
      "Có mấy loại học phần?" — so the LLM can decompose enumeration,
      comparison, and multi-aspect questions correctly.
    """
    words = question.split()
    if len(words) > 22:
        return True
    q_norm = _unaccent(question) + " "
    if any(kw in q_norm for kw in _CONJUNCTIONS):
        return True
    if any(kw in q_norm for kw in _COMPLEX_KW_NORM):
        return True
    # Trigger LLM for any query mentioning a domain entity, regardless of length.
    # Short domain queries (e.g. "How many course types are there?") need LLM decomposition to
    # produce sub_queries for enumeration / listing — fast path misses this.
    if any(key in q_norm for key, _ in _DOMAIN_ENTITIES):
        return True
    return False

# LLM-backed query processor (complex queries only)

class QueryProcessor:
    def __init__(self, client: Client) -> None:
        self.client = client

    def process(self, question: str) -> ProcessedQuery:
        """Fast path: heuristic; Slow path: LLM decomposition."""
        if not _needs_llm_processing(question):
            return _fast_process(question)

        try:
            resp = self.client.chat(
                model=settings.ollama_model,
                messages=[{
                    "role": "user",
                    "content": QUERY_PROCESS_PROMPT.format(question=question),
                }],
                options={"temperature": 0.0},
            )
            raw  = resp["message"]["content"].strip()
            raw  = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(raw)

            def _to_list(val, fallback: list) -> list:
                """LLM đôi khi trả dict/str thay vì list — normalize về list."""
                if isinstance(val, list):
                    return [str(v) for v in val if v]
                if isinstance(val, dict):
                    return list(val.values())
                if isinstance(val, str):
                    return [val] if val else fallback
                return fallback

            return ProcessedQuery(
                original=question,
                entities=_to_list(data.get("entities"), []),
                sub_queries=_to_list(data.get("sub_queries"), [question]),
                expanded_terms=_to_list(data.get("expanded_terms"), []),
            )
        except Exception:
            return _fast_process(question)

# Thread-safe LRU cache

class LRUCache:
    """256-entry LRU cache keyed on normalised query text.  Thread-safe."""

    def __init__(self, maxsize: int = 256) -> None:
        self._cache: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self._maxsize = maxsize
        self._lock    = threading.RLock()

    @staticmethod
    def _key(query: str) -> str:
        return " ".join(query.lower().split())

    def get(self, query: str) -> tuple[str, str] | None:
        k = self._key(query)
        with self._lock:
            if k not in self._cache:
                return None
            self._cache.move_to_end(k)
            return self._cache[k]

    def put(self, query: str, value: tuple[str, str]) -> None:
        k = self._key(query)
        with self._lock:
            if k in self._cache:
                self._cache.move_to_end(k)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
            self._cache[k] = value

    def invalidate(self, query: str) -> None:
        k = self._key(query)
        with self._lock:
            self._cache.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

# Sentence-level context compressor

_SENT_SPLIT = re.compile(r"(?<=[.!?。\n])\s+")

def _sent_score(sentence: str, query_tokens: set[str]) -> float:
    """
    Keyword-overlap relevance using BOTH original and unaccented tokens.
    Scores on unaccented form so Vietnamese diacritics don't block matching.
    sqrt-normalised by sentence length to avoid penalising long sentences.
    """
    # Score on unaccented tokens so diacritics are ignored ("học" matches "hoc")
    sent_norm = _unaccent(sentence)
    toks = set(re.findall(r"\w+", sent_norm))
    if not toks:
        return 0.0
    return len(toks & query_tokens) / (max(len(toks), 1) ** 0.5)

def compress_text(text: str, query: str, budget: int = 450) -> str:
    """
    Return the most query-relevant sentences within budget characters.

    Changes vs original:
    - Joins OCR soft line breaks before splitting (fixes mid-sentence \n)
    - Uses unaccented token matching (fixes Vietnamese diacritic mismatch)
    - Always keeps first sentence as structural anchor (title/header)
    - Falls back to first `budget` chars instead of discarding content
    - Minimum budget guard: never compress below 200 chars
    """
    budget = max(budget, 200)

    # Join OCR soft line breaks: letter\nletter → letter letter
    # e.g. soft line-break inside a phrase → join with space
    text = re.sub(
        r'(?<=[a-zA-ZÀ-ỹà-ỹ])\n[ \t]*(?=[a-zà-ỹ])',
        ' ', text
    )

    if len(text) <= budget:
        return text

    sents = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 10]
    if not sents:
        return text[:budget]

    # Unaccented query tokens for matching
    q_norm = _unaccent(query)
    qtoks  = set(re.findall(r"\w+", q_norm))

    # Score all sentences; give sentence[0] a small bonus (structural anchor)
    scored = sorted(
        enumerate(sents),
        key=lambda x: _sent_score(x[1], qtoks) + (0.05 if x[0] == 0 else 0.0),
        reverse=True,
    )

    # Greedily pick highest-scoring sentences within budget, preserving order
    kept: list[tuple[int, str]] = []
    used = 0
    for i, s in scored:
        needed = len(s) + 2
        if used + needed > budget:
            # If budget almost full, stop rather than skipping too many
            if used >= budget * 0.6:
                break
            continue
        kept.append((i, s))
        used += needed

    if not kept:
        return text[:budget]

    kept.sort(key=lambda x: x[0])
    return ". ".join(s for _, s in kept)

# Organizer — context assembly with MMR diversity

class Organizer:

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _mmr_select(candidates: list[dict], k: int, lambda_: float = 0.7) -> list[dict]:
        """Maximal Marginal Relevance with Jaccard redundancy."""
        if len(candidates) <= k:
            return candidates

        token_sets = [
            set(re.findall(r"\w+",
                (c.get("text") or c.get("text_preview") or "").lower()))
            for c in candidates
        ]

        selected:  list[int] = []
        remaining: list[int] = list(range(len(candidates)))

        while len(selected) < k and remaining:
            if not selected:
                best = max(remaining, key=lambda i: candidates[i].get("score", 0.0))
            else:
                def mmr_score(i: int) -> float:
                    rel = candidates[i].get("score", 0.0)
                    red = max(
                        Organizer._jaccard(token_sets[i], token_sets[j])
                        for j in selected
                    )
                    return lambda_ * rel - (1.0 - lambda_) * red

                best = max(remaining, key=mmr_score)

            selected.append(best)
            remaining.remove(best)

        return [candidates[i] for i in selected]

    def organize(
        self,
        query:           str,
        processed:       ProcessedQuery,
        dense_hits:      list[dict],
        graph_hits:      list[dict],
        memory_hits:     list[dict],
        reinforced_hits: list[dict],
        recent:          list[dict],
        critic_feedback: str = "",
    ) -> str:
        sections: list[str] = []

        # 1. Unified hybrid hits
        # Dedup by CONTENT hash (first 120 normalised chars) — not by chunk-id,
        # because different retrieval methods (dense / bm25 / qdap_s) can return
        # the same chunk with different metadata IDs, causing duplicates.
        seen_text_hashes: set[str] = set()
        unique_dense: list[dict] = []
        for item in sorted(dense_hits, key=lambda x: x.get("score", 0), reverse=True):
            raw_text   = item.get("text", "")
            # normalise: lowercase, collapse whitespace, strip OCR pipe artefacts
            norm_text  = re.sub(r"\s*\|\s*", " ", raw_text).lower()
            norm_text  = re.sub(r"\s+", " ", norm_text).strip()
            text_key   = norm_text[:120]
            if text_key not in seen_text_hashes:
                seen_text_hashes.add(text_key)
                unique_dense.append(item)

        final_dense = self._mmr_select(
            unique_dense,
            k=settings.top_k,
            lambda_=settings.mmr_lambda,
        )

        if final_dense:
            has_ce  = any("+ce" in (h.get("retrieval_type") or "") for h in final_dense)
            has_ehrag = any(h.get("ehrag_entity") is not None for h in final_dense)
            label = "Hybrid: Dense + BM25 + Graph → QDAP-S + Doc-boost + MMR"
            if has_ce:
                label += " + CE-rerank"
            if has_ehrag:
                label += " + EHRAG-topic"
            sections.append(f"== Tài liệu liên quan ({label}) ==")

            total_score = sum(h.get("score", 0.0) for h in final_dense) or 1.0
            n_chunks    = len(final_dense)
            total_chars = 0

            for item in final_dense:
                if total_chars >= settings.max_context_chars:
                    break
                weight       = item.get("score", 0.0) / total_score
                chunk_budget = int(settings.max_chars_per_chunk * weight * n_chunks)
                chunk_budget = max(
                    settings.min_chars_per_chunk,
                    min(settings.max_chars_per_chunk, chunk_budget),
                )
                raw_chunk = item.get("text", "")
                # Strip OCR pipe artefacts (e.g. "năng | lực" → "năng lực")
                clean_chunk = re.sub(r"\s*\|\s*", " ", raw_chunk)
                clean_chunk = re.sub(r"\s{2,}", " ", clean_chunk).strip()
                text    = compress_text(clean_chunk, query, budget=chunk_budget)
                src_tag = (
                    f"[{item.get('source', '')} | tr.{item.get('page', '')} | "
                    f"score={item.get('score', 0):.4f}]"
                )
                sections.append(f"{src_tag}\n{text}")
                total_chars += len(text)

        # 2. Graph relation paths
        graph_ids_in_dense = {
            str(item.get("id") or item.get("text", "")[:60])
            for item in final_dense
        }
        graph_paths = [
            g for g in graph_hits
            if g.get("relation_path")
            and str(g.get("chunk_id", "")) not in graph_ids_in_dense
        ]
        if graph_paths:
            sections.append("\n== Quan hệ tri thức (Knowledge Graph — PPR paths) ==")
            for g in graph_paths[:4]:
                src_tag  = (
                    f"[{g.get('source', '')} | tr.{g.get('page', '')} | "
                    f"{g.get('section', '') or 'N/A'}]"
                )
                sections.append(f"{src_tag}\nPath: {g['relation_path']}")

        # 3. Memory recall
        # Raise threshold and filter out "not mentioned" answers — these are
        # known-bad answers that would poison the LLM's reasoning if recalled.
        _POISON_PHRASES = (
            "không đề cập", "không có thông tin", "không tìm thấy",
            "không được đề cập", "không có trong tài liệu",
        )
        relevant_mem = [
            m for m in memory_hits
            if m.get("score", 0) >= 0.72
            and not any(p in m.get("content", "").lower() for p in _POISON_PHRASES)
        ]
        if relevant_mem:
            sections.append("\n== Hội thoại liên quan trước đó ==")
            for m in relevant_mem[:2]:
                sections.append(f"{m.get('role', '')}: {m.get('content', '')[:200]}")

        # 4. Reinforced answers
        if reinforced_hits:
            sections.append("\n== Câu trả lời đã xác nhận chất lượng cao ==")
            for r in reinforced_hits[:2]:
                sections.append(
                    f"Hỏi: {r.get('user_query', '')[:150]}\n"
                    f"Đáp: {r.get('assistant_answer', '')[:150]}"
                )

        # 5. Recent conversation (only used for follow-up queries)
        # Follow-up = short question with a referential pronoun to the previous turn.
        # Fully new questions do NOT need history — prevents LLM from repeating the previous answer.
        _FOLLOWUP_MARKERS = (
            "đó", "cái đó", "cái này", "nó", "vậy thì", "còn về",
            "tiếp theo", "thêm nữa", "còn gì", "ý trên", "trường hợp trên",
            "câu hỏi trên", "điều đó", "như vậy", "theo đó", "nếu vậy",
        )
        q_lower = query.lower()
        is_followup = (
            len(query.split()) < 12
            and any(m in q_lower for m in _FOLLOWUP_MARKERS)
        )
        if recent and is_followup:
            sections.append("\n== Lịch sử hội thoại gần đây ==")
            for m in recent[-2:]:   # only last 1 turn (2 messages)
                sections.append(f"{m['role']}: {m['content'][:200]}")

        # 6. [NEW] Critic feedback note
        if critic_feedback and critic_feedback.strip():
            sections.append(
                f"\n[Ghi chú tra cứu bổ sung: {critic_feedback[:200]}]"
            )

        return "\n".join(sections)

# Agent

class Agent:
    """
    STELLAR-RAG v4 Agent — full EHRAG + HybGRAG pipeline.

    Key additions over v3:
    - Critic validation loop (HybGRAG): up to 3 retrieval-refinement iterations.
    - EHRAG hypergraph scoring: automatic via GraphRAG.query().
    - Extended debug_info: critic_iterations, critic_feedback.
    """

    def __init__(self) -> None:
        # Use unified LLMClient (supports Ollama + Cloud LLM based on LLM_BACKEND)
        self.llm_client      = LLMClient()
        # Keep self.client as Ollama for backward-compat with QueryProcessor
        self.client          = Client(host=settings.ollama_host)
        self.graphrag        = GraphRAG()
        self.memory          = Memory()
        self.query_proc      = QueryProcessor(self.client)
        self.router          = QueryRouter()
        self.organizer       = Organizer()
        self.cache           = LRUCache(maxsize=settings.query_cache_size)

        _classifier = (
            LLMSafetyClassifier(self.client)
            if settings.guardrail_llm_classify
            else None
        )
        self.input_guardrail = InputGuardrail(classifier=_classifier)
        self.out_guardrail   = OutputGuardrail()
        self.query_expander  = QueryExpander(self.client)

        # HybGRAG critic — lazy-init (fast model, shares Ollama client)
        self.critic: Critic | None = (
            Critic(self.client)
            if settings.critic_enabled
            else None
        )

        # Debug/eval instrumentation
        self.debug_info: dict = {}

        if self.graphrag.exists():
            self.graphrag.load()

    # HyDE

    def _hyde_expand(self, question: str) -> str:
        """Generate a short hypothetical document passage for dense embedding."""
        if not settings.hyde_enabled:
            return question
        try:
            resp = self.client.chat(
                model=settings.ollama_model,
                messages=[{
                    "role": "user",
                    "content": HYDE_PROMPT.format(question=question),
                }],
                options={
                    "num_predict": settings.hyde_max_tokens,
                    "temperature": 0.3,
                },
            )
            hyp = resp["message"]["content"].strip()
            if len(hyp) > 20:
                return f"{question}\n{hyp}"
        except Exception:
            pass
        return question

    # Core retrieval + context assembly (DRY)

    def _retrieve_and_build_context(
        self,
        user_query: str,
        processed:  ProcessedQuery,
        top_k:      int,
        hops:       int,
        use_graph:  bool,
        hyde_query: str | None,
        critic_feedback: str = "",
    ) -> tuple[str, list[dict], list[dict]]:
        """
        Run hybrid retrieval → CE rerank → Self-RAG expansion → context assembly.
        Returns (context_str, dense_hits, graph_hits).

        Args:
            critic_feedback: Feedback from the critic commenter (appended to
                             context and used for query enrichment).
        """
        dense_hits: list[dict] = []
        graph_hits: list[dict] = []

        if self.graphrag.exists():
            if len(processed.sub_queries) > 1:
                res = self.graphrag.query_batch(
                    questions=processed.sub_queries,
                    k=top_k, hops=hops, use_graph=use_graph,
                    hyde_query=hyde_query,
                )
                dense_hits.extend(res.get("dense_hits", []))
                graph_hits.extend(res.get("graph_hits", []))
            else:
                # Fallback to user_query when sub_queries is empty (QP parse failed)
                primary_query = processed.sub_queries[0] if processed.sub_queries else user_query
                res = self.graphrag.query(
                    primary_query,
                    k=top_k, hops=hops, use_graph=use_graph,
                    hyde_query=hyde_query,
                )
                dense_hits.extend(res.get("dense_hits", []))
                if use_graph:
                    graph_hits.extend(res.get("graph_hits", []))

            # Expanded synonym terms
            for term in processed.expanded_terms[:2]:
                res = self.graphrag.query(term, k=2, hops=0, use_graph=False)
                dense_hits.extend(res.get("dense_hits", []))

            # Dedup before reranking — identical text chunks waste CE slots
            _seen_text: set[str] = set()
            deduped: list[dict] = []
            for h in dense_hits:
                key = re.sub(r"\s+", " ", h.get("text", "")).strip()[:150]
                if key not in _seen_text:
                    _seen_text.add(key)
                    deduped.append(h)
            dense_hits = deduped

            # Cross-encoder reranking
            reranker = Reranker.get()
            if reranker is not None and dense_hits:
                dense_hits = reranker.rerank(
                    user_query, dense_hits, top_k=settings.reranker_top_k
                )

        memory_hits     = self.memory.recall(user_query, k=4)
        reinforced_hits = self.memory.reinforced_recall(user_query, k=3)
        recent          = self.memory.recent(n=8)

        context = self.organizer.organize(
            query=user_query,
            processed=processed,
            dense_hits=dense_hits,
            graph_hits=graph_hits,
            memory_hits=memory_hits,
            reinforced_hits=reinforced_hits,
            recent=recent,
            critic_feedback=critic_feedback,
        )

        # Self-RAG quality check + one-shot expansion
        if (
            settings.self_rag_enabled
            and self.graphrag.exists()
            and dense_hits
        ):
            quality = _estimate_context_quality(context, user_query)
            self.debug_info["self_rag_quality"] = round(float(quality), 4)
            if quality < settings.self_rag_threshold:
                self.debug_info["self_rag_triggered"] = True
                exp_k    = min(top_k * 2, 20)
                exp_hops = min(hops + 1, 3)
                print(
                    f"[Self-RAG] quality={quality:.2f} < {settings.self_rag_threshold}"
                    f" → expanding k={exp_k} hops={exp_hops}"
                )
                exp_res = self.graphrag.query_batch(
                    questions=processed.sub_queries[:2],
                    k=exp_k, hops=exp_hops, use_graph=True,
                    hyde_query=None,
                )
                extra = exp_res.get("dense_hits", [])
                if extra:
                    reranker = Reranker.get()
                    if reranker is not None:
                        extra = reranker.rerank(
                            user_query, extra, top_k=settings.reranker_top_k
                        )
                    dense_hits = dense_hits + extra
                    graph_hits = graph_hits + exp_res.get("graph_hits", [])
                    context = self.organizer.organize(
                        query=user_query,
                        processed=processed,
                        dense_hits=dense_hits,
                        graph_hits=graph_hits,
                        memory_hits=memory_hits,
                        reinforced_hits=reinforced_hits,
                        recent=recent,
                        critic_feedback=critic_feedback,
                    )

        return context, dense_hits, graph_hits

    # HybGRAG critic loop (NEW)

    def _retrieve_with_critic(
        self,
        user_query:  str,
        processed:   ProcessedQuery,
        top_k:       int,
        hops:        int,
        use_graph:   bool,
        hyde_query:  str | None,
        complexity:  str = "simple",
    ) -> tuple[str, list[dict], list[dict]]:
        """
        Wraps _retrieve_and_build_context in the HybGRAG critic loop.

        Algorithm (max settings.critic_max_iterations):
        1. Retrieve context.
        2. Build verbalized reasoning paths from top graph hits.
        3. Critic.validate(query, context, paths)
           → True: sufficient, break and return.
           → False: generate feedback, enrich query, repeat.

        Falls back to the base retrieval context if critic is disabled,
        not available, or the index is empty.

        Args:
            user_query: Original user query.
            processed:  ProcessedQuery from query processor.
            top_k, hops, use_graph, hyde_query: Retrieval parameters.

        Returns:
            (context, dense_hits, graph_hits) — same as _retrieve_and_build_context.
        """
        query    = user_query
        feedback = ""

        for iteration in range(settings.critic_max_iterations):
            context, dense_hits, graph_hits = self._retrieve_and_build_context(
                query, processed, top_k, hops, use_graph, hyde_query,
                critic_feedback=feedback,
            )

            # Skip critic for simple queries — small model hallucinates on
            # single-entity factual lookups and only adds noise to context.
            if complexity == "simple":
                break

            # Only run critic when enabled, available, and index exists
            if not (settings.critic_enabled and self.critic and self.graphrag.exists()):
                break

            # Fast-path: if Self-RAG quality is already high enough, the context
            # is clearly sufficient — skip the LLM validator call entirely.
            # This avoids 2-3 unnecessary critic iterations for simple queries
            # where the retrieved context already covers all query tokens well.
            quality = _estimate_context_quality(context, user_query)
            if quality >= settings.critic_skip_threshold:
                self.debug_info["self_rag_quality"] = round(float(quality), 4)
                break  # context adequate — skip critic

            # Build verbalized graph paths for validator
            reasoning_paths = self._verbalize_paths(graph_hits[:3])

            if self.critic.validate(user_query, context, reasoning_paths):
                # Context is sufficient — stop iterating
                break

            # Context insufficient — generate feedback and enrich query
            feedback = self.critic.comment(user_query, context)
            if not feedback:
                break   # commenter returned empty → stop

            print(
                f"[CRITIC iter {iteration + 1}/{settings.critic_max_iterations}] "
                f"feedback: {feedback[:80]}"
            )
            self.debug_info["critic_iterations"] = iteration + 1
            self.debug_info["critic_feedback"]   = feedback

            # Enrich query for next retrieval iteration
            enriched = Critic.enrich_query(user_query, feedback)
            if enriched == query:
                break   # no change — stop to avoid infinite loop
            query = enriched

        return context, dense_hits, graph_hits

    def _verbalize_paths(self, graph_hits: list[dict]) -> str:
        """
        Build a compact verbalized reasoning paths string from graph hits.

        Uses the 'relation_path' field already computed by GraphRAG._graph_retrieve().
        Falls back gracefully if GraphRAG provides a verbalize helper.
        """
        try:
            return self.graphrag.verbalize_graph_paths(graph_hits)
        except AttributeError:
            # Fallback: collect 'relation_path' fields directly
            paths = [
                h.get("relation_path", "")
                for h in graph_hits
                if h.get("relation_path")
            ]
            return "\n".join(p for p in paths if p.strip())

    # Prompt builder (shared)

    def _build_messages(
        self,
        user_query: str,
        processed:  ProcessedQuery,
        context:    str,
    ) -> list[dict]:
        focus = ""
        if processed.entities:
            focus += f"Thực thể: {', '.join(processed.entities)}\n"
        if len(processed.sub_queries) > 1:
            focus += f"Khía cạnh: {'; '.join(processed.sub_queries)}\n"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Câu hỏi: {user_query}\n\n"
                    f"{focus}"
                    f"Ngữ cảnh:\n{context}"
                ),
            },
        ]

    # Debug printer (shared)

    def _debug_print(
        self,
        user_query:  str,
        complexity:  str,
        top_k:       int,
        hops:        int,
        hyde_query:  str | None,
        dense_hits:  list[dict],
        graph_hits:  list[dict],
        messages:    list[dict],
        reranked:    bool = False,
    ) -> None:
        ctx_chars = sum(len(m["content"]) for m in messages)
        critic_iters = self.debug_info.get("critic_iterations", 0)
        tags = [
            complexity,
            f"top_k={top_k}",
            f"hops={hops}",
            "QP=" + ("LLM" if _needs_llm_processing(user_query) else "fast"),
            "HyDE=" + ("Y" if hyde_query else "N"),
            "CE=" + ("Y" if reranked else "N"),
            f"critic={critic_iters}iter",
            f"ctx={ctx_chars}c",
            f"dense={len(dense_hits)}",
            f"graph={len(graph_hits)}",
        ]
        print(f"\n[STELLAR-RAG v4] {' | '.join(tags)}")

    # answer() — blocking

    def answer(self, user_query: str) -> tuple[str, str]:
        """
        Full blocking answer with EHRAG + HybGRAG critic loop.

        Returns (answer_str, turn_id).
        """
        turn_id = str(uuid.uuid4())

        self.debug_info = {
            "guardrail_action":  "allow",
            "guardrail_reason":  "",
            "query_complexity":  "simple",
            "expansion_variants": [user_query],
            "num_dense_hits":    0,
            "num_graph_hits":    0,
            "avg_dense_score":   0.0,
            "max_dense_score":   0.0,
            "unique_sources":    [],
            "context":           "",
            "context_length":    0,
            "self_rag_triggered": False,
            "self_rag_quality":  1.0,
            "hyde_used":         False,
            "llm_messages":      [],
            "dense_hits":        [],
            "graph_hits":        [],
            # NEW: critic fields
            "critic_iterations": 0,
            "critic_feedback":   "",
        }

        # 0a. Input guardrail
        if settings.guardrail_enabled:
            gr = self.input_guardrail.check(user_query)
            self.debug_info["guardrail_action"] = gr.action
            self.debug_info["guardrail_reason"] = gr.reason
            if gr.action == "block":
                block_msg = f"Yêu cầu bị từ chối: {gr.reason}"
                print(f"[GUARDRAIL BLOCK] {gr.reason}")
                self.memory.add("user",      user_query, turn_id=turn_id)
                self.memory.add("assistant", block_msg,  turn_id=turn_id)
                return block_msg, turn_id
            if gr.action == "warn":
                print(f"[GUARDRAIL WARN] {gr.reason}")
                if settings.guardrail_block_ood:
                    warn_msg = (
                        f"{gr.reason}\n\n"
                        "Vui lòng đặt câu hỏi về thông tin trường đại học."
                    )
                    return warn_msg, turn_id
            user_query = gr.sanitized_query

        # 0b. Cache check
        cached = self.cache.get(user_query)
        if cached:
            cached_answer, _ = cached
            print(f"[CACHE HIT] '{user_query[:60]}'")
            self.memory.add("user",      user_query,    turn_id=turn_id)
            self.memory.add("assistant", cached_answer, turn_id=turn_id)
            return cached_answer, turn_id

        # 1. Query processing
        processed = self.query_proc.process(user_query)

        # 2. Routing FIRST — drives both expansion and HyDE decisions
        complexity = self.router.classify(processed)
        self.debug_info["query_complexity"] = complexity
        params     = self.router.retrieval_params(complexity)
        top_k, hops, use_graph = params["top_k"], params["hops"], params["use_graph"]

        # 1b. Query expansion — skip for simple factual queries
        # Simple queries (single-entity factual lookups) don't benefit from
        # paraphrase variants; expansion only adds LLM latency.
        if (settings.query_expansion_enabled
                and complexity != "simple"
                and len(processed.sub_queries) == 1):
            expanded_variants = self.query_expander.expand(user_query)
            self.debug_info["expansion_variants"] = expanded_variants
            if len(expanded_variants) > 1:
                processed = ProcessedQuery(
                    original=processed.original,
                    entities=processed.entities,
                    sub_queries=expanded_variants,
                    expanded_terms=processed.expanded_terms,
                )
                print(f"[EXPAND] {len(expanded_variants)} variants")

        # 3. HyDE — only for analytically complex queries
        # HyDE helps when the query asks for reasoning/analysis/comparison.
        # For factual lookups (tables, numbers, dates) it can hurt by generating
        # a hypothetical passage that doesn't match the document's exact format.
        hyde_query: str | None = None
        if settings.hyde_enabled and _should_hyde(user_query, complexity):
            hyde_query = self._hyde_expand(user_query)
        self.debug_info["hyde_used"] = hyde_query is not None

        # 4–5. Retrieval + critic loop
        context, dense_hits, graph_hits = self._retrieve_with_critic(
            user_query, processed, top_k, hops, use_graph, hyde_query
        )

        # Populate debug_info
        scores  = [h.get("score", 0.0) for h in dense_hits if h.get("score") is not None]
        sources = list({h.get("source", "") for h in dense_hits if h.get("source")})
        self.debug_info.update({
            "num_dense_hits":  len(dense_hits),
            "num_graph_hits":  len(graph_hits),
            "avg_dense_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "max_dense_score": round(max(scores), 4) if scores else 0.0,
            "unique_sources":  sources,
            "context":         context,
            "context_length":  len(context),
            "dense_hits":      dense_hits,
            "graph_hits":      graph_hits,
        })

        # 6. Build prompt
        messages = self._build_messages(user_query, processed, context)
        self.debug_info["llm_messages"] = messages
        reranked = any("+ce" in (h.get("retrieval_type") or "") for h in dense_hits)
        self._debug_print(user_query, complexity, top_k, hops,
                          hyde_query, dense_hits, graph_hits, messages, reranked)

        # 7. LLM generation
        resp   = self.llm_client.chat(model=settings.ollama_model, messages=messages)
        answer = resp["message"]["content"]

        # 7b. Auto-retry when answer says "not mentioned" with higher top_k
        _NO_INFO_PHRASES = (
            "tài liệu không đề cập", "không có thông tin",
            "không tìm thấy", "không được đề cập", "không có trong tài liệu",
        )
        answer_lower = answer.lower()
        is_no_info   = any(ph in answer_lower for ph in _NO_INFO_PHRASES)

        if is_no_info and complexity != "complex":
            # Upgrade one complexity tier and retry once
            fb_params = self.router.fallback_params(complexity)
            fb_context, fb_dense, fb_graph = self._retrieve_with_critic(
                user_query, processed,
                fb_params["top_k"], fb_params["hops"], fb_params["use_graph"],
                hyde_query,
            )
            if fb_context.strip():
                fb_messages = self._build_messages(user_query, processed, fb_context)
                fb_resp     = self.llm_client.chat(model=settings.ollama_model, messages=fb_messages)
                fb_answer   = fb_resp["message"]["content"]
                fb_lower    = fb_answer.lower()
                # Only use retry answer if it no longer says "not mentioned"
                if not any(ph in fb_lower for ph in _NO_INFO_PHRASES):
                    answer = fb_answer
                    print(f"[RETRY] upgraded {complexity}→{fb_params} found answer")

        # 7c. Output guardrail
        if settings.guardrail_enabled and settings.guardrail_output_check and context:
            out_gr = self.out_guardrail.check(answer, context)
            if out_gr.action == "warn":
                print(f"[OUTPUT GUARDRAIL] {out_gr.reason}")

        # 8. Cache + memory
        self.cache.put(user_query, (answer, turn_id))
        self.memory.add("user",      user_query, turn_id=turn_id)
        self.memory.add("assistant", answer,     turn_id=turn_id)

        return answer, turn_id

    # answer_stream() — streaming

    def answer_stream(
        self, user_query: str
    ) -> Generator[tuple[str, bool], None, None]:
        """
        Streaming answer generator.

        Yields ``(token: str, is_final: bool)`` tuples.
        While generating: ``(token_fragment, False)``
        After last token:  ``(full_answer, True)``

        The critic loop runs before streaming begins (retrieval phase is always
        blocking).  Streaming only covers the LLM generation phase.
        """
        turn_id = str(uuid.uuid4())

        # 0a. Input guardrail
        if settings.guardrail_enabled:
            gr = self.input_guardrail.check(user_query)
            if gr.action == "block":
                block_msg = f"Yêu cầu bị từ chối: {gr.reason}"
                print(f"[GUARDRAIL BLOCK] {gr.reason}")
                self.memory.add("user",      user_query, turn_id=turn_id)
                self.memory.add("assistant", block_msg,  turn_id=turn_id)
                yield block_msg, True
                return
            if gr.action == "warn":
                print(f"[GUARDRAIL WARN] {gr.reason}")
                if settings.guardrail_block_ood:
                    warn_msg = (
                        f"{gr.reason}\n\n"
                        "Vui lòng đặt câu hỏi về thông tin trường đại học."
                    )
                    yield warn_msg, True
                    return
            user_query = gr.sanitized_query

        # 0b. Cache check
        cached = self.cache.get(user_query)
        if cached:
            cached_answer, _ = cached
            print(f"[CACHE HIT] '{user_query[:60]}'")
            self.memory.add("user",      user_query,    turn_id=turn_id)
            self.memory.add("assistant", cached_answer, turn_id=turn_id)
            yield cached_answer, True
            return

        # 1. Query processing + 2. Routing (before expansion)
        processed  = self.query_proc.process(user_query)
        complexity = self.router.classify(processed)
        params     = self.router.retrieval_params(complexity)
        top_k, hops, use_graph = params["top_k"], params["hops"], params["use_graph"]

        # 1b. Expansion — skip for simple
        if (settings.query_expansion_enabled
                and complexity != "simple"
                and len(processed.sub_queries) == 1):
            expanded_variants = self.query_expander.expand(user_query)
            if len(expanded_variants) > 1:
                processed = ProcessedQuery(
                    original=processed.original,
                    entities=processed.entities,
                    sub_queries=expanded_variants,
                    expanded_terms=processed.expanded_terms,
                )
                print(f"[EXPAND] {len(expanded_variants)} variants")

        # 3. HyDE (analytical complex only) + critic loop
        hyde_query: str | None = None
        if settings.hyde_enabled and _should_hyde(user_query, complexity):
            hyde_query = self._hyde_expand(user_query)

        self.debug_info = {"critic_iterations": 0, "critic_feedback": ""}
        context, dense_hits, graph_hits = self._retrieve_with_critic(
            user_query, processed, top_k, hops, use_graph, hyde_query
        )

        messages = self._build_messages(user_query, processed, context)
        reranked = any("+ce" in (h.get("retrieval_type") or "") for h in dense_hits)
        self._debug_print(user_query, complexity, top_k, hops,
                          hyde_query, dense_hits, graph_hits, messages, reranked)

        # 6. Streaming LLM generation
        tokens: list[str] = []
        try:
            for chunk in self.client.chat(
                model=settings.ollama_model,
                messages=messages,
                stream=True,
            ):
                token = chunk["message"]["content"]
                if token:
                    tokens.append(token)
                    yield token, False
        except Exception as exc:
            print(f"[WARN] Streaming failed ({exc}), falling back to blocking")
            resp  = self.client.chat(model=settings.ollama_model, messages=messages)
            token = resp["message"]["content"]
            tokens = [token]
            yield token, False

        full_answer = "".join(tokens)

        # 7b. Output guardrail
        if settings.guardrail_enabled and settings.guardrail_output_check and context:
            out_gr = self.out_guardrail.check(full_answer, context)
            if out_gr.action == "warn":
                print(f"[OUTPUT GUARDRAIL] {out_gr.reason}")

        # 8. Cache + memory
        self.cache.put(user_query, (full_answer, turn_id))
        self.memory.add("user",      user_query,  turn_id=turn_id)
        self.memory.add("assistant", full_answer, turn_id=turn_id)

        yield full_answer, True

    # answer_dual() — Ollama + Cloud LLM in parallel, same context

    def answer_dual(self, user_query: str) -> tuple[str, str, str]:
        """
        Chạy retrieval pipeline 1 lần, sau đó gửi cùng prompt đến
        Run Ollama AND Cloud LLM in parallel (ThreadPoolExecutor).

        Returns:
            (ollama_answer, cloud_answer, turn_id)

        Nếu một backend lỗi, trả về chuỗi mô tả lỗi cho backend đó.
        Câu trả lời Ollama được lưu vào cache + memory (primary).
        """
        turn_id = str(uuid.uuid4())

        # 0a. Input guardrail
        if settings.guardrail_enabled:
            gr = self.input_guardrail.check(user_query)
            if gr.action == "block":
                msg = f"Yêu cầu bị từ chối: {gr.reason}"
                self.memory.add("user",      user_query, turn_id=turn_id)
                self.memory.add("assistant", msg,        turn_id=turn_id)
                return msg, msg, turn_id
            if gr.action == "warn" and settings.guardrail_block_ood:
                msg = f"{gr.reason}\n\nVui lòng đặt câu hỏi về thông tin trường đại học."
                return msg, msg, turn_id
            user_query = gr.sanitized_query

        # 0b. Cache check
        cached = self.cache.get(user_query)
        if cached:
            cached_answer, _ = cached
            print(f"[CACHE HIT dual] '{user_query[:60]}'")
            self.memory.add("user",      user_query,    turn_id=turn_id)
            self.memory.add("assistant", cached_answer, turn_id=turn_id)
            return cached_answer, cached_answer, turn_id

        # 1–5. Retrieval (same as answer())
        processed  = self.query_proc.process(user_query)
        complexity = self.router.classify(processed)
        params     = self.router.retrieval_params(complexity)
        top_k, hops, use_graph = params["top_k"], params["hops"], params["use_graph"]

        if (settings.query_expansion_enabled
                and complexity != "simple"
                and len(processed.sub_queries) == 1):
            expanded = self.query_expander.expand(user_query)
            if len(expanded) > 1:
                processed = ProcessedQuery(
                    original=processed.original,
                    entities=processed.entities,
                    sub_queries=expanded,
                    expanded_terms=processed.expanded_terms,
                )

        hyde_query: str | None = None
        if settings.hyde_enabled and _should_hyde(user_query, complexity):
            hyde_query = self._hyde_expand(user_query)

        self.debug_info = {"critic_iterations": 0, "critic_feedback": ""}
        context, dense_hits, graph_hits = self._retrieve_with_critic(
            user_query, processed, top_k, hops, use_graph, hyde_query,
            complexity=complexity,
        )

        messages = self._build_messages(user_query, processed, context)
        reranked = any("+ce" in (h.get("retrieval_type") or "") for h in dense_hits)
        self._debug_print(user_query, complexity, top_k, hops,
                          hyde_query, dense_hits, graph_hits, messages, reranked)
        # Store for ?debug command
        self.debug_info["context"]      = context
        self.debug_info["llm_messages"] = messages
        self.debug_info["dense_hits"]   = dense_hits

        # 6. Dual generation (parallel)
        # Ensure both backends are initialised
        self.llm_client.ensure_dual()

        ollama_resp, cloud_resp, errors = self.llm_client.chat_dual(
            model=settings.ollama_model,
            messages=messages,
            options={"temperature": 0.0},
        )

        ollama_answer = (
            ollama_resp["message"]["content"]
            if ollama_resp is not None
            else f"[Ollama lỗi: {errors.get('ollama', 'không khả dụng')}]"
        )
        cloud_label = settings.cloud_provider.capitalize()
        cloud_answer = (
            cloud_resp.content
            if cloud_resp is not None
            else f"[{cloud_label} lỗi: {errors.get('cloud', 'không khả dụng — kiểm tra CLOUD_API_KEY')}]"
        )

        # 7. Output guardrail (applied to both answers)
        if settings.guardrail_enabled and settings.guardrail_output_check and context:
            for ans_text, label in ((ollama_answer, "Ollama"), (cloud_answer, cloud_label)):
                out_gr = self.out_guardrail.check(ans_text, context)
                if out_gr.action == "warn":
                    print(f"[OUTPUT GUARDRAIL/{label}] {out_gr.reason}")

        # 8. Cache (store Ollama answer as primary) + memory
        primary = ollama_answer if ollama_resp is not None else cloud_answer
        self.cache.put(user_query, (primary, turn_id))
        self.memory.add("user",      user_query, turn_id=turn_id)
        self.memory.add("assistant", primary,    turn_id=turn_id)

        print(f"\n[DUAL] Ollama={len(ollama_answer)}c  {cloud_label}={len(cloud_answer)}c")
        return ollama_answer, cloud_answer, turn_id

    # Online feedback — QDAP-S reinforcement

    def update_qdap_feedback(self, reward: float) -> None:
        """
        Feed a user rating back into the QDAP-S predictor.

        Args:
            reward: Float in [-1, +1], derived from a 1-5 star rating as
                    ``(rating - 3) / 2.0`` so that:
                      5 → +1.0  (reinforce the α blend that was used)
                      4 → +0.5
                      3 →  0.0  (neutral — no update)
                      2 → -0.5
                      1 → -1.0  (push toward neutral α = 0.5)

        Delegates to GraphRAG which holds the QDAP predictor instance and
        the last query vector.  No-ops silently if graphrag has no data yet.
        """
        try:
            self.graphrag.update_qdap_online(reward)
        except Exception as exc:
            print(f"[QDAP-S] Feedback update skipped: {exc}")
