"""
STELLAR-RAG v4 — GraphRAG core with EHRAG hypergraph enhancement.

v4 additions over v3:
  - Rich entity extraction: ALL named entities per chunk with type classification
    (RULE, SUBJECT, AMOUNT, DATE, ORG, PERSON, CONDITION, PROCESS) in addition
    to the existing triplet extraction.
  - Co-occurrence edges: all entity pairs within the same chunk are connected
    with a 'co_occurs_with' edge, not just triplet subjects/objects.
  - Entity normalization: merges entities with the same unaccented name to
    reduce duplicate nodes.
  - EHRAG hypergraph: after graph build, constructs structural + semantic
    hyperedges via EntityHypergraph.build().
  - Hypergraph diffusion: after FAISS/BM25/graph retrieval, runs hybrid
    diffusion to compute entity weights and cluster scores, then applies
    topic-aware 3-component re-scoring.
  - All existing features retained: QDAP-S fusion, RRF, doc-type boost,
    local-subgraph PPR, parallel retrieval, BM25, contextual embedding.
"""
from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any

import networkx as nx
import numpy as np
from ollama import Client

from config import settings
from embedding import Embedder
from hypergraph import EntityHypergraph
from pdf_pipeline import Chunk
from qdap import QDAPSmall
from vector_store import FaissStore

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# Triplet extraction prompt (unchanged from v3)

TRIPLET_EXTRACTION_PROMPT = """Bạn là hệ thống trích xuất tri thức từ tài liệu đại học.
Từ đoạn văn bản, hãy trích xuất các quan hệ quan trọng dưới dạng triplet.

Tự xác định tên quan hệ phù hợp nhất với nội dung văn bản (snake_case không dấu).
Ví dụ gợi ý (không bắt buộc, tự do đặt tên phù hợp):
  co_tien_quyet, quan_ly, ap_dung_cho, quy_dinh_ve, yeu_cau,
  co_dieu_kien, thuoc_ve, lien_quan_den, thay_the, to_chuc_boi,
  bat_buoc_doi_voi, chi_dinh_boi, xac_dinh_boi, ...

Chỉ trả về JSON thuần túy, không giải thích, không markdown:
{{"triplets": [{{"subject": "...", "relation": "...", "object": "..."}}]}}

Tối đa {max_triplets} triplets. Chỉ trích xuất quan hệ rõ ràng, có trong văn bản.

Văn bản:
{text}
"""

# NEW: Rich entity extraction prompt (EHRAG enhancement)

ENTITY_EXTRACTION_PROMPT = """Từ đoạn văn bản này, trích xuất TẤT CẢ thực thể quan trọng.

Tự xác định loại thực thể (type) phù hợp với nội dung — không bắt buộc dùng nhãn cố định.
Ví dụ gợi ý: quy_dinh, hoc_phan, so_luong, thoi_han, to_chuc, dieu_kien, quy_trinh, chuc_danh

Ưu tiên thực thể có giá trị tra cứu cao: điều khoản cụ thể, số liệu, điều kiện, tên quy định.
Bỏ qua từ chung chung không có giá trị thông tin.

Chỉ trả JSON (không markdown): {{"entities": [{{"name": "tên cụ thể", "type": "loại tự xác định"}}]}}

Văn bản: {text}"""

# Edge weights for graph traversal

RELATION_WEIGHTS: dict[str, float] = {
    # Structural (fixed — these come from code, not LLM)
    "has_section":     0.8,
    "has_chunk":       0.7,
    "mentioned_in":    0.6,
    "contains_entity": 0.6,
    "co_occurs_with":  0.9,
    "is_type":         0.5,
    "defines":         1.5,
    # Semantic (suggested examples — LLM may produce other names)
    # Any relation NOT in this dict gets _DEFAULT_RELATION_WEIGHT
    "co_tien_quyet":   2.0,
    "quy_dinh_ve":     1.8,
    "yeu_cau":         1.7,
    "co_dieu_kien":    1.5,
    "ap_dung_cho":     1.5,
    "thay_the":        1.4,
    "thuoc_ve":        1.3,
    "quan_ly":         1.2,
    "to_chuc_boi":     1.1,
    "lien_quan_den":   1.0,
    "relates_to":      1.0,
}

# Dynamic fallback: unknown LLM-generated relation names get this weight
_DEFAULT_RELATION_WEIGHT = 1.0

# Vietnamese unaccent helper

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
    """Strip Vietnamese diacritics -> ASCII lowercase."""
    return text.lower().translate(_UNACCENT_MAP)

# Doc-type boost embeddings

_DOC_TYPE_DESCRIPTIONS: dict[str, str] = {
    "hoc_phi":      "học phí, chi phí học tập, tiền đóng học, miễn giảm học phí",
    "quy_che":      "quy chế đào tạo, quy định, điều khoản, chính sách giáo dục",
    "chuong_trinh": "chương trình đào tạo, kế hoạch học tập, môn học, học phần, tín chỉ",
    "lich_hoc":     "lịch học, thời khóa biểu, lịch thi, lịch giảng dạy",
    "tuyen_sinh":   "tuyển sinh, nhập học, xét tuyển, điểm chuẩn, đăng ký",
    "thong_bao":    "thông báo, thông tin cập nhật, thông cáo, thư ngỏ",
}

# Vietnamese tokenizer (BM25)

_VI_PATTERN = re.compile(
    r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợ'
    r'úùủũụưứừửữựýỳỷỹỵđa-z0-9_]+',
    re.UNICODE,
)

def _tokenize_vi(text: str) -> list[str]:
    """Tokenize Vietnamese text for BM25.
    Keeps tokens >= 2 chars OR pure digits so article numbers like '9', '15'
    are not dropped — they are critical for article-specific lookups.
    """
    return [t for t in _VI_PATTERN.findall(text.lower())
            if len(t) >= 2 or t.isdigit()]

# GraphRAG

class GraphRAG:
    """
    STELLAR-RAG v4 GraphRAG — full hybrid retrieval with EHRAG hypergraph.

    Architecture
    ------------
    Build phase:
      1. Dense FAISS index (contextual embeddings).
      2. BM25 index (raw text).
      3. Knowledge graph (triplets + rich entities + co-occurrence edges).
      4. Entity embedding index.
      5. EHRAG EntityHypergraph (structural + semantic hyperedges).

    Query phase:
      1. Parallel BM25 + Dense + Graph retrieval.
      2. QDAP-S or RRF fusion.
      3. Doc-type intent boost.
      4. [NEW] Hypergraph diffusion re-scoring (topic-aware 3-component).
    """

    def __init__(self) -> None:
        self.embedder   = Embedder()
        self.vector     = FaissStore(settings.vector_index_path, settings.vector_meta_path)
        self.graph: nx.DiGraph = nx.DiGraph()
        self.client     = Client(host=settings.ollama_host)

        self.bm25: Any            = None
        self.bm25_meta: list[dict] = []

        self.entity_names: list[str]        = []
        self.entity_vecs:  np.ndarray | None = None
        # O(1) name → index lookup; rebuilt whenever entity_names changes
        self._entity_name_to_idx: dict[str, int] = {}

        self._doc_type_vecs:  np.ndarray | None = None
        self._doc_type_names: list[str]          = []

        self._qdap_predictor: QDAPSmall | None = None

        # Online learning state — set by _qdap_fuse(), read by update_qdap_online()
        self._last_qv:         np.ndarray | None = None
        self._last_qdap_alpha: float             = 0.5

        # EHRAG hypergraph — built after graph construction
        self.hypergraph: EntityHypergraph = EntityHypergraph()

        # Cache: chunk texts keyed by chunk_id for topic scoring
        self._chunk_text_cache: dict[str, str] = {}

    # Build

    def build(self, chunks: list[Chunk]) -> None:
        """
        Build all indices from a list of Chunk objects.

        Steps:
        1. Dense FAISS (contextual or raw text embeddings).
        2. BM25 (raw text).
        3. Knowledge graph (triplets + rich entities + co-occurrence).
        4. Entity embedding index.
        5. EHRAG hypergraph.
        """
        meta = [asdict(c) for c in chunks]

        # 1. Dense FAISS
        if settings.contextual_embedding:
            embed_texts = [self._enrich_chunk_text(c) for c in chunks]
            print("Contextual embedding: ON")
        else:
            embed_texts = [c.text for c in chunks]

        vectors = self.embedder.encode(embed_texts)
        self.vector.build(vectors, meta)
        self.vector.save()
        print(f"Dense index built: {len(chunks)} chunks")

        # 2. BM25
        if HAS_BM25:
            raw_texts = [c.text for c in chunks]
            tokenized = [_tokenize_vi(t) for t in raw_texts]
            self.bm25      = BM25Okapi(tokenized, k1=settings.bm25_k1, b=settings.bm25_b)
            self.bm25_meta = meta
            _bm25_path = settings.storage / "bm25_index.pkl"
            with open(_bm25_path, "wb") as fh:
                pickle.dump({"bm25": self.bm25, "meta": self.bm25_meta}, fh)
            print(f"BM25 index built: {len(chunks)} docs")
        else:
            print("[WARN] rank_bm25 not installed — pip install rank_bm25")

        # 3. Knowledge graph
        self.graph = nx.DiGraph()
        for chunk in chunks:
            self._add_chunk_to_graph(chunk)
            # Cache chunk text for hypergraph topic scoring
            self._chunk_text_cache[chunk.id] = chunk.text

        # 4. Entity embedding index
        self._build_entity_index()

        nx.write_graphml(self.graph, settings.graph_path)
        print(
            f"Graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

        # 5. EHRAG hypergraph
        self._build_hypergraph(chunks)

    # Contextual text enrichment

    @staticmethod
    def _enrich_chunk_text(chunk: Chunk) -> str:
        """Prepend metadata context before chunk text for contextual embedding."""
        parts: list[str] = []
        if chunk.doc_type and chunk.doc_type != "general":
            parts.append(f"Loại: {chunk.doc_type}")
        if chunk.section:
            parts.append(f"Mục: {chunk.section[:100]}")
        if chunk.source:
            parts.append(f"Nguồn: {chunk.source}")
        prefix = " | ".join(parts)
        return f"[{prefix}]\n{chunk.text}" if prefix else chunk.text

    # Graph construction

    def _add_chunk_to_graph(self, chunk: Chunk) -> None:
        """
        Add a chunk to the knowledge graph.

        For each chunk:
        1. Create doc → section → chunk hierarchy nodes/edges.
        2. Extract triplets (LLM) and add entity nodes + relation edges.
        3. [NEW] Extract rich named entities with type classification.
        4. [NEW] Add co-occurrence edges between all entity pairs in chunk.
        5. [NEW] Add entity_type nodes and is_type edges.
        """
        doc_node = f"doc::{chunk.source}"
        if not self.graph.has_node(doc_node):
            self.graph.add_node(doc_node, kind="document",
                                name=chunk.source, doc_type=chunk.doc_type)

        section_node: str | None = None
        if chunk.section:
            section_node = f"section::{chunk.source}::{chunk.section[:80]}"
            if not self.graph.has_node(section_node):
                self.graph.add_node(section_node, kind="section",
                                    name=chunk.section, source=chunk.source)
            if not self.graph.has_edge(doc_node, section_node):
                self.graph.add_edge(doc_node, section_node,
                                    relation="has_section", weight=0.8)

        chunk_node = f"chunk::{chunk.id}"
        self.graph.add_node(
            chunk_node, kind="chunk",
            source=chunk.source, doc_type=chunk.doc_type,
            page=chunk.page, section=chunk.section,
            text=chunk.text[:200],
        )
        parent = section_node or doc_node
        self.graph.add_edge(parent, chunk_node, relation="has_chunk", weight=0.7)

        # Triplet extraction
        triplet_entities: set[str] = set()
        for t in self._extract_triplets(chunk.text):
            subj = (t.get("subject") or "").strip()
            rel  = (t.get("relation")  or "").strip()
            obj  = (t.get("object")   or "").strip()
            if not subj or not rel or not obj:
                continue
            subj = self._normalize_entity(subj)
            obj  = self._normalize_entity(obj)
            rw   = RELATION_WEIGHTS.get(rel, 1.0)
            subj_node = f"entity::{subj}"
            obj_node  = f"entity::{obj}"
            for en, ename in ((subj_node, subj), (obj_node, obj)):
                if not self.graph.has_node(en):
                    self.graph.add_node(en, kind="entity", name=ename)
            self.graph.add_edge(subj_node, obj_node,  relation=rel,              weight=rw)
            self.graph.add_edge(subj_node, chunk_node, relation="mentioned_in",  weight=0.6)
            self.graph.add_edge(chunk_node, subj_node, relation="contains_entity", weight=0.6)
            triplet_entities.add(subj)
            triplet_entities.add(obj)

        # Rich entity extraction (EHRAG)
        chunk_entities: list[tuple[str, str]] = []   # (name, type)
        if settings.entity_extract_enabled:
            raw_entities = self._extract_entities(chunk.text)
            for e in raw_entities[:settings.max_entities_per_chunk]:
                ename = self._normalize_entity((e.get("name") or "").strip())
                etype = (e.get("type") or "").strip().upper()
                if not ename:
                    continue
                chunk_entities.append((ename, etype))

                # Create entity node
                en = f"entity::{ename}"
                if not self.graph.has_node(en):
                    self.graph.add_node(en, kind="entity", name=ename,
                                        entity_type=etype)
                elif etype:
                    # Update type if not set
                    nd = self.graph.nodes[en]
                    if not nd.get("entity_type"):
                        self.graph.nodes[en]["entity_type"] = etype

                # entity ↔ chunk edges
                if not self.graph.has_edge(en, chunk_node):
                    self.graph.add_edge(en, chunk_node, relation="mentioned_in", weight=0.6)
                if not self.graph.has_edge(chunk_node, en):
                    self.graph.add_edge(chunk_node, en, relation="contains_entity", weight=0.6)

                # entity_type node
                if etype:
                    type_node = f"type::{etype}"
                    if not self.graph.has_node(type_node):
                        self.graph.add_node(type_node, kind="entity_type", name=etype)
                    if not self.graph.has_edge(en, type_node):
                        self.graph.add_edge(en, type_node, relation="is_type", weight=0.5)

        # Co-occurrence edges (EHRAG enhancement)
        # All entity pairs within the same chunk get a co-occurrence edge.
        # Includes both triplet entities and rich-extracted entities.
        all_chunk_entity_names = list(triplet_entities) + [n for n, _ in chunk_entities]
        # Deduplicate, preserving order
        seen: set[str] = set()
        unique_entities: list[str] = []
        for e in all_chunk_entity_names:
            if e not in seen:
                seen.add(e)
                unique_entities.append(e)

        for idx_a in range(len(unique_entities)):
            for idx_b in range(idx_a + 1, len(unique_entities)):
                ea = f"entity::{unique_entities[idx_a]}"
                eb = f"entity::{unique_entities[idx_b]}"
                if self.graph.has_node(ea) and self.graph.has_node(eb):
                    if not self.graph.has_edge(ea, eb):
                        self.graph.add_edge(ea, eb,
                                            relation="co_occurs_with", weight=0.9)
                    if not self.graph.has_edge(eb, ea):
                        self.graph.add_edge(eb, ea,
                                            relation="co_occurs_with", weight=0.9)

    def _normalize_entity(self, name: str) -> str:
        """
        Normalize entity name by stripping leading/trailing punctuation and
        collapsing whitespace.  Does NOT strip diacritics — normalization is
        text-preserving; only deduplication uses unaccented keys.
        """
        return re.sub(r"\s+", " ", name.strip().strip(".,;:\"'()[]"))

    def _build_entity_index(self) -> None:
        """Build the FAISS-style numpy entity embedding index."""
        names = [
            d.get("name", "")
            for _, d in self.graph.nodes(data=True)
            if d.get("kind") == "entity" and d.get("name")
        ]
        if not names:
            return

        # Entity normalization: deduplicate by unaccented name
        seen_norm: dict[str, str] = {}
        unique_names: list[str] = []
        for n in names:
            norm = _unaccent(n)
            if norm not in seen_norm:
                seen_norm[norm] = n
                unique_names.append(n)

        self.entity_names = unique_names
        self._entity_name_to_idx = {n: i for i, n in enumerate(unique_names)}

        # Batch embed in chunks of embed_batch_size
        batch = settings.embed_batch_size
        all_vecs: list[np.ndarray] = []
        for start in range(0, len(unique_names), batch):
            batch_names = unique_names[start:start + batch]
            try:
                vecs = self.embedder.encode(batch_names)
                all_vecs.append(vecs)
            except Exception as exc:
                print(f"[WARN] Entity embedding batch {start} failed: {exc}")
                # Zero vectors as fallback — use actual embed_dim, not hardcoded 768
                if all_vecs:
                    dim = all_vecs[0].shape[1]
                else:
                    dim = self.embedder.embed_dim
                all_vecs.append(np.zeros((len(batch_names), dim), dtype=np.float32))

        if all_vecs:
            self.entity_vecs = np.vstack(all_vecs)
            np.save(str(settings.entity_vecs_path), self.entity_vecs)
            settings.entity_names_path.write_text(
                json.dumps(unique_names, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Entity index: {len(unique_names)} entities")
        else:
            self.entity_vecs = None

    # EHRAG hypergraph construction

    def _build_hypergraph(self, chunks: list[Chunk]) -> None:
        """
        Build the EHRAG EntityHypergraph after the knowledge graph is ready.

        Steps:
        1. Collect entity_names and entity_vecs from existing entity index.
        2. Build chunk_entity_map: {chunk_id → [entity_names]} from graph.
        3. Embed all chunks in batches (uses embed_batch_size guard).
        4. Call EntityHypergraph.build() and save artefacts.
        """
        if self.entity_vecs is None or not self.entity_names:
            print("[Hypergraph] No entity vectors — skipping hypergraph build.")
            return

        # Build chunk_entity_map from graph edges
        chunk_entity_map: dict[str, list[str]] = {}
        for node, data in self.graph.nodes(data=True):
            if data.get("kind") == "chunk":
                chunk_id = str(node).replace("chunk::", "")
                entities_in_chunk: list[str] = []
                for nbr in self.graph.successors(node):
                    nd = self.graph.nodes[nbr]
                    if nd.get("kind") == "entity":
                        ename = nd.get("name", "")
                        if ename:
                            entities_in_chunk.append(ename)
                chunk_entity_map[chunk_id] = entities_in_chunk

        if not chunk_entity_map:
            print("[Hypergraph] No chunk-entity mapping found — skipping hypergraph.")
            return

        # Build ordered chunk_ids and embed chunk texts in batches
        ordered_chunk_ids = sorted(chunk_entity_map.keys())
        chunk_texts: list[str] = []
        for cid in ordered_chunk_ids:
            text = self._chunk_text_cache.get(cid, cid)
            if settings.contextual_embedding:
                # Find the chunk in our cache — use raw text for hypergraph
                chunk_texts.append(text)
            else:
                chunk_texts.append(text)

        # Batch embed chunk texts
        batch        = settings.embed_batch_size
        chunk_vecs_list: list[np.ndarray] = []
        print(f"[Hypergraph] Embedding {len(chunk_texts)} chunks in batches of {batch}...")

        for start in range(0, len(chunk_texts), batch):
            batch_texts = chunk_texts[start:start + batch]
            try:
                vecs = self.embedder.encode(batch_texts)
                chunk_vecs_list.append(vecs)
            except Exception as exc:
                print(f"[WARN] Chunk embedding batch {start} failed: {exc}")
                if chunk_vecs_list:
                    dim = chunk_vecs_list[0].shape[1]
                else:
                    dim = self.entity_vecs.shape[1]
                chunk_vecs_list.append(
                    np.zeros((len(batch_texts), dim), dtype=np.float32)
                )

        if not chunk_vecs_list:
            print("[Hypergraph] No chunk vectors produced — skipping.")
            return

        chunk_vecs = np.vstack(chunk_vecs_list)

        # Reorder chunk_entity_map to match ordered_chunk_ids
        ordered_map = {cid: chunk_entity_map[cid] for cid in ordered_chunk_ids}

        # Build hypergraph
        try:
            print(
                f"[Hypergraph] Building hypergraph  "
                f"E={len(self.entity_names)}  C={len(ordered_chunk_ids)}"
            )
            self.hypergraph.build(
                entity_names=self.entity_names,
                entity_vecs=self.entity_vecs,
                chunk_entity_map=ordered_map,
                chunk_vecs=chunk_vecs,
            )

            # Save hypergraph and chunk vecs
            settings.ensure_dirs()
            self.hypergraph.save(settings.hypergraph_path)
            np.save(str(settings.chunk_vecs_path), chunk_vecs)
            settings.chunk_ids_path.write_text(
                json.dumps(ordered_chunk_ids, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[Hypergraph] Saved to {settings.hypergraph_path}")

        except MemoryError:
            print("[Hypergraph] MemoryError during build — hypergraph disabled.")
        except Exception as exc:
            print(f"[Hypergraph] Build error: {exc} — hypergraph disabled.")

    # Load

    def load(self) -> None:
        """Load all indices from storage."""
        self.vector.load()

        # Dimension guard
        # If the stored FAISS index was built with a different embedding model
        # (different output dimension), queries will throw a FAISS shape error.
        # Catch this early with a clear message so the user knows to re-ingest.
        if self.vector.index is not None:
            faiss_dim  = self.vector.index.d
            embed_dim  = self.embedder.embed_dim
            if faiss_dim != embed_dim:
                raise RuntimeError(
                    f"[GraphRAG] Embedding dimension mismatch: "
                    f"stored index d={faiss_dim} vs. current model d={embed_dim}.\n"
                    f"The embed_model was likely changed (e.g. nomic-embed-text→BAAI/bge-m3).\n"
                    f"Re-ingest by running:  python ingest.py"
                )

        self.graph = nx.read_graphml(settings.graph_path)

        bm25_path = settings.storage / "bm25_index.pkl"
        if HAS_BM25 and bm25_path.exists():
            with open(bm25_path, "rb") as fh:
                data = pickle.load(fh)
                self.bm25      = data["bm25"]
                self.bm25_meta = data["meta"]

        if settings.entity_vecs_path.exists() and settings.entity_names_path.exists():
            self.entity_vecs  = np.load(str(settings.entity_vecs_path))
            self.entity_names = json.loads(
                settings.entity_names_path.read_text(encoding="utf-8")
            )
            self._entity_name_to_idx = {n: i for i, n in enumerate(self.entity_names)}

        # Load hypergraph artefacts
        if self.hypergraph.exists(settings.hypergraph_path):
            loaded = self.hypergraph.load(settings.hypergraph_path)
            if loaded:
                # Load chunk vecs if not already in hypergraph
                if (self.hypergraph.chunk_vecs is None
                        and settings.chunk_vecs_path.exists()):
                    self.hypergraph.chunk_vecs = np.load(str(settings.chunk_vecs_path))
                print(
                    f"[Hypergraph] Loaded  K={self.hypergraph.n_clusters}  "
                    f"E={len(self.hypergraph.entity_names)}"
                )
            else:
                print("[Hypergraph] Load failed — topic scoring disabled.")
        else:
            print("[Hypergraph] No saved hypergraph found — topic scoring disabled.")

        # Load chunk text cache for topic scoring (from graphml node data)
        for node, data in self.graph.nodes(data=True):
            if data.get("kind") == "chunk":
                cid  = str(node).replace("chunk::", "")
                text = data.get("text", "")
                if cid and text:
                    self._chunk_text_cache[cid] = text

    def exists(self) -> bool:
        """Return True if the minimum required indices exist."""
        return self.vector.exists() and settings.graph_path.exists()

    # Query batch

    def query_batch(
        self,
        questions:  list[str],
        k:          int,
        hops:       int,
        use_graph:  bool,
        hyde_query: str | None = None,
    ) -> dict[str, list[Any]]:
        """
        Encode all sub-queries in one embedder call and run retrieval + hypergraph
        re-scoring for each.
        """
        if not questions:
            return {"dense_hits": [], "graph_hits": []}

        embed_texts: list[str] = [
            (hyde_query if (i == 0 and hyde_query) else q)
            for i, q in enumerate(questions)
        ]
        all_vecs = self.embedder.encode(embed_texts)

        all_dense: list[dict] = []
        all_graph: list[dict] = []

        for i, q in enumerate(questions):
            qv = all_vecs[i : i + 1]

            if settings.parallel_retrieval:
                d_hits, b_hits, g_hits = self._retrieve_parallel(q, qv, k, hops, use_graph)
            else:
                d_hits = self._dense_search(qv, k * 2)
                b_hits = self._bm25_search(q,  k * 2)
                g_hits = self._graph_retrieve(q, qv, k, hops) if use_graph else []

            for h in g_hits:
                h.setdefault("id",    h.get("chunk_id", ""))
                h.setdefault("text",  h.get("text_preview", ""))
                h.setdefault("score", h.get("graph_score", 0.0))

            fused = self._fuse(d_hits, b_hits, g_hits, qv=qv)
            fused = self._doc_type_boost(fused, q)
            fused = self._hypergraph_rescore(fused, qv=qv)

            all_dense.extend(fused[:k])
            all_graph.extend(g_hits)

        return {"dense_hits": all_dense, "graph_hits": all_graph}

    # Query — single query entry point

    def query(
        self,
        question:   str,
        k:          int | None = None,
        hops:       int | None = None,
        use_graph:  bool = True,
        hyde_query: str | None = None,
    ) -> dict[str, Any]:
        """
        Full hybrid retrieval with EHRAG topic-aware re-scoring.

        Args:
            question:   Raw query text.
            k:          Number of final results.
            hops:       Graph traversal depth.
            use_graph:  Whether to run graph retrieval.
            hyde_query: HyDE-augmented query for dense embedding.
        """
        k    = k    or settings.top_k
        hops = hops if hops is not None else settings.graph_hops

        q_text = (question.get("query", str(question))
                  if isinstance(question, dict) else str(question))

        embed_text = hyde_query if hyde_query else q_text
        qv = self.embedder.encode([embed_text])

        if settings.parallel_retrieval:
            dense_hits, bm25_hits, graph_hits = self._retrieve_parallel(
                q_text, qv, k, hops, use_graph
            )
        else:
            dense_hits = self._dense_search(qv, k * 2)
            bm25_hits  = self._bm25_search(q_text, k * 2)
            graph_hits = self._graph_retrieve(q_text, qv, k, hops) if use_graph else []

        for h in graph_hits:
            h.setdefault("id",    h.get("chunk_id", ""))
            h.setdefault("text",  h.get("text_preview", ""))
            h.setdefault("score", h.get("graph_score", 0.0))

        fused = self._fuse(dense_hits, bm25_hits, graph_hits, qv=qv)
        fused = self._doc_type_boost(fused, q_text)
        # EHRAG topic-aware re-scoring
        fused = self._hypergraph_rescore(fused, qv=qv)

        return {
            "dense_hits": fused[:k],
            "graph_hits": graph_hits,
        }

    # EHRAG hypergraph re-scoring

    def _hypergraph_rescore(
        self,
        hits: list[dict],
        qv:   np.ndarray | None = None,
    ) -> list[dict]:
        """
        Apply EHRAG hybrid diffusion + topic-aware 3-component re-scoring.

        If the hypergraph is not available or an error occurs, returns hits
        unchanged (fail-open).

        Args:
            hits: Fused hit dicts (must have 'score' field).
            qv:   Query vector, shape (1, d).

        Returns:
            Re-scored hits, sorted by new score descending.
        """
        if not self.hypergraph.is_built() or qv is None or not hits:
            return hits

        try:
            # Build seed entity scores from top dense hits via entity linking
            seed_entity_scores: dict[str, float] = {}
            if self.entity_vecs is not None and self.entity_names:
                linked_nodes = self._entity_link_embedding(qv, top_k=8)
                # Use cached O(1) dict (built at load/ingest time)
                _name_to_idx = self._entity_name_to_idx or {
                    n: i for i, n in enumerate(self.entity_names)
                }
                for node in linked_nodes:
                    ename = node.replace("entity::", "")
                    idx = _name_to_idx.get(ename)
                    if idx is not None:
                        sim = float(self.entity_vecs[idx] @ qv.squeeze())
                        seed_entity_scores[ename] = max(0.0, sim)

            if not seed_entity_scores:
                return hits

            # Run diffusion
            entity_weights, cluster_scores = self.hypergraph.diffuse(
                query_vec=qv.squeeze(),
                seed_entity_scores=seed_entity_scores,
                chunk_vecs=self.hypergraph.chunk_vecs,
            )

            if not entity_weights and not cluster_scores:
                return hits

            # Apply topic-aware scoring
            rescored = self.hypergraph.topic_score_chunks(
                hits=hits,
                entity_weights=entity_weights,
                cluster_scores=cluster_scores,
            )
            return rescored

        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug(
                f"[GraphRAG] Hypergraph rescore failed: {exc}"
            )
            return hits

    # Parallel retrieval

    def _retrieve_parallel(
        self,
        q_text: str,
        qv: np.ndarray,
        k: int,
        hops: int,
        use_graph: bool,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Submit BM25, Dense, and Graph searches concurrently."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_dense = executor.submit(self._dense_search,  qv,     k * 2)
            f_bm25  = executor.submit(self._bm25_search,   q_text, k * 2)
            f_graph = (
                executor.submit(self._graph_retrieve, q_text, qv, k, hops)
                if use_graph else None
            )

            dense_hits = f_dense.result()
            bm25_hits  = f_bm25.result()
            graph_hits = f_graph.result() if f_graph is not None else []

        return dense_hits, bm25_hits, graph_hits

    # Individual search methods

    def _dense_search(self, qv: np.ndarray, k: int) -> list[dict]:
        hits = self.vector.search(qv, k=k)
        for h in hits:
            h["retrieval_type"] = "dense"
        return hits

    def _bm25_search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        if not self.bm25 or not self.bm25_meta:
            return []
        tokens = _tokenize_vi(query)
        if not tokens:
            return []
        scores  = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_idx:
            if float(scores[idx]) <= 0 or idx >= len(self.bm25_meta):
                continue
            item = dict(self.bm25_meta[idx])
            item["score"]          = float(scores[idx])
            item["retrieval_type"] = "bm25"
            results.append(item)
        return results

    # RRF fusion

    @staticmethod
    def _rrf_fuse(
        hit_lists: list[list[dict]],
        id_key:    str = "id",
        rrf_k:     int = 60,
    ) -> list[dict]:
        """RRF(d) = Σ_i  1 / (rrf_k + rank_i(d))"""
        rrf_scores: dict[str, float] = defaultdict(float)
        registry:   dict[str, dict]  = {}

        for hit_list in hit_lists:
            for rank, item in enumerate(hit_list):
                doc_id = str(item.get(id_key) or item.get("text", "")[:60])
                rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)
                if doc_id not in registry:
                    registry[doc_id] = item

        fused = []
        for doc_id in sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True):
            item               = dict(registry[doc_id])
            item["score"]      = rrf_scores[doc_id]
            item["rrf_score"]  = rrf_scores[doc_id]
            fused.append(item)
        return fused

    # Unified fusion dispatcher

    def _fuse(
        self,
        dense_hits: list[dict],
        bm25_hits:  list[dict],
        graph_hits: list[dict],
        qv:         np.ndarray | None = None,
        id_key:     str = "id",
    ) -> list[dict]:
        """Dispatch to QDAP-S or RRF fusion."""
        non_empty = [l for l in [dense_hits, bm25_hits, graph_hits] if l]

        if settings.fusion_method == "qdap_s" and qv is not None and len(non_empty) >= 2:
            return self._qdap_fuse(dense_hits, bm25_hits, graph_hits, qv, id_key)

        if len(non_empty) > 1:
            return self._rrf_fuse(non_empty, id_key=id_key, rrf_k=settings.rrf_k)
        return dense_hits or bm25_hits or graph_hits or []

    # QDAP-S fusion

    @staticmethod
    def _minmax_normalize(score_dict: dict[str, float]) -> dict[str, float]:
        """Min-max normalise a {doc_id: score} mapping to [0, 1]."""
        if not score_dict:
            return {}
        vals  = list(score_dict.values())
        s_min = min(vals)
        s_max = max(vals)
        if s_max == s_min:
            return {k: 0.5 for k in score_dict}
        span = s_max - s_min
        return {k: (v - s_min) / span for k, v in score_dict.items()}

    def _qdap_fuse(
        self,
        dense_hits: list[dict],
        bm25_hits:  list[dict],
        graph_hits: list[dict],
        qv:         np.ndarray,
        id_key:     str = "id",
    ) -> list[dict]:
        """QDAP-S hybrid fusion (dense × BM25) + graph blend."""
        if self._qdap_predictor is None:
            embed_dim = int(qv.shape[-1])
            model_path = str(settings.qdap_model_path)
            self._qdap_predictor = QDAPSmall(
                embed_dim=embed_dim, model_path=model_path
            )
            print(
                f"[QDAP-S] Initialised  embed_dim={embed_dim}  "
                f"trained={self._qdap_predictor.is_trained}"
            )

        alpha: float = self._qdap_predictor.predict_alpha(qv)

        # Persist for online learning (read by update_qdap_online)
        self._last_qv         = qv
        self._last_qdap_alpha = alpha

        registry:     dict[str, dict]  = {}
        dense_scores: dict[str, float] = {}
        bm25_scores:  dict[str, float] = {}
        graph_scores: dict[str, float] = {}

        for item in dense_hits:
            doc_id = str(item.get(id_key) or item.get("text", "")[:60])
            s = item.get("score", 0.0)
            if s > dense_scores.get(doc_id, -1.0):
                dense_scores[doc_id] = s
                registry[doc_id]     = item

        for item in bm25_hits:
            doc_id = str(item.get(id_key) or item.get("text", "")[:60])
            s = item.get("score", 0.0)
            if s > bm25_scores.get(doc_id, -1.0):
                bm25_scores[doc_id] = s
                if doc_id not in registry:
                    registry[doc_id] = item

        for item in graph_hits:
            doc_id = str(item.get(id_key) or item.get("chunk_id", "") or item.get("text", "")[:60])
            s = item.get("score", item.get("graph_score", 0.0))
            if s > graph_scores.get(doc_id, -1.0):
                graph_scores[doc_id] = s
                if doc_id not in registry:
                    registry[doc_id] = item

        all_ids = set(dense_scores) | set(bm25_scores) | set(graph_scores)
        if not all_ids:
            return []

        dense_n = self._minmax_normalize(dense_scores)
        bm25_n  = self._minmax_normalize(bm25_scores)
        graph_n = self._minmax_normalize(graph_scores)

        w_g = settings.qdap_graph_weight if graph_scores else 0.0
        w_d = 1.0 - w_g

        results: list[dict] = []
        for doc_id in all_ids:
            s_d = dense_n.get(doc_id, 0.0)
            s_b = bm25_n.get(doc_id, 0.0)
            s_g = graph_n.get(doc_id, 0.0)

            if doc_id in dense_n or doc_id in bm25_n:
                s_db = alpha * s_d + (1.0 - alpha) * s_b
            else:
                s_db = 0.0

            hybrid = w_d * s_db + w_g * s_g

            item = dict(registry[doc_id])
            item["score"]            = round(float(hybrid), 6)
            item["qdap_alpha"]       = round(float(alpha),  4)
            item["retrieval_type"]   = item.get("retrieval_type", "qdap_s")
            results.append(item)

        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results

    # Doc-type intent boost

    def _doc_type_boost(self, hits: list[dict], query: str) -> list[dict]:
        """Multiply score by boost factor when query intent matches doc_type."""
        boost = settings.doc_type_boost

        if self._doc_type_vecs is None:
            try:
                descs = list(_DOC_TYPE_DESCRIPTIONS.values())
                self._doc_type_names = list(_DOC_TYPE_DESCRIPTIONS.keys())
                self._doc_type_vecs  = self.embedder.encode(descs)
            except Exception:
                return hits

        try:
            qv   = self.embedder.encode([query])
            sims = (self._doc_type_vecs @ qv.squeeze())
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim < 0.40:
                return hits
            best_dt = self._doc_type_names[best_idx]
        except Exception:
            return hits

        for item in hits:
            if item.get("doc_type", "general") == best_dt:
                item["score"] = item.get("score", 0.0) * boost

        hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return hits

    # Entity linking (embedding similarity)

    def _entity_link_embedding(self, qv: np.ndarray, top_k: int = 8) -> list[str]:
        """Return top-k entity node names linked to query via embedding similarity."""
        if self.entity_vecs is None or not self.entity_names:
            return []
        sims    = self.entity_vecs @ qv.squeeze()
        top_idx = np.argsort(sims)[::-1][:top_k]
        thr     = settings.entity_link_threshold
        linked: list[str] = []
        for idx in top_idx:
            if float(sims[idx]) < thr:
                break
            node = f"entity::{self.entity_names[idx]}"
            if self.graph.has_node(node):
                linked.append(node)
        return linked

    # Graph retrieval — entity linking + ego-graph + local PPR

    def _graph_retrieve(
        self,
        question: str,
        qv:       np.ndarray,
        seed_k:   int = 5,
        hops:     int = 2,
    ) -> list[dict[str, Any]]:
        """
        Graph-based retrieval: entity linking → ego-graph extraction → local PPR.

        The ego-graph extraction (radius ≤ 2 hops from seed entities) replaces
        the raw BFS used in v3, aligning with HybGRAG's recommendation.
        """
        # 1. Entity linking via embedding similarity
        seed_nodes = self._entity_link_embedding(qv, top_k=seed_k * 2)

        # Substring fallback
        if not seed_nodes:
            q_lower = question.lower()
            for n, d in self.graph.nodes(data=True):
                if d.get("kind") == "entity":
                    name = str(d.get("name", "")).lower()
                    if len(name) >= 3 and name in q_lower:
                        seed_nodes.append(n)

        # Section keyword fallback
        if not seed_nodes:
            q_words = set(re.findall(r"\w+", question.lower()))
            for n, d in self.graph.nodes(data=True):
                if d.get("kind") == "section":
                    sw = set(re.findall(r"\w+", str(d.get("name", "")).lower()))
                    if len(sw & q_words) >= 2:
                        seed_nodes.append(n)

        if not seed_nodes:
            return []

        seed_nodes = seed_nodes[:seed_k]

        # 2. Local-subgraph PPR
        chunk_scores = self._ppr_local(seed_nodes, hops=hops, top_k=seed_k * 3)

        # Fallback: weighted BFS
        if not chunk_scores:
            chunk_scores = self._weighted_bfs(seed_nodes, hops=hops, top_k=seed_k * 3)

        # 3. Build result dicts
        results: list[dict[str, Any]] = []
        for node, score in chunk_scores:
            data     = dict(self.graph.nodes[node])
            path_str = self._find_relation_path(seed_nodes, node)
            results.append({
                "chunk_id":      node.replace("chunk::", ""),
                "source":        data.get("source", ""),
                "doc_type":      data.get("doc_type", ""),
                "page":          data.get("page", ""),
                "section":       data.get("section", ""),
                "text_preview":  data.get("text", ""),
                "relation_path": path_str,
                "graph_score":   score,
            })
        return results

    # Local-subgraph Personalised PageRank

    def _ppr_local(
        self,
        seed_nodes: list[str],
        hops:       int   = 2,
        top_k:      int   = 15,
        alpha:      float = 0.85,
    ) -> list[tuple[str, float]]:
        """Run PPR on a local subgraph around seed entities."""
        valid_seeds = [n for n in seed_nodes if self.graph.has_node(n)]
        if not valid_seeds:
            return []

        local: set[str]    = set(valid_seeds)
        frontier: set[str] = set(valid_seeds)
        cap = settings.ppr_max_subgraph

        for _ in range(hops + 1):
            if len(local) >= cap:
                break
            nxt: set[str] = set()
            for n in frontier:
                nxt.update(self.graph.successors(n))
                nxt.update(self.graph.predecessors(n))
            nxt -= local
            nxt = set(list(nxt)[: cap - len(local)])
            local.update(nxt)
            frontier = nxt

        subgraph = self.graph.subgraph(local)
        personal = {
            n: 1.0 / len(valid_seeds)
            for n in valid_seeds
            if n in subgraph
        }
        if not personal:
            return []

        try:
            ppr = nx.pagerank(
                subgraph,
                alpha=alpha,
                personalization=personal,
                max_iter=50,
                tol=1e-5,
                weight="weight",
            )
        except Exception:
            return []

        scores = [(n, s) for n, s in ppr.items() if str(n).startswith("chunk::")]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # Weighted BFS (PPR fallback)

    def _weighted_bfs(
        self,
        seed_nodes: list[str],
        hops:       int = 2,
        top_k:      int = 15,
    ) -> list[tuple[str, float]]:
        """Weighted BFS traversal as fallback when PPR produces no results."""
        node_scores: dict[str, float] = {}
        frontier: list[tuple[str, float]] = [
            (n, 1.0) for n in seed_nodes if self.graph.has_node(n)
        ]
        visited: set[str] = {n for n, _ in frontier}

        for hop in range(hops):
            decay = 0.7 ** (hop + 1)
            nxt: list[tuple[str, float]] = []
            for node, pw in frontier:
                for nbr in self.graph.successors(node):
                    ed  = self.graph.get_edge_data(node, nbr) or {}
                    rw  = RELATION_WEIGHTS.get(ed.get("relation", ""), _DEFAULT_RELATION_WEIGHT)
                    s   = pw * rw * decay
                    node_scores[nbr] = node_scores.get(nbr, 0.0) + s
                    if nbr not in visited:
                        visited.add(nbr)
                        nxt.append((nbr, s))
            frontier = nxt

        scores = [
            (n, s) for n, s in node_scores.items() if str(n).startswith("chunk::")
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # LLM triplet extraction

    def _extract_triplets(self, text: str) -> list[dict[str, str]]:
        """Extract knowledge triplets from text using LLM."""
        prompt = TRIPLET_EXTRACTION_PROMPT.format(
            max_triplets=settings.max_triplets_per_chunk,
            text=text[:1200],
        )
        try:
            resp = self.client.chat(
                model=settings.ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp["message"]["content"].strip()
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(raw).get("triplets", [])
        except Exception:
            return []

    # NEW: Rich entity extraction (EHRAG)

    def _extract_entities(self, text: str) -> list[dict[str, str]]:
        """
        Extract ALL named entities with type classification using LLM.

        Entity types: RULE, SUBJECT, AMOUNT, DATE, ORG, PERSON, CONDITION, PROCESS.
        Returns list of {name, type} dicts.  Returns [] on any error.
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:1200])
        try:
            resp = self.client.chat(
                model=settings.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            raw = resp["message"]["content"].strip()
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(raw)
            return data.get("entities", [])
        except Exception:
            return []

    # Verbalized relation paths (for critic context)

    def _find_relation_path(self, seed_nodes: list[str], target: str) -> str:
        """Find and format the shortest relation path from any seed to target."""
        for seed in seed_nodes:
            try:
                path = nx.shortest_path(self.graph, source=seed, target=target)
                labels: list[str] = []
                for i in range(len(path) - 1):
                    ed   = self.graph.get_edge_data(path[i], path[i + 1]) or {}
                    rel  = ed.get("relation", "→")
                    name = self.graph.nodes[path[i]].get("name", path[i])
                    labels.append(f"{name} --[{rel}]-->")
                return " ".join(labels)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        return ""

    # Online QDAP-S learning

    def update_qdap_online(self, reward: float) -> None:
        """
        Perform one REINFORCE update on the QDAP-S predictor using the
        last query vector and predicted α.

        Called from Agent.update_qdap_feedback() after the user rates an answer.

        Args:
            reward: Scalar in [-1, +1].
                    Derived from user rating r ∈ {1..5}  as  (r - 3) / 2.0.
                    +1 = perfect answer → reinforce the α that was used.
                    -1 = wrong answer   → push α toward neutral 0.5.
                     0 = neutral rating → no-op.
        """
        if self._qdap_predictor is None or self._last_qv is None:
            return   # no query has been processed yet — nothing to update

        self._qdap_predictor.update_online(
            query_embedding=self._last_qv,
            alpha_used=self._last_qdap_alpha,
            reward=reward,
        )

        # Persist updated weights after every feedback so progress survives restarts
        try:
            self._qdap_predictor.save(str(settings.qdap_model_path))
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                f"[QDAP-S] Could not save updated weights: {exc}"
            )

    def verbalize_graph_paths(self, graph_hits: list[dict]) -> str:
        """
        Generate verbalized reasoning paths for the critic validator.

        Format: "{entity} --[relation]--> {entity} --[relation]--> ..."

        Args:
            graph_hits: List of graph hit dicts (should have 'relation_path' field).

        Returns:
            Multi-line string of reasoning paths, or empty string if none available.
        """
        paths: list[str] = []
        for h in graph_hits:
            rp = h.get("relation_path", "")
            if rp and rp.strip():
                paths.append(rp.strip())
        return "\n".join(paths[:5])
