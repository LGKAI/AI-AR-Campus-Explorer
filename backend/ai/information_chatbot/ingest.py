"""
STELLAR-RAG v4 — Ingest Pipeline

Improvements over v1
--------------------
* Article-boundary chunking (Dieu X.) -> no repeated headers, clean embeddings
* PyMuPDF direct extraction for digital PDFs, EasyOCR fallback for scanned
* NER pre-pass (NlpHUST/ner-vietnamese-electra-base) populates entity list locally
* LLaMA 70B (Groq) extracts relations only (entities already known from NER)
  -> ~50% fewer tokens per LLM call vs full entity+relation extraction
* TPM-guarded Cloud LLM calls: 8s gap for 70B, exponential back-off on 429
* EHRAG hypergraph rebuilt from richer graph data
* Detailed build report saved to storage/ingest_report.json

Usage
-----
    python ingest.py                    # NER entities + LLaMA 70B relations (default)
    python ingest.py --no-ner           # LLM extracts both entities and relations (old)
    python ingest.py --skip-graph       # skip all LLM graph extraction (fast regex only)
    python ingest.py --dry-run          # show what would be processed
    python ingest.py --limit N          # process only first N PDFs
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

# Path setup 
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))
os.chdir(_ROOT)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger("ingest")

from config import settings

# Main pipeline

def run(args: argparse.Namespace) -> None:
    t_start = time.time()

    #  0. Collect PDFs 
    raw_dir = _ROOT / "data" / "raw"
    pdfs    = sorted(raw_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    logger.info(f"Found {len(pdfs)} PDF(s) in {raw_dir}")
    for p in pdfs:
        logger.info(f"  {p.name}  ({p.stat().st_size // 1024} KB)")

    if args.dry_run:
        logger.info("[DRY RUN] exiting.")
        return

    #  1. Extract & chunk 
    from pdf_extractor import PDFExtractor
    extractor = PDFExtractor()
    all_chunks: list = []
    for pdf in pdfs:
        logger.info(f"Extracting: {pdf.name} …")
        chunks = extractor.extract(pdf)
        logger.info(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")
    _save_chunks_json(all_chunks)

    #  2. Embeddings + FAISS 
    logger.info("Building FAISS index …")
    embedder = _build_faiss_index(all_chunks)

    #  3. BM25 index 
    logger.info("Building BM25 index …")
    _build_bm25_index(all_chunks)

    #  4. Knowledge graph
    # Default: NER pre-pass (local) for entities + LLaMA 70B (cloud) for relations.
    # --no-ner: LLM extracts both entities and relations (original behavior).
    # --skip-graph: skip all LLM, use regex-only fast extraction.
    if not args.skip_graph:
        if settings.cloud_api_key:
            use_ner = settings.ner_enabled and not getattr(args, "no_ner", False)
            if use_ner:
                logger.info(
                    f"Building knowledge graph: NER ({settings.ner_model}) + "
                    f"LLaMA 70B relations ({settings.cloud_relation_model}) …"
                )
                graph = _build_graph_ner_llm(all_chunks)
            else:
                logger.info(
                    f"Building knowledge graph with Cloud LLM ({settings.cloud_provider}) "
                    f"[--no-ner: LLM extracts entities + relations] …"
                )
                graph = _build_graph_cloud(all_chunks)
        else:
            logger.info("No CLOUD_API_KEY — falling back to fast NER graph extraction.")
            graph = _build_graph_fast(all_chunks)
    else:
        logger.info("Skipping LLM graph extraction (--skip-graph) — using fast NER")
        graph = _build_graph_fast(all_chunks)

    _save_graph(graph)

    #  5. Entity embeddings 
    logger.info("Building entity embeddings …")
    _build_entity_embeddings(graph, embedder)

    #  6. EHRAG Hypergraph 
    logger.info("Building EHRAG hypergraph …")
    _build_hypergraph(all_chunks, embedder)

    #  7. Report 
    elapsed = time.time() - t_start
    report  = {
        "timestamp":    datetime.now().isoformat(),
        "pdfs":         [p.name for p in pdfs],
        "total_chunks": len(all_chunks),
        "elapsed_s":    round(elapsed, 1),
        "skip_graph":   args.skip_graph,
        "graph_mode":   (
            "skip"       if args.skip_graph else
            "no_ner"     if getattr(args, "no_ner", False) else
            "ner_llm70b" if settings.cloud_api_key and settings.ner_enabled else
            "fast_regex"
        ),
        "ner_model":    settings.ner_model if settings.ner_enabled else None,
        "relation_model": settings.cloud_relation_model if settings.cloud_api_key else None,
    }
    (settings.storage / "ingest_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Done in {elapsed:.1f}s. Report → storage/ingest_report.json")

# Step implementations

def _save_chunks_json(chunks) -> None:
    """Save processed chunks to data/processed/chunks.json"""
    out = []
    for c in chunks:
        out.append({
            "text":           c.text,
            "source":         c.source,
            "page":           c.page,
            "article_number": c.article_number,
            "article_name":   c.article_name,
            "section":        c.section or (
                f"Điều {c.article_number}. {c.article_name}" if c.article_number else ""
            ),
            "doc_type":       c.doc_type,
            "chunk_index":    c.chunk_index,
        })
    out_path = _ROOT / "data" / "processed" / "chunks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"  Saved chunks → {out_path}")

def _build_faiss_index(chunks) -> object:
    """Build FAISS + docs_meta from chunks. Returns the Embedder."""
    import numpy as np
    import faiss

    from embedding import Embedder
    embedder = Embedder()

    texts = [c.text for c in chunks]
    logger.info(f"  Encoding {len(texts)} chunks …")

    # Batch to avoid OOM
    batch_size = 4
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        v = embedder.encode(batch)
        vecs.append(v)
        logger.info(f"  Encoded {min(i+batch_size, len(texts))}/{len(texts)}")

    all_vecs = np.vstack(vecs).astype("float32")
    faiss.normalize_L2(all_vecs)

    dim   = all_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_vecs)

    settings.storage.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.storage / "docs.faiss"))

    meta = []
    for c in chunks:
        meta.append({
            "text":           c.text,
            "source":         c.source,
            "page":           c.page,
            "article_number": c.article_number,
            "article_name":   c.article_name,
            "section":        c.section or (
                f"Điều {c.article_number}. {c.article_name}" if c.article_number else ""
            ),
            "doc_type":       c.doc_type,
        })
    (settings.storage / "docs_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"  FAISS index: {index.ntotal} vectors, dim={dim}")
    return embedder

def _build_bm25_index(chunks) -> None:
    import re
    from rank_bm25 import BM25Okapi  # type: ignore

    def tokenize(text: str) -> list[str]:
        # Keep tokens >= 2 chars OR pure digits (e.g. "9", "15") so article
        # numbers like "Dieu 9" are not lost in BM25 lookup.
        return [t for t in re.findall(r"[\w]+", text.lower())
                if len(t) >= 2 or t.isdigit()]

    corpus = [tokenize(c.text) for c in chunks]
    bm25   = BM25Okapi(corpus, k1=settings.bm25_k1, b=settings.bm25_b)

    meta = [{"text": c.text, "source": c.source, "page": c.page,
             "section": c.section, "doc_type": c.doc_type} for c in chunks]

    out_path = settings.storage / "bm25_index.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"bm25": bm25, "meta": meta}, f)
    logger.info(f"  BM25 index: {len(corpus)} docs → {out_path.name}")

def _build_graph_ner_llm(chunks) -> "nx.DiGraph":
    """
    Build knowledge graph using a two-stage pipeline:
      Stage 1 — NER (local model, no API): extract entities from every chunk.
      Stage 2 — LLaMA 70B (Groq, cloud): extract relations between known entities.

    Token savings vs full LLM extraction:
      Before: LLM -> entities + relations (~1000 tokens/call)
      After:  NER -> entities (0 tokens)  + LLM -> relations (~500 tokens/call)
      Result: ~50% fewer API tokens, higher entity coverage, fewer hallucinations.

    Rate limits (Groq free tier, llama-3.3-70b-versatile):
      6,000 TPM -> with ~500 tok/call, safe at min_gap=8s (7-8 calls/min).
      The CloudLLMClient TPM guard enforces this automatically.
    """
    import re
    from collections import defaultdict
    import networkx as nx
    from cloud_llm_client import CloudLLMClient, recommended_min_gap
    from ner_extractor import ViNERExtractor

    RELATION_PROMPT = """\
You are a Vietnamese university regulation expert.

The following named entities were identified in this text:
{entity_list}

Text:
{text}

Extract ONLY the relations between the entities listed above.
Use snake_case relation names without diacritics (e.g. co_tien_quyet, yeu_cau, ap_dung_cho, quy_dinh_ve).
Only extract relations explicitly stated in the text — do not infer.

Return JSON only (no markdown, no explanation):
{{"relations":[{{"from":"entity_name","relation":"relation_name","to":"entity_name"}}]}}"""

    MAX_CHARS_PER_CALL = 1500   # ~375 Vietnamese tokens input

    def _parse_relations(raw: str) -> list[dict]:
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return []
        try:
            return json.loads(m.group(0)).get("relations", [])
        except Exception:
            return []

    # Stage 1: NER pre-pass (local, no API calls)
    logger.info(f"  [Stage 1] Running NER ({settings.ner_model}) on {len(chunks)} chunks ...")
    ner = ViNERExtractor(
        model_name = settings.ner_model,
        device     = settings.ner_device,
        min_score  = settings.ner_min_score,
        max_chars  = settings.ner_max_chars,
    )
    chunk_entities: dict[str, list[dict]] = ner.extract_chunks(chunks)
    ner.unload()   # free ~400 MB before cloud LLM phase

    total_ner_entities = sum(len(v) for v in chunk_entities.values())
    logger.info(
        f"  [Stage 1] Done — {total_ner_entities} entity spans across {len(chunks)} chunks"
    )

    # Stage 2: Group chunks by article, call LLaMA 70B for relations only
    ArticleKey = tuple  # (source, article_number or page)
    groups: dict[ArticleKey, list] = defaultdict(list)
    for chunk in chunks:
        art  = getattr(chunk, "article_number", "") or ""
        page = getattr(chunk, "page", 1)
        key  = (chunk.source, art if art else f"p{page}")
        groups[key].append(chunk)

    # Build chunk_id -> entities lookup (same ID formula as ViNERExtractor)
    def _chunk_id(chunk) -> str:
        return (
            f"{chunk.source}"
            f"::Điều{getattr(chunk, 'article_number', '')}"
            f"::p{chunk.page}"
            f"::i{chunk.chunk_index}"
        )

    gap = recommended_min_gap(settings.cloud_provider, settings.cloud_relation_model)
    client = CloudLLMClient(
        model     = settings.cloud_relation_model,
        min_gap_s = gap,
        max_rpm   = 12,    # 12K TPM / ~700 tok/call = 17 max → cap at 12 for headroom
    )
    graph = nx.DiGraph()
    total = len(groups)

    logger.info(
        f"  [Stage 2] {len(chunks)} chunks -> {total} article groups "
        f"via {settings.cloud_provider.capitalize()} "
        f"model={settings.cloud_relation_model} min_gap={gap}s"
    )

    for idx, (key, grp) in enumerate(groups.items(), 1):
        source   = grp[0].source
        art      = getattr(grp[0], "article_number", "") or ""
        art_name = getattr(grp[0], "article_name",   "") or ""
        page     = getattr(grp[0], "page", 1)
        label    = f"Dieu {art}" if art else f"tr.{page}"

        # Collect NER entities for all chunks in this group
        group_entities: list[dict] = []
        seen_names: set[str] = set()
        for chunk in grp:
            for ent in chunk_entities.get(_chunk_id(chunk), []):
                name = ent["name"].strip()
                if name and name.lower() not in seen_names:
                    seen_names.add(name.lower())
                    group_entities.append(ent)

        # Build article node + entity nodes from NER (no LLM needed for entities)
        art_node: str | None = None
        if art and art_name:
            art_node = f"Dieu {art}"
            graph.add_node(art_node, type="article", name=art_name,
                           page=page, source=source)

        for ent in group_entities:
            name  = ent["name"].strip()
            etype = ent.get("type", "concept")
            if not name or len(name) < 2:
                continue
            graph.add_node(name, type=etype, source=source, article=art, page=page)
            if art_node and name != art_node:
                graph.add_edge(art_node, name, relation="defines")

        # Skip LLM call if no entities or too few to have relations
        if len(group_entities) < 2:
            logger.info(
                f"  [{idx}/{total}] {label} — {len(group_entities)} entities, "
                "skipping relation call (need >= 2)"
            )
            continue

        # Build relation prompt with entity list
        entity_list_str = ", ".join(
            f'"{e["name"]}"' for e in group_entities[:30]   # cap to avoid overshooting context
        )
        combined = "\n\n".join(c.text for c in grp)
        if len(combined) > MAX_CHARS_PER_CALL:
            combined = combined[:MAX_CHARS_PER_CALL]

        logger.info(
            f"  [{idx}/{total}] {label} — {len(group_entities)} NER entities, "
            f"{len(combined)}c text → relation call"
        )

        try:
            prompt   = RELATION_PROMPT.format(entity_list=entity_list_str, text=combined)
            messages = [{"role": "user", "content": prompt}]
            raw_text = client.chat(messages, temperature=0.0, max_tokens=400)
            relations = _parse_relations(raw_text)
        except Exception as exc:
            logger.warning(f"  Relation LLM failed for {label}: {exc}. Skipping relations.")
            relations = []

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            src   = str(rel.get("from",     "")).strip()
            tgt   = str(rel.get("to",       "")).strip()
            rtype = str(rel.get("relation", "relates_to")).strip()
            if src and tgt and src != tgt and len(src) >= 2 and len(tgt) >= 2:
                # Only add edge if at least one endpoint is in the graph
                # (prevents hallucinated entity names from polluting the graph)
                if graph.has_node(src) or graph.has_node(tgt):
                    if not graph.has_node(src):
                        graph.add_node(src, type="concept", source=source)
                    if not graph.has_node(tgt):
                        graph.add_node(tgt, type="concept", source=source)
                    graph.add_edge(src, tgt, relation=rtype, source=source)

    logger.info(
        f"  Graph (NER+LLM): {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )
    return graph


def _build_graph_cloud(chunks) -> "nx.DiGraph":
    """
    Build knowledge graph using Cloud LLM (Groq/Llama 70B, DeepSeek, etc.).

    Key optimisation: GROUP chunks by article before calling the LLM.
    641 individual chunks → ~60-80 article groups → 8-12x fewer API calls.
    Each group's text is truncated to MAX_CHARS_PER_CALL to stay within
    the model's context and Groq's TPM quota.

    Prompt is open-ended — entity types and relation names are NOT hardcoded;
    the LLM discovers them from the document context.
    """
    import re
    from collections import defaultdict
    import networkx as nx
    from cloud_llm_client import CloudLLMClient

    GRAPH_PROMPT = """\
Bạn là chuyên gia phân tích văn bản pháp lý đại học Việt Nam.

Từ văn bản điều khoản sau, trích xuất:
1. CÁC THỰC THỂ quan trọng — tên điều khoản, khái niệm, giá trị số, điều kiện, quy trình, chủ thể
2. MỐI QUAN HỆ giữa các thực thể (đặt tên bằng snake_case không dấu, ví dụ: co_tien_quyet, yeu_cau, ap_dung_cho)

Nguyên tắc:
- Loại thực thể (type): tự xác định từ nội dung, KHÔNG dùng nhãn cố định
- Chỉ trích xuất quan hệ RÕ RÀNG có trong văn bản, không suy đoán
- Ưu tiên: điều khoản cụ thể, số liệu, điều kiện bắt buộc, tên quy định

VĂN BẢN:
{text}

Trả về JSON (không markdown, không giải thích):
{{"entities":[{{"name":"...","type":"...","description":"..."}}],
  "relations":[{{"from":"...","relation":"...","to":"..."}}]}}"""

    # llama-3.1-8b-instant: 30K TPM. Each call: ~400 tokens in + 300 out = 700 tokens.
    # Safe at 5s gap: 12 calls/min × 700 = 8400 tokens/min < 30K TPM ✓
    MAX_CHARS_PER_CALL = 1500   # ~375 tokens input (4 chars/token avg Vietnamese)

    def _parse_json(raw: str) -> dict:
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {"entities": [], "relations": []}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {"entities": [], "relations": []}

    #  Group chunks by (source, article_number) 
    # Chunks from the same article contain overlapping content → one LLM call
    # covers the whole article, saving 3-10x API calls.
    ArticleKey = tuple  # (source, article_number or page)
    groups: dict[ArticleKey, list] = defaultdict(list)

    for chunk in chunks:
        art  = getattr(chunk, 'article_number', '') or ''
        page = getattr(chunk, 'page', 1)
        key  = (chunk.source, art if art else f"p{page}")
        groups[key].append(chunk)

    # Use dedicated graph model (smaller, higher TPM quota)
    graph_model = settings.cloud_graph_model or settings.cloud_model or ""
    client = CloudLLMClient(
        model    = graph_model,
        min_gap_s = 5.0,   # 12 calls/min × 700 tokens = 8400 TPM < 30K limit
        max_rpm  = 25,
    )
    graph  = nx.DiGraph()
    total  = len(groups)

    logger.info(
        f"  {len(chunks)} chunks → {total} article groups "
        f"({len(chunks)//max(total,1):.1f} chunks/article avg) "
        f"via {settings.cloud_provider.capitalize()} graph_model={client.model}"
    )

    for idx, (key, grp) in enumerate(groups.items(), 1):
        source = grp[0].source
        art    = getattr(grp[0], 'article_number', '') or ''
        art_name = getattr(grp[0], 'article_name', '') or ''
        page   = getattr(grp[0], 'page', 1)
        label  = f"Điều {art}" if art else f"tr.{page}"

        # Concatenate chunk texts up to MAX_CHARS_PER_CALL
        combined = "\n\n".join(c.text for c in grp)
        if len(combined) > MAX_CHARS_PER_CALL:
            combined = combined[:MAX_CHARS_PER_CALL]

        logger.info(f"  [{idx}/{total}] {label} ({len(combined)}c) — {source[:35]}")

        try:
            messages = [{"role": "user", "content": GRAPH_PROMPT.format(text=combined)}]
            raw_text = client.chat(messages, temperature=0.0, max_tokens=512)
            data     = _parse_json(raw_text)
        except Exception as exc:
            logger.warning(f"  Cloud LLM failed for {label}: {exc}. Skipping.")
            continue

        entities  = data.get("entities", [])
        relations = data.get("relations", [])

        for ent in entities:
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name", "")).strip()
            if not name or len(name) < 2:
                continue
            graph.add_node(name,
                type        = str(ent.get("type", "concept")),
                description = str(ent.get("description", "")),
                source      = source,
                article     = art,
                page        = page,
            )

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            src   = str(rel.get("from", "")).strip()
            tgt   = str(rel.get("to",   "")).strip()
            rtype = str(rel.get("relation", "relates_to")).strip()
            if src and tgt and src != tgt and len(src) >= 2 and len(tgt) >= 2:
                graph.add_edge(src, tgt, relation=rtype, source=source)

        # Article node links all entities to their article
        if art and art_name:
            art_node = f"Điều {art}"
            graph.add_node(art_node, type="article", name=art_name,
                           page=page, source=source)
            for ent in entities:
                name = str(ent.get("name", "")).strip() if isinstance(ent, dict) else ""
                if name and name != art_node and len(name) >= 2:
                    graph.add_edge(art_node, name, relation="defines")

    logger.info(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph

def _build_graph_fast(chunks) -> "nx.DiGraph":
    """
    Fallback graph builder (no LLM) using NER heuristics.
    Extracts article names, key numbers, and simple co-occurrence edges.
    """
    import re
    import networkx as nx

    graph = nx.DiGraph()
    _NUM_PAT = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(tín chỉ|tiết|giờ|năm|học kỳ|%)\b")
    _DIEU_REF = re.compile(r"Điều\s+(\d+)")

    for chunk in chunks:
        art_node = f"Điều {chunk.article_number}" if chunk.article_number else None
        if art_node:
            graph.add_node(
                art_node,
                type   = "article",
                name   = chunk.article_name,
                page   = chunk.page,
                source = chunk.source,
            )

        # Extract numeric facts
        for m in _NUM_PAT.finditer(chunk.text):
            ent_name = f"{m.group(1)} {m.group(2)}"
            graph.add_node(ent_name, type="value", source=chunk.source)
            if art_node:
                graph.add_edge(art_node, ent_name, relation="specifies")

        # Cross-reference edges
        for ref in _DIEU_REF.findall(chunk.text):
            ref_node = f"Điều {ref}"
            if art_node and ref_node != art_node:
                graph.add_edge(art_node, ref_node, relation="references")

    logger.info(
        f"  Fast graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )
    return graph

def _save_graph(graph) -> None:
    import networkx as nx
    out = settings.storage / "knowledge.graphml"
    nx.write_graphml(graph, str(out))
    logger.info(f"  Graph saved → {out.name}")

def _build_entity_embeddings(graph, embedder) -> None:
    import numpy as np

    entity_names = list(graph.nodes())
    if not entity_names:
        logger.warning("  No entities in graph — skipping entity embeddings")
        np.save(str(settings.storage / "entity_vecs.npy"), np.zeros((0, 1024)))
        (settings.storage / "entity_names.json").write_text("[]")
        return

    logger.info(f"  Embedding {len(entity_names)} entities …")
    batch_size = 4
    vecs = []
    for i in range(0, len(entity_names), batch_size):
        batch = entity_names[i: i + batch_size]
        v = embedder.encode(batch)
        vecs.append(v)

    all_vecs = np.vstack(vecs).astype("float32")
    np.save(str(settings.storage / "entity_vecs.npy"), all_vecs)
    (settings.storage / "entity_names.json").write_text(
        json.dumps(entity_names, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"  Entity embeddings: {all_vecs.shape}")

def _build_hypergraph(chunks, embedder) -> None:
    """
    Build EHRAG hypergraph from chunks + embedder.

    Requires chunk_entity_map — {chunk_id: [entity_names]} — derived from the
    knowledge graph to construct the structural incidence matrix H^str.
    Without this map H^str is all-zero and diffusion produces no signal.
    """
    try:
        import networkx as nx
        from hypergraph import EntityHypergraph
        import numpy as np

        #  Load entity names + vecs (built in step 5) 
        ent_path  = settings.storage / "entity_names.json"
        evec_path = settings.storage / "entity_vecs.npy"
        if not ent_path.exists() or not evec_path.exists():
            logger.warning("  Entity names/vecs missing — skipping hypergraph.")
            return

        entity_names = json.loads(ent_path.read_text(encoding="utf-8"))
        entity_vecs  = np.load(str(evec_path))

        if not entity_names:
            logger.warning("  No entities — skipping hypergraph.")
            return

        #  Build chunk_entity_map from saved graph 
        # This is the critical missing piece: H^str[e,c] = 1 iff entity e
        # appears in chunk c.  We derive it from the knowledge.graphml edges.
        graph_path = settings.storage / "knowledge.graphml"
        chunk_entity_map: dict[str, list[str]] = {}

        if graph_path.exists():
            G = nx.read_graphml(str(graph_path))
            entity_name_set = set(entity_names)

            # For each entity node, find which chunks mention it via edges
            entity_to_chunks: dict[str, list[str]] = {}
            for u, v, data in G.edges(data=True):
                rel = data.get("relation", "")
                # "defines" edge: article_node → entity
                # We map entities to chunks via article → chunk lookup
                if rel in ("defines", "relates_to", "mentions"):
                    if v in entity_name_set:
                        entity_to_chunks.setdefault(v, [])

            # Build chunk_entity_map: for each chunk, which entities appear?
            # Use chunk source+article+page as chunk_id (matches how we name them)
            chunk_id_set: dict[str, list[str]] = {}
            for chunk in chunks:
                cid = f"{chunk.source}::Điều{getattr(chunk,'article_number','')}::p{chunk.page}::i{chunk.chunk_index}"
                # Find entities whose article node matches this chunk's article
                art = getattr(chunk, 'article_number', '')
                art_node = f"Điều {art}" if art else None
                ents_in_chunk: list[str] = []
                if art_node and G.has_node(art_node):
                    for nbr in G.successors(art_node):
                        if nbr in entity_name_set:
                            ents_in_chunk.append(nbr)
                # Also: any entity node whose source matches this chunk's source
                chunk_id_set[cid] = ents_in_chunk

            chunk_entity_map = chunk_id_set
            logger.info(
                f"  chunk_entity_map: {len(chunk_entity_map)} chunks, "
                f"avg {sum(len(v) for v in chunk_entity_map.values())//max(len(chunk_entity_map),1)} entities/chunk"
            )
        else:
            # Fallback: assign entities to chunks by text overlap
            logger.warning("  No knowledge.graphml — using text-match fallback for chunk_entity_map")
            entity_name_set = set(entity_names)
            for chunk in chunks:
                cid = f"{chunk.source}::Điều{getattr(chunk,'article_number','')}::p{chunk.page}::i{chunk.chunk_index}"
                chunk_entity_map[cid] = [
                    e for e in entity_name_set if e.lower() in chunk.text.lower()
                ]

        # Embed chunk texts
        chunk_texts = [c.text for c in chunks]
        chunk_ids   = list(chunk_entity_map.keys())   # ordered

        # Re-order chunks to match chunk_ids order
        cid_to_text = {}
        for chunk in chunks:
            cid = f"{chunk.source}::Điều{getattr(chunk,'article_number','')}::p{chunk.page}::i{chunk.chunk_index}"
            cid_to_text[cid] = chunk.text

        ordered_texts = [cid_to_text.get(cid, "") for cid in chunk_ids]

        batch_size = 4
        chunk_vecs_list: list[np.ndarray] = []
        for i in range(0, len(ordered_texts), batch_size):
            v = embedder.encode(ordered_texts[i: i + batch_size])
            chunk_vecs_list.append(v)

        if not chunk_vecs_list:
            logger.warning("  No chunk vectors — skipping hypergraph.")
            return

        chunk_vecs = np.vstack(chunk_vecs_list).astype("float32")

        # Save for runtime use
        np.save(str(settings.storage / "chunk_vecs.npy"), chunk_vecs)
        (settings.storage / "chunk_ids.json").write_text(
            json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8"
        )

        # Build hypergraph
        hg = EntityHypergraph()
        hg.build(
            entity_names     = entity_names,
            entity_vecs      = entity_vecs,
            chunk_entity_map = chunk_entity_map,
            chunk_vecs       = chunk_vecs,
        )
        hg.save(settings.storage / "hypergraph")
        logger.info(
            f"  Hypergraph built: K={hg.n_clusters}  E={len(entity_names)}  "
            f"C={len(chunk_ids)}  H_str_nnz={hg._H_str.nnz if hg._H_str is not None else 0}"
        )

    except Exception as exc:
        logger.warning(f"  Hypergraph build failed (non-fatal): {exc}")

# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="STELLAR-RAG Ingest Pipeline")
    parser.add_argument(
        "--skip-graph", action="store_true",
        help="Skip all LLM graph extraction — use fast regex NER only",
    )
    parser.add_argument(
        "--no-ner", action="store_true",
        help="Disable NER pre-pass; LLM extracts both entities and relations (original mode)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files and exit without processing",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only first N PDF files",
    )
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
