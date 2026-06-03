# STELLAR-RAG v4 — Retrieval Pipeline

This document explains every stage of the query-time retrieval pipeline: from raw user input to the assembled context passed to the LLM.

---

## 1. Pipeline Overview

```
User query
    │
    ▼
[Guard]     InputGuardrail  →  block / warn / sanitise
    │
    ▼
[Cache]     LRUCache 256-entry  →  cache hit: return early
    │
    ▼
[Route]     QueryRouter  →  complexity: simple | medium | complex
    │
    ▼
[Expand]    QueryExpander  →  paraphrase variants (skip if simple)
    │
    ▼
[HyDE]      _should_hyde()?  →  hypothetical passage (analytical+complex only)
    │
    ▼
╔══ HybGRAG Critic Loop  (max 3 iterations) ═══════════════════╗
║                                                               ║
║  [Retrieve]  Parallel: Dense FAISS │ BM25 │ Graph (PPR)      ║
║      │                                                        ║
║  [Fuse]      QDAP-S or RRF                                   ║
║      │                                                        ║
║  [Boost]     Doc-type intent cosine boost                    ║
║      │                                                        ║
║  [EHRAG]     Hypergraph diffusion rescore                     ║
║      │                                                        ║
║  [Rerank]    Cross-encoder (top-20)                          ║
║      │                                                        ║
║  [Self-RAG]  Quality < 0.15 → re-retrieve k*2, hops+1       ║
║      │                                                        ║
║  [Critic]    Validator YES → break                           ║
║              Validator NO  → Commenter → enrich query        ║
╚══════════════════════════════════════════════════════════════╝
    │
    ▼
[MMR]       Diversity selection (Jaccard, λ=0.7)
    │
    ▼
[Organizer] Context assembly with score-proportional budget
    │
    ▼
    context string  →  LLM
```

---

## 2. Query Routing

**Source**: `src/router.py`

The router assigns a complexity level to drive all downstream decisions.

| Level | Criteria | top_k | hops | use_graph | Expander | HyDE |
|-------|----------|-------|------|-----------|----------|------|
| `simple` | Short, single entity, no analytical keywords | 4 | 1 | False | Skip | No |
| `medium` | Multiple entities or mild complexity | 6 | 2 | True | Run | No |
| `complex` | Long, multi-entity, analytical keywords, conjunctions | 8 | 3 | True | Run | If analytical |

Routing happens **before** expansion and HyDE so that expensive LLM calls are only triggered when justified.

---

## 3. Query Expansion

**Source**: `src/query_expander.py`
**Condition**: `complexity != 'simple'` and `len(sub_queries) == 1`

The LLM generates 2–3 paraphrase variants. These become `sub_queries` in `ProcessedQuery`. Downstream retrieval runs `query_batch()` over all variants, widening the recall net.

---

## 4. HyDE — Hypothetical Document Embedding

**Source**: `src/agent.py` — `_hyde_expand()`
**Condition**: `complexity == 'complex'` AND (analytical keyword in query OR ≥25 words)

The LLM generates a short 2–3 sentence passage that would *answer* the query. The concatenation of `query + hypothetical_passage` is used as the dense query vector.

```
Original:    "Tại sao sinh viên bị cảnh báo học tập?"
HyDE text:   "Sinh viên bị cảnh báo học tập khi điểm trung bình tích lũy
              giảm xuống dưới ngưỡng quy định..."
Dense query: "Tại sao sinh viên bị cảnh báo học tập?\n[hypothetical text]"
```

**When NOT to use HyDE**: factual table lookups that happen to be "complex" (e.g., "Điều 15 quy định gì?"). HyDE generates prose that moves the query vector away from structured document chunks, hurting recall. The analytical-keyword gate prevents this.

---

## 5. Dense Retrieval

**Source**: `src/graphrag.py` — `_dense_search()`
**Index**: FAISS `IndexFlatIP` (n < 500) or `IndexHNSWFlat` M=32 efSearch=64 (n ≥ 500)
**Model**: BAAI/bge-m3 (1024-dim, multilingual)

Search returns the top 2k nearest neighbours by inner product (= cosine similarity for L2-normalised vectors).

**Contextual embedding**: each chunk is embedded with a metadata prefix at ingest time:

```
[Loại: quy_che | Mục: Điều 15 | Nguồn: quy_che_2021.pdf]
{chunk text}
```

---

## 6. BM25 Sparse Retrieval

**Source**: `src/graphrag.py` — `_bm25_search()`
**Implementation**: `rank_bm25.BM25Okapi`, k1=1.5, b=0.75

Vietnamese tokenisation (`_tokenize_vi`): extract Unicode alphanumeric tokens of length ≥ 2. IDF naturally handles high-frequency terms.

Returns top 2k by BM25 score. Complements dense retrieval for exact-match terms: article numbers ("Điều 15"), amounts ("4.5 triệu"), dates.

---

## 7. Graph Retrieval

**Source**: `src/graphrag.py` — `_graph_retrieve()`

### 7.1 Entity Linking

At query time, find the top entities by cosine similarity:

$$\text{linked} = \lbrace e : \hat{\mathbf{v}}_e \cdot \hat{\mathbf{q}} \geq 0.45,\; e \in \text{top-10} \rbrace$$

Fallbacks (in order): substring match → section keyword overlap.

### 7.2 Local Subgraph PPR

BFS up to `hops` steps (capped at 500 nodes), then personalised PageRank on this subgraph. The personalisation vector places equal weight on seed entities.

**Relation weights** driving the random walk:

| Relation | Weight |
|----------|--------|
| `co_tien_quyet` (prerequisite) | 2.0 |
| `quy_dinh_ve` (regulates) | 1.8 |
| `yeu_cau` (requires) | 1.7 |
| `co_occurs_with` | 0.9 |
| `has_chunk` | 0.7 |

### 7.3 Weighted BFS Fallback

If PPR returns no chunk nodes:

$$s_{\text{BFS}}(v, t+1) = \sum_{u \to v} s_{\text{BFS}}(u, t) \cdot w_{u \to v} \cdot 0.7^{t+1}$$

---

## 8. QDAP-S Fusion

**Source**: `src/graphrag.py` — `_qdap_fuse()`
**Reference**: [MATH_FOUNDATIONS.md §4](MATH_FOUNDATIONS.md)

1. Predict α from query embedding via QDAP-S predictor.
2. Min-max normalise dense, BM25, and graph scores independently.
3. Blend dense and BM25: `s_db = α * s_dense + (1-α) * s_BM25`
4. Blend with graph: `s_final = 0.85 * s_db + 0.15 * s_graph`

Each result carries `qdap_alpha` in its metadata for debugging.

---

## 9. Doc-Type Intent Boost

**Source**: `src/graphrag.py` — `_doc_type_boost()`

Classify query intent by cosine similarity to pre-embedded document-type descriptions. Only activates when best-type similarity ≥ 0.40. Matching chunks get score × 1.35.

| Type | Description embedded |
|------|---------------------|
| `hoc_phi` | tuition, fees, payment, exemption |
| `quy_che` | training regulations, rules, policies |
| `chuong_trinh` | curriculum, study plan, credits |
| `lich_hoc` | schedule, timetable, exam calendar |
| `tuyen_sinh` | admissions, enrolment, cut-off scores |
| `thong_bao` | announcements, updates, notices |

---

## 10. EHRAG Hypergraph Rescore

**Source**: `src/graphrag.py` — `_hypergraph_rescore()`
**Reference**: [HYPERGRAPH_EHRAG.md](HYPERGRAPH_EHRAG.md)

1. Link top-8 entities to the query via embedding similarity.
2. Build `seed_entity_scores = {name: cosine_sim}`.
3. Run `EntityHypergraph.diffuse()` → `entity_weights`, `cluster_scores`.
4. Run `topic_score_chunks(hits, entity_weights, cluster_scores)` → re-sorted hits.

`S(d) = S_dense + λ1 * entity_evidence + λ2 * cluster_term`

Fails open: any exception returns the original hits unchanged.

---

## 11. Cross-Encoder Reranking

**Source**: `src/reranker.py`
**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22 MB, lazy-loaded singleton)

Scores the top `reranker_top_k = 20` fused candidates jointly as `(query, chunk_text)` pairs. The cross-encoder attends across query and document, catching relevance nuances missed by bi-encoder cosine similarity. Results annotated with `+ce` suffix in `retrieval_type`.

Enabled by default (`RERANKER_ENABLED=true`). ~100 ms on CPU, ~10–20 ms on GPU.

---

## 12. Self-RAG Quality Expansion

**Source**: `src/agent.py` — `_retrieve_and_build_context()`

After context assembly:

$$q_{\text{quality}} = \frac{|T_q \cap T_c|}{|T_q|}$$

If $\,q_{\text{quality}} < 0.15$: re-retrieve with k' = min(2k, 20) and hops' = min(hops+1, 3).

---

## 13. HybGRAG Critic Loop

**Source**: `src/agent.py` — `_retrieve_with_critic()`, `src/critic.py`
**Paper**: HybGRAG (arXiv 2412.16311)

### 13.1 Fast-Path Bypass

If $\,q_{\text{quality}} \geq 0.5$: break, skip critic call entirely.

### 13.2 Validator (C_val)

`qwen2.5:0.5b`, temperature 0, max 8 tokens. Answers YES/NO. YES → proceed. NO → trigger Commenter.

Fail-open: LLM error returns True so the pipeline never stalls.

### 13.3 Commenter (C_com)

Structured single-line feedback, temperature 0.2, max 120 tokens:

```
Thiếu thực thể: [entity name]
Thiếu điều khoản: [article reference]
Thiếu bảng số liệu: [table name]
```

### 13.4 Query Enrichment

```python
enriched = f"{original_query} [Cần thêm: {feedback}]"
```

The structured feedback terms become additional query tokens in the next iteration's dense and BM25 retrieval.

### 13.5 Stop Conditions

- Validator returns YES
- Self-RAG fast-path triggered (quality ≥ 0.5)
- `critic_max_iterations = 3` reached
- Commenter returns empty string
- Enriched query equals previous query (no change)

---

## 14. Context Assembly (Organizer)

**Source**: `src/agent.py` — `Organizer.organize()`

### 14.1 MMR Selection

From the full fused hit list, apply MMR to select top_k=6 diverse, relevant chunks:

$$\text{MMR}(d_i) = \lambda \cdot s(d_i) - (1 - \lambda) \cdot \max_{d_j \in S} J(d_i, d_j), \quad \lambda = 0.7$$

### 14.2 Score-Proportional Budget

Total budget: `max_context_chars = 6000`. Per-chunk allocation proportional to score, clipped to [min_chars=150, max_chars_per_chunk=500].

### 14.3 Sentence-Level Compression

Score: `score(s) = |T_s ∩ T_q| / sqrt(|T_s|)`. Greedy selection of highest-scoring sentences within budget.

### 14.4 Context Sections (in order)

1. **Tài liệu liên quan** — top-k hybrid chunks (source/page/section/score tags)
2. **Quan hệ tri thức** — graph relation paths from PPR traversal
3. **Hội thoại liên quan** — relevant conversation memory (cosine ≥ 0.5)
4. **Câu trả lời chất lượng cao** — reinforced-recall high-rated answers
5. **Lịch sử hội thoại** — last 4 turns
6. **Ghi chú tra cứu** — critic feedback note (if applicable)
