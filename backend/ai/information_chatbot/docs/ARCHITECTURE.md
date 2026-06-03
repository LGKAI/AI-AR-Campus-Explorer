# STELLAR-RAG v4 — System Architecture

> **Stack**: Python 3.11+ · PyMuPDF · EasyOCR · FAISS · NetworkX · Ollama · sentence-transformers · HuggingFace transformers
> **Papers implemented**: EHRAG (arXiv 2604.17458) · HybGRAG (arXiv 2412.16311)

---

## 1. Bird's-Eye View

STELLAR-RAG v4 is a **hybrid retrieval-augmented generation** system for Vietnamese university Q&A. Three independent retrieval signals — dense semantic search (FAISS/bge-m3), BM25 keyword search, and knowledge-graph traversal (PPR) — are fused by QDAP-S, post-processed by EHRAG hypergraph diffusion, validated by a HybGRAG critic loop, and optionally reranked by a cross-encoder.

```
╔══════════════════════════════════════════════════════════════════╗
║  INGEST PHASE                                                    ║
║                                                                  ║
║  PDF ──► PDFExtractor v4 ──► Chunks                              ║
║              │                   │                               ║
║   article-boundary split     ┌───┼───┬─────────┐                 ║
║   merge-forward (600c)       ▼   ▼   ▼         ▼                 ║
║   seam overlap               FAISS BM25  KG    Hypergraph        ║
║   OCR normalisation          bge-m3  BM25Okapi Graph + EHRAG     ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║  QUERY PHASE                                                     ║
║                                                                  ║
║  User query                                                      ║
║      │                                                           ║
║      ├─ InputGuardrail ──[block]──► Rejected                     ║
║      ├─ LRUCache ──[hit]──► Cached answer                        ║
║      ├─ QueryRouter → simple | medium | complex                  ║
║      ├─ [medium/complex] QueryExpander → paraphrase variants     ║
║      ├─ [analytical+complex] HyDE → hypothetical passage         ║
║      │                                                           ║
║      └─ HybGRAG Critic Loop (max 3 iterations)                   ║
║              │                                                   ║
║         ┌────▼──── Parallel Retrieval ─────────────────────┐     ║
║         │  Dense FAISS/bge-m3 │ BM25 │ Graph (entity+PPR)  │     ║
║         └──────────────────────┬──────────────────────────-┘     ║
║                                 │                                ║
║                           QDAP-S fusion                          ║
║                                 │                                ║
║                           Doc-type boost                         ║
║                                 │                                ║
║                     EHRAG hypergraph rescore                     ║
║                                 │                                ║
║                     Cross-encoder rerank                         ║
║                                 │                                ║
║              Self-RAG quality ──[low]──► expand k, hops          ║
║                                 │                                ║
║              Critic Validator YES ──► break                      ║
║                                 │ NO                             ║
║              Critic Commenter → enrich query → next iter         ║
║                                 │                                ║
║                    MMR diversity selection                       ║
║                    Context assembly (Organizer)                  ║
║                    LLM generation (Ollama or Cloud LLM)          ║
║                    OutputGuardrail                               ║
║                    Cache + Memory update                         ║
║                                 │                                ║
║         [Optional] Rating 1-5 ──► QDAP-S online update           ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 2. Module Reference

| Module | Responsibility |
|--------|----------------|
| `config.py` | Centralised settings; all parameters overridable via env-vars |
| `pdf_extractor.py` | PDFExtractor v4: article-boundary split, merge-forward chunking, seam overlap, OCR normalisation |
| `pdf_pipeline.py` | Legacy PDF pipeline (kept for reference) |
| `embedding.py` | Text → L2-normalised float32 vectors (BAAI/bge-m3 or Ollama) |
| `vector_store.py` | FAISS index with auto FlatIP / HNSWFlat selection |
| `graphrag.py` | Core orchestrator: all indices, fusion, rescoring |
| `hypergraph.py` | EHRAG entity hypergraph (H^str + H^sem + diffusion + scoring) |
| `qdap.py` | QDAP-S α predictor (Linear → Conv1D → Softmax → E[α]) |
| `critic.py` | HybGRAG Validator + Commenter |
| `agent.py` | End-to-end pipeline: guardrail → retrieval → generation |
| `llm_client.py` | Unified LLM client (Ollama + Cloud LLM, dual-mode parallel) |
| `cloud_llm_client.py` | Cloud LLM backend (Groq / DeepSeek / OpenRouter via OpenAI API) |
| `memory.py` | SQLite + FAISS conversational memory + reinforced recall |
| `reranker.py` | Singleton cross-encoder (ms-marco-MiniLM-L-6-v2) |
| `router.py` | Heuristic query complexity classifier |
| `guardrail.py` | Input sanitisation + output grounding check |
| `query_expander.py` | LLM paraphrase variant generation |
| `ner_extractor.py` | Vietnamese NER (HuggingFace token-classification, CPU-safe, unloadable) |

---

## 3. Ingest Data Flow

```
PDF file
  │
  ├─ fitz.open()
  │    ├─ page.get_text("text")  →  native_text
  │    └─ page.get_pixmap(dpi=220) → PIL Image → EasyOCR
  │         ├─ detail=1: (bbox, text, confidence) per token
  │         ├─ row clustering: sort by y-centre, group within row_tol
  │         ├─ table mode: ≥40% rows with ≥2 cells → " | " separator
  │         └─ OCR normalisation (noise filter, soft line-break join)
  │
  └─ PDFExtractor v4:
       ├─ article-boundary split (Điều X.)
       ├─ merge-forward (TARGET=600 chars, merge short chunks forward)
       ├─ seam overlap (append 2 leading sentences of next chunk)
       └─ OCR normalisation (pipe artefact removal, soft hyphen joins)

All Chunks → ingest.py pipeline:
  ├─ FAISS index (bge-m3 1024-dim, batch_size=4)
  ├─ BM25Okapi (k1=1.5, b=0.75)
  ├─ Knowledge Graph — two-stage NER+LLM pipeline:
  │    ├─ Stage 1 (NER, local): NlpHUST/ner-vietnamese-electra-base
  │    │    ├─ Extracts PER / ORG / LOC / MISC entities from every chunk
  │    │    ├─ Domain regex adds QUANTITY and ARTICLE entities (no model)
  │    │    ├─ Runs on CPU — does not compete with bge-m3 for GPU VRAM
  │    │    └─ Model unloaded after pass (~400 MB RAM freed)
  │    ├─ Stage 2 (LLM, cloud): Groq llama-3.3-70b-versatile
  │    │    ├─ Receives known entity list → extracts relations ONLY
  │    │    ├─ ~50% fewer tokens vs full entity+relation extraction
  │    │    ├─ TPM guard: min_gap=8s, max_rpm=7 (fits 6,000 TPM free tier)
  │    │    └─ Exponential back-off on 429 (up to 300s)
  │    └─ Fallback: --no-ner → LLM extracts entities+relations (original mode)
  │             --skip-graph → fast regex NER only (no cloud API needed)
  ├─ Entity embeddings (bge-m3, batch_size=4)
  └─ EHRAG hypergraph
       ├─ chunk_entity_map from knowledge graph edges
       ├─ H^str (E×C): structural incidence matrix (scipy.sparse)
       └─ H^sem (E×K): BIRCH clusters → Gaussian-weighted edges
```

---

## 4. Dual LLM Mode

The system supports running two LLMs in parallel for comparative answers:

```
answer_dual(query)
  │
  ├─ [Thread 1] Ollama (qwen2.5:7b-instruct, local)
  │       → ollama_answer
  │
  └─ [Thread 2] Cloud LLM (e.g. llama-3.3-70b-versatile via Groq)
          → cloud_answer

Both use the same retrieved context.
app.py prints both answers side-by-side.
TTS and RLHF rating use the Ollama answer.
```

Dual mode is activated by choosing option **2** at startup, or by passing `--dual` to `eval/pipeline.py`.

---

## 5. Query Data Flow (Detailed)

```
Agent.answer(user_query)
│
├─ [1] InputGuardrail.check()
│
├─ [2] LRUCache.get()
│
├─ [3] QueryProcessor.process()
│       ├─ fast heuristic path (zero LLM, most queries)
│       └─ LLM path for complex/domain queries
│
├─ [4] QueryRouter.classify() → simple | medium | complex
│
├─ [5] QueryExpander.expand()  [skipped if simple]
│
├─ [6] _should_hyde()? → _hyde_expand()
│       └─ gate: complex AND (analytical keyword OR ≥25 words)
│
└─ [7] _retrieve_with_critic()  (HybGRAG loop, max 3 iters)
         │
         └─ for t in range(critic_max_iterations):
              │
              ├─ _retrieve_and_build_context()
              │    ├─ GraphRAG.query() or .query_batch(sub_queries)
              │    │    ├─ _dense_search(qv, k*2)       [FAISS]
              │    │    ├─ _bm25_search(q, k*2)         [BM25Okapi]
              │    │    ├─ _graph_retrieve(q, qv, k, hops) [PPR]
              │    │    ├─ _fuse(dense, bm25, graph, qv)   [QDAP-S]
              │    │    ├─ _doc_type_boost()
              │    │    └─ _hypergraph_rescore(hits, qv)   [EHRAG]
              │    ├─ Reranker.rerank(query, hits, top_k=20)
              │    ├─ Self-RAG quality < 0.15 → re-retrieve k*2, hops+1
              │    └─ Organizer.organize() → context string
              │
              ├─ FAST-PATH: Self-RAG quality ≥ 0.5 → break
              │
              ├─ Critic.validate(query, context, reasoning_paths)
              │    ├─ YES → break
              │    └─ NO  → continue
              │
              └─ Critic.comment() → feedback
                   └─ Critic.enrich_query() → enriched_query → next t
         │
         ├─ _build_messages() → [{system: SYSTEM_PROMPT}, {user: question+context}]
         ├─ LLMClient.chat() or .chat_dual() → answer(s)
         ├─ OutputGuardrail.check(answer, context)
         ├─ LRUCache.put()
         ├─ Memory.add(user) + Memory.add(assistant)
         └─ [if rating] QDAP-S online update
```

---

## 6. Storage Layout

```
storage/
├── docs.faiss                FAISS index (FlatIP n<500 | HNSWFlat M=32 n≥500)
├── docs_meta.json            Chunk metadata list
├── bm25_index.pkl            BM25Okapi object + metadata
├── knowledge.graphml         NetworkX DiGraph (GraphML format)
├── entity_vecs.npy           Entity embedding matrix (E, 1024) float32
├── entity_names.json         Entity name list, len=E
├── qdap_s.npz                QDAP-S weights: W, b
├── memory.sqlite             SQLite conversation history
├── memory.faiss              Memory FAISS index
├── memory_meta.json          Memory metadata
├── reward_memory.faiss       High-rated answer index
├── reward_memory_meta.json
├── chunk_vecs.npy            Chunk embeddings for hypergraph (C, 1024)
├── chunk_ids.json            Chunk ID list, len=C
└── hypergraph/
    ├── hgraph_H_str.npz      Structural incidence matrix (scipy sparse, E×C)
    ├── hgraph_H_sem.npz      Semantic incidence matrix   (scipy sparse, E×K)
    ├── hgraph_meta.npz       cluster_ids, cluster_centroids, n_clusters
    ├── hgraph_chunk_ids.json
    └── hgraph_entity_names.json
```

---

## 7. Configuration Quick Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `BAAI/bge-m3` | Embedding model (1024-dim, multilingual) |
| `EMBEDDING_BACKEND` | `sentence_transformers` | `sentence_transformers` or `ollama` |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Main LLM for generation |
| `CRITIC_MODEL` | `qwen2.5:0.5b` | Fast LLM for critic (50–150 ms/call) |
| `CHUNK_SIZE` | `600` | Target chunk size in characters (merge-forward) |
| `TOP_K` | `6` | Final retrieved chunks per query |
| `FUSION_METHOD` | `qdap_s` | `qdap_s` or `rrf` |
| `QDAP_GRAPH_WEIGHT` | `0.15` | Graph contribution weight |
| `CRITIC_ENABLED` | `true` | Enable HybGRAG critic loop |
| `CRITIC_MAX_ITERATIONS` | `3` | Max retrieval-refinement rounds |
| `CRITIC_SKIP_THRESHOLD` | `0.5` | Self-RAG quality threshold to skip critic |
| `RERANKER_ENABLED` | `true` | Cross-encoder reranking |
| `HYDE_ENABLED` | `true` | HyDE (analytical complex queries only) |
| `EMBED_BATCH_SIZE` | `4` | Batch size for embedding (OOM prevention) |
| `BIRCH_THRESHOLD` | `0.5` | BIRCH merge distance |
| `HYPERGRAPH_TAU` | `1.0` | Gaussian temperature τ |
| `HYPERGRAPH_GAMMA` | `0.5` | Semantic expansion decay γ |
| `HYPERGRAPH_DIFFUSE_T` | `3` | Structural propagation iterations |
| `HYPERGRAPH_LAMBDA1` | `0.3` | Entity evidence weight λ₁ |
| `HYPERGRAPH_LAMBDA2` | `0.2` | Cluster topic weight λ₂ |
| `LLM_BACKEND` | `ollama` | `ollama` \| `cloud` \| `both` |
| `CLOUD_PROVIDER` | — | `groq` \| `deepseek` \| `openrouter` |
| `CLOUD_API_KEY` | — | Cloud provider API key |
| `CLOUD_MODEL` | — | Cloud model for chat generation |
| `CLOUD_GRAPH_MODEL` | — | Cloud model for graph extraction |
