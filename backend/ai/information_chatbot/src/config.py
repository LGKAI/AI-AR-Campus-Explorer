"""
STELLAR-RAG v4 — Unified settings with EHRAG and HybGRAG extensions.

All new settings follow the same pattern as the originals:
  - Class-level attribute with a sensible default
  - Overridable via environment variable (same name, UPPER_CASE)
  - Documented inline
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

class Settings:
    # Ollama / LLM
    ollama_host:  str = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

    # Gemini API (legacy — kept for graph extraction only)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model:   str = os.getenv("GEMINI_MODEL",   "gemini-1.5-flash")

    # Cloud LLM (dual-answer second generator)
    # provider: "groq" | "deepseek" | "openrouter" | "together"
    cloud_provider: str = os.getenv("CLOUD_PROVIDER", "groq").lower()
    cloud_api_key:  str = os.getenv("CLOUD_API_KEY",  "")
    # Leave blank to use provider default (set in cloud_llm_client._PROVIDERS)
    cloud_api_base: str = os.getenv("CLOUD_API_BASE", "")
    cloud_model:    str = os.getenv("CLOUD_MODEL",    "")
    # Separate model for graph extraction — must have high TPM (many calls).
    # llama-3.1-8b-instant: 6K TPM, safe for automated pipelines.
    # llama-3.3-70b-versatile: 12K TPM but only 100K tok/day — too low for bulk extraction.
    cloud_graph_model: str = os.getenv("CLOUD_GRAPH_MODEL", "llama-3.1-8b-instant")

    # Model for relation-only extraction in NER+LLM ingest pipeline.
    # Default: 8b-instant — faster, reliable, 6K TPM allows normal throughput.
    cloud_relation_model: str = os.getenv("CLOUD_RELATION_MODEL", "llama-3.1-8b-instant")

    # LLM backend: "ollama" | "cloud" | "both"
    llm_backend: str = os.getenv("LLM_BACKEND", "ollama").lower()

    # Embedding
    embed_model:       str  = os.getenv("EMBED_MODEL",        "BAAI/bge-m3")
    embedding_backend: str  = os.getenv("EMBEDDING_BACKEND",  "sentence_transformers").lower()
    use_gpu:           bool = os.getenv("USE_GPU", "true").lower() == "true"

    # Batch size for embedding calls (used in hypergraph chunk embedding)
    embed_batch_size:  int  = int(os.getenv("EMBED_BATCH_SIZE", "4"))

    # Prepend doc/section context to chunk text before embedding.
    # Improves dense retrieval quality — requires re-ingest to take effect.
    contextual_embedding: bool = os.getenv("CONTEXTUAL_EMBEDDING", "true").lower() == "true"

    # PDF ingestion
    pdf_dpi:       int = int(os.getenv("PDF_DPI",       "220"))
    chunk_size:    int = int(os.getenv("CHUNK_SIZE",    "750"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    doc_type_map: dict[str, str] = {
        "quy_che":      "quy_che",
        "quyche":       "quy_che",
        "regulation":   "quy_che",
        "tuyen_sinh":   "tuyen_sinh",
        "tuyensinh":    "tuyen_sinh",
        "admission":    "tuyen_sinh",
        "chuong_trinh": "chuong_trinh",
        "ctdt":         "chuong_trinh",
        "curriculum":   "chuong_trinh",
        "lich":         "lich_hoc",
        "schedule":     "lich_hoc",
        "hoc_phi":      "hoc_phi",
        "hocphi":       "hoc_phi",
        "fee":          "hoc_phi",
        "thong_bao":    "thong_bao",
        "notice":       "thong_bao",
    }
    default_doc_type: str = "general"

    # Graph construction
    max_triplets_per_chunk: int   = int(os.getenv("MAX_TRIPLETS_PER_CHUNK", "10"))
    graph_hops:             int   = int(os.getenv("GRAPH_HOPS",             "2"))
    graph_score_threshold:  float = float(os.getenv("GRAPH_SCORE_THRESHOLD", "0.0"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "6"))

    # BM25
    bm25_k1: float = float(os.getenv("BM25_K1", "1.5"))
    bm25_b:  float = float(os.getenv("BM25_B",  "0.75"))

    # RRF
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # Fusion method
    # "qdap_s" — QDAP-S min-max fusion with predicted alpha (recommended)
    # "rrf"    — Reciprocal Rank Fusion (legacy fallback)
    fusion_method:     str   = os.getenv("FUSION_METHOD",    "qdap_s")
    # Fixed graph-retrieval contribution weight in QDAP fusion [0-1].
    # 0.15 = 15% graph, 85% QDAP(dense+BM25). Set 0.0 to exclude graph.
    qdap_graph_weight: float = float(os.getenv("QDAP_GRAPH_WEIGHT", "0.15"))

    # Entity linking
    entity_link_threshold: float = float(os.getenv("ENTITY_LINK_THRESHOLD", "0.45"))

    # Run BM25 + Dense + Graph in parallel threads (safe — all read-only)
    parallel_retrieval: bool = os.getenv("PARALLEL_RETRIEVAL", "true").lower() == "true"

    # PPR subgraph size cap — limits PageRank to local neighbourhood for speed
    ppr_max_subgraph: int = int(os.getenv("PPR_MAX_SUBGRAPH", "500"))

    # Doc-type boost multiplier (applied when query intent matches doc_type)
    doc_type_boost: float = float(os.getenv("DOC_TYPE_BOOST", "1.35"))

    # HyDE — Hypothetical Document Embedding (complex queries only)
    hyde_enabled:    bool = os.getenv("HYDE_ENABLED",    "true").lower() == "true"
    hyde_max_tokens: int  = int(os.getenv("HYDE_MAX_TOKENS", "80"))

    # MMR diversity (Jaccard-based in Organizer)
    # 1.0 = pure relevance, 0.0 = pure diversity; 0.7 is recommended
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.7"))

    # LRU query cache
    query_cache_size: int = int(os.getenv("QUERY_CACHE_SIZE", "256"))
    # Set to 0 to disable caching
    query_cache_ttl_turns: int = int(os.getenv("QUERY_CACHE_TTL_TURNS", "0"))

    # Reranker — cross-encoder (~22 MB, lazy-loaded, silently disabled on import failure)
    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
    reranker_model:   str  = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # How many top fused candidates to score with the CE model
    reranker_top_k:   int  = int(os.getenv("RERANKER_TOP_K", "20"))

    # Self-RAG — adaptive retrieval expansion
    # If context quality (query-token overlap) < threshold,
    # re-retrieves with wider top_k + more graph hops.
    self_rag_enabled:   bool  = os.getenv("SELF_RAG_ENABLED",   "true").lower()  == "true"
    self_rag_threshold: float = float(os.getenv("SELF_RAG_THRESHOLD", "0.15"))

    # Semantic cache — cosine similarity fallback for LRU cache
    # When enabled, paraphrase queries (similarity > threshold) hit cache
    semantic_cache_enabled:   bool  = os.getenv("SEMANTIC_CACHE_ENABLED",   "false").lower() == "true"
    semantic_cache_threshold: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))

    # Organizer / context budget
    max_context_chars:   int = int(os.getenv("MAX_CONTEXT_CHARS",   "8000"))
    max_chars_per_chunk: int = int(os.getenv("MAX_CHARS_PER_CHUNK", "800"))
    # Minimum chars guaranteed to every included chunk
    min_chars_per_chunk: int = int(os.getenv("MIN_CHARS_PER_CHUNK", "300"))

    # Guardrail — input/output safety and quality
    # Layers: injection detection (regex) → LLM classifier OR
    #         regex toxic+OOD fallback → sanitisation (input);
    #         grounding + hallucination (output)
    guardrail_enabled:       bool  = os.getenv("GUARDRAIL_ENABLED",       "true").lower()  == "true"
    # When True, OOD queries are blocked (default: warn-only / let through)
    guardrail_block_ood:     bool  = os.getenv("GUARDRAIL_BLOCK_OOD",     "false").lower() == "true"
    # Check LLM output for grounding + hallucination markers
    guardrail_output_check:  bool  = os.getenv("GUARDRAIL_OUTPUT_CHECK",  "true").lower()  == "true"
    # Hard upper limit on raw query length (characters)
    guardrail_max_query_len: int   = int(os.getenv("GUARDRAIL_MAX_QUERY_LEN", "2000"))
    # Use a lightweight LLM model (via Ollama) for semantic harmful-content
    # and intent classification (replaces regex-only toxic + OOD layers).
    guardrail_llm_classify:   bool = os.getenv("GUARDRAIL_LLM_CLASSIFY",   "false").lower() == "true"
    guardrail_classify_model: str  = os.getenv("GUARDRAIL_CLASSIFY_MODEL", "qwen2.5:0.5b")

    # Query expansion — paraphrase robustness
    query_expansion_enabled: bool = os.getenv("QUERY_EXPANSION_ENABLED", "true").lower() == "true"

    # Memory
    memory_rebuild_threshold: int = int(os.getenv("MEMORY_REBUILD_THRESHOLD", "10"))

    # Hypergraph settings (EHRAG — arxiv 2604.17458)

    # BIRCH clustering for semantic hyperedges.
    # threshold controls merge distance; n_clusters=None → auto-detect.
    birch_threshold:  float     = float(os.getenv("BIRCH_THRESHOLD",   "0.5"))
    birch_n_clusters: int | None = (
        int(os.getenv("BIRCH_N_CLUSTERS"))
        if os.getenv("BIRCH_N_CLUSTERS") else None
    )

    # Semantic hyperedge: top-D nearest entity neighbours per cluster
    hypergraph_top_d:    int   = int(os.getenv("HYPERGRAPH_TOP_D",   "10"))

    # Gaussian weight temperature tau for H^sem: exp(-||x - c||^2 / tau)
    hypergraph_tau:      float = float(os.getenv("HYPERGRAPH_TAU",   "1.0"))

    # Semantic expansion decay coefficient gamma (EHRAG eq. 3)
    hypergraph_gamma:    float = float(os.getenv("HYPERGRAPH_GAMMA", "0.5"))

    # Number of structural propagation iterations T
    hypergraph_diffuse_T: int  = int(os.getenv("HYPERGRAPH_DIFFUSE_T", "3"))

    # Top-L sentences (chunks) in the query-gating matrix per iteration
    hypergraph_L:        int   = int(os.getenv("HYPERGRAPH_L", "50"))

    # Activation threshold epsilon — activations below epsilon are zeroed
    hypergraph_epsilon:  float = float(os.getenv("HYPERGRAPH_EPSILON", "0.01"))

    # Topic-aware scoring: lambda1 explicit entity evidence, lambda2 semantic cluster
    hypergraph_lambda1:  float = float(os.getenv("HYPERGRAPH_LAMBDA1", "0.3"))
    hypergraph_lambda2:  float = float(os.getenv("HYPERGRAPH_LAMBDA2", "0.2"))

    # Critic settings (HybGRAG — arxiv 2412.16311)

    # Set CRITIC_ENABLED=false to disable the critic loop entirely
    critic_enabled:        bool = os.getenv("CRITIC_ENABLED", "true").lower() == "true"

    # Maximum retrieval-refinement iterations (latency guard)
    critic_max_iterations: int  = int(os.getenv("CRITIC_MAX_ITERATIONS", "3"))

    # Fast small model for critic validation (shares Ollama server)
    # qwen2.5:0.5b ~300 MB, ~50-150 ms per call — keeps overhead low
    critic_model:          str  = os.getenv("CRITIC_MODEL", "qwen2.5:0.5b")

    # Self-RAG quality threshold for critic fast-path.
    # If token-overlap quality >= this value after the first retrieval,
    # skip the critic loop entirely (context already sufficient).
    # Range [0.0, 1.0]. Default 0.5 = 50% query-token coverage in context.
    critic_skip_threshold: float = float(os.getenv("CRITIC_SKIP_THRESHOLD", "0.5"))

    # Rich entity extraction (EHRAG enhancement)

    # Enable LLM-based rich entity extraction in addition to triplet extraction.
    # Extracts typed named entities (RULE, SUBJECT, AMOUNT, DATE, ORG, etc.)
    # and adds co-occurrence edges between all entity pairs in same chunk.
    entity_extract_enabled: bool = os.getenv("ENTITY_EXTRACT_ENABLED", "true").lower() == "true"

    # Max entities to extract per chunk (guards against LLM over-extraction)
    max_entities_per_chunk: int  = int(os.getenv("MAX_ENTITIES_PER_CHUNK", "20"))

    # NER pre-pass settings (ingest pipeline)

    # Enable PhoBERT-based NER pre-pass during ingest (step 4a).
    # When True, entities come from NER; LLM only extracts relations (step 4b).
    # When False, LLM extracts both entities and relations (original behavior).
    ner_enabled: bool = os.getenv("NER_ENABLED", "true").lower() == "true"

    # HuggingFace token-classification model for Vietnamese NER.
    # NlpHUST/ner-vietnamese-electra-base: ~270MB, VLSP tagset (PER/ORG/LOC/MISC).
    # Swap for any AutoModelForTokenClassification model compatible with Vietnamese.
    ner_model: str = os.getenv("NER_MODEL", "NlpHUST/ner-vietnamese-electra-base")

    # Device for NER inference. -1 = CPU (recommended to save GPU for bge-m3).
    ner_device: int = int(os.getenv("NER_DEVICE", "-1"))

    # Minimum confidence score to accept an NER span (0.0-1.0).
    ner_min_score: float = float(os.getenv("NER_MIN_SCORE", "0.70"))

    # Max characters per chunk passed to NER model (keeps sub-token count < 512).
    ner_max_chars: int = int(os.getenv("NER_MAX_CHARS", "500"))

    # Paths
    data_raw:       Path = ROOT_DIR / "data" / "raw"
    data_processed: Path = ROOT_DIR / "data" / "processed"
    storage:        Path = ROOT_DIR / "storage"

    vector_index_path: Path = storage / "docs.faiss"
    vector_meta_path:  Path = storage / "docs_meta.json"
    graph_path:        Path = storage / "knowledge.graphml"

    memory_db_path:    Path = storage / "memory.sqlite"
    memory_index_path: Path = storage / "memory.faiss"
    memory_meta_path:  Path = storage / "memory_meta.json"

    reward_index_path: Path = storage / "reward_memory.faiss"
    reward_meta_path:  Path = storage / "reward_memory_meta.json"

    entity_vecs_path:  Path = storage / "entity_vecs.npy"
    entity_names_path: Path = storage / "entity_names.json"

    # QDAP-S trained weights (optional — absent = untrained alpha=0.5 fallback)
    qdap_model_path:   Path = storage / "qdap_s.npz"

    # EHRAG hypergraph artefacts
    hypergraph_path:        Path = storage / "hypergraph"
    chunk_vecs_path:        Path = storage / "chunk_vecs.npy"
    chunk_ids_path:         Path = storage / "chunk_ids.json"

    def ensure_dirs(self) -> None:
        self.data_raw.mkdir(parents=True, exist_ok=True)
        self.data_processed.mkdir(parents=True, exist_ok=True)
        self.storage.mkdir(parents=True, exist_ok=True)
        self.hypergraph_path.mkdir(parents=True, exist_ok=True)

    def resolve_doc_type(self, filename: str) -> str:
        name = filename.lower()
        for keyword, doc_type in self.doc_type_map.items():
            if keyword in name:
                return doc_type
        return self.default_doc_type

settings = Settings()
