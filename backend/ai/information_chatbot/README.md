# STELLAR-RAG v4

Vietnamese university Q&A system with EHRAG hypergraph retrieval and HybGRAG critic validation.

**Papers implemented:**
- EHRAG: Entity Hypergraph for Retrieval-Augmented Generation (arXiv 2604.17458)
- HybGRAG: Hybrid Retrieval-Augmented Generation (arXiv 2412.16311)

---

## Architecture overview

```
Query
  |
[InputGuardrail]       -- injection detection + safety classification
  |
[LRU Cache]            -- 256-entry thread-safe cache
  |
[QueryProcessor]       -- entity extraction + sub-query split
  |
[QueryRouter]          -- complexity: simple | medium | complex
  |
[QueryExpander]        -- LLM paraphrase variants (skip if simple)
  |
[HyDE]                 -- hypothetical document embedding (analytical+complex only)
  |
+-- HybGRAG Critic Loop (max 3 iterations) ---+
|                                              |
|  +-- Parallel retrieval --+                 |
|  | BM25 | FAISS/bge-m3 | Graph |           |
|  +----------+-------------+                 |
|             |                               |
|  [QDAP-S Fusion]   -- query-adaptive blend  |
|  [Doc-type Boost]  -- intent boost          |
|  [EHRAG Rescore]   -- hypergraph diffusion  |
|  [CE Reranker]     -- CrossEncoder top-20   |
|  [Self-RAG]        -- quality check         |
|  [Critic Validator] -- sufficient? -> break |
|  [Critic Commenter] -> enrich query         |
+----------------------------------------------+
  |
[Context Assembly]     -- MMR diversity + score-proportional budget
  |
[LLM Generation]       -- Ollama (primary) or Cloud LLM (secondary/dual)
  |
[OutputGuardrail]      -- grounding check
  |
Answer
```

---

## Project structure

```
STELLAR-RAG/
+-- README.md
+-- requirements.txt
+-- .env.example
+-- .gitignore
+-- app.py                       # interactive chat entry point
+-- ingest.py                    # ingest pipeline (PDF -> FAISS + BM25 + Graph + Hypergraph)
+-- src/                         # core library modules
|   +-- agent.py
|   +-- graphrag.py
|   +-- hypergraph.py
|   +-- critic.py
|   +-- llm_client.py
|   +-- cloud_llm_client.py
|   +-- config.py
|   +-- embedding.py
|   +-- vector_store.py
|   +-- qdap.py
|   +-- memory.py
|   +-- reranker.py
|   +-- router.py
|   +-- guardrail.py
|   +-- query_expander.py
|   +-- pdf_extractor.py
|   +-- pdf_pipeline.py
|   +-- ner_extractor.py
|   +-- speech.py
|   +-- stt_worker.py
|   +-- rlhf.py
+-- scripts/                     # operational/maintenance scripts
|   +-- rlhf_train.py
|   +-- verify_chunks.py
+-- eval/                        # evaluation suite
|   +-- evaluate.py
|   +-- pipeline.py
|   +-- metrics.py
|   +-- qa_dataset.json
|   +-- results/
+-- data/
|   +-- raw/                     # input PDF files
|   +-- processed/               # chunks.json + full OCR text
+-- storage/                     # index and model files (git-ignored, built at runtime)
+-- docs/
    +-- ARCHITECTURE.md
    +-- RETRIEVAL_PIPELINE.md
    +-- IMPROVEMENTS.md
    +-- HYPERGRAPH_EHRAG.md
    +-- MATH_FOUNDATIONS.md
```

---

## Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- Pull required models:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:0.5b
```

### Setup

```bash
cd STELLAR-RAG

python -m venv .venv

# With GPU (CUDA 12.4)
.venv\Scripts\pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# CPU-only
.venv\Scripts\pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and edit:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
EMBEDDING_BACKEND=sentence_transformers
EMBED_MODEL=BAAI/bge-m3

# Optional: Cloud LLM for dual-answer mode and graph extraction
CLOUD_PROVIDER=groq
CLOUD_API_KEY=gsk_...
CLOUD_MODEL=llama-3.3-70b-versatile
CLOUD_GRAPH_MODEL=llama-3.1-8b-instant        # entity+relation mode (--no-ner)
CLOUD_RELATION_MODEL=llama-3.3-70b-versatile  # relation-only mode (NER default)

# NER settings — ingest pipeline only, no effect at query time
NER_MODEL=NlpHUST/ner-vietnamese-electra-base
NER_DEVICE=-1          # -1 = CPU (recommended), 0 = first GPU
NER_MIN_SCORE=0.70     # minimum confidence to accept an NER span
```

---

## Usage

### 1. Ingest PDFs

Place PDF files in `data/raw/` then run:

```bash
# Default: NER entities (local) + LLaMA 70B relations (Groq cloud)
.venv\Scripts\python ingest.py

# Original mode: LLM extracts both entities and relations (no NER pre-pass)
.venv\Scripts\python ingest.py --no-ner

# Skip all LLM graph extraction: fast regex NER only (no API key needed)
.venv\Scripts\python ingest.py --skip-graph

# Dry run: list files and exit
.venv\Scripts\python ingest.py --dry-run
```

**Graph extraction modes:**

| Mode | Entity source | Relation source | API tokens | Notes |
|------|--------------|-----------------|------------|-------|
| Default (NER+LLM) | NER model (local) | LLaMA 70B (Groq) | ~500/call | Recommended |
| `--no-ner` | LLM (cloud) | LLM (cloud) | ~1000/call | Original behavior |
| `--skip-graph` | Regex only | Regex only | 0 | No API key needed |

The NER model (`NlpHUST/ner-vietnamese-electra-base`, ~270 MB) downloads automatically on first ingest. It runs on CPU and is unloaded before the cloud LLM phase to minimise peak RAM usage.

Ingest builds:

| File | Description |
|------|-------------|
| `storage/docs.faiss` | FAISS dense index |
| `storage/docs_meta.json` | Chunk metadata |
| `storage/bm25_index.pkl` | BM25 index |
| `storage/knowledge.graphml` | Knowledge graph (NetworkX DiGraph) |
| `storage/entity_vecs.npy` | Entity embeddings |
| `storage/entity_names.json` | Entity name list |
| `storage/chunk_vecs.npy` | Chunk embeddings for hypergraph diffusion |
| `storage/chunk_ids.json` | Chunk ID list |
| `storage/hypergraph/` | EHRAG hypergraph artefacts (H^str, H^sem) |

### 2. Verify chunk quality

```bash
.venv\Scripts\python scripts/verify_chunks.py
```

### 3. Chat

```bash
.venv\Scripts\python app.py
```

On startup, choose:

- **T** -- Text mode (keyboard input)
- **S** -- Speech mode (Vietnamese voice input via PhoWhisper + edge-tts output)

Then choose answer mode:

- **1** -- Single LLM (follows `LLM_BACKEND` setting)
- **2** -- Dual mode (Ollama + Cloud LLM in parallel)

Special commands during chat:

| Command | Action |
|---------|--------|
| `exit` / `quit` | Exit the chat |
| `?debug` | Show retrieval context, hit scores, and LLM prompt |
| `?clear-memory` | Wipe conversation memory and cache |

After each answer, type a rating `1-5` to update QDAP-S online (press Enter to skip).

### 4. Evaluation

Two eval scripts — pick based on what you need:

| Script | LLM judge | Retrieval trace | Use when |
|--------|-----------|-----------------|----------|
| `evaluate.py` | Yes (Ollama scores 0–10) | No | Final quality assessment |
| `pipeline.py` | No (auto nDCG/grounding) | Yes (top chunks, scores) | Daily retrieval debugging |

---

**`evaluate.py`** — full eval with LLM-as-judge + composite scoring:

```bash
# Single LLM (Ollama only)
.venv\Scripts\python eval/evaluate.py

# Dual mode: compare Ollama vs Cloud LLM side-by-side
.venv\Scripts\python eval/evaluate.py --dual

# Filter by category; limit to 10 questions
.venv\Scripts\python eval/evaluate.py --dual --category medium --limit 10

# Start from question N (skip Q1–Q9)
.venv\Scripts\python eval/evaluate.py --start 10

# Continue a stopped run: start from Q10, merge Q1–Q9 from checkpoint
.venv\Scripts\python eval/evaluate.py --start 10 --resume eval/results/eval_single_20260531_143022.json

# Same in dual mode
.venv\Scripts\python eval/evaluate.py --dual --start 10 --resume eval/results/eval_dual_20260531_143022.json
```

> Results are auto-saved to `eval/results/eval_single_YYYYMMDD_HHMMSS.json` (every 5 questions).  
> Pass that file to `--resume` to continue exactly where you stopped.

---

**`pipeline.py`** — lightweight debug pipeline (retrieval trace, nDCG, grounding):

```bash
# Run all questions (qa_dataset.json, no eval_prompts.txt needed)
.venv\Scripts\python eval/pipeline.py eval/qa_dataset.json

# Dual mode (Ollama + Cloud)
.venv\Scripts\python eval/pipeline.py eval/qa_dataset.json --dual

# Filter by category; first 5 questions only
.venv\Scripts\python eval/pipeline.py eval/qa_dataset.json --category hard --limit 5

# Start from question 10
.venv\Scripts\python eval/pipeline.py eval/qa_dataset.json --start 10

# Continue stopped run: start from Q10, merge Q1–Q9 from prior report
.venv\Scripts\python eval/pipeline.py eval/qa_dataset.json --start 10 --resume eval/eval_report.json

# Dual mode from Q10
.venv\Scripts\python eval/pipeline.py eval/qa_dataset.json --dual --start 10 --resume eval/eval_report.json
```

> Output: `eval/eval_report.json` (all results merged) + `eval/eval_log.txt` (detailed trace per question).

---

**Shared flags (both scripts):**

| Flag | Default | Description |
|------|---------|-------------|
| `--start N` | 1 | Start from question number N (1-based), skip Q1..Q(N-1) |
| `--resume FILE` | — | Load prior results from FILE and merge into output |
| `--dual` | off | Run Ollama + Cloud LLM in parallel |
| `--category` | all | Filter: `easy` / `medium` / `hard` |
| `--limit N` | 0 (all) | Stop after N questions |

**Typical workflow for a stopped run:**

```
Run → stopped at Q9 → results saved
         ↓
--start 10 --resume <saved_file>   # continue from Q10, final file has Q1–Q60
```

### 5. Offline RLHF training

```bash
# Full batch update + export JSONL
.venv\Scripts\python scripts/rlhf_train.py

# Incremental (only new feedback since last run)
.venv\Scripts\python scripts/rlhf_train.py --incremental

# Export JSONL only, no QDAP update
.venv\Scripts\python scripts/rlhf_train.py --export-only

# Dry run (stats only)
.venv\Scripts\python scripts/rlhf_train.py --dry-run
```

RLHF exports:

| File | Format |
|------|--------|
| `eval/rlhf_exports/preference_pairs.jsonl` | DPO pairs (chosen / rejected) |
| `eval/rlhf_exports/sft_positives.jsonl` | SFT positives (reward >= 4) |

---

## Speech mode (STT + TTS)

| Component | Technology | Notes |
|-----------|-----------|-------|
| STT | PhoWhisper-medium (VinAI) | Subprocess isolation -- full VRAM freed before Ollama |
| TTS | edge-tts vi-VN-HoaiMyNeural | HTTP API, no GPU, requires internet |
| Recording | sounddevice + soundfile | 16 kHz mono WAV |
| Playback | pygame.mixer | Overwrites `storage/speech_output.mp3` |

Install speech dependencies:

```bash
.venv\Scripts\pip install transformers sounddevice soundfile edge-tts pygame torch
```

---

## Configuration reference

### Core retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Main generation model |
| `EMBED_MODEL` | `BAAI/bge-m3` | Embedding model (1024-dim, multilingual) |
| `EMBEDDING_BACKEND` | `sentence_transformers` | `sentence_transformers` or `ollama` |
| `TOP_K` | `6` | Final retrieved chunks per query |
| `FUSION_METHOD` | `qdap_s` | `qdap_s` or `rrf` |
| `QDAP_GRAPH_WEIGHT` | `0.15` | Graph contribution weight |
| `HYDE_ENABLED` | `true` | HyDE (analytical complex queries only) |
| `SELF_RAG_ENABLED` | `true` | Self-RAG quality expansion |
| `RERANKER_ENABLED` | `true` | CrossEncoder reranking |
| `EMBED_BATCH_SIZE` | `4` | Embedding batch size (OOM prevention) |

### HybGRAG critic

| Variable | Default | Description |
|----------|---------|-------------|
| `CRITIC_ENABLED` | `true` | Enable/disable critic loop |
| `CRITIC_MAX_ITERATIONS` | `3` | Max retrieval-refinement iterations |
| `CRITIC_MODEL` | `qwen2.5:0.5b` | Fast model for critic calls |
| `CRITIC_SKIP_THRESHOLD` | `0.5` | Self-RAG quality threshold to skip critic |

### EHRAG hypergraph

| Variable | Default | Description |
|----------|---------|-------------|
| `BIRCH_THRESHOLD` | `0.5` | BIRCH merge distance |
| `HYPERGRAPH_DIFFUSE_T` | `3` | Structural propagation iterations |
| `HYPERGRAPH_LAMBDA1` | `0.3` | Entity evidence weight lambda1 |
| `HYPERGRAPH_LAMBDA2` | `0.2` | Cluster topic weight lambda2 |
| `HYPERGRAPH_TAU` | `1.0` | Gaussian temperature tau |
| `HYPERGRAPH_GAMMA` | `0.5` | Semantic expansion decay gamma |

### Cloud LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `ollama` | `ollama` \| `cloud` \| `both` |
| `CLOUD_PROVIDER` | -- | `groq` \| `deepseek` \| `openrouter` \| `together` |
| `CLOUD_API_KEY` | -- | API key for the chosen provider |
| `CLOUD_MODEL` | -- | Model for chat (blank = provider default) |
| `CLOUD_GRAPH_MODEL` | -- | Smaller model for graph extraction (e.g. `llama-3.1-8b-instant`) |

---

## Hardware requirements

| Setup | Notes |
|-------|-------|
| CPU-only | Fully supported. Set `EMBEDDING_BACKEND=sentence_transformers`. |
| GPU | Ollama handles LLM inference. PyTorch GPU used for embeddings. |
| RAM | 8 GB minimum recommended. |
| Disk | ~500 MB for models + index files (typical 50-document corpus). |
