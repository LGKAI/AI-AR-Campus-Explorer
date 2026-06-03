"""
STELLAR-RAG — Evaluation Pipeline

Reads prompts from a plain-text file OR qa_dataset.json, runs each through
the full RAG pipeline, and produces detailed per-prompt metrics + a summary.

Input formats (auto-detected by file extension)
  .txt   One question per line; lines starting with # are skipped.
  .json  qa_dataset.json format: {"questions":[{"id","question","gold_answer",
         "key_concepts","category"}]}. When JSON is used, key-concept coverage
         is computed against gold_answer automatically.

Metrics captured

Retrieval quality
  num_dense_hits      Number of dense-retrieval chunks after RRF fusion
  num_graph_hits      Number of graph-retrieval hits
  avg_dense_score     Mean RRF/reranker score of retrieved chunks
  max_dense_score     Best score in the retrieved set
  unique_sources      Number of distinct source documents
  context_length      Total chars in the context window sent to the LLM
  self_rag_triggered  Whether Self-RAG triggered a second wider retrieval
  self_rag_quality    Context quality score before Self-RAG decision [0–1]
  hyde_used           Whether HyDE (hypothetical doc) was applied

Query processing
  query_complexity    simple / compound / complex (router decision)
  num_variants        Number of query variants from the expander
  expansion_variants  The actual variant strings

Answer quality
  answer_length       Character count of the LLM's answer
  grounding_overlap   Token overlap between answer and context [0–1]
                      Good answers have >= 0.10; < 0.10 may indicate hallucination
  has_hallucination   Whether speculative markers were detected in the answer
  hallucination_text  The specific marker phrase (if any)
  guardrail_action    allow / warn / block from input guardrail
  key_concept_hits    (JSON mode) number of gold key_concepts found in answer
  key_concept_total   (JSON mode) total key_concepts for the question
  key_concept_pct     (JSON mode) key_concept_hits / key_concept_total [0–1]
  category            (JSON mode) easy / medium / hard

Performance
  total_ms            End-to-end wall-clock time including LLM generation

Usage
-----
  cd <project-root>

  # Use qa_dataset.json directly (recommended — no eval_prompts.txt needed)
  .venv\\Scripts\\python eval/pipeline.py eval/qa_dataset.json

  # Plain-text fallback (one question per line)
  .venv\\Scripts\\python eval/pipeline.py eval/eval_prompts.txt

  # Dual mode (Ollama + Cloud LLM)
  .venv\\Scripts\\python eval/pipeline.py eval/qa_dataset.json --dual

  # Limit to N questions
  .venv\\Scripts\\python eval/pipeline.py eval/qa_dataset.json --limit 10

  # Filter by category (easy / medium / hard)
  .venv\\Scripts\\python eval/pipeline.py eval/qa_dataset.json --category hard

  Output files: eval_report.json, eval_log.txt  (written next to the input file)
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ── Force UTF-8 stdout on Windows (cp1252 can't print Vietnamese)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Project imports
sys.path.insert(0, str(Path(os.path.dirname(os.path.abspath(__file__))).parent / "src"))

from agent import Agent
from guardrail import OutputGuardrail, _no_accent

# Result dataclass

@dataclass
class EvalResult:
    prompt: str

    # Guardrail
    guardrail_action: str = "allow"
    guardrail_reason: str = ""

    # Query processing
    query_complexity:   str        = "simple"
    num_variants:       int        = 1
    expansion_variants: list[str]  = field(default_factory=list)

    # Retrieval
    num_dense_hits:     int        = 0
    num_graph_hits:     int        = 0
    avg_dense_score:    float      = 0.0
    max_dense_score:    float      = 0.0
    unique_sources:     list[str]  = field(default_factory=list)
    context_length:     int        = 0
    self_rag_triggered: bool       = False
    self_rag_quality:   float      = 1.0
    hyde_used:          bool       = False

    # Single-mode answer
    answer:              str   = ""
    answer_length:       int   = 0
    grounding_overlap:   float = 0.0
    has_hallucination:   bool  = False
    hallucination_text:  str   = ""

    # Dual-mode answers
    dual_mode:           bool  = False
    answer_ollama:       str   = ""
    answer_cloud:        str   = ""
    cloud_label:         str   = ""

    # Ollama metrics (dual)
    ollama_length:       int   = 0
    ollama_grounding:    float = 0.0
    ollama_hallucination:bool  = False
    ollama_ndcg:         float = 0.0

    # Cloud metrics (dual)
    cloud_length:        int   = 0
    cloud_grounding:     float = 0.0
    cloud_hallucination: bool  = False
    cloud_ndcg:          float = 0.0

    # Retrieval ranking quality
    ndcg_at_10:  float = 0.0
    qdap_alpha:  float = -1.0

    # Performance
    total_ms: float = 0.0

    # JSON-mode fields (populated only when input is qa_dataset.json)
    question_id:       str        = ""
    category:          str        = ""
    gold_answer:       str        = ""
    key_concepts:      list[str]  = field(default_factory=list)
    key_concept_hits:  int        = 0
    key_concept_total: int        = 0
    key_concept_pct:   float      = 0.0

    # Detailed trace
    retrieved_chunks_log: list[str] = field(default_factory=list)
    context_preview:      str       = ""

# Grounding / hallucination helpers (mirrors OutputGuardrail logic)

_MIN_GROUNDING = 0.10

_HALLUCINATION_PATTERNS: list = [
    re.compile(r"theo\s+(toi|minh)\s+(biet|nghi|hieu|suy\s*nghi|ung\s*doan)", re.I),
    re.compile(r"toi\s+(nghi|doan|tuong|cho\s+rang|uoc\s+tinh)\b", re.I),
    re.compile(r"\bi\s+(think|believe|guess|assume|suppose|imagine)\b", re.I),
    re.compile(r"\b(probably|likely|perhaps|maybe|possibly)\b.{0,20}\b(is|are|was|were|will)\b",
               re.I | re.S),
    re.compile(r"i[''`]m\s+not\s+(sure|certain|100%|fully)", re.I),
    re.compile(r"\b(co\s*le\s*la|duong\s*nhu\s*la|co\s*the\s*la)\b", re.I),
    re.compile(r"khong\s*chinh\s*xac\s*nhung", re.I),
    re.compile(r"\b(as\s+far\s+as\s+i\s+know|to\s+my\s+knowledge|i\s+recall)\b", re.I),
    re.compile(r"(based\s+on\s+my\s+training|my\s+knowledge\s+cutoff)", re.I),
]

def _compute_grounding(answer: str, context: str) -> float:
    if not context.strip():
        return 1.0   # no context available → no check
    a_norm = _no_accent(answer)
    c_norm = _no_accent(context)
    a_tok = set(re.findall(r"\w{3,}", a_norm))
    if not a_tok:
        return 1.0
    c_tok = set(re.findall(r"\w{3,}", c_norm))
    return round(len(a_tok & c_tok) / len(a_tok), 4)

def _detect_hallucination(answer: str) -> str:
    norm = _no_accent(answer)
    for pat in _HALLUCINATION_PATTERNS:
        m = pat.search(norm)
        if m:
            return m.group(0)[:80]
    return ""

# nDCG@k — ranking quality metric (paper Section 3.3)

def compute_ndcg_at_k(hits: list[dict], answer: str, k: int = 10) -> float:
    """
    Compute nDCG@k using answer-grounding as a graded relevance proxy.

    Since we have no external ground-truth labels, we approximate relevance
    by measuring how much each retrieved chunk's text overlaps with the LLM's
    final answer — chunks that contributed to the answer are marked relevant.

    Graded relevance scale (matches paper's 0–3 scheme):
      3  ≥ 5 content tokens shared with the answer  (strongly relevant)
      2  3–4 shared tokens                           (moderately relevant)
      1  1–2 shared tokens                           (weakly relevant)
      0  no shared tokens                            (not relevant)

    DCG@k  = Σ_{i=1}^{k}  rel_i / log₂(i+1)
    IDCG@k = DCG of ideal ordering (sort rel descending)
    nDCG@k = DCG@k / IDCG@k

    Returns 0.0 if no chunks are retrieved or the answer is empty.
    """
    if not hits or not answer.strip():
        return 0.0

    a_norm = _no_accent(answer)
    a_tok  = set(re.findall(r"\w{3,}", a_norm))
    if not a_tok:
        return 0.0

    rel: list[int] = []
    for h in hits[:k]:
        text   = h.get("text") or h.get("text_preview") or ""
        c_norm = _no_accent(text)
        c_tok  = set(re.findall(r"\w{3,}", c_norm))
        n      = len(a_tok & c_tok)
        rel.append(3 if n >= 5 else 2 if n >= 3 else 1 if n >= 1 else 0)

    # Pad to exactly k positions with 0 (missing hits = irrelevant)
    while len(rel) < k:
        rel.append(0)

    dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rel, reverse=True)))

    return round(dcg / idcg, 4) if idcg > 0 else 0.0

# Log formatting

def _fmt_bar(label: str, value: float, width: int = 30) -> str:
    filled = int(value * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"{label:20s} |{bar}| {value:.3f}"

def _render_prompt_log(idx: int, r: EvalResult) -> str:
    lines: list[str] = []
    sep = "═" * 72

    lines.append(f"\n{sep}")
    lines.append(f"PROMPT {idx:02d}: {r.prompt}")
    lines.append(sep)

    lines.append("\n── GUARDRAIL ──────────────────────────────────────────────")
    lines.append(f"  Action : {r.guardrail_action.upper()}")
    if r.guardrail_reason:
        lines.append(f"  Reason : {r.guardrail_reason}")

    lines.append("\n── QUERY PROCESSING ────────────────────────────────────────")
    lines.append(f"  Complexity : {r.query_complexity}")
    lines.append(f"  Variants   : {r.num_variants}")
    for i, v in enumerate(r.expansion_variants):
        lines.append(f"    [{i}] {v}")

    lines.append("\n── RETRIEVAL ───────────────────────────────────────────────")
    lines.append(f"  Dense hits     : {r.num_dense_hits}")
    lines.append(f"  Graph hits     : {r.num_graph_hits}")
    lines.append(f"  Avg score      : {r.avg_dense_score:.4f}")
    lines.append(f"  Max score      : {r.max_dense_score:.4f}")
    lines.append(f"  Unique sources : {len(r.unique_sources)}  →  {', '.join(r.unique_sources) or '—'}")
    lines.append(f"  Context length : {r.context_length:,} chars")
    lines.append(f"  HyDE           : {'YES' if r.hyde_used else 'NO'}")
    lines.append(f"  Self-RAG       : quality={r.self_rag_quality:.3f}  triggered={r.self_rag_triggered}")

    if r.retrieved_chunks_log:
        lines.append("\n── TOP RETRIEVED CHUNKS ────────────────────────────────────")
        for cl in r.retrieved_chunks_log:
            lines.append(cl)

    lines.append("\n── CONTEXT SENT TO LLM (first 600 chars) ──────────────────")
    lines.append(f"  {(r.context_preview or '(empty)')[:600]}")

    if r.key_concept_total > 0:
        lines.append("\n── KEY CONCEPTS (gold answer) ──────────────────────────────")
        lines.append(f"  Coverage: {r.key_concept_hits}/{r.key_concept_total} ({r.key_concept_pct:.0%})")
        answer_check = _unaccent_simple(r.answer)
        for kc in r.key_concepts:
            found  = _unaccent_simple(kc) in answer_check
            marker = "OK  " if found else "MISS"
            lines.append(f"    [{marker}] {kc}")

    if r.dual_mode:
        lines.append(f"\n── ANSWER — Ollama ({r.answer_ollama[:600]})")
        lines.append(f"\n── ANSWER — {r.cloud_label} ({r.answer_cloud[:600]})")
        lines.append("\n── METRICS (DUAL) ──────────────────────────────────────────")
        lines.append(f"  {'':20s}  {'Ollama':>10}  {r.cloud_label:>12}")
        lines.append(f"  {'Answer length':20s}  {r.ollama_length:>10}  {r.cloud_length:>12}")
        lines.append(f"  {'Grounding overlap':20s}  {r.ollama_grounding:>10.3f}  {r.cloud_grounding:>12.3f}")
        lines.append(f"  {'Hallucination':20s}  {'YES' if r.ollama_hallucination else 'NO':>10}  "
                     f"{'YES' if r.cloud_hallucination else 'NO':>12}")
        lines.append(f"  {'nDCG@10':20s}  {r.ollama_ndcg:>10.4f}  {r.cloud_ndcg:>12.4f}")
    else:
        lines.append(f"\n── ANSWER ──────────────────────────────────────────────────")
        lines.append(f"  {r.answer[:800]}")
        lines.append("\n── METRICS ─────────────────────────────────────────────────")
        overlap_lbl = "GOOD" if r.grounding_overlap >= _MIN_GROUNDING else "LOW"
        ndcg_lbl    = "GOOD" if r.ndcg_at_10 >= 0.70 else ("OK" if r.ndcg_at_10 >= 0.40 else "LOW")
        lines.append(f"  Answer length      : {r.answer_length}")
        lines.append(f"  nDCG@10            : {r.ndcg_at_10:.4f}  ({ndcg_lbl})")
        lines.append(f"  Grounding overlap  : {r.grounding_overlap:.3f}  ({overlap_lbl})")
        lines.append(f"  Hallucination      : {'YES — «' + r.hallucination_text + '»' if r.has_hallucination else 'NONE'}")

    lines.append(f"  Total time         : {r.total_ms:.0f} ms")
    lines.append("")
    return "\n".join(lines)

def _render_summary(results: list[EvalResult]) -> str:
    lines: list[str] = []
    sep  = "═" * 72
    dual = any(r.dual_mode for r in results)

    lines.append(f"\n{sep}")
    lines.append(f"SUMMARY  ({len(results)} prompts)  mode={'DUAL' if dual else 'SINGLE'}")
    lines.append(sep)

    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    if dual:
        cloud_label = next((r.cloud_label for r in results if r.cloud_label), "Cloud")
        lines.append(f"\n  {'#':>3}  {'O-nDCG':>7}  {'O-Grd':>6}  {'O-H':>4}  "
                     f"{'C-nDCG':>7}  {'C-Grd':>6}  {'C-H':>4}  "
                     f"{'ms':>6}  Prompt")
        lines.append("  " + "─" * 78)
        for i, r in enumerate(results, 1):
            lines.append(
                f"  {i:>3}  "
                f"{r.ollama_ndcg:>7.4f}  {r.ollama_grounding:>6.3f}  {'Y' if r.ollama_hallucination else 'N':>4}  "
                f"{r.cloud_ndcg:>7.4f}  {r.cloud_grounding:>6.3f}  {'Y' if r.cloud_hallucination else 'N':>4}  "
                f"{r.total_ms:>6.0f}  {r.prompt[:38]}"
            )
        lines.append(f"\n  Columns: O=Ollama  C={cloud_label}  nDCG=nDCG@10  Grd=grounding  H=hallucination")

        o_ndcg = avg([r.ollama_ndcg for r in results])
        c_ndcg = avg([r.cloud_ndcg  for r in results])
        o_grd  = avg([r.ollama_grounding for r in results])
        c_grd  = avg([r.cloud_grounding  for r in results])
        o_hall = sum(1 for r in results if r.ollama_hallucination)
        c_hall = sum(1 for r in results if r.cloud_hallucination)

        lines.append(f"\n  {'':20s}  {'Ollama':>10}  {cloud_label:>12}")
        lines.append(f"  {'Avg nDCG@10':20s}  {o_ndcg:>10.4f}  {c_ndcg:>12.4f}"
                     + (f"  ← {'Ollama' if o_ndcg > c_ndcg else cloud_label} wins" if abs(o_ndcg-c_ndcg) > 0.02 else "  ← TIE"))
        lines.append(f"  {'Avg grounding':20s}  {o_grd:>10.3f}  {c_grd:>12.3f}"
                     + (f"  ← {'Ollama' if o_grd > c_grd else cloud_label} wins" if abs(o_grd-c_grd) > 0.02 else "  ← TIE"))
        lines.append(f"  {'Hallucinations':20s}  {o_hall:>10}  {c_hall:>12}")
    else:
        lines.append(f"\n  {'#':>3}  {'nDCG':>6}  {'α':>5}  {'Grd':>5}  {'H':>2}  "
                     f"{'Src':>3}  {'Ctx':>5}  {'SRq':>5}  {'ms':>6}  Prompt")
        lines.append("  " + "─" * 78)
        for i, r in enumerate(results, 1):
            alpha = f"{r.qdap_alpha:>5.3f}" if r.qdap_alpha >= 0 else "  N/A"
            lines.append(
                f"  {i:>3}  {r.ndcg_at_10:>6.4f}  {alpha}  "
                f"{r.grounding_overlap:>5.3f}  {'Y' if r.has_hallucination else 'N':>2}  "
                f"{len(r.unique_sources):>3}  {r.context_length:>5}  "
                f"{r.self_rag_quality:>5.3f}  {r.total_ms:>6.0f}  {r.prompt[:46]}"
            )
        ndcg = [r.ndcg_at_10 for r in results]
        grd  = [r.grounding_overlap for r in results]
        hall = [r for r in results if r.has_hallucination]
        lat  = [r.total_ms for r in results]
        lines.append(f"\n  Avg nDCG@10    : {avg(ndcg):.4f}")
        lines.append(f"  Avg grounding  : {avg(grd):.3f}")
        lines.append(f"  Hallucinations : {len(hall)}/{len(results)}")
        lines.append(f"  Avg latency    : {avg(lat):.0f} ms")

    blk = [r for r in results if r.guardrail_action == "block"]
    srt = [r for r in results if r.self_rag_triggered]
    lat = [r.total_ms for r in results]
    lines.append(f"\n  Guardrail blocks   : {len(blk)}/{len(results)}")
    lines.append(f"  Self-RAG triggered : {len(srt)}/{len(results)}")
    lines.append(f"  Avg latency        : {avg(lat):.0f} ms  |  "
                 f"Min/Max: {min(lat):.0f}/{max(lat):.0f} ms")
    lines.append(f"\n{sep}")
    return "\n".join(lines)

# Main evaluation loop

def _unaccent_simple(text: str) -> str:
    """Strip Vietnamese diacritics for keyword matching."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    ).replace("đ", "d")


def _count_key_concept_hits(answer: str, key_concepts: list[str]) -> int:
    """Count how many gold key_concepts appear in the answer (diacritic-insensitive)."""
    if not key_concepts:
        return 0
    norm_answer = _unaccent_simple(answer)
    hits = 0
    for concept in key_concepts:
        if _unaccent_simple(concept) in norm_answer:
            hits += 1
    return hits


def load_prompts(path: str) -> list[str]:
    """Load questions from a plain-text file (one per line, # lines skipped)."""
    prompts: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def load_dataset(path: str) -> list[dict]:
    """
    Load questions from qa_dataset.json.

    Returns a list of dicts with keys: question, id, category, gold_answer, key_concepts.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("questions", [])


def load_input(
    path: str,
    category: str = "all",
    limit: int = 0,
) -> tuple[list[dict], bool]:
    """
    Load questions from either a .txt or .json file.

    Returns:
        (items, is_json)
        items    — list of dicts with at least {"question": str}
                   JSON items also have id, category, gold_answer, key_concepts.
        is_json  — True when the source was a JSON dataset.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        items = load_dataset(path)
        if category != "all":
            items = [q for q in items if q.get("category") == category]
        if limit > 0:
            items = items[:limit]
        return items, True
    else:
        # Plain-text: wrap each line in a minimal dict
        questions = load_prompts(path)
        if limit > 0:
            questions = questions[:limit]
        return [{"question": q} for q in questions], False

def _fill_retrieval_fields(r: EvalResult, di: dict[str, Any], prompt: str) -> None:
    """Populate retrieval fields on EvalResult from agent.debug_info."""
    r.guardrail_action   = di.get("guardrail_action", "allow")
    r.guardrail_reason   = di.get("guardrail_reason", "")
    r.query_complexity   = di.get("query_complexity", "simple")
    r.expansion_variants = di.get("expansion_variants", [prompt])
    r.num_variants       = len(r.expansion_variants)
    r.num_dense_hits     = di.get("num_dense_hits", 0)
    r.num_graph_hits     = di.get("num_graph_hits", 0)
    r.avg_dense_score    = di.get("avg_dense_score", 0.0)
    r.max_dense_score    = di.get("max_dense_score", 0.0)
    r.unique_sources     = di.get("unique_sources", [])
    r.context_length     = di.get("context_length", 0)
    r.self_rag_triggered = di.get("self_rag_triggered", False)
    r.self_rag_quality   = di.get("self_rag_quality", 1.0)
    r.hyde_used          = di.get("hyde_used", False)

    dense_hits = di.get("dense_hits", [])
    r.retrieved_chunks_log = []
    for rank, h in enumerate(dense_hits[:8], 1):
        src   = h.get("source", "?")
        page  = h.get("page", "?")
        sec   = (h.get("section") or "")[:50]
        score = h.get("score", 0.0)
        text  = (h.get("text") or "")[:120].replace("\n", " ")
        r.retrieved_chunks_log.append(
            f"  [{rank}] score={score:.4f} | {src} | p{page}"
            + (f" | {sec}" if sec else "")
            + f"\n       \"{text}\""
        )
    r.context_preview = di.get("context", "")[:1200]

def run_evaluation(
    prompts_file: str  = "eval/qa_dataset.json",
    report_file:  str  = "eval_report.json",
    log_file:     str  = "eval_log.txt",
    dual:         bool = False,
    category:     str  = "all",
    limit:        int  = 0,
    start:        int  = 1,
    resume_file:  str  = "",
) -> list[EvalResult]:
    base = os.path.dirname(os.path.abspath(__file__))

    # Resolve path: try as-is first, then relative to eval/ dir
    input_path = prompts_file if os.path.isabs(prompts_file) else os.path.join(base, prompts_file)
    if not os.path.exists(input_path):
        input_path = os.path.join(os.path.dirname(base), prompts_file)

    report_file = os.path.join(base, report_file)
    log_file    = os.path.join(base, log_file)

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        print("  Tip: run  python eval/pipeline.py eval/qa_dataset.json")
        sys.exit(1)

    items, is_json = load_input(input_path, category=category, limit=limit)
    if not items:
        print("[ERROR] No questions found."); sys.exit(1)

    # --resume: load prior results to merge + skip already-done IDs
    prior_results: list[EvalResult] = []
    done_ids: set[str] = set()
    if resume_file and os.path.exists(resume_file):
        try:
            with open(resume_file, encoding="utf-8") as fh:
                saved = json.load(fh)
            for row in saved:
                prior_results.append(EvalResult(**{
                    k: v for k, v in row.items()
                    if k in EvalResult.__dataclass_fields__
                }))
                if row.get("question_id"):
                    done_ids.add(row["question_id"])
            print(f"[EVAL] Resumed: {len(prior_results)} prior results from {resume_file}")
        except Exception as exc:
            print(f"[WARN] Could not load resume file: {exc}")

    # --start: skip to question index N (1-based, applied after --resume dedup)
    start_idx = max(1, start) - 1
    if start_idx > 0:
        if start_idx >= len(items):
            print(f"[ERROR] --start {start} exceeds dataset size ({len(items)})")
            sys.exit(1)
        items = items[start_idx:]

    # Skip IDs already completed (from --resume)
    if done_ids and is_json:
        items = [it for it in items if it.get("id", "") not in done_ids]

    total_original = start_idx + len(items) + len(prior_results)
    src_fmt  = "JSON (qa_dataset)" if is_json else "TXT"
    mode_str = "DUAL (Ollama + Cloud)" if dual else "SINGLE"
    print(f"[EVAL] source={src_fmt}  |  mode={mode_str}")
    if start_idx > 0 or prior_results:
        print(f"[EVAL] Start Q{start}  |  prior={len(prior_results)}  |  remaining={len(items)}")
    else:
        print(f"[EVAL] {len(items)} questions to run")
    print("[EVAL] Initialising agent…")

    agent = Agent()
    if dual:
        agent.llm_client.ensure_dual()
        from config import settings as _s
        cloud_label = _s.cloud_provider.capitalize()
        print(f"[EVAL] Ollama: {_s.ollama_model}  |  Cloud: {cloud_label}")

    results:   list[EvalResult] = []
    log_parts: list[str]        = []

    for idx, item in enumerate(items, 1):
        prompt       = item.get("question", item.get("prompt", ""))
        gold_answer  = item.get("gold_answer", "")
        key_concepts = item.get("key_concepts", [])
        qid          = item.get("id", f"Q{idx:03d}")
        cat          = item.get("category", "")

        print(f"\n[{idx:02d}/{len(items)}] [{qid}] {prompt[:65]}")
        r  = EvalResult(prompt=prompt)
        t0 = time.perf_counter()

        # Populate JSON-mode metadata
        r.question_id    = qid
        r.category       = cat
        r.gold_answer    = gold_answer
        r.key_concepts   = key_concepts

        if dual:
            # Dual mode
            try:
                ollama_ans, cloud_ans, _ = agent.answer_dual(prompt)
            except Exception as exc:
                ollama_ans = cloud_ans = f"[ERROR: {exc}]"
                print(f"  [!] {exc}")

            elapsed_ms = (time.perf_counter() - t0) * 1000
            r.total_ms = round(elapsed_ms, 1)
            r.dual_mode    = True
            r.cloud_label  = cloud_label
            r.answer_ollama = ollama_ans
            r.answer_cloud  = cloud_ans
            r.answer        = ollama_ans   # primary for guardrail fields

            di = agent.debug_info
            _fill_retrieval_fields(r, di, prompt)
            context = di.get("context", "")
            dense_hits = di.get("dense_hits", [])

            # Metrics for each backend
            r.ollama_length       = len(ollama_ans)
            r.ollama_grounding    = _compute_grounding(ollama_ans, context)
            h_o = _detect_hallucination(ollama_ans)
            r.ollama_hallucination = bool(h_o)
            r.ollama_ndcg         = compute_ndcg_at_k(dense_hits, ollama_ans, k=10)

            r.cloud_length        = len(cloud_ans)
            r.cloud_grounding     = _compute_grounding(cloud_ans, context)
            h_c = _detect_hallucination(cloud_ans)
            r.cloud_hallucination = bool(h_c)
            r.cloud_ndcg          = compute_ndcg_at_k(dense_hits, cloud_ans, k=10)

            # Key concept coverage (JSON mode only)
            if key_concepts:
                r.key_concept_total = len(key_concepts)
                r.key_concept_hits  = _count_key_concept_hits(ollama_ans, key_concepts)
                r.key_concept_pct   = round(r.key_concept_hits / r.key_concept_total, 3)

            print(
                f"  done {elapsed_ms:.0f}ms"
                f" | Ollama nDCG={r.ollama_ndcg:.4f} grd={r.ollama_grounding:.3f}"
                f" hall={'Y' if r.ollama_hallucination else 'N'}"
                + (f" kw={r.key_concept_pct:.0%}" if key_concepts else "")
                + f" | {cloud_label} nDCG={r.cloud_ndcg:.4f} grd={r.cloud_grounding:.3f}"
                f" hall={'Y' if r.cloud_hallucination else 'N'}"
            )
        else:
            # Single mode
            try:
                answer, _ = agent.answer(prompt)
            except Exception as exc:
                answer = f"[EVAL ERROR] {exc}"
                print(f"  [!] {exc}")

            elapsed_ms = (time.perf_counter() - t0) * 1000
            r.total_ms = round(elapsed_ms, 1)
            r.answer   = answer

            di = agent.debug_info
            _fill_retrieval_fields(r, di, prompt)
            context    = di.get("context", "")
            dense_hits = di.get("dense_hits", [])

            r.answer_length     = len(answer)
            r.grounding_overlap = _compute_grounding(answer, context)
            halluc              = _detect_hallucination(answer)
            r.has_hallucination  = bool(halluc)
            r.hallucination_text = halluc
            r.ndcg_at_10        = compute_ndcg_at_k(dense_hits, answer, k=10)
            r.qdap_alpha        = next(
                (float(h.get("qdap_alpha", -1.0)) for h in dense_hits if "qdap_alpha" in h),
                -1.0,
            )

            # Key concept coverage (JSON mode only)
            if key_concepts:
                r.key_concept_total = len(key_concepts)
                r.key_concept_hits  = _count_key_concept_hits(answer, key_concepts)
                r.key_concept_pct   = round(r.key_concept_hits / r.key_concept_total, 3)

            ndcg_lbl = "GOOD" if r.ndcg_at_10 >= 0.70 else ("OK" if r.ndcg_at_10 >= 0.40 else "LOW")
            grd_lbl  = "OK" if r.grounding_overlap >= _MIN_GROUNDING else "LOW"
            print(
                f"  done {elapsed_ms:.0f}ms"
                f" | nDCG@10={r.ndcg_at_10:.4f}[{ndcg_lbl}]"
                f" | grd={r.grounding_overlap:.3f}[{grd_lbl}]"
                f" | hall={'Y' if r.has_hallucination else 'N'}"
                f" | dense={r.num_dense_hits} ctx={r.context_length}c"
                + (f" | kw={r.key_concept_hits}/{r.key_concept_total}({r.key_concept_pct:.0%})" if key_concepts else "")
            )

        results.append(r)
        log_parts.append(_render_prompt_log(idx, r))

    # Merge prior results (from --resume) before saving + summary
    all_results = prior_results + results

    summary_str = _render_summary(results)   # summary over THIS run only
    print(summary_str)
    log_parts.append(summary_str)

    if prior_results:
        print(f"\n[EVAL] Merged with {len(prior_results)} prior results "
              f"→ {len(all_results)} total in output file")

    with open(report_file, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in all_results], fh, ensure_ascii=False, indent=2)
    print(f"[EVAL] JSON report → {report_file}  ({len(all_results)} results)")

    with open(log_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_parts))
    print(f"[EVAL] Detail log  → {log_file}")

    # Print key concept summary across ALL results
    kw_results = [r for r in all_results if r.key_concept_total > 0]
    if kw_results:
        avg_kw = sum(r.key_concept_pct for r in kw_results) / len(kw_results)
        print(f"\n[EVAL] Key concept coverage ({len(kw_results)} questions): {avg_kw:.1%}")
        by_cat: dict[str, list[float]] = {}
        for r in kw_results:
            by_cat.setdefault(r.category or "?", []).append(r.key_concept_pct)
        for cat, vals in sorted(by_cat.items()):
            print(f"  {cat:10s}: {sum(vals)/len(vals):.1%}  ({len(vals)} questions)")

    return all_results

# Entry point

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="STELLAR-RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all questions
  python eval/pipeline.py eval/qa_dataset.json

  # Run single mode, start from question 10
  python eval/pipeline.py eval/qa_dataset.json --start 10

  # Continue a stopped run: start from Q10 and merge Q1-Q9 from prior file
  python eval/pipeline.py eval/qa_dataset.json --start 10 --resume eval/eval_report.json

  # Dual mode (Ollama + Cloud), hard questions only, from Q5
  python eval/pipeline.py eval/qa_dataset.json --dual --category hard --start 5
        """,
    )
    p.add_argument(
        "input_file", nargs="?", default="eval/qa_dataset.json",
        help="Path to qa_dataset.json or plain-text prompts file",
    )
    p.add_argument(
        "--dual", action="store_true",
        help="Evaluate both Ollama and Cloud LLM side-by-side",
    )
    p.add_argument(
        "--category", default="all",
        choices=["all", "easy", "medium", "hard"],
        help="Filter by question category (JSON mode only)",
    )
    p.add_argument(
        "--limit", type=int, default=0,
        help="Stop after N questions (0 = run all)",
    )
    p.add_argument(
        "--start", type=int, default=1,
        help="Start from question number N (1-based). E.g. --start 10 skips Q1-Q9.",
    )
    p.add_argument(
        "--resume", default="",
        dest="resume_file",
        help="Path to a prior eval_report.json to append to. "
             "Use with --start to continue a stopped run.",
    )
    args = p.parse_args()
    run_evaluation(
        prompts_file=args.input_file,
        dual=args.dual,
        category=args.category,
        limit=args.limit,
        start=args.start,
        resume_file=args.resume_file,
    )
