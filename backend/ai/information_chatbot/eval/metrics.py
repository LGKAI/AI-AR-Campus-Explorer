"""
Evaluation Metrics — STELLAR-RAG v4.

Metrics for evaluating RAG system answer quality:

  1. llm_judge          — LLM scoring (0-10), comparing agent answer vs gold answer
  2. rouge_l            — ROUGE-L (Longest Common Subsequence)
  3. keyword_coverage   — % of key concepts from the gold answer present in agent answer
  4. exact_match        — Exact match after normalisation
  5. citation_check     — Whether the agent cites the relevant article/page
  6. answer_length_ratio — agent/gold length ratio (completeness proxy)
  7. compute_all        — Aggregate all metrics for a single question
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

# ─
# Utility
# ─

def _normalize(text: str) -> str:
    """Lowercase + strip punctuation + normalise Unicode for comparison."""
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _tokenize_vi(text: str) -> list[str]:
    """Simple space-based tokeniser for Vietnamese text."""
    return _normalize(text).split()

# ─
# 1. ROUGE-L
# ─

def _lcs_length(a: list, b: list) -> int:
    """Longest Common Subsequence length — O(|a|×|b|)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Rolling array for memory efficiency
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]

def rouge_l(reference: str, hypothesis: str) -> dict[str, float]:
    """
    Compute ROUGE-L (F1, Precision, Recall) based on token-level LCS.

    Returns dict: {"f1": float, "precision": float, "recall": float}
    """
    ref_tokens = _tokenize_vi(reference)
    hyp_tokens = _tokenize_vi(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    recall    = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4)}

# ─
# 2. Keyword Coverage
# ─

def keyword_coverage(key_concepts: list[str], agent_answer: str) -> float:
    """
    Fraction of key concepts from the gold answer that appear in the agent answer.

    Returns float 0.0 – 1.0.
    """
    if not key_concepts:
        return 1.0

    norm_answer = _normalize(agent_answer)
    hits = sum(1 for kw in key_concepts if _normalize(kw) in norm_answer)
    return round(hits / len(key_concepts), 4)

# ─
# 3. Exact Match
# ─

def exact_match(reference: str, hypothesis: str) -> bool:
    """Exact match after normalising both strings."""
    return _normalize(reference) == _normalize(hypothesis)

# ─
# 4. Citation Check
# ─

def citation_check(agent_answer: str, relevant_article: str) -> bool:
    """
    Check whether the agent cites the relevant article/page in its answer.

    relevant_article example: "Điều 6", "Điều 15"
    """
    if not relevant_article:
        return False
    # Extract number from "Điều 6" → "6"
    match = re.search(r"\d+", relevant_article)
    if not match:
        return False
    article_num = match.group()
    # Match patterns like "điều 6", "Điều 6", "tr.6", "trang 6"
    patterns = [
        rf"[Ðđ][iì][eề]u\s*{article_num}\b",
        rf"tr\.{article_num}\b",
        rf"trang\s*{article_num}\b",
    ]
    for pat in patterns:
        if re.search(pat, agent_answer, re.IGNORECASE):
            return True
    return False

# ─
# 5. Answer Length Ratio
# ─

def answer_length_ratio(gold_answer: str, agent_answer: str) -> float:
    """
    Ratio of agent token count to gold token count.
    > 1.0 : agent is longer (possibly verbose)
    0.5-1.5: acceptable range
    < 0.3  : agent is too short (missing information)
    """
    gold_len  = len(_tokenize_vi(gold_answer))
    agent_len = len(_tokenize_vi(agent_answer))
    if gold_len == 0:
        return 1.0
    return round(agent_len / gold_len, 4)

# ─
# 6. Semantic Similarity (SequenceMatcher fallback when Ollama is unavailable)
# ─

def sequence_similarity(reference: str, hypothesis: str) -> float:
    """
    SequenceMatcher similarity (character level) as a proxy for semantic similarity.
    Used as fallback when Ollama is not available.
    """
    return round(
        SequenceMatcher(None, _normalize(reference), _normalize(hypothesis)).ratio(),
        4,
    )

# ─
# 7. LLM Judge (Ollama)
# ─

_JUDGE_PROMPT = """\
Bạn là giám khảo KHẮT KHE đánh giá chất lượng câu trả lời của hệ thống AI về quy chế đào tạo đại học.

## Câu hỏi
{question}

## Câu trả lời CHUẨN (gold answer)
{gold}

## Câu trả lời của AI cần chấm
{answer}

## HƯỚNG DẪN CHẤM ĐIỂM NGHIÊM NGẶT

Bước 1 — Liệt kê các điểm KEY trong gold answer (số liệu, tên điều, thuật ngữ cụ thể).
Bước 2 — Với mỗi điểm KEY: AI có trả lời đúng không? (✓ đúng / ✗ sai / ~ thiếu)
Bước 3 — Tính % điểm KEY đúng → xác định mức điểm bên dưới.

## THANG ĐIỂM (0–10)
- 9–10 : Đúng và đầy đủ tất cả điểm KEY, không có thông tin sai
- 7–8  : Đúng phần lớn (>70% KEY), thiếu 1–2 chi tiết không quan trọng
- 5–6  : Đúng khoảng 50% KEY, thiếu một số điểm quan trọng
- 3–4  : Đúng <50% KEY hoặc có thông tin SAI đáng kể
- 1–2  : Sai hầu hết, thông tin lạc đề hoặc bịa đặt
- 0    : Không trả lời, từ chối trả lời, hoặc sai hoàn toàn

## BẮT BUỘC cho điểm THẤP khi
- AI nói "không có thông tin", "tài liệu không đề cập" → overall ≤ 3
- AI đưa con số/tên sai so với gold (ví dụ: sai tên Điều, sai giờ, sai GPA) → accuracy ≤ 4
- Câu trả lời chỉ 1 câu chung chung, không có chi tiết cụ thể như gold → completeness ≤ 5

Chỉ trả về JSON hợp lệ (không giải thích, không markdown):
{{"accuracy": 6, "completeness": 5, "relevance": 8, "overall": 6, "comment": "lý do ngắn 1 câu"}}"""

def llm_judge(
    question: str,
    gold_answer: str,
    agent_answer: str,
    ollama_client: Any,
    model: str = "qwen2.5:7b-instruct",
) -> dict[str, Any]:
    """
    Use LLM (Ollama) to score the agent answer against the gold answer.

    Returns dict: {"accuracy": float, "completeness": float,
                   "relevance": float, "overall": float, "comment": str}
    Returns None if Ollama does not respond.
    """
    if agent_answer.strip() == "" or agent_answer.startswith("[Không tìm"):
        return {
            "accuracy": 0.0, "completeness": 0.0, "relevance": 0.0,
            "overall": 0.0, "comment": "Agent không trả lời được"
        }

    prompt = _JUDGE_PROMPT.format(
        question=question,
        gold=gold_answer,
        answer=agent_answer,
    )

    try:
        resp = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        raw = resp.message.content.strip()
        # Extract JSON block from output
        json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if json_match:
            import json
            data = json.loads(json_match.group())
            return {
                "accuracy":     float(data.get("accuracy", 0)),
                "completeness": float(data.get("completeness", 0)),
                "relevance":    float(data.get("relevance", 0)),
                "overall":      float(data.get("overall", 0)),
                "comment":      str(data.get("comment", "")),
            }
    except Exception as e:
        return {
            "accuracy": -1, "completeness": -1, "relevance": -1,
            "overall": -1, "comment": f"LLM error: {e}"
        }

    return {
        "accuracy": -1, "completeness": -1, "relevance": -1,
        "overall": -1, "comment": "JSON parse failed"
    }

# ─
# 8. Compute All Metrics
# ─

def compute_all(
    item: dict,
    agent_answer: str,
    ollama_client: Any | None = None,
    model: str = "qwen2.5:7b-instruct",
    use_llm_judge: bool = True,
) -> dict[str, Any]:
    """
    Compute all metrics for a single question.

    item: dict from qa_dataset.json with keys: question, gold_answer, key_concepts, relevant_article
    agent_answer: answer string from STELLAR-RAG

    Returns a dict with all metrics.
    """
    gold   = item["gold_answer"]
    q      = item["question"]
    keys   = item.get("key_concepts", [])
    art    = item.get("relevant_article", "")

    rl   = rouge_l(gold, agent_answer)
    kwc  = keyword_coverage(keys, agent_answer)
    em   = exact_match(gold, agent_answer)
    cit  = citation_check(agent_answer, art)
    alr  = answer_length_ratio(gold, agent_answer)
    sim  = sequence_similarity(gold, agent_answer)

    result: dict[str, Any] = {
        "id":               item["id"],
        "category":         item["category"],
        "question":         q,
        "gold_answer":      gold,
        "agent_answer":     agent_answer,
        "rouge_l_f1":       rl["f1"],
        "rouge_l_precision":rl["precision"],
        "rouge_l_recall":   rl["recall"],
        "keyword_coverage": kwc,
        "exact_match":      em,
        "citation_ok":      cit,
        "length_ratio":     alr,
        "seq_similarity":   sim,
    }

    # Detect "no answer" responses
    _no_answer_phrases = (
        "không có thông tin", "không tìm thấy", "tài liệu không đề cập",
        "không được đề cập", "không có trong tài liệu",
        "không có thông tin chi tiết", "không đủ thông tin",
    )
    agent_lower = agent_answer.lower()
    is_no_answer = any(ph in agent_lower for ph in _no_answer_phrases)

    # LLM Judge (optional, slower)
    if use_llm_judge and ollama_client is not None:
        judge = llm_judge(q, gold, agent_answer, ollama_client, model)
        result["llm_accuracy"]     = judge["accuracy"]
        result["llm_completeness"] = judge["completeness"]
        result["llm_relevance"]    = judge["relevance"]
        result["llm_overall"]      = judge["overall"]
        result["llm_comment"]      = judge["comment"]
    else:
        proxy = round(sim * 10, 2)
        result["llm_accuracy"]     = proxy
        result["llm_completeness"] = proxy
        result["llm_relevance"]    = proxy
        result["llm_overall"]      = proxy
        result["llm_comment"]      = "(no LLM judge)"

    # Sanity cap: objective metrics override overly-generous LLM judge
    # Rule 1: ROUGE < 0.05 + keyword = 0% → answer is off-topic → cap <= 4
    if rl["f1"] < 0.05 and kwc == 0.0 and result["llm_overall"] > 4.0:
        result["llm_overall"]      = min(result["llm_overall"], 4.0)
        result["llm_accuracy"]     = min(result["llm_accuracy"], 4.0)
        result["llm_completeness"] = min(result["llm_completeness"], 3.0)
        result["llm_comment"]     += " [cap≤4: ROUGE<0.05 & kw=0%]"

    # Rule 2: is_no_answer → cap ≤ 3
    if is_no_answer and result["llm_overall"] > 3.0:
        result["llm_overall"]      = min(result["llm_overall"], 3.0)
        result["llm_accuracy"]     = min(result["llm_accuracy"], 2.0)
        result["llm_completeness"] = min(result["llm_completeness"], 2.0)
        result["llm_comment"]     += " [cap≤3: agent said no info]"

    # Rule 3: length_ratio < 0.15 (too short) + kw < 0.2 → cap <= 5
    if alr < 0.15 and kwc < 0.20 and result["llm_overall"] > 5.0:
        result["llm_overall"]      = min(result["llm_overall"], 5.0)
        result["llm_completeness"] = min(result["llm_completeness"], 4.0)
        result["llm_comment"]     += " [cap≤5: too short & low KW]"

    # Composite score (0–10)
    # Weights: LLM(50%) + ROUGE(20%) + KW(20%) + SeqSim(10%)
    result["composite_score"] = round(
        0.50 * result["llm_overall"] +
        0.20 * (result["rouge_l_f1"] * 10) +
        0.20 * (result["keyword_coverage"] * 10) +
        0.10 * (result["seq_similarity"] * 10),
        4,
    )

    return result

# ─
# 9. Aggregation
# ─

def aggregate(results: list[dict]) -> dict[str, Any]:
    """
    Aggregate evaluation results: averages by category and overall.

    Returns a dict with summary statistics.
    """
    import statistics

    def _mean(vals: list[float]) -> float:
        valid = [v for v in vals if v >= 0]
        return round(statistics.mean(valid), 4) if valid else 0.0

    def _stats(key: str, subset: list[dict]) -> dict:
        vals = [r[key] for r in subset if key in r]
        valid = [v for v in vals if v >= 0]
        if not valid:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(statistics.mean(valid), 4),
            "std":  round(statistics.stdev(valid), 4) if len(valid) > 1 else 0.0,
            "min":  round(min(valid), 4),
            "max":  round(max(valid), 4),
        }

    cats = {"medium", "hard", "comprehensive"}
    by_cat: dict[str, list[dict]] = {c: [r for r in results if r["category"] == c] for c in cats}

    summary: dict[str, Any] = {
        "total_evaluated": len(results),
        "overall": {},
        "by_category": {},
    }

    metric_keys = [
        "llm_overall", "llm_accuracy", "llm_completeness", "llm_relevance",
        "composite_score", "rouge_l_f1", "keyword_coverage", "seq_similarity",
    ]

    # Overall
    for k in metric_keys:
        summary["overall"][k] = _stats(k, results)

    # Binary metrics
    summary["overall"]["exact_match_rate"] = round(
        sum(1 for r in results if r.get("exact_match")) / max(len(results), 1), 4
    )
    summary["overall"]["citation_rate"] = round(
        sum(1 for r in results if r.get("citation_ok")) / max(len(results), 1), 4
    )
    summary["overall"]["pass_rate_6"] = round(   # % score >= 6/10
        sum(1 for r in results if r.get("llm_overall", 0) >= 6) / max(len(results), 1), 4
    )
    summary["overall"]["pass_rate_7"] = round(
        sum(1 for r in results if r.get("llm_overall", 0) >= 7) / max(len(results), 1), 4
    )

    # Per category
    for cat, subset in by_cat.items():
        if not subset:
            continue
        summary["by_category"][cat] = {}
        for k in metric_keys:
            summary["by_category"][cat][k] = _stats(k, subset)
        summary["by_category"][cat]["count"] = len(subset)
        summary["by_category"][cat]["pass_rate_6"] = round(
            sum(1 for r in subset if r.get("llm_overall", 0) >= 6) / max(len(subset), 1), 4
        )
        summary["by_category"][cat]["citation_rate"] = round(
            sum(1 for r in subset if r.get("citation_ok")) / max(len(subset), 1), 4
        )
        summary["by_category"][cat]["keyword_coverage_mean"] = round(
            _mean([r.get("keyword_coverage", 0) for r in subset]), 4
        )

    # Top/Bottom performers
    valid_results = [r for r in results if r.get("llm_overall", -1) >= 0]
    if valid_results:
        sorted_r = sorted(valid_results, key=lambda x: x["llm_overall"], reverse=True)
        summary["top_5"]    = [{"id": r["id"], "score": r["llm_overall"], "q": r["question"][:60]} for r in sorted_r[:5]]
        summary["bottom_5"] = [{"id": r["id"], "score": r["llm_overall"], "q": r["question"][:60]} for r in sorted_r[-5:]]

    return summary
