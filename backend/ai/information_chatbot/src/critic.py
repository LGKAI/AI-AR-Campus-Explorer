"""
HybGRAG Critic module — implements the divide-and-conquer critic from
"HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational
Knowledge Bases" (arxiv 2412.16311).

The critic consists of two cooperating LLM components:

Validator (C_val)
-----------------
  Binary classification: "YES, the retrieved context is sufficient to answer
  the query" / "NO, the context is insufficient or missing key information".

  Input:  query + context + (optional) verbalized reasoning paths
  Output: bool — True = sufficient, False = need more

Commenter (C_com)
-----------------
  Generates structured corrective feedback when validation fails.

  Error taxonomy (Vietnamese domain):
    - "Thiếu thực thể: X"     — Missing Entity
    - "Quan hệ sai: Y"        — Incorrect Relation
    - "Ngữ cảnh không đủ: Z"  — Insufficient Context

  Output: structured feedback string used to enrich the next retrieval query.

Design decisions
----------------
  * Uses the fast critic model (default: qwen2.5:0.5b, ~300 MB) for both
    components to keep per-iteration overhead ≤ 200 ms.
  * Fail-open: any LLM error causes validate() to return True (sufficient)
    so the pipeline always delivers a context without blocking.
  * Temperature 0 for validator (deterministic YES/NO), 0.2 for commenter
    (minimal creative variability for structured output).
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollama import Client as OllamaClient

from config import settings

logger = logging.getLogger(__name__)

class Critic:
    """
    HybGRAG Critic — validator + commenter for agentic retrieval refinement.

    Usage::

        critic = Critic(ollama_client)
        if not critic.validate(query, context, reasoning_paths):
            feedback = critic.comment(query, context)
            enriched = critic.enrich_query(original_query, feedback)
    """

    # Vietnamese validator prompt
    VALIDATOR_PROMPT: str = (
        "Bạn là hệ thống đánh giá chất lượng ngữ cảnh cho hệ thống hỏi đáp.\n\n"
        "Nhiệm vụ: Đánh giá xem ngữ cảnh được cung cấp có ĐỦ để trả lời câu hỏi không.\n\n"
        "Tiêu chí ngữ cảnh ĐỦ:\n"
        "  - Có thông tin trực tiếp liên quan đến câu hỏi\n"
        "  - Chứa dữ kiện, con số hoặc điều khoản cần thiết\n"
        "  - Không cần thêm tra cứu để trả lời\n\n"
        "Tiêu chí ngữ cảnh KHÔNG ĐỦ:\n"
        "  - Thiếu thông tin về thực thể chính trong câu hỏi\n"
        "  - Không có dữ kiện cụ thể (ngày, số tiền, điều khoản)\n"
        "  - Ngữ cảnh quá mơ hồ hoặc không liên quan\n\n"
        "Câu hỏi: {query}\n\n"
        "Ngữ cảnh:\n{context}\n\n"
        "{reasoning_section}"
        "Chỉ trả lời đúng 1 từ (YES / NO), không giải thích:"
    )

    # Vietnamese commenter prompt
    # Forced single-line structured output so the enriched query contains
    # concrete retrieval terms instead of narrative explanation.
    COMMENTER_PROMPT: str = (
        "Xác định thông tin CỤ THỂ còn thiếu để trả lời câu hỏi này.\n\n"
        "Câu hỏi: {query}\n\n"
        "Ngữ cảnh hiện tại:\n{context}\n\n"
        "Chọn đúng 1 loại và điền tên cụ thể (KHÔNG giải thích, KHÔNG mở đầu bằng 'Dựa trên'):\n"
        "  Thiếu thực thể: [tên thực thể/khái niệm cần tìm]\n"
        "  Thiếu điều khoản: [số điều / tên quyết định cần tra]\n"
        "  Thiếu bảng số liệu: [tên bảng hoặc loại dữ liệu cần]\n\n"
        "Ví dụ tốt:\n"
        "  Thiếu bảng số liệu: bảng xếp loại tốt nghiệp khóa 2020 trở về trước\n"
        "  Thiếu thực thể: điều kiện tốt nghiệp chương trình tiên tiến\n\n"
        "Chỉ trả lời đúng 1 dòng theo mẫu trên, không thêm gì khác:"
    )

    def __init__(
        self,
        client:  "OllamaClient",
        model:   str = "",
    ) -> None:
        """
        Args:
            client: Shared ``ollama.Client`` instance.
            model:  Ollama model for critic calls.  Defaults to
                    ``settings.critic_model`` (env ``CRITIC_MODEL``,
                    default ``"qwen2.5:0.5b"``).
        """
        self.client = client
        self.model  = model or settings.critic_model

    # Validator

    def validate(
        self,
        query:           str,
        context:         str,
        reasoning_paths: str = "",
    ) -> bool:
        """
        Validate whether the retrieved context is sufficient to answer the query.

        Args:
            query:           Original user query.
            context:         Assembled retrieval context string.
            reasoning_paths: Optional verbalized graph paths for additional
                             evidence.  Empty string = not used.

        Returns:
            True  — context is sufficient (proceed to answer generation)
            False — context is insufficient (trigger refinement loop)

        Fail-open: returns True on any LLM error so the pipeline is never
        blocked by critic unavailability.
        """
        if not context or not context.strip():
            return False

        # Build optional reasoning section
        reasoning_section = ""
        if reasoning_paths and reasoning_paths.strip():
            reasoning_section = (
                f"Đường dẫn suy luận:\n{reasoning_paths[:500]}\n\n"
            )

        prompt = self.VALIDATOR_PROMPT.format(
            query=query[:400],
            context=context[:1500],
            reasoning_section=reasoning_section,
        )

        try:
            resp = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 8},
            )
            raw = resp["message"]["content"].strip().upper()

            # Accept YES/NO with optional punctuation
            for token in re.split(r"[\s,.!?]+", raw):
                token = token.strip(".,!?:'\"")
                if token == "YES":
                    return True
                if token == "NO":
                    return False

            # Unrecognised response — fail-open (treat as sufficient)
            logger.debug(f"[Critic] Unrecognised validator response: {raw!r}")
            return True

        except Exception as exc:
            logger.debug(f"[Critic] validate() error: {exc}")
            return True   # fail-open

    # Commenter

    def comment(
        self,
        query:   str,
        context: str,
    ) -> str:
        """
        Generate structured corrective feedback describing what information is
        missing from the current context.

        Args:
            query:   Original user query.
            context: Assembled retrieval context string that was deemed insufficient.

        Returns:
            Feedback string, e.g. "Thiếu thực thể: học phí CNTT 2024.
            Cần thêm thông tin về mức học phí cụ thể theo học kỳ."
            Returns empty string on any error.
        """
        prompt = self.COMMENTER_PROMPT.format(
            query=query[:400],
            context=context[:1500],
        )

        try:
            resp = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 120},
            )
            feedback = resp["message"]["content"].strip()

            # Strict format validation — must start with one of the template prefixes.
            # If the small model hallucinates free text, discard it entirely so
            # garbage feedback never contaminates the retrieval context.
            _VALID_PREFIXES = (
                "Thiếu thực thể:",
                "Thiếu điều khoản:",
                "Thiếu bảng số liệu:",
            )
            if not any(feedback.startswith(p) for p in _VALID_PREFIXES):
                logger.debug(f"[Critic] commenter output rejected (bad format): {feedback[:80]!r}")
                return ""

            return feedback

        except Exception as exc:
            logger.debug(f"[Critic] comment() error: {exc}")
            return ""

    # Query enrichment

    @staticmethod
    def enrich_query(original_query: str, feedback: str) -> str:
        """
        Enrich the retrieval query with critic feedback for the next iteration.

        Appends a structured "[Cần thêm: …]" suffix so that both BM25 and
        dense retrieval see the expanded query terms.

        Args:
            original_query: The user's original query.
            feedback:        Feedback from ``comment()``.

        Returns:
            Enriched query string, or the original query if feedback is empty.
        """
        if not feedback or not feedback.strip():
            return original_query
        return f"{original_query} [Cần thêm: {feedback.strip()}]"
