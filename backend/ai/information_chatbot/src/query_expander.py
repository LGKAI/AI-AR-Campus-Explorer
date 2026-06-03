"""
STELLAR-RAG — LLM-based query expansion for paraphrase robustness.

Replaces the previous hardcoded abbreviation map + synonym-group approach
with a lightweight Ollama LLM call.  The model:

  - Expands any Vietnamese abbreviation it knows (CNTT, GPA, KTX, TKB, …)
    without a hand-written dictionary — understands context and intent.
  - Generates semantically equivalent paraphrase variants so retrieval
    covers the full meaning space even when the user types short or
    abbreviated queries.
  - Works for any new abbreviation or term without code changes.

Fail-open design

When Ollama is unavailable or returns unparseable output, the expander
returns only the original query — retrieval quality degrades gracefully
without crashing.

Usage::

    from query_expander import QueryExpander
    from ollama import Client

    expander = QueryExpander(Client())
    variants = expander.expand("Học phí CNTT bao nhiêu?")
    # → ["Học phí CNTT bao nhiêu?",
    #    "Học phí ngành công nghệ thông tin là bao nhiêu?",
    #    "Chi phí học kỳ ngành CNTT?"]

    # Without client (tests / offline):
    expander_offline = QueryExpander()
    variants = expander_offline.expand("GPA tối thiểu?")
    # → ["GPA tối thiểu?"]
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ollama import Client as OllamaClient

from config import settings

def _nfc(text: str) -> str:
    """NFC-normalise text for consistent code-point representation."""
    return unicodedata.normalize("NFC", text)

class LLMQueryExpander:
    """
    LLM-based query expansion — no hardcoded dictionaries.

    Sends a single low-cost Ollama request (temperature=0.3, num_predict=180)
    asking the model to generate up to 2 semantically equivalent alternative
    phrasings for the query.

    The model naturally handles:
    * Abbreviation expansion: "CNTT" → "công nghệ thông tin"
    * Synonym substitution:   "học phí" → "chi phí đào tạo"
    * Register variation:     colloquial → formal academic Vietnamese
    * Any new domain term without code changes

    Returns
    -------
    list[str]
        Always starts with the NFC-normalised original query.
        May contain up to 2 additional LLM-generated variants.
        Length is 1–3.
    """

    _PROMPT_TEMPLATE: str = (
        "Bạn là chuyên gia ngôn ngữ Việt Nam cho hệ thống hỏi đáp về quy chế đào tạo đại học.\n\n"
        "Sinh ra TỐI ĐA 2 biến thể diễn đạt khác cho câu hỏi dưới đây.\n"
        "Yêu cầu mỗi biến thể:\n"
        "  - Giữ nguyên ý nghĩa và phạm vi câu hỏi gốc\n"
        "  - ƯU TIÊN dùng ngôn ngữ văn bản quy chế/pháp lý (không phải ngôn ngữ thường)\n"
        "    Ví dụ: 'không đi học' → 'không tham gia học phần đã đăng ký'\n"
        "           'bỏ học' → 'tự ý bỏ học'\n"
        "           'bị đuổi' → 'buộc thôi học'\n"
        "           'rớt môn' → 'không đạt học phần'\n"
        "           'thi lại' → 'thi kết thúc học phần lần hai'\n"
        "  - Mở rộng viết tắt nếu có (GPA → điểm trung bình tích lũy, CNTT → công nghệ thông tin)\n"
        "  - Ngắn gọn, không thêm thông tin mới\n\n"
        'Câu hỏi gốc: "{query}"\n\n'
        "Trả về JSON thuần túy (không giải thích, không markdown):\n"
        '{{"variants": ["biến thể 1", "biến thể 2"]}}\n\n'
        "Nếu không có biến thể phù hợp:\n"
        '{{"variants": []}}'
    )

    def __init__(
        self,
        ollama_client: "OllamaClient | None" = None,
        model: str = "",
    ) -> None:
        """
        Args:
            ollama_client: Shared ``ollama.Client`` instance.
                           When *None*, ``expand()`` returns only the original
                           query (offline / test mode — no LLM call).
            model:         Ollama model to use.  Defaults to
                           ``settings.ollama_model`` (the main chat model).
                           Using a smaller dedicated model is fine too.
        """
        self.client = ollama_client
        self.model  = model or settings.ollama_model

    def expand(self, query: str) -> list[str]:
        """
        Expand *query* into up to 3 variants (original + ≤2 LLM variants).

        Always returns a list starting with the NFC-normalised original.
        Returns ``[original]`` if the LLM call fails or client is *None*.
        """
        base = _nfc(query.strip())

        if self.client is None:
            return [base]

        try:
            resp = self.client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": self._PROMPT_TEMPLATE.format(query=base[:300]),
                }],
                options={"temperature": 0.3, "num_predict": 180},
            )
            raw = resp["message"]["content"].strip()

            # Parse JSON (model may add prose, backticks, or extra spaces)
            m = re.search(r'\{[^{}]*"variants"\s*:\s*\[[^\]]*\][^{}]*\}', raw, re.S)
            if not m:
                m = re.search(r'\{.*\}', raw, re.S)
            if not m:
                return [base]

            data     = json.loads(m.group(0))
            variants = data.get("variants", [])

            result: list[str] = [base]
            for v in variants[:2]:
                v = _nfc(str(v).strip())
                if v and v != base and v not in result:
                    result.append(v)
            return result

        except Exception:
            # Any error (network, JSON parse, model unavailable) → fail-open
            return [base]

# Public alias — backward-compatible name used throughout the codebase
QueryExpander = LLMQueryExpander
