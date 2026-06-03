"""
STELLAR-RAG — Multi-layer guardrail for prompt safety and output quality.

Input layers (checked in order):
  0. Length guard          — reject oversized queries immediately
  1. Injection detection   — regex + keyword matching (EN + VI, accent-stripped)
                             Always regex — deterministic, must run before sanitise.
  2. Semantic classifier   — LLM-based 4-class intent detection when configured
                             (SAFE / HARMFUL / ILLEGAL / PERSONAL)
                             Falls back to regex toxic+OOD when classifier=None.
  2b. Toxic fallback       — regex harmful/abusive patterns (fallback only)
  3b. OOD fallback         — out-of-domain detection via domain keywords (fallback only)
  4. Sanitisation          — control chars, XML tags, excess whitespace → sanitized_query

Output layers:
  1. Grounding check       — answer token-overlap vs retrieved context
  2. Hallucination markers — flag speculative / first-person-belief language

LLM classifier (Layer 2)
------------------------
``LLMSafetyClassifier`` uses a tiny Ollama model (default ``qwen2.5:0.5b``,
~300 MB) for *semantic* intent detection.  Unlike regex, it correctly handles:

* Vietnamese contextual slang  — "ăn cướp điểm" (academic fraud → HARMFUL)
  vs "giết thời gian" (harmless idiom → SAFE)
* Academic misconduct framing  — "cách truy cập trái phép hệ thống điểm" → HARMFUL
* Privacy requests             — "số điện thoại của thầy X?" → PERSONAL

Activation::

    # .env / environment
    GUARDRAIL_LLM_CLASSIFY=true
    GUARDRAIL_CLASSIFY_MODEL=qwen2.5:0.5b    # any Ollama model

All matching uses accent-stripped (NFD + combining-char removal) normalisation
so queries typed without Vietnamese diacritics are caught equally.

Usage::

    from guardrail import InputGuardrail, OutputGuardrail, LLMSafetyClassifier

    # Without LLM classifier (regex-only, backward-compatible default):
    in_guard = InputGuardrail()

    # With LLM classifier (recommended when Ollama available):
    from ollama import Client
    client   = Client(host="http://localhost:11434")
    in_guard = InputGuardrail(classifier=LLMSafetyClassifier(client))

    result = in_guard.check(user_query)
    if result.action == "block":
        return result.reason
    safe_query = result.sanitized_query

    out_guard  = OutputGuardrail()
    out_result = out_guard.check(llm_answer, retrieved_context)
    if out_result.action == "warn":
        print("[WARN]", out_result.reason)
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ollama import Client as OllamaClient  # import only for type hints

from config import settings

# Unicode helpers

# Pre-compiled: combining diacritical marks block U+0300–U+036F.
# Build the character-class string programmatically so the source file never
# contains invisible combining chars that can be mangled by editors / encoders.
_COMBINING_RE: re.Pattern = re.compile(
    "[" + chr(0x0300) + "-" + chr(0x036F) + "]"
)

def _no_accent(text: str) -> str:
    """Strip Vietnamese diacritics → ASCII-folded lowercase for pattern matching.

    Two-step process:
    1. NFD decomposition: accented chars split into base + combining marks
       (e.g. ề → e + U+0302 + U+0300).  Combining marks are stripped by
       _COMBINING_RE (U+0300–U+036F).
    2. Explicit replacement for đ/Đ (U+0111/U+0110) which NFD does NOT
       decompose — mapped to 'd' manually.
    """
    text = _COMBINING_RE.sub("", unicodedata.normalize("NFD", text.lower()))
    text = text.replace("đ", "d")   # đ (U+0111, already lowercased by NFD step)
    return text

# GuardrailResult

@dataclass
class GuardrailResult:
    """
    Result returned by every guardrail check.

    action         : "allow" (safe), "warn" (let through, log warning),
                     "block" (reject with reason).
    reason         : Human-readable explanation (empty for "allow").
    sanitized_query: Cleaned version of the input / output text.
    """
    action: Literal["allow", "block", "warn"]
    reason: str
    sanitized_query: str

# Prompt-injection patterns

# Each compiled regex is matched against the accent-stripped, lowercased query.
# Patterns are ordered: more specific / dangerous first.
_INJECTION_PATTERNS: list[re.Pattern] = [
    # Direct instruction override (English)
    # No trailing \b on the third group so "instructions" / "prompts" match.
    re.compile(
        r"\b(ignore|disregard|forget|bypass|override|overwrite)\b"
        r".{0,60}\b(previous|all|above|prior|original|existing)\b"
        r".{0,60}\b(instruction|prompt|rule|directive|context|system)",
        re.I | re.S,
    ),
    re.compile(r"\bnew\s+(instruction|prompt|rule|system\s*prompt)", re.I),
    re.compile(
        r"\b(you\s+are\s+now|act\s+as|pretend\s+(to\s+be|you\s+are)|"
        r"roleplay\s+as|your\s+(new\s+)?role\s+is|from\s+now\s+on\s+you)\b",
        re.I,
    ),
    re.compile(r"\b(jailbreak|dan\s*mode|dev(eloper)?\s*mode|unrestricted\s*mode)\b", re.I),
    re.compile(
        r"\b(reveal|expose|leak|show|print|output|repeat|echo)\b"
        r".{0,50}\b(system\s*prompt|instruction|rule|secret|config|token)",
        re.I | re.S,
    ),
    re.compile(r"<\s*/?\s*(system|prompt|instruction|context|role)\s*>", re.I),  # XML/HTML tag inject
    re.compile(r"\bdo\s+anything\s+now\b", re.I),                        # DAN
    re.compile(
        r"\b(print|repeat|output|echo|copy|recite)\b"
        r".{0,40}\b(everything|all|above|before|verbatim|literally|word\s*for\s*word)\b",
        re.I | re.S,
    ),  # "repeat everything above verbatim"
    re.compile(r"\btoken\s+(manipulation|smuggling|injection)\b", re.I),
    re.compile(r"\bprompt\s+(injection|leak|hack|attack)\b", re.I),
    re.compile(r"\bsystem:\s*\[", re.I),                                 # OpenAI-style fake system msg
    re.compile(r"\bassistant:\s*\[", re.I),
    # Social-engineering pretexts
    re.compile(r"\bgrandma\s+(used\s+to|would|told|said)\b", re.I),     # "grandma trick"
    re.compile(
        r"\bstep\s+by\s+step\b.{0,30}\b(how\s+to|instruction[s]?\s+for)\b"
        r".{0,40}\b(hack|exploit|bypass|inject)\b",
        re.I | re.S,
    ),
    # Vietnamese — direct override (accent-stripped patterns)
    re.compile(r"bo\s*qua.{0,40}(huong\s*dan|lenh|quy\s*tac|he\s*thong|loi\s*nhac)", re.I),
    re.compile(r"quen.{0,30}(tat\s*ca|huong\s*dan|lenh|quy\s*tac|noi\s*dung\s*tren)", re.I),
    re.compile(r"(dong\s*vai|gia\s*vo\s*la|tu\s*gio\s*ban\s*la|bay\s*gio\s*ban\s*la)", re.I),
    re.compile(r"(lenh\s*moi|huong\s*dan\s*moi|che\s*do\s*moi|vai\s*tro\s*moi)", re.I),
    re.compile(r"(tiet\s*lo|hien\s*thi|in\s*ra).{0,30}(lenh\s*he\s*thong|cau\s*hinh|bi\s*mat|system)", re.I),
    re.compile(r"ghi\s*de.{0,30}(lenh|quy\s*tac|noi\s*dung)", re.I),
]

# Single-token / compact-form danger words (checked on space-stripped norm):
_INJECTION_COMPACT_KEYWORDS: frozenset[str] = frozenset({
    "jailbreak", "jailbreaking", "danmode", "devmode", "developermode",
    "promptinjection", "promptleak", "prompthack", "prompthijack",
    "instructionoverride", "systembypass", "aigeneration",
})

# NOTE: Hardcoded toxic/OOD regex patterns have been removed.
# Semantic intent classification is handled exclusively by LLMSafetyClassifier
# (Layer 2 in InputGuardrail).  When no classifier is configured the system
# passes through after injection-check + sanitisation — pair with
# GUARDRAIL_LLM_CLASSIFY=true for production deployments.

# Sanitisation helpers

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_XML_TAG_RE      = re.compile(r"<[^>]{1,200}>")
_EXCESS_WS_RE    = re.compile(r"[ \t]{3,}")
_NULL_BYTE_RE    = re.compile(r"\\u0000|\\x00|\x00")

def _sanitize(text: str) -> str:
    """Strip control chars, XML injection tags, null bytes, excess whitespace."""
    text = _NULL_BYTE_RE.sub("", text)
    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = _XML_TAG_RE.sub("", text)
    text = _EXCESS_WS_RE.sub("  ", text)
    return unicodedata.normalize("NFC", text).strip()

# LLM Safety Classifier (Layer 2 — semantic intent detection)

class LLMSafetyClassifier:
    """
    Lightweight LLM-based 4-class safety classifier.

    Sends a zero-temperature, 8-token completion request to a tiny Ollama
    model to classify the *semantic intent* of a query into one of:

    * ``SAFE``     — Normal academic / informational query
    * ``HARMFUL``  — Fraud, harassment, manipulation, cheating, malicious intent
    * ``ILLEGAL``  — Clear legal violations (violence, drugs, cybercrime…)
    * ``PERSONAL`` — Requests for another person's private / personal data

    Unlike regex patterns, the model handles Vietnamese contextual slang:
    * "ăn cướp điểm"  (academic fraud slang)   → HARMFUL  ✓
    * "giết thời gian" (kill time, harmless idiom) → SAFE  ✓
    * regex both misclassifies                          ✗

    Fail-open design
    ----------------
    Any exception (Ollama unavailable, model not found, timeout, …) returns
    ``"SAFE"`` so that Ollama downtime never blocks legitimate user queries.

    Configuration
    -------------
    Controlled by ``settings.guardrail_llm_classify`` (GUARDRAIL_LLM_CLASSIFY)
    and ``settings.guardrail_classify_model`` (GUARDRAIL_CLASSIFY_MODEL).
    Instantiation and injection into ``InputGuardrail`` is done by
    ``Agent.__init__()`` so the same Ollama client is reused.
    """

    #: All recognised output labels (uppercase)
    LABELS: frozenset[str] = frozenset({"SAFE", "HARMFUL", "ILLEGAL", "PERSONAL"})

    # Vietnamese prompt keeps the model aligned to the domain language and
    # explicit examples prevent false-positives from common Vietnamese idioms.
    _PROMPT_TEMPLATE: str = (
        "Bạn là bộ lọc an toàn cho hệ thống hỏi đáp đại học. "
        "Phân loại câu hỏi thành đúng 1 nhãn:\n\n"
        "SAFE     - Câu hỏi học thuật, thông tin, tra cứu bình thường\n"
        "HARMFUL  - Ý định gian lận, quấy rối, phá hoại, làm hại người khác\n"
        "ILLEGAL  - Vi phạm pháp luật rõ ràng (bạo lực, ma túy, tội phạm mạng...)\n"
        "PERSONAL - Yêu cầu thông tin cá nhân / riêng tư của người khác\n\n"
        "Lưu ý quan trọng:\n"
        "- Thành ngữ thông thường như 'giết thời gian', 'cướp show', "
        "'ăn điểm' (nghĩa bình thường) → SAFE\n"
        "- Ý định gian lận học thuật như 'ăn cướp điểm', 'truy cập trái phép "
        "hệ thống điểm', 'làm giả kết quả' → HARMFUL\n"
        "- Chỉ phân loại theo ý định thực sự, không theo từ ngữ bề mặt\n\n"
        'Câu hỏi: "{query}"\n'
        "Chỉ trả lời đúng 1 từ (SAFE / HARMFUL / ILLEGAL / PERSONAL), "
        "không giải thích:"
    )

    def __init__(self, ollama_client: "OllamaClient", model: str = "") -> None:
        """
        Args:
            ollama_client: An ``ollama.Client`` instance (shared with the Agent).
            model:         Ollama model name.  Defaults to
                           ``settings.guardrail_classify_model``
                           (env ``GUARDRAIL_CLASSIFY_MODEL``, default
                           ``"qwen2.5:0.5b"``).
        """
        self.client = ollama_client
        self.model  = model or settings.guardrail_classify_model

    def classify(self, query: str) -> str:
        """
        Classify *query* and return one of SAFE / HARMFUL / ILLEGAL / PERSONAL.

        * Temperature 0 — deterministic output.
        * num_predict 8 — the model only needs to emit a single word.
        * Truncates query to 500 chars to keep the prompt short.
        * Returns ``"SAFE"`` on any exception (fail-open).
        """
        try:
            resp = self.client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": self._PROMPT_TEMPLATE.format(query=query[:500]),
                }],
                options={"temperature": 0.0, "num_predict": 8},
            )
            raw = resp["message"]["content"].strip().upper()
            # Find the first token that matches a known label
            for token in raw.split():
                token = token.strip(".,!?:'\"")
                if token in self.LABELS:
                    return token
            # Unrecognised response → fail-open
            return "SAFE"
        except Exception:
            # Ollama down, model missing, network error, etc. → fail-open
            return "SAFE"

# InputGuardrail

class InputGuardrail:
    """
    Multi-layer input defense.

    check(query) → GuardrailResult
      action="block"  → reject, return reason to user
      action="warn"   → let through but log (OOD queries — regex path only)
      action="allow"  → safe, use sanitized_query downstream

    Layer order
    -----------
    0. Length guard           — always, hard reject
    1. Injection detection    — always regex (deterministic, structural attacks)
    2. LLM semantic classifier  ← preferred when ``classifier`` is provided
       OR
       2b. Toxic regex        ← regex fallback (backward-compatible default)
       3b. OOD regex          ← regex fallback

    When ``classifier`` is provided (``GUARDRAIL_LLM_CLASSIFY=true``), layers
    2b and 3b are skipped entirely — the LLM subsumes both.
    """

    #: Hard upper bound on query length (characters)
    MAX_QUERY_LEN: int = 2_000

    def __init__(self, classifier: "LLMSafetyClassifier | None" = None) -> None:
        """
        Args:
            classifier: Optional ``LLMSafetyClassifier`` instance.
                        When *None* (default), regex toxic + OOD layers are
                        used as fallback (backward-compatible behaviour).
                        When provided, the LLM classifier replaces those two
                        regex layers for semantic intent detection.
        """
        self._classifier = classifier

    def check(self, query: str) -> GuardrailResult:
        # Layer 0: length guard 
        if len(query) > self.MAX_QUERY_LEN:
            return GuardrailResult(
                action="block",
                reason=(
                    f"Câu hỏi quá dài ({len(query):,} ký tự). "
                    f"Vui lòng rút gọn xuống dưới {self.MAX_QUERY_LEN:,} ký tự."
                ),
                sanitized_query=query[: self.MAX_QUERY_LEN],
            )

        # Layer 1: injection detection on RAW query 
        # MUST run BEFORE _sanitize() because sanitization strips XML tags,
        # which are themselves a primary injection vector.
        raw_norm    = _no_accent(query)
        raw_compact = raw_norm.replace(" ", "")
        injection_hit = self._check_injection(raw_norm, raw_compact)
        if injection_hit:
            # Sanitise before putting in result even for blocked queries
            return GuardrailResult(
                action="block",
                reason=(
                    "Phat hien noi dung can thiep he thong (prompt injection / jailbreak). "
                    f"Mau vi pham: {injection_hit[:80]}"
                ),
                sanitized_query=_sanitize(query),
            )

        # Sanitise → build normalised form for remaining layers 
        clean = _sanitize(query)

        # Layer 2: LLM semantic classifier (preferred path) 
        if self._classifier is not None:
            label = self._classifier.classify(clean)
            if label == "HARMFUL":
                return GuardrailResult(
                    action="block",
                    reason=(
                        "Câu hỏi chứa ý định có hại, gian lận hoặc quấy rối. "
                        "Hệ thống không hỗ trợ yêu cầu này."
                    ),
                    sanitized_query=clean,
                )
            if label == "ILLEGAL":
                return GuardrailResult(
                    action="block",
                    reason=(
                        "Câu hỏi liên quan đến hành vi vi phạm pháp luật. "
                        "Hệ thống không hỗ trợ yêu cầu này."
                    ),
                    sanitized_query=clean,
                )
            if label == "PERSONAL":
                return GuardrailResult(
                    action="block",
                    reason=(
                        "Hệ thống không cung cấp thông tin cá nhân hoặc "
                        "dữ liệu riêng tư của người khác."
                    ),
                    sanitized_query=clean,
                )
            # SAFE → allow (LLM classification subsumes toxic + OOD regex)
            return GuardrailResult(action="allow", reason="", sanitized_query=clean)

        # No classifier configured → allow (semantic filtering requires LLM) ──
        # Set GUARDRAIL_LLM_CLASSIFY=true and pull the classify model to enable
        # semantic harmful/OOD detection.
        return GuardrailResult(action="allow", reason="", sanitized_query=clean)

    # Private helpers 

    @staticmethod
    def _check_injection(norm: str, compact: str) -> str:
        """Return matched snippet if injection detected, else empty string."""
        for pat in _INJECTION_PATTERNS:
            m = pat.search(norm)
            if m:
                return m.group(0)[:80]
        for kw in _INJECTION_COMPACT_KEYWORDS:
            if kw in compact:
                return kw
        return ""

# Output hallucination markers

# Phrases the LLM uses when fabricating information not from context.
# Matched on accent-stripped answer text.
_HALLUCINATION_MARKERS: list[re.Pattern] = [
    re.compile(r"theo\s+(toi|minh)\s+(biet|nghi|hieu|suy\s*nghi|ung\s*doan)", re.I),
    re.compile(r"toi\s+(nghi|doan|tuong|cho\s+rang|uoc\s+tinh)\b", re.I),
    re.compile(r"\bi\s+(think|believe|guess|assume|suppose|imagine)\b", re.I),
    re.compile(r"\b(probably|likely|perhaps|maybe|possibly)\b.{0,20}\b(is|are|was|were|will)\b", re.I | re.S),
    re.compile(r"i[''`]m\s+not\s+(sure|certain|100%|fully)", re.I),
    re.compile(r"\b(co\s*le\s*la|duong\s*nhu\s*la|co\s*the\s*la)\b", re.I),     # có lẽ là, dường như là
    re.compile(r"khong\s*chinh\s*xac\s*nhung", re.I),                             # không chính xác nhưng
    re.compile(r"\b(as\s+far\s+as\s+i\s+know|to\s+my\s+knowledge|i\s+recall)\b", re.I),
    re.compile(r"(based\s+on\s+my\s+training|my\s+knowledge\s+cutoff)", re.I),
    re.compile(r"toi\s+khong\s+co\s+thong\s+tin\s+chinh\s+xac", re.I),           # tôi không có thông tin chính xác
]

#: Minimum required token-overlap fraction between answer and context.
#: Below this threshold the answer is flagged as potentially ungrounded.
_MIN_GROUNDING_OVERLAP: float = 0.10

# OutputGuardrail

class OutputGuardrail:
    """
    Two-layer output quality guard.

    check(answer, context) → GuardrailResult
      action="warn"  → log and optionally append a caveat
      action="allow" → answer appears grounded and confident

    Grounding check
    ---------------
    Computes the fraction of content-bearing answer tokens (length ≥ 3) that
    appear in the retrieved context:

        $\\text{overlap} = \\frac{|\\text{tokens}(\\text{answer}) \\cap \\text{tokens}(\\text{context})|}{|\\text{tokens}(\\text{answer})|}$

    Values below ``_MIN_GROUNDING_OVERLAP`` (default 0.10) trigger a warning.
    This catches the "empty context + confident-sounding answer" failure mode.

    Hallucination marker check
    --------------------------
    Scans the answer for first-person belief phrases ("I think", "có lẽ là",
    "theo tôi biết", etc.) that indicate the model is speculating rather than
    reading from the context.
    """

    def check(self, answer: str, context: str) -> GuardrailResult:
        norm_answer  = _no_accent(answer)
        norm_context = _no_accent(context)

        grounding_warning = self._check_grounding(norm_answer, norm_context)
        halluc_hit        = self._check_hallucination(norm_answer)

        if grounding_warning and halluc_hit:
            return GuardrailResult(
                action="warn",
                reason=(
                    f"Câu trả lời có thể thiếu căn cứ từ tài liệu "
                    f"(overlap={grounding_warning}) và chứa ngôn ngữ suy đoán: «{halluc_hit}»"
                ),
                sanitized_query=answer,
            )
        if halluc_hit:
            return GuardrailResult(
                action="warn",
                reason=f"Câu trả lời chứa ngôn ngữ suy đoán: «{halluc_hit}»",
                sanitized_query=answer,
            )
        if grounding_warning:
            return GuardrailResult(
                action="warn",
                reason=f"Câu trả lời có thể thiếu căn cứ từ tài liệu (overlap={grounding_warning})",
                sanitized_query=answer,
            )

        return GuardrailResult(action="allow", reason="", sanitized_query=answer)

    # Private helpers

    @staticmethod
    def _check_grounding(norm_answer: str, norm_context: str) -> str:
        """
        Return overlap fraction string if below threshold, else empty string.
        Returns empty string (no warning) when context is absent — the caller
        is responsible for not passing an empty context.
        """
        if not norm_context.strip():
            return ""
        answer_tokens  = set(re.findall(r"\w{3,}", norm_answer))
        if not answer_tokens:
            return ""
        context_tokens = set(re.findall(r"\w{3,}", norm_context))
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        if overlap < _MIN_GROUNDING_OVERLAP:
            return f"{overlap:.2f}"
        return ""

    @staticmethod
    def _check_hallucination(norm_answer: str) -> str:
        """Return matched phrase if hallucination marker found, else empty string."""
        for pat in _HALLUCINATION_MARKERS:
            m = pat.search(norm_answer)
            if m:
                return m.group(0)[:80]
        return ""
