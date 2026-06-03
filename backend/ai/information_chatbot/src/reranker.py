"""
STELLAR-RAG v3 — Cross-encoder reranker.

Uses a lightweight cross-encoder (ms-marco-MiniLM-L-6-v2, ~22 MB) to jointly
score (query, passage) pairs after the bi-encoder + BM25 + graph RRF fusion
step.  Literature result: MRR@3 jumps ~0.10 versus bi-encoder-only retrieval.

Design decisions

* **Lazy singleton** — model is downloaded and loaded only on first call when
  RERANKER_ENABLED=true.  Import-time cost is zero.
* **Top-K pooling** — only the top `reranker_top_k` fused candidates are
  rescored; the tail is left unchanged.  This keeps latency bounded even for
  large candidate pools.
* **Graceful fallback** — any ImportError / load failure silently disables the
  reranker; the rest of the pipeline is unaffected.
* **Score transparency** — original bi-encoder score is preserved as
  'dense_score'; 'score' is overwritten with the CE logit so downstream
  MMR + Organizer see the better signal.

Typical latency: ~15-40 ms for 20 candidates on CPU (MiniLM-L6).
"""
from __future__ import annotations

import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

class Reranker:
    """
    Cross-encoder reranker — lazy-loaded singleton.

    Usage::

        reranker = Reranker.get()
        if reranker:
            candidates = reranker.rerank(query, candidates, top_k=20)
    """

    _instance:  "Reranker | None" = None
    _available: bool | None = None   # None = not yet tested

    # Singleton factory

    @classmethod
    def get(cls) -> "Reranker | None":
        """
        Return the shared Reranker instance, or None if disabled / unavailable.
        Thread-safe for the common read path (CPython GIL covers the attribute
        read; worst case two threads race to create the instance — harmless).
        """
        if not settings.reranker_enabled:
            return None
        if cls._available is False:
            return None
        if cls._instance is not None:
            return cls._instance
        try:
            cls._instance  = cls._create()
            cls._available = True
        except Exception as exc:
            logger.warning(f"[Reranker] disabled — {exc}")
            cls._available = False
        return cls._instance

    @classmethod
    def _create(cls) -> "Reranker":
        obj = object.__new__(cls)
        # Lazy import: sentence-transformers is an optional dependency
        from sentence_transformers import CrossEncoder  # type: ignore[import]
        obj._model = CrossEncoder(
            settings.reranker_model,
            max_length=512,
            device="cpu",
        )
        logger.info(f"[Reranker] loaded '{settings.reranker_model}'")
        return obj

    # Rerank

    def rerank(
        self,
        query:      str,
        candidates: list[dict[str, Any]],
        top_k:      int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Cross-encoder rerank the top `top_k` candidates.

        Args:
            query:      Raw user query string.
            candidates: Fused hit dicts with at least 'text' or 'text_preview'.
            top_k:      How many candidates to score.  Defaults to
                        ``settings.reranker_top_k``.

        Returns:
            Same list, with 'score' replaced by CE logit for the pool and
            'dense_score' holding the original bi-encoder score.
            Items beyond top_k retain their original scores and are appended
            after the reranked pool.
        """
        if not candidates:
            return candidates

        pool_size = top_k if top_k is not None else settings.reranker_top_k
        pool      = candidates[:pool_size]
        tail      = candidates[pool_size:]

        # Build (query, passage) pairs
        pairs: list[tuple[str, str]] = [
            (query, (c.get("text") or c.get("text_preview") or "")[:512])
            for c in pool
        ]

        # Skip if all passages are empty
        if not any(p[1] for p in pairs):
            return candidates

        try:
            ce_scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            logger.warning(f"[Reranker] predict failed — {exc}")
            return candidates

        for item, ce_score in zip(pool, ce_scores):
            item["dense_score"]    = item.get("score", 0.0)
            item["score"]          = float(ce_score)
            item["retrieval_type"] = (item.get("retrieval_type") or "") + "+ce"

        pool.sort(key=lambda x: x["score"], reverse=True)
        return pool + tail
