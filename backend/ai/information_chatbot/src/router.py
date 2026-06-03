"""
Adaptive Query Router — classifies query complexity to determine retrieval depth.

Three tiers (inspired by Adaptive RAG, Jeong et al. 2024):
  simple  → single-entity factual lookup; BM25 + dense, no graph needed
  medium  → 2-entity or 1-hop relational; light graph traversal (1 hop)
  complex → multi-entity, multi-hop reasoning; full PPR + deep graph

Routing is based on the LLM-extracted ``ProcessedQuery`` attributes
(entities, sub_queries) produced by ``QueryProcessor``.  No hardcoded
keyword lists — if the LLM thinks the query is complex, it will extract
multiple entities / sub-queries, and the router reflects that judgment.
"""
from __future__ import annotations

import re

class QueryRouter:
    """Structural router — uses LLM-extracted entities and sub-queries."""

    # Retrieval parameters per complexity tier
    # top_k bumped: short enumeration queries still need enough chunks to cover
    # all relevant items (e.g. listing all types of học phần).
    _PARAMS: dict[str, dict] = {
        "simple":  {"top_k": 6,  "hops": 0, "use_graph": False},
        "medium":  {"top_k": 10, "hops": 1, "use_graph": True},
        "complex": {"top_k": 15, "hops": 2, "use_graph": True},
    }

    # Fallback tier when the first retrieval attempt returns no useful answer
    _FALLBACK: dict[str, str] = {
        "simple":  "medium",
        "medium":  "complex",
        "complex": "complex",  # already at max
    }

    def fallback_params(self, complexity: str) -> dict:
        """Return params for the next tier up — used on retry after no-answer."""
        fallback = self._FALLBACK.get(complexity, "complex")
        return dict(self._PARAMS[fallback])

    def classify(self, processed_query: object) -> str:
        """
        Returns ``'simple'`` | ``'medium'`` | ``'complex'``.

        Decision is based entirely on the LLM-produced ``ProcessedQuery``
        output — entity count, sub-query count, and raw word count.
        No hardcoded keyword lists.
        """
        entities:  list[str] = getattr(processed_query, "entities",    [])
        sub_qs:    list[str] = getattr(processed_query, "sub_queries",  [])
        original:  str       = getattr(processed_query, "original",     "")

        n_entities    = len(entities)
        n_sub_queries = len(sub_qs)
        word_count    = len(re.findall(r"\w+", original.lower()))

        # Complex: many LLM-extracted entities / sub-queries, or very long query
        if n_sub_queries >= 3 or n_entities >= 3 or word_count >= 30:
            return "complex"

        # Medium: two entities / aspects, or a moderately rich query
        if n_sub_queries >= 2 or n_entities >= 2 or word_count >= 15:
            return "medium"

        return "simple"

    def retrieval_params(self, complexity: str) -> dict:
        """Return ``{top_k, hops, use_graph}`` for the given complexity tier."""
        return dict(self._PARAMS.get(complexity, self._PARAMS["medium"]))
