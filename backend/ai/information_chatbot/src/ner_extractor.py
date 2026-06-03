"""
STELLAR-RAG — Vietnamese NER Extractor

Uses a HuggingFace token-classification pipeline (default: NlpHUST/ner-vietnamese-electra-base)
to extract named entities from Vietnamese text without LLM API calls.

Memory profile:
  Model weights : ~270 MB on disk, ~400 MB in RAM during inference
  Inference     : CPU-only by default (device=-1) to avoid competing with the
                  BAAI/bge-m3 embedding model for GPU VRAM
  After ingest  : call .unload() to release ~400 MB before the LLM relation step

NER tagset (VLSP 2016/2018):
  PER  → PERSON          ORG  → ORGANIZATION
  LOC  → LOCATION        MISC → MISC

Domain entities (always extracted via regex — no model needed):
  QUANTITY : "120 tín chỉ", "3 học kỳ", "70%"
  ARTICLE  : "Điều 15", "Điều 3a"
"""
from __future__ import annotations

import gc
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Map VLSP BIO tag prefixes → canonical type names
_LABEL_MAP: dict[str, str] = {
    "PER":  "PERSON",
    "ORG":  "ORGANIZATION",
    "LOC":  "LOCATION",
    "MISC": "MISC",
}

# Regex for academic/legal domain entities the NER model may miss
_QUANTITY_PAT = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(tín chỉ|tiết|giờ|năm|học kỳ|học phần|%|điểm GPA|điểm)\b",
    re.IGNORECASE,
)
_ARTICLE_PAT = re.compile(r"\bĐiều\s+(\d+[a-z]?)\b")
_KHOAT_PAT   = re.compile(r"\bKhoản\s+(\d+)\b")


class ViNERExtractor:
    """
    Vietnamese Named Entity Recognizer.

    Wraps a HuggingFace token-classification pipeline with:
      - Lazy model loading (loaded on first .extract() call)
      - Domain regex for quantities and article references
      - Confidence filtering (default threshold: 0.70)
      - Deduplication by case-insensitive name
      - .unload() to release RAM between ingest phases

    Example::
        ner = ViNERExtractor()
        entities = ner.extract("Sinh viên cần tích lũy 120 tín chỉ theo Điều 15.")
        # [{"name": "120 tín chỉ", "type": "QUANTITY"},
        #  {"name": "Điều 15",      "type": "ARTICLE"}]
        ner.unload()
    """

    def __init__(
        self,
        model_name:  str   = "NlpHUST/ner-vietnamese-electra-base",
        device:      int   = -1,    # -1 = CPU, 0 = first GPU
        batch_size:  int   = 1,
        min_score:   float = 0.70,
        max_chars:   int   = 500,   # chars before tokenization; keeps sub-tokens < 512
    ) -> None:
        self.model_name = model_name
        self.device     = device
        self.batch_size = batch_size
        self.min_score  = min_score
        self.max_chars  = max_chars
        self._pipeline: Any = None

    # Model lifecycle

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(
                f"[NER] Loading {self.model_name} "
                f"(device={'CPU' if self.device < 0 else f'GPU:{self.device}'}) ..."
            )
            self._pipeline = hf_pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",   # merge B/I tokens into spans
                device=self.device,
                batch_size=self.batch_size,
            )
            logger.info("[NER] Model ready.")
        except Exception as exc:
            logger.warning(
                f"[NER] Could not load '{self.model_name}': {exc}. "
                "Falling back to regex-only entity extraction."
            )
            self._pipeline = None

    def unload(self) -> None:
        """Release model weights from RAM. Call after NER pre-pass is complete."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            logger.info("[NER] Model unloaded — RAM freed.")

    # Extraction

    def extract(self, text: str) -> list[dict[str, str]]:
        """
        Extract named entities from a single text.

        Always runs domain regex first (fast, no model dependency).
        Neural NER is layered on top if the model is available.

        Returns: [{"name": str, "type": str}, ...]
        """
        entities: list[dict[str, str]] = []
        entities.extend(self._regex_entities(text))

        self._load()
        if self._pipeline is not None:
            truncated = text[: self.max_chars]
            try:
                raw = self._pipeline(truncated)
                for item in raw:
                    name  = item.get("word", "").strip()
                    label = item.get("entity_group", "")
                    score = float(item.get("score", 0.0))
                    if not name or len(name) < 2 or score < self.min_score:
                        continue
                    # Strip HuggingFace subword artefacts
                    name = re.sub(r"\s+##", "", name).strip()
                    etype = _LABEL_MAP.get(label.upper().lstrip("BI-"), label.upper())
                    entities.append({"name": name, "type": etype})
            except Exception as exc:
                logger.debug(f"[NER] Inference failed: {exc}")

        return self._deduplicate(entities)

    def extract_chunks(self, chunks: list[Any]) -> dict[str, list[dict[str, str]]]:
        """
        Run NER on a list of Chunk objects in a single pass.

        Returns: {chunk_id: [{"name": str, "type": str}]}
        where chunk_id = f"{source}::Điều{article_number}::p{page}::i{chunk_index}"
        """
        self._load()
        result: dict[str, list[dict[str, str]]] = {}
        total = len(chunks)
        for idx, chunk in enumerate(chunks, 1):
            cid = (
                f"{chunk.source}"
                f"::Điều{getattr(chunk, 'article_number', '')}"
                f"::p{chunk.page}"
                f"::i{chunk.chunk_index}"
            )
            result[cid] = self.extract(chunk.text)
            if idx % 50 == 0 or idx == total:
                logger.info(f"  [NER] {idx}/{total} chunks processed")
        return result

    # Internal helpers

    @staticmethod
    def _regex_entities(text: str) -> list[dict[str, str]]:
        """Extract domain entities via regex (quantities, article refs)."""
        entities: list[dict[str, str]] = []
        for m in _QUANTITY_PAT.finditer(text):
            entities.append({"name": m.group(0).strip(), "type": "QUANTITY"})
        for m in _ARTICLE_PAT.finditer(text):
            entities.append({"name": f"Điều {m.group(1)}", "type": "ARTICLE"})
        for m in _KHOAT_PAT.finditer(text):
            entities.append({"name": f"Khoản {m.group(1)}", "type": "CLAUSE"})
        return entities

    @staticmethod
    def _deduplicate(entities: list[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[str] = set()
        unique: list[dict[str, str]] = []
        for e in entities:
            key = e["name"].lower().strip()
            if key and key not in seen and len(key) >= 2:
                seen.add(key)
                unique.append(e)
        return unique
