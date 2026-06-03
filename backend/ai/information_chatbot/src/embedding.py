"""
STELLAR-RAG v3 — Embedder with parallel Ollama batch encoding.

For the sentence-transformers backend, batch encoding is already handled
natively (batch_size=4 default to avoid OOM on bge-m3).

For the Ollama backend, each embedding call is a separate HTTP request.
v3 parallelises these with a ThreadPoolExecutor (max 8 workers) so a batch
of N texts takes ≈ latency_of_one instead of N × latency_of_one.
"""
from __future__ import annotations

import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from ollama import Client
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)

# Ollama batch: cap workers to avoid hammering the local server
_OLLAMA_MAX_WORKERS = 8

# bge-m3 max_seq_length cap — reduces attention memory from O(n²) to O(cap²)
_MAX_SEQ_LENGTH = 512

class Embedder:
    def __init__(self) -> None:
        self.backend = settings.embedding_backend
        if self.backend == "ollama":
            self.client = Client(host=settings.ollama_host)
            self.model  = None
        else:
            device = "cuda" if settings.use_gpu else "cpu"
            self.model = SentenceTransformer(settings.embed_model, device=device)
            # Cap sequence length to prevent O(n²) attention OOM
            if hasattr(self.model, "max_seq_length"):
                self.model.max_seq_length = min(
                    self.model.max_seq_length, _MAX_SEQ_LENGTH
                )
            self.client = None

    # Public API

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts and return an L2-normalised float32 array
        of shape (len(texts), embedding_dim).

        Ollama backend: parallel HTTP requests (up to _OLLAMA_MAX_WORKERS).
        ST backend:     native batched inference (batch_size=64, GPU if set).
        """
        if self.backend == "ollama":
            return self._encode_ollama(texts)
        return self._encode_st(texts)

    # Backends

    def _embed_one_ollama(self, text: str) -> list[float]:
        resp = self.client.embeddings(model=settings.embed_model, prompt=text)
        return resp["embedding"]

    def _encode_ollama(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        if n == 0:
            return np.empty((0, 0), dtype="float32")

        if n == 1:
            # Fast path: skip thread overhead for single text
            vec = self._embed_one_ollama(texts[0])
            arr = np.array([vec], dtype="float32")
        else:
            # Parallel: submit all requests concurrently, collect in order
            workers = min(n, _OLLAMA_MAX_WORKERS)
            results: dict[int, list[float]] = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futs = {
                    executor.submit(self._embed_one_ollama, text): idx
                    for idx, text in enumerate(texts)
                }
                for fut in as_completed(futs):
                    idx = futs[fut]
                    try:
                        results[idx] = fut.result()
                    except Exception as exc:
                        logger.warning(f"[Embedder] failed idx={idx}: {exc}")
                        # Use zero vector as fallback — will score low in cosine search
                        results[idx] = [0.0] * len(next(iter(results.values()), [0.0]))

            arr = np.array([results[i] for i in range(n)], dtype="float32")

        # L2 normalise
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms

    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension for the active backend/model."""
        if self.backend == "ollama":
            if not hasattr(self, "_cached_dim"):
                try:
                    v = self._embed_one_ollama("a")
                    self._cached_dim: int = len(v)
                except Exception:
                    self._cached_dim = 768   # safe fallback
            return self._cached_dim
        else:
            return int(self.model.get_sentence_embedding_dimension())

    def _encode_st(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts in small sub-batches with GC between each to prevent
        CPU/GPU OOM on large models like bge-m3.
        """
        import torch

        sub_bs = settings.embed_batch_size  # default 4
        all_vecs: list[np.ndarray] = []

        for i in range(0, len(texts), sub_bs):
            sub = texts[i : i + sub_bs]
            with torch.no_grad():
                v = self.model.encode(
                    sub,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    batch_size=sub_bs,
                )
            all_vecs.append(v.astype("float32"))
            # Free intermediate tensors between sub-batches
            gc.collect()
            if settings.use_gpu:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        return np.vstack(all_vecs) if len(all_vecs) > 1 else all_vecs[0]
