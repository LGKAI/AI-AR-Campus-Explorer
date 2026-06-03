"""
QDAP-S — Query-Distribution Adaptive Predictor (Small variant)
From: "Query-Adaptive Hybrid Search" (make-08-00091-v3)
      Section 3.2 — QDAP architecture

Architecture

    query_embedding(d)
        │
        ▼
    Linear(d → 101)          W: (101, d), b: (101,)
        │
        ▼
    Conv1D(kernel=7, padding=3)  smooths the logit histogram
        │
        ▼
    Softmax                  → 101-bin probability distribution over α
        │
        ▼
    E[α] = Σ p_i · α_i       α_i = i/100,  i ∈ {0,…,100}

α ∈ [0, 1] is the dense weight in the final hybrid score:
    s_hybrid = α · s'_dense + (1 − α) · s'_sparse

α → 0 : lean sparse/BM25  (exact-match, factual queries)
α → 1 : lean dense/semantic (conceptual, paraphrased queries)

Untrained defaults

Zero linear weights → uniform logits → uniform softmax → E[α] = 0.5
(balanced hybrid — a reasonable baseline without fine-tuning)

Training (optional, offline)

  Loss:     L = 0.62 · L_CE + 0.38 · L_WD
                (cross-entropy + 1-D Wasserstein distance)
  Negatives: antagonist sampling — select negatives that outrank the
             positive in BOTH dense and sparse score spaces.
  Fit on query–document pairs with graded relevance labels,
  then save weights to storage/qdap_s.npz.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

class QDAPSmall:
    """
    QDAP-S: Linear(d, 101) → Conv1D(k=7) → Softmax → E[α].

    Thread-safe for concurrent ``predict_alpha()`` calls (numpy is GIL-safe
    for read-only ops; all state is read-only after __init__/load).
    """

    N_BINS:            int   = 101
    CONV_KERNEL_SIZE:  int   = 7
    _ALPHA_GRID: np.ndarray  = np.linspace(0.0, 1.0, 101, dtype=np.float32)

    def __init__(
        self,
        embed_dim:  int            = 768,
        model_path: Optional[str]  = None,
    ) -> None:
        self.embed_dim = embed_dim

        # Linear layer: W (101, d),  b (101,)
        self._W: np.ndarray = np.zeros((self.N_BINS, embed_dim), dtype=np.float32)
        self._b: np.ndarray = np.zeros(self.N_BINS, dtype=np.float32)

        # Uniform Conv1D kernel = simple moving average over 7 bins
        k = self.CONV_KERNEL_SIZE
        self._kernel: np.ndarray = np.ones(k, dtype=np.float32) / float(k)

        self._trained: bool = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    # Inference

    def predict_alpha(self, query_embedding: np.ndarray) -> float:
        """
        Predict the optimal dense/sparse mixing coefficient α for a query.

        Args:
            query_embedding: shape ``(d,)`` or ``(1, d)`` — the query's
                             dense vector produced by the same embedder
                             used for document indexing.

        Returns:
            α ∈ [0.0, 1.0]  (expected value of the predicted distribution)
        """
        emb = np.asarray(query_embedding, dtype=np.float32).squeeze()
        if emb.ndim == 0 or emb.shape[-1] != self.embed_dim:
            return 0.5   # shape mismatch — return neutral fallback

        #  Linear projection: (101, d) @ (d,) + (101,) → (101,) 
        logits: np.ndarray = self._W @ emb + self._b

        #  Conv1D: edge-pad → convolve → trim to N_BINS 
        pad     = self.CONV_KERNEL_SIZE // 2          # 3
        padded  = np.pad(logits, pad, mode="edge")
        smoothed = np.convolve(padded, self._kernel, mode="valid")[: self.N_BINS]

        #  Softmax (numerically stable) 
        smoothed = smoothed - smoothed.max()
        probs    = np.exp(smoothed)
        probs   /= probs.sum()

        #  Expected α = Σ p_i · α_i 
        return float(np.dot(probs, self._ALPHA_GRID))

    # Online learning (REINFORCE)

    def update_online(
        self,
        query_embedding: np.ndarray,
        alpha_used:      float,
        reward:          float,
        learning_rate:   float = 0.001,
    ) -> None:
        """
        Update linear weights via one REINFORCE step.

        Args:
            query_embedding: The query vector that produced this prediction.
                             Shape ``(d,)`` or ``(1, d)``.
            alpha_used:      The α value that was actually used when the user
                             rated the answer (from ``predict_alpha()``).
            reward:          Scalar in ``[-1, +1]``.
                             > 0  →  reinforce the current α.
                             < 0  →  push toward the neutral baseline α = 0.5.
                             = 0  →  no-op (skip tiny updates that add noise).
            learning_rate:   Step size for the gradient update.

        Algorithm:
            1. Replay the forward pass (logits → conv → softmax → probs).
            2. Determine target bin:
               - If reward > 0 → target = bin matching alpha_used (reinforce).
               - If reward < 0 → target = bin 50 (α = 0.5, neutral fallback).
            3. Gradient w.r.t. smoothed logits: grad_s = (target_one_hot − probs).
            4. Backprop through Conv1D (symmetric uniform kernel):
                   grad_logits = conv(pad(grad_s, 3, edge), kernel)[:N_BINS]
            5. Update: W += lr * |reward| * outer(grad_logits, emb)
                       b += lr * |reward| * grad_logits

        Thread-safety: NOT thread-safe.  Call only from the feedback
        handling path, which is single-threaded in the current app.
        """
        if abs(reward) < 1e-6:
            return   # skip zero-reward updates — no signal

        emb = np.asarray(query_embedding, dtype=np.float32).squeeze()
        if emb.ndim == 0 or emb.shape[-1] != self.embed_dim:
            return   # shape mismatch — skip silently

        #  Forward pass (same as predict_alpha) 
        logits   = self._W @ emb + self._b
        pad      = self.CONV_KERNEL_SIZE // 2
        padded   = np.pad(logits, pad, mode="edge")
        smoothed = np.convolve(padded, self._kernel, mode="valid")[: self.N_BINS]
        smoothed = smoothed - smoothed.max()
        probs    = np.exp(smoothed)
        probs   /= probs.sum()

        #  Target bin
        if reward > 0:
            target_bin = int(round(alpha_used * 100))
        else:
            target_bin = 50   # neutral: α = 0.5
        target_bin = max(0, min(self.N_BINS - 1, target_bin))

        target_one_hot = np.zeros(self.N_BINS, dtype=np.float32)
        target_one_hot[target_bin] = 1.0

        #  REINFORCE gradient — backprop through Conv1D 
        # grad_smoothed = (target_one_hot − probs) is the gradient w.r.t. the
        # Conv1D output.  To get the gradient w.r.t. the linear-layer logits,
        # we must propagate back through the moving-average Conv1D.
        # The Conv1D is symmetric (uniform kernel), so the backward pass is
        # the same operation: convolve grad_smoothed with the same kernel.
        grad_smoothed = target_one_hot - probs          # (N_BINS,)
        pad_g         = self.CONV_KERNEL_SIZE // 2      # 3
        padded_grad   = np.pad(grad_smoothed, pad_g, mode="edge")
        grad_logits   = np.convolve(padded_grad, self._kernel, mode="valid")[: self.N_BINS]

        scale = learning_rate * abs(reward)

        self._W      += scale * np.outer(grad_logits, emb)   # (N_BINS, d)
        self._b      += scale * grad_logits                   # (N_BINS,)
        self._trained = True

    # Persistence

    def save(self, path: str) -> None:
        """Save linear weights to ``<path>.npz``."""
        np.savez(path, W=self._W, b=self._b, embed_dim=np.array(self.embed_dim))
        print(f"[QDAP-S] Weights saved → {path}")

    def load(self, path: str) -> None:
        """
        Load weights from a ``.npz`` file.
        Silently falls back to untrained defaults on shape mismatch or
        any I/O error — the system continues working with α=0.5.
        """
        try:
            data = np.load(path)
            W = data["W"].astype(np.float32)
            b = data["b"].astype(np.float32)
            if W.shape == self._W.shape and b.shape == self._b.shape:
                self._W       = W
                self._b       = b
                self._trained = True
                print(f"[QDAP-S] Weights loaded from {path}  (embed_dim={self.embed_dim})")
            else:
                print(
                    f"[QDAP-S] Weight shape mismatch "
                    f"(file: W={W.shape}, expected W={self._W.shape}) — "
                    "using untrained weights (α=0.5)"
                )
        except Exception as exc:
            print(f"[QDAP-S] Could not load weights from {path}: {exc} — using α=0.5")

    # Dunder helpers

    @property
    def is_trained(self) -> bool:
        return self._trained

    def __repr__(self) -> str:
        return (
            f"QDAPSmall("
            f"embed_dim={self.embed_dim}, "
            f"trained={self._trained})"
        )
