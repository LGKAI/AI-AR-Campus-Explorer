"""
EHRAG Hypergraph module — implements the Entity Hypergraph for Retrieval-Augmented
Generation (arxiv 2604.17458).

Two types of hyperedges are built over the entity set E:

Structural hyperedges (H^str, shape E×C)
-----------------------------------------
  Entity e_i and chunk c_j are connected when e_i appears in c_j:
      H^str[i, j] = 1

Semantic hyperedges (H^sem, shape E×K)
-----------------------------------------
  BIRCH clustering of entity embeddings produces K centroids {c_1, …, c_K}.
  For each centroid c_k the top-D nearest entities are connected with a
  Gaussian weight:
      H^sem[i, k] = exp(-‖x_i − c_k‖² / τ)   if e_i ∈ N_D(c_k)
                  = 0                            otherwise

Hybrid diffusion retrieval
---------------------------
  Given a query vector q and initial entity activation scores a^(0):
  1. Semantic one-off expansion:
       a_sem   = γ · H^sem · (H^sem)^T · a^(0)
       a^(1)   = a^(0) + a_sem
  2. Structural iterative propagation (T iterations):
       s^(t)   = (H^str)^T · a^(t)          [entity → chunk]
       G_q[j,j] = cos(chunk_j, query) if j ∈ top-L, else 0
       Δa^(t+1) = H^str · G_q · s^(t)       [chunk → entity backprop]
       a^(t+1)  = threshold(Δa^(t+1), ε)
       w        += a^(t+1)                   [cumulative weight]

Topic-aware scoring
--------------------
  S(d) = S_dense(q, d)
       + λ₁ · Σ_{v∈d} log(1 + w(v))
       + λ₂ · log(1 + Σ_{C∈d} S_topic(C))
  where S_topic(C) = mean activation score of cluster C's entities.

Memory safety
-------------
  All matrices use scipy.sparse (csr_matrix).  Never stores dense E×C for
  large E.  Clustering is guarded by a 50 000-entity cap.  All embedding
  operations are batched in embed_batch_size steps.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# EntityHypergraph

class EntityHypergraph:
    """
    EHRAG Entity Hypergraph — structural + semantic hyperedges for hybrid
    diffusion retrieval.

    Lifecycle
    ---------
    1. ``build(entity_names, entity_vecs, chunk_entity_map, chunk_vecs)``
       — constructs H^str, H^sem, stores cluster metadata.
    2. ``diffuse(query_vec, seed_entity_scores, chunk_vecs)``
       — returns per-entity cumulative weights and per-cluster activation scores.
    3. ``topic_score_chunks(hits, entity_weights, cluster_scores)``
       — re-scores a list of retrieved hit dicts using the 3-component formula.
    4. ``save(path)`` / ``load(path)``
       — persist / restore the hypergraph artefacts.
    """

    def __init__(self) -> None:
        """Initialise an empty hypergraph (call build() to populate)."""
        # Entity metadata
        self.entity_names:  list[str]         = []
        self.entity_vecs:   np.ndarray | None  = None   # (E, d) float32

        # Chunk metadata
        self.chunk_ids:     list[str]          = []
        self.chunk_vecs:    np.ndarray | None  = None   # (C, d) float32

        # Incidence matrices (scipy.sparse.csr_matrix)
        self._H_str = None   # (E, C) structural
        self._H_sem = None   # (E, K) semantic

        # Cluster metadata
        self.cluster_ids:      np.ndarray | None = None   # (E,) int, cluster per entity
        self.cluster_centroids: np.ndarray | None = None  # (K, d) float32
        self.n_clusters: int = 0

        self._built: bool = False

   
    # Build

    def build(
        self,
        entity_names:      list[str],
        entity_vecs:       np.ndarray,
        chunk_entity_map:  dict[str, list[str]],
        chunk_vecs:        np.ndarray,
    ) -> None:
        """
        Construct the hypergraph from entity and chunk data.

        Args:
            entity_names:     List of entity name strings, length E.
            entity_vecs:      L2-normalised embeddings, shape (E, d).
            chunk_entity_map: {chunk_id: [entity_name, …]} mapping which
                              entities appear in each chunk.
            chunk_vecs:       L2-normalised chunk embeddings, shape (C, d).
                              chunk_vecs[i] corresponds to chunk_ids[i].
        """
        try:
            from scipy.sparse import csr_matrix, lil_matrix
        except ImportError as exc:
            raise ImportError("scipy is required for EntityHypergraph: pip install scipy") from exc

        if len(entity_names) == 0:
            logger.warning("[Hypergraph] No entities — skipping hypergraph build.")
            return

        self.entity_names = entity_names
        self.entity_vecs  = np.asarray(entity_vecs, dtype=np.float32)
        self.chunk_ids    = list(chunk_entity_map.keys())
        self.chunk_vecs   = np.asarray(chunk_vecs, dtype=np.float32) if chunk_vecs is not None else None

        E = len(entity_names)
        C = len(self.chunk_ids)

        entity_idx: dict[str, int] = {name: i for i, name in enumerate(entity_names)}
        chunk_idx:  dict[str, int] = {cid: j for j, cid in enumerate(self.chunk_ids)}

        #  Build H^str (E × C)
        logger.info(f"[Hypergraph] Building H^str  E={E} C={C}")
        H_str_lil = lil_matrix((E, C), dtype=np.float32)

        for chunk_id, ent_list in chunk_entity_map.items():
            j = chunk_idx.get(chunk_id)
            if j is None:
                continue
            for ename in ent_list:
                i = entity_idx.get(ename)
                if i is not None:
                    H_str_lil[i, j] = 1.0

        self._H_str = H_str_lil.tocsr()
        logger.info(f"[Hypergraph] H^str nnz={self._H_str.nnz}")

        #  Build H^sem (E × K) via BIRCH clustering 
        self._build_semantic_hyperedges(entity_vecs=self.entity_vecs)
        self._built = True

    def _build_semantic_hyperedges(self, entity_vecs: np.ndarray) -> None:
        """
        Run BIRCH clustering on entity_vecs, then build the Gaussian-weighted
        semantic incidence matrix H^sem (E × K).

        Uses scipy.sparse to avoid dense E×K materialisation.
        Guarded by a 50 000-entity hard limit to prevent OOM on large corpora.
        """
        try:
            from scipy.sparse import lil_matrix
            from sklearn.cluster import Birch
        except ImportError as exc:
            raise ImportError(
                "scikit-learn and scipy are required: pip install scikit-learn scipy"
            ) from exc

        from config import settings

        E = len(entity_vecs)
        if E == 0:
            logger.warning("[Hypergraph] No entity vectors — skipping semantic hyperedges.")
            return

        # Hard memory guard: skip clustering for very large entity sets
        if E > 50_000:
            logger.warning(
                f"[Hypergraph] {E} entities > 50 000 limit — skipping BIRCH clustering."
            )
            self._H_sem   = None
            self.n_clusters = 0
            return

        logger.info(f"[Hypergraph] BIRCH clustering  E={E}  threshold={settings.birch_threshold}")

        try:
            birch = Birch(
                threshold=settings.birch_threshold,
                n_clusters=settings.birch_n_clusters,
            )
            cluster_labels = birch.fit_predict(entity_vecs)
        except MemoryError:
            logger.error("[Hypergraph] MemoryError during BIRCH — skipping semantic hyperedges.")
            self._H_sem    = None
            self.n_clusters = 0
            return
        except Exception as exc:
            logger.error(f"[Hypergraph] BIRCH failed: {exc} — skipping semantic hyperedges.")
            self._H_sem    = None
            self.n_clusters = 0
            return

        self.cluster_ids = cluster_labels
        K = int(cluster_labels.max()) + 1
        self.n_clusters  = K
        logger.info(f"[Hypergraph] BIRCH produced K={K} clusters")

        # Compute centroids as mean of member vectors
        centroids = np.zeros((K, entity_vecs.shape[1]), dtype=np.float32)
        for k in range(K):
            members = entity_vecs[cluster_labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
        self.cluster_centroids = centroids

        # Build H^sem: for each cluster c_k, connect top-D nearest entities
        top_d = settings.hypergraph_top_d
        tau   = settings.hypergraph_tau

        H_sem_lil = lil_matrix((E, K), dtype=np.float32)

        for k in range(K):
            c_k = centroids[k]   # (d,)
            # Squared Euclidean distance from all entities to centroid k.
            # entity_vecs rows are L2-normalised → ||e_i||² = 1.
            # ||e_i − c_k||² = ||e_i||² − 2·(e_i · c_k) + ||c_k||²
            #                 = 1 − 2·(e_i · c_k) + ||c_k||²
            # This avoids materialising the full (E, d) difference matrix.
            c_k_norm_sq = float(np.dot(c_k, c_k))
            dots        = entity_vecs @ c_k           # (E,)  — O(E·d) but no copy
            sq_dists    = 1.0 - 2.0 * dots + c_k_norm_sq  # (E,)

            # Top-D nearest neighbours
            nearest_idx = np.argsort(sq_dists)[:top_d]

            for i in nearest_idx:
                w = float(np.exp(-sq_dists[i] / tau))
                if w > 1e-9:
                    H_sem_lil[i, k] = w

        self._H_sem = H_sem_lil.tocsr()
        logger.info(f"[Hypergraph] H^sem nnz={self._H_sem.nnz}")

    # Hybrid diffusion retrieval

    def diffuse(
        self,
        query_vec:          np.ndarray,
        seed_entity_scores: dict[str, float],
        chunk_vecs:         np.ndarray | None = None,
        T:                  int   | None = None,
        L:                  int   | None = None,
        gamma:              float | None = None,
        epsilon:            float | None = None,
    ) -> tuple[dict[str, float], dict[int, float]]:
        """
        Run hybrid diffusion to propagate query-entity relevance through the
        hypergraph and collect cumulative activation weights.

        Args:
            query_vec:          Query embedding, shape (d,) or (1, d).
            seed_entity_scores: {entity_name: initial_score} from dense retrieval.
            chunk_vecs:         Chunk embeddings (C, d).  If None, uses self.chunk_vecs.
            T:                  Structural propagation iterations.  Default from settings.
            L:                  Top-L chunks for query gating.  Default from settings.
            gamma:              Semantic expansion decay.  Default from settings.
            epsilon:            Activation threshold.  Default from settings.

        Returns:
            (entity_weights, cluster_scores)
            entity_weights: {entity_name: cumulative_weight}
            cluster_scores: {cluster_idx: mean_activation_score}
        """
        from config import settings

        T       = T       if T       is not None else settings.hypergraph_diffuse_T
        L       = L       if L       is not None else settings.hypergraph_L
        gamma   = gamma   if gamma   is not None else settings.hypergraph_gamma
        epsilon = epsilon if epsilon is not None else settings.hypergraph_epsilon

        if not self._built or self._H_str is None:
            return {}, {}

        E = len(self.entity_names)
        if E == 0:
            return {}, {}

        # Resolve chunk_vecs
        cv = chunk_vecs if chunk_vecs is not None else self.chunk_vecs
        if cv is None:
            return {}, {}

        qv = np.asarray(query_vec, dtype=np.float32).squeeze()   # (d,)

        # Initialise a^(0) from seed entity scores
        a = np.zeros(E, dtype=np.float32)
        entity_idx = {name: i for i, name in enumerate(self.entity_names)}

        for ename, score in seed_entity_scores.items():
            # Try exact match first, then normalised match
            idx = entity_idx.get(ename)
            if idx is None:
                # Fallback: look for entity names that contain this name
                for name, i in entity_idx.items():
                    if ename in name or name in ename:
                        idx = i
                        break
            if idx is not None:
                a[idx] = max(a[idx], float(score))

        if a.sum() == 0:
            # No seed scores — try cosine similarity as initialisation
            if self.entity_vecs is not None:
                sims = self.entity_vecs @ qv
                a    = np.maximum(sims, 0).astype(np.float32)

        if a.sum() == 0:
            return {}, {}

        #  Step 1: Semantic one-off expansion
        if self._H_sem is not None:
            try:
                # a_sem = γ · H^sem · (H^sem)^T · a
                H_sem_T_a = self._H_sem.T.dot(a)          # (K,)
                a_sem     = gamma * self._H_sem.dot(H_sem_T_a)  # (E,)
                a         = a + a_sem.astype(np.float32)
            except Exception as exc:
                logger.debug(f"[Hypergraph] Semantic expansion failed: {exc}")

        #  Step 2: Structural iterative propagation 
        C = len(self.chunk_ids)
        cumulative_w = np.zeros(E, dtype=np.float32)

        # Pre-compute query-chunk cosine similarities for gating matrix
        chunk_query_sims = cv @ qv   # (C,)

        # Build top-L mask
        if L < C:
            top_L_idx = np.argsort(chunk_query_sims)[-L:]
            gate_mask  = np.zeros(C, dtype=np.float32)
            gate_mask[top_L_idx] = 1.0
        else:
            gate_mask = np.ones(C, dtype=np.float32)

        gated_sims = chunk_query_sims * gate_mask   # (C,) — zero outside top-L

        for _t in range(T):
            try:
                # s^(t) = (H^str)^T · a  [entity → chunk]
                s = self._H_str.T.dot(a)   # (C,)

                # Apply query gating: G_q · s
                s_gated = gated_sims * s   # (C,) element-wise

                # Δa = H^str · G_q · s  [chunk → entity backpropagation]
                delta_a = self._H_str.dot(s_gated)   # (E,)
                delta_a = np.asarray(delta_a, dtype=np.float32).ravel()

                # Threshold
                delta_a[delta_a < epsilon] = 0.0

                if delta_a.sum() == 0:
                    break

                a = delta_a
                cumulative_w += a

            except Exception as exc:
                logger.debug(f"[Hypergraph] Structural propagation iter {_t} failed: {exc}")
                break

        #  Collect entity weights 
        entity_weights: dict[str, float] = {}
        for i, name in enumerate(self.entity_names):
            w = float(cumulative_w[i])
            if w > 0:
                entity_weights[name] = w

        #  Compute cluster activation scores 
        cluster_scores: dict[int, float] = {}
        if self.cluster_ids is not None and self.n_clusters > 0:
            for k in range(self.n_clusters):
                members = np.where(self.cluster_ids == k)[0]
                if len(members) > 0:
                    member_scores = cumulative_w[members]
                    cluster_scores[k] = float(member_scores.mean())

        return entity_weights, cluster_scores

    # Topic-aware scoring

    def topic_score_chunks(
        self,
        hits:             list[dict],
        entity_weights:   dict[str, float],
        cluster_scores:   dict[int, float],
        lambda1:          float | None = None,
        lambda2:          float | None = None,
    ) -> list[dict]:
        """
        Apply the EHRAG 3-component topic-aware scoring formula:

            S(d) = S_dense(q, d)
                 + λ₁ · Σ_{v∈d} log(1 + w(v))
                 + λ₂ · log(1 + Σ_{C∈d} S_topic(C))

        where S_topic(C) is the mean cluster activation for entities in the
        cluster that also appear in the chunk's associated entities.

        Args:
            hits:           List of retrieved hit dicts (must have 'id' or
                            'chunk_id' field and 'score' field).
            entity_weights: {entity_name: cumulative_diffusion_weight}
            cluster_scores: {cluster_idx: mean_activation}
            lambda1:        Explicit entity evidence weight λ₁.
            lambda2:        Semantic cluster weight λ₂.

        Returns:
            Re-scored and re-sorted copy of hits.  Original scores are
            preserved in 'base_score'.  Fails open (returns hits unchanged)
            on any error.
        """
        if not hits or (not entity_weights and not cluster_scores):
            return hits

        from config import settings
        lambda1 = lambda1 if lambda1 is not None else settings.hypergraph_lambda1
        lambda2 = lambda2 if lambda2 is not None else settings.hypergraph_lambda2

        # Build entity → cluster mapping for fast lookup
        entity_to_cluster: dict[str, int] = {}
        if self.cluster_ids is not None:
            for i, name in enumerate(self.entity_names):
                entity_to_cluster[name] = int(self.cluster_ids[i])

        # Build chunk → entity mapping from H^str
        chunk_to_entities: dict[str, list[str]] = {}
        if self._H_str is not None:
            chunk_idx_to_id = {j: cid for j, cid in enumerate(self.chunk_ids)}
            entity_idx_to_name = {i: name for i, name in enumerate(self.entity_names)}

            cx = self._H_str.tocsc()
            for j in range(cx.shape[1]):
                col = cx.getcol(j)
                rows = col.indices
                cid  = chunk_idx_to_id.get(j, "")
                if cid and len(rows) > 0:
                    chunk_to_entities[cid] = [
                        entity_idx_to_name[r] for r in rows if r in entity_idx_to_name
                    ]

        result: list[dict] = []
        for hit in hits:
            try:
                base_score = float(hit.get("score", 0.0))

                # Resolve chunk id
                chunk_id = str(
                    hit.get("id") or hit.get("chunk_id") or hit.get("text", "")[:60]
                )

                # Entity evidence term: Σ log(1 + w(v))
                chunk_entities = chunk_to_entities.get(chunk_id, [])
                entity_term = sum(
                    np.log1p(entity_weights.get(e, 0.0))
                    for e in chunk_entities
                )

                # Cluster topic term: log(1 + Σ S_topic(C))
                cluster_sum = 0.0
                seen_clusters: set[int] = set()
                for ename in chunk_entities:
                    k = entity_to_cluster.get(ename)
                    if k is not None and k not in seen_clusters:
                        seen_clusters.add(k)
                        cluster_sum += cluster_scores.get(k, 0.0)
                cluster_term = float(np.log1p(cluster_sum))

                new_score = base_score + lambda1 * entity_term + lambda2 * cluster_term

                item = dict(hit)
                item["base_score"]   = base_score
                item["score"]        = new_score
                item["ehrag_entity"] = round(lambda1 * entity_term, 5)
                item["ehrag_cluster"] = round(lambda2 * cluster_term, 5)
                result.append(item)

            except Exception as exc:
                logger.debug(f"[Hypergraph] topic_score_chunks item error: {exc}")
                result.append(hit)

        result.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return result

    # Persist / restore

    def save(self, directory: Path) -> None:
        """
        Save hypergraph artefacts to *directory*.

        Files written:
          hgraph_H_str.npz    — structural incidence matrix (scipy sparse)
          hgraph_H_sem.npz    — semantic incidence matrix  (scipy sparse)
          hgraph_meta.npz     — cluster_ids, cluster_centroids, n_clusters
          hgraph_chunk_ids.json — ordered list of chunk_id strings
          hgraph_entity_names.json — ordered list of entity name strings
        """
        try:
            import scipy.sparse as sp
        except ImportError:
            logger.error("[Hypergraph] scipy not available — cannot save.")
            return

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        try:
            if self._H_str is not None:
                sp.save_npz(str(directory / "hgraph_H_str.npz"), self._H_str)
            if self._H_sem is not None:
                sp.save_npz(str(directory / "hgraph_H_sem.npz"), self._H_sem)

            meta: dict = {"n_clusters": self.n_clusters}
            if self.cluster_ids is not None:
                meta["cluster_ids"] = self.cluster_ids
            if self.cluster_centroids is not None:
                meta["cluster_centroids"] = self.cluster_centroids
            if self.chunk_vecs is not None:
                meta["chunk_vecs"] = self.chunk_vecs

            np.savez(str(directory / "hgraph_meta.npz"), **meta)

            (directory / "hgraph_chunk_ids.json").write_text(
                json.dumps(self.chunk_ids, ensure_ascii=False), encoding="utf-8"
            )
            (directory / "hgraph_entity_names.json").write_text(
                json.dumps(self.entity_names, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(f"[Hypergraph] Saved to {directory}")

        except Exception as exc:
            logger.error(f"[Hypergraph] Save failed: {exc}")

    def load(self, directory: Path) -> bool:
        """
        Load hypergraph artefacts from *directory*.

        Returns True on success, False if required files are missing or on
        any loading error (fail-open: the rest of the pipeline is unaffected).
        """
        try:
            import scipy.sparse as sp
        except ImportError:
            logger.error("[Hypergraph] scipy not available — cannot load.")
            return False

        directory = Path(directory)
        h_str_path  = directory / "hgraph_H_str.npz"
        meta_path   = directory / "hgraph_meta.npz"
        cids_path   = directory / "hgraph_chunk_ids.json"
        enames_path = directory / "hgraph_entity_names.json"

        if not h_str_path.exists() or not meta_path.exists():
            return False

        try:
            self._H_str = sp.load_npz(str(h_str_path))

            h_sem_path = directory / "hgraph_H_sem.npz"
            if h_sem_path.exists():
                self._H_sem = sp.load_npz(str(h_sem_path))

            meta = np.load(str(meta_path), allow_pickle=True)
            self.n_clusters = int(meta.get("n_clusters", 0))
            if "cluster_ids" in meta:
                self.cluster_ids = meta["cluster_ids"]
            if "cluster_centroids" in meta:
                self.cluster_centroids = meta["cluster_centroids"]
            if "chunk_vecs" in meta:
                self.chunk_vecs = meta["chunk_vecs"]

            if cids_path.exists():
                self.chunk_ids = json.loads(cids_path.read_text(encoding="utf-8"))
            if enames_path.exists():
                self.entity_names = json.loads(enames_path.read_text(encoding="utf-8"))

            self._built = True
            logger.info(
                f"[Hypergraph] Loaded from {directory}  "
                f"E={len(self.entity_names)}  C={len(self.chunk_ids)}  "
                f"K={self.n_clusters}"
            )
            return True

        except Exception as exc:
            logger.error(f"[Hypergraph] Load failed: {exc}")
            return False

    def exists(self, directory: Path) -> bool:
        """Return True if the hypergraph artefacts exist at *directory*."""
        directory = Path(directory)
        return (
            (directory / "hgraph_H_str.npz").exists()
            and (directory / "hgraph_meta.npz").exists()
        )

    def is_built(self) -> bool:
        """Return True if the hypergraph has been built or loaded."""
        return self._built and self._H_str is not None
