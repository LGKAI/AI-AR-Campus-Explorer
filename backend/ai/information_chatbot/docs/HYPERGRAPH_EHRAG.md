# STELLAR-RAG v4 — EHRAG Entity Hypergraph

**Paper**: "Entity Hypergraph for Retrieval-Augmented Generation" (arXiv 2604.17458)  
**Source**: `src/hypergraph.py`, `src/graphrag.py` (`_build_hypergraph`, `_hypergraph_rescore`)

---

## 1. Motivation

Standard knowledge graphs model pairwise relations (edges between two nodes). However, many real-world facts are **multi-way**: an exam regulation connects multiple rules, students, deadlines, and fees simultaneously. Hypergraphs extend edges to **hyperedges** that connect arbitrary subsets of nodes.

EHRAG builds two complementary hyperedge types over the entity set $E$:

| Type | Hyperedge captures | Dimension |
|------|-------------------|-----------|
| $H^{\text{str}}$ — structural | Which entities co-occur in the same chunk | $E \times C$ |
| $H^{\text{sem}}$ — semantic | Which entities are semantically similar (same cluster) | $E \times K$ |

---

## 2. Notation Specific to This Module

| Symbol | Meaning |
|--------|---------|
| $E$ | Number of unique entities |
| $C$ | Number of chunks |
| $K$ | Number of BIRCH semantic clusters |
| $D$ | Hypergraph top-D neighbours per cluster (`hypergraph_top_d = 10`) |
| $\tau$ | Gaussian temperature (`hypergraph_tau = 1.0`) |
| $\gamma$ | Semantic expansion decay (`hypergraph_gamma = 0.5`) |
| $T$ | Structural propagation iterations (`hypergraph_diffuse_T = 3`) |
| $L$ | Query-gated top-L chunks (`hypergraph_L = 50`) |
| $\varepsilon$ | Activation threshold (`hypergraph_epsilon = 0.01`) |
| $\lambda_1$ | Entity evidence weight (`hypergraph_lambda1 = 0.3`) |
| $\lambda_2$ | Cluster topic weight (`hypergraph_lambda2 = 0.2`) |
| $\mathbf{v}_e \in \mathbb{R}^d$ | L2-normalised embedding of entity $e$ |
| $\mathbf{c}_k \in \mathbb{R}^d$ | Centroid of cluster $k$ |
| $\mathbf{q} \in \mathbb{R}^d$ | L2-normalised query embedding |

---

## 3. Structural Hyperedge Matrix $H^{\text{str}}$

$H^{\text{str}}$ is a binary incidence matrix of shape $E \times C$:

$$H^{\text{str}}_{e,\, c} = \begin{cases} 1 & \text{entity } e \text{ appears in chunk } c \cr 0 & \text{otherwise} \end{cases}$$

Built from the knowledge graph: for each chunk node, follow outgoing `contains_entity` edges to find which entities it contains.

**Sparse representation**: Stored as `scipy.sparse.csr_matrix`. For a corpus with $E = 5000$ entities and $C = 10000$ chunks, the dense matrix would require $5000 \times 10000 \times 4 = 200$ MB. The sparse CSR representation stores only the non-zeros (roughly $2{-}10\%$ fill), reducing memory by 10-50×.

---

## 4. Semantic Hyperedge Matrix $H^{\text{sem}}$

### 4.1 BIRCH Clustering

BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) groups the $E$ entity embeddings into $K$ clusters without requiring $K$ to be specified in advance.

The algorithm builds a Clustering Feature Tree (CF Tree) online. Each CF node stores the triple $(N, \mathbf{LS}, SS)$:
- $N$: number of points in the subcluster
- $\mathbf{LS} = \sum_{i=1}^{N} \mathbf{v}_i$: linear sum of embeddings
- $SS = \sum_{i=1}^{N} \lVert\mathbf{v}_i\rVert^2$: sum of squared norms

Centroid and radius of a subcluster:

$$\mathbf{c} = \frac{\mathbf{LS}}{N}, \qquad R = \sqrt{\frac{SS}{N} - \lVert\mathbf{c}\rVert^2}$$

A new point is merged into the nearest subcluster if $\,R_{\text{new}} \leq \text{threshold}$ (default 0.5); otherwise a new subcluster is created.

After tree construction, $K$ is the number of leaf subclusters. The centroid of cluster $k$ is computed as the mean of its member embeddings:

$$\mathbf{c}_k = \frac{1}{|E_k|} \sum_{e \in E_k} \mathbf{v}_e$$

**Memory guard**: BIRCH is skipped if $E > 50{,}000$ to prevent OOM.

### 4.2 Gaussian-Weighted Semantic Edges

For each cluster $k$, select the $D$ entities nearest to centroid $\mathbf{c}_k$:

$$N_D(k) = \text{argtop-}D\left\lbrace e : -\lVert\mathbf{v}_e - \mathbf{c}_k\rVert^2\right\rbrace$$

Assign weights via a Gaussian kernel:

$$H^{\text{sem}}_{e,\, k} = \begin{cases} \exp\left(-\dfrac{\lVert\mathbf{v}_e - \mathbf{c}_k\rVert^2}{\tau}\right) & e \in N_D(k) \cr 0 & \text{otherwise} \end{cases}$$

**Interpretation**: entities close to the cluster centroid get high weight; entities at the fringes get exponentially lower weight. $\tau$ controls the falloff — larger $\tau$ gives a flatter distribution, smaller $\tau$ concentrates weight near the centroid.

---

## 5. Hybrid Diffusion Algorithm

Given query embedding $\mathbf{q}$ and seed entity scores $\mathbf{a}^{(0)} \in \mathbb{R}^E$ (from entity linking):

### Step 0 — Initialise Seed Scores

For each entity $e$ linked to the query with similarity $s$:

$$a^{(0)}_e = \max\left(0,\; \hat{\mathbf{v}}_e \cdot \mathbf{q}\right)$$

If no entity linking succeeds, fall back to cosine similarity over all entities.

### Step 1 — Semantic One-Off Expansion

Propagate scores through the semantic hyperedges:

$$\mathbf{h} = (H^{\text{sem}})^\top \mathbf{a}^{(0)} \in \mathbb{R}^K \qquad \text{(entity} \to \text{cluster)}$$

$$\mathbf{a}_{\text{sem}} = \gamma \cdot H^{\text{sem}} \mathbf{h} \in \mathbb{R}^E \qquad \text{(cluster} \to \text{entity)}$$

$$\mathbf{a}^{(1)} = \mathbf{a}^{(0)} + \mathbf{a}_{\text{sem}}$$

**Intuition**: if entity $e$ has a high seed score, all semantically related entities (in the same cluster) receive a fraction $\gamma$ of that score. This expands recall to entities not directly linked by the query but semantically co-present.

### Step 2 — Structural Iterative Propagation

Pre-compute query-chunk gating scores:

$$g_c = \hat{\mathbf{v}}_c \cdot \mathbf{q}, \quad c = 1, \ldots, C$$

Build top-$L$ gate mask (zero out chunks below top-L by cosine to query):

$$G_q = \text{diag}(g_1 \cdot \mathbf{1}[c \in \text{top-}L],\; \ldots,\; g_C \cdot \mathbf{1}[c \in \text{top-}L]) \in \mathbb{R}^{C \times C}$$

For $t = 1, \ldots, T$:

$$\mathbf{s}^{(t)} = (H^{\text{str}})^\top \mathbf{a}^{(t)} \in \mathbb{R}^C \qquad \text{(entity} \to \text{chunk projection)}$$

$$\tilde{\mathbf{s}}^{(t)} = G_q \, \mathbf{s}^{(t)} \in \mathbb{R}^C \qquad \text{(query gating: suppress off-topic chunks)}$$

$$\Delta\mathbf{a}^{(t+1)} = H^{\text{str}} \, \tilde{\mathbf{s}}^{(t)} \in \mathbb{R}^E \qquad \text{(chunk} \to \text{entity backpropagation)}$$

$$a^{(t+1)}_e = \begin{cases} \Delta a^{(t+1)}_e & \text{if } \Delta a^{(t+1)}_e \geq \varepsilon \cr 0 & \text{otherwise} \end{cases} \qquad \text{(threshold)}$$

Accumulate cumulative entity weights:

$$\mathbf{w} \mathrel{+}= \mathbf{a}^{(t+1)}$$

**Intuition**: each iteration propagates relevance from entities into chunks they appear in (weighted by query-chunk cosine similarity), then back into entities that co-occur in those query-relevant chunks. Entities that are mentioned alongside many query-relevant chunks receive high cumulative weight.

**Early termination**: if $\sum_e a^{(t+1)}_e = 0$ (all activations below $\varepsilon$), propagation stops.

### Step 3 — Cluster Activation Scores

For each cluster $k$:

$$S_{\text{cluster}}(k) = \frac{1}{|E_k|} \sum_{e \in E_k} w_e$$

This gives the average diffusion weight of entities in the cluster.

---

## 6. Topic-Aware Three-Component Scoring

**Source**: `src/hypergraph.py` — `topic_score_chunks()`

After diffusion, re-score each retrieved chunk $d$ using three terms:

$$S(d) = S_{\text{dense}}(q, d) + \lambda_1 \sum_{e \in E(d)} \log(1 + w_e) + \lambda_2 \log\left(1 + \sum_{k \in \mathcal{K}(d)} S_{\text{cluster}}(k)\right)$$

| Term | Formula | Captures |
|------|---------|---------|
| Base score | $\,S_{\text{dense}}(q, d)$ | QDAP-S fused hybrid score |
| Entity evidence | $\lambda_1 \sum_{e \in E(d)} \log(1 + w_e)$ | Entities in chunk $d$ with high diffusion weight |
| Cluster topic | $\lambda_2 \log(1 + \sum_k S_{\text{cluster}}(k))$ | Semantic clusters relevant to the query |

where:
- $E(d)$: entities linked to chunk $d$ via $H^{\text{str}}$
- $\mathcal{K}(d)$: clusters of entities in $E(d)$ (de-duplicated)
- $\log(1 + x)$ (log1p): monotone, concave — rewards high-weight entities but with diminishing returns, preventing any single entity from dominating

**Log1p choice**: a chunk with 10 medium-weight entities scores higher than a chunk with 1 high-weight entity of the same total weight. This rewards **breadth** of relevant entity coverage.

---

## 7. Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Build $H^{\text{str}}$ | $O(E \cdot C_{\text{avg}})$ | $O(\text{nnz}(H^{\text{str}}))$ sparse |
| BIRCH clustering | $O(E \cdot d \cdot \log K)$ | $O(E \cdot d)$ |
| Build $H^{\text{sem}}$ | $O(K \cdot E \cdot d)$ | $O(K \cdot D)$ sparse |
| Semantic expansion (Step 1) | $O(\text{nnz}(H^{\text{sem}}))$ | $O(E + K)$ |
| Structural propagation (Step 2, per iter) | $O(\text{nnz}(H^{\text{str}}) + C)$ | $O(E + C)$ |
| Topic scoring | $O(\lvert\text{hits}\rvert \cdot C_{\text{avg}})$ | $O(E)$ |

All matrix multiplications use scipy sparse routines (CSR format), avoiding dense $E \times C$ materialisation. This keeps memory below ~100 MB for typical university corpora ($E \approx 5000$, $C \approx 10000$).

---

## 8. Failure Modes and Safeguards

| Failure | Safeguard |
|---------|-----------|
| $E > 50{,}000$ → OOM in BIRCH | Hard skip: `_H_sem = None`, hypergraph continues without semantic edges |
| `scipy` not installed | `ImportError` raised early with install hint |
| BIRCH produces $K = 1$ cluster | Degenerate but handled — all entities in one hyperedge |
| No seed entity scores from linking | Fall back to cosine similarity over all entities |
| Diffusion accumulates NaN/Inf | `try/except` per iteration; returns empty weights (fail-open) |
| `topic_score_chunks` error on a hit | Keeps original hit unchanged (per-item try/except) |
