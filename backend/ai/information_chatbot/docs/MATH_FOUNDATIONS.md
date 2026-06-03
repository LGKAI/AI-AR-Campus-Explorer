# STELLAR-RAG v4 — Mathematical Foundations

This document derives every formula used in the system from first principles, with references to the source modules.

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $d$ | Embedding dimension (BAAI/bge-m3: 1024) |
| $n$ | Number of documents / chunks in the corpus |
| $E$ | Number of unique entities |
| $C$ | Number of chunks |
| $K$ | Number of BIRCH clusters |
| $\mathbf{q} \in \mathbb{R}^d$ | Query embedding vector (L2-normalised) |
| $\mathbf{v}_i \in \mathbb{R}^d$ | Document/entity embedding (L2-normalised) |
| $\hat{\mathbf{x}}$ | L2-normalised form of $\mathbf{x}$: $\hat{\mathbf{x}} = \mathbf{x} / \lVert\mathbf{x}\rVert_2$ |
| $\alpha \in [0,1]$ | Dense/sparse mixing weight (QDAP-S output) |
| $\,w_g \in [0,1]$ | Graph-retrieval contribution weight |
| $\lambda_1, \lambda_2$ | EHRAG topic scoring weights |
| $\gamma$ | Semantic diffusion decay |
| $\tau$ | Gaussian temperature for semantic hyperedges |
| $\varepsilon$ | Activation threshold (structural diffusion) |

---

## 2. Embeddings and Cosine Similarity

### 2.1 L2 Normalisation

All embeddings — query, document, entity, chunk — are L2-normalised before storage and use:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\lVert\mathbf{v}\rVert_2 + \delta}, \quad \delta = 10^{-12} \text{ (numerical stability)}$$

### 2.2 Cosine Similarity via Inner Product

After L2 normalisation, the inner product equals cosine similarity:

$$\cos(\mathbf{q}, \mathbf{v}) = \frac{\mathbf{q} \cdot \mathbf{v}}{\lVert\mathbf{q}\rVert_2 \lVert\mathbf{v}\rVert_2} = \hat{\mathbf{q}} \cdot \hat{\mathbf{v}}$$

This is why FAISS is configured with `METRIC_INNER_PRODUCT` — it computes exact cosine similarity with no extra normalisation cost at query time.

### 2.3 Dense Retrieval Score

$$s_{\text{dense}}(d) = \hat{\mathbf{q}} \cdot \hat{\mathbf{v}}_d \in [-1, 1]$$

In practice all corpus embeddings are non-negative after the ReLU-like behaviour of the model, so scores lie in $[0, 1]$.

---

## 3. BM25 Sparse Retrieval

BM25 (Best Match 25) is a probabilistic bag-of-words ranking function. For query $q = \lbrace t_1, \ldots, t_m \rbrace$ and document $d$:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \left(1 - b + b \cdot \dfrac{|d|}{\text{avgdl}}\right)}$$

where:
- $f(t, d)$ — term frequency of $t$ in document $d$
- $|d|$ — document length in tokens
- $\text{avgdl}$ — average document length across the corpus
- $\,k_1 = 1.5$ — term-frequency saturation parameter
- $b = 0.75$ — length normalisation parameter

The IDF (inverse document frequency) with smoothing:

$$\text{IDF}(t) = \log\!\left(\frac{N - n_t + 0.5}{n_t + 0.5} + 1\right)$$

where $N$ is the total number of documents and $\,n_t$ is the number containing term $t$.

**Why BM25 complements dense retrieval**: BM25 rewards exact keyword matches and penalises common terms via IDF. Dense retrieval captures semantic similarity but may miss exact numeric/code matches (e.g., "Điều 15", "4.5 GPA"). The two signals are complementary.

---

## 4. QDAP-S: Query-Adaptive Dense/Sparse Predictor

**Source**: `src/qdap.py`  
**Paper**: "Query-Adaptive Hybrid Search", Section 3.2

QDAP-S learns to predict the optimal $\alpha$ for blending dense and BM25 scores, conditioned on the query embedding.

### 4.1 Architecture

$$\mathbf{e} \xrightarrow{\text{Linear}} \mathbf{l} \xrightarrow{\text{Conv1D}} \tilde{\mathbf{l}} \xrightarrow{\text{Softmax}} \mathbf{p} \xrightarrow{E[\cdot]} \alpha$$

**Step 1 — Linear projection** to 101-bin logit histogram:

$$\mathbf{l} = W \mathbf{e} + \mathbf{b}, \quad W \in \mathbb{R}^{101 \times d},\; \mathbf{b} \in \mathbb{R}^{101}$$

**Step 2 — Conv1D smoothing** (moving average, kernel size 7, edge padding):

$$\tilde{l}_i = \frac{1}{7} \sum_{j=-3}^{3} l_{\text{pad}(i+j)}, \quad i = 0, \ldots, 100$$

Edge padding: $\,l_{\text{pad}(i)} = l_0$ for $i < 0$, $\,l_{100}$ for $i > 100$.

**Step 3 — Numerically stable Softmax**:

$$p_i = \frac{\exp(\tilde{l}_i - \max_j \tilde{l}_j)}{\sum_{j=0}^{100} \exp(\tilde{l}_j - \max_j \tilde{l}_j)}$$

**Step 4 — Expected $\alpha$** over the uniform grid $\lbrace\frac{i}{100}\rbrace_{i=0}^{100}$:

$$\alpha = \sum_{i=0}^{100} p_i \cdot \frac{i}{100}$$

**Untrained defaults**: $W = 0$, $\mathbf{b} = 0$ → uniform logits → uniform $\mathbf{p}$ → $\alpha = 0.5$ (balanced blend).

### 4.2 Hybrid Fusion Formula

Let $s'_\star$ denote the min-max normalised score for retrieval type $\star$:

$$s'_\star(d) = \frac{s_\star(d) - \min_{d'} s_\star(d')}{\max_{d'} s_\star(d') - \min_{d'} s_\star(d')}, \quad \text{with } s'_\star = 0.5 \text{ if all scores equal}$$

Dense/BM25 blend:

$$s_{\text{db}}(d) = \begin{cases} \alpha \cdot s'_{\text{dense}}(d) + (1 - \alpha) \cdot s'_{\text{BM25}}(d) & \text{if } d \text{ appears in dense or BM25} \cr 0 & \text{otherwise} \end{cases}$$

Three-way blend with graph:

$$s_{\text{final}}(d) = (1 - w_g) \cdot s_{\text{db}}(d) + w_g \cdot s'_{\text{graph}}(d), \quad w_g = 0.15$$

When $\alpha \to 1$: lean dense/semantic (good for paraphrased, analytical queries).  
When $\alpha \to 0$: lean sparse/BM25 (good for exact-match factual queries).

---

## 5. QDAP-S Online Learning (REINFORCE)

**Source**: `src/qdap.py` — `update_online()`  
**Algorithm**: Policy gradient (Williams, 1992)

After the user rates an answer (1–5 stars), we map to reward signal:

$$r = \frac{\text{rating} - 3}{2} \in \lbrace -1.0,\; -0.5,\; 0.0,\; +0.5,\; +1.0 \rbrace$$

Treat the QDAP-S softmax output as a stochastic policy $\pi_\theta(\alpha) = p_{\lfloor\alpha \cdot 100\rceil}$.

The REINFORCE policy gradient estimator:

$$\nabla_\theta \mathcal{J}(\theta) \approx r \cdot \nabla_\theta \log \pi_\theta(\alpha^*)$$

For the softmax policy, the log-probability gradient is:

$$\nabla_W \log p_{\text{bin}} = (\mathbf{1}_{\text{bin}} - \mathbf{p}) \otimes \mathbf{e}$$

where $\mathbf{1}_{\text{bin}}$ is a one-hot vector at the target bin and $\otimes$ denotes the outer product.

**Target bin selection**:
- $r > 0$: reinforce the α that was used → $\text{bin}^* = \lfloor \alpha_{\text{used}} \times 100 \rceil$
- $r < 0$: push toward the neutral baseline → $\text{bin}^* = 50$ (i.e., $\alpha = 0.5$)
- $r = 0$: no update (skip)

**Update rules**:

$$W \leftarrow W + \eta \, |r| \, (\mathbf{1}_{\text{bin}^*} - \mathbf{p}) \otimes \mathbf{e}$$

$$\mathbf{b} \leftarrow \mathbf{b} + \eta \, |r| \, (\mathbf{1}_{\text{bin}^*} - \mathbf{p})$$

where $\eta = 0.001$ (learning rate), $\mathbf{p}$ is the softmax output during the rated query, and $\mathbf{e}$ is the query embedding.

**Intuition**: If the answer was rated highly ($r > 0$), increase the probability mass at the bin corresponding to the $\alpha$ that was used. If rated poorly ($r < 0$), pull the distribution toward the neutral $\alpha = 0.5$ because the chosen blend did not help.

---

## 6. Reciprocal Rank Fusion (RRF)

**Source**: `src/graphrag.py` — `_rrf_fuse()`  
**Paper**: Cormack et al., 2009

Given $L$ ranked lists, the RRF score of document $d$ is:

$$\text{RRF}(d) = \sum_{i=1}^{L} \frac{1}{k_{\text{RRF}} + r_i(d)}$$

where $\,r_i(d)$ is the rank of $d$ in list $i$ (1-indexed) and $\,k_{\text{RRF}} = 60$ (default).

**Properties**:
- Scale-invariant: raw score magnitudes do not matter, only rank.
- Robust to outlier scores in any single list.
- Used as fallback when `fusion_method = "rrf"`.

---

## 7. Personalised PageRank (PPR)

**Source**: `src/graphrag.py` — `_ppr_local()`

PPR is run on a local subgraph around seed entities (within `ppr_max_subgraph = 500` nodes).

Let $A$ be the row-stochastic adjacency matrix of the subgraph. The PPR vector $\mathbf{r}$ satisfies:

$$\mathbf{r} = \alpha_{\text{pr}} \, A^\top \mathbf{r} + (1 - \alpha_{\text{pr}}) \, \mathbf{p}$$

where:
- $\alpha_{\text{pr}} = 0.85$ (damping factor, teleport probability $= 0.15$)
- $\mathbf{p}$ is the personalisation vector: uniform over the seed entity nodes

Solved iteratively (max 50 iterations, tolerance $10^{-5}$) via `nx.pagerank()`.

After convergence, extract chunk nodes from $\mathbf{r}$ and rank by PPR score.

**Why PPR over plain BFS**: PPR propagates relevance globally within the subgraph via all paths, not just the shortest. A chunk connected to many relevant entities via different paths scores higher. Edge weights (from `RELATION_WEIGHTS`) are respected.

---

## 8. Maximal Marginal Relevance (MMR)

**Source**: `src/agent.py` — `Organizer._mmr_select()`  
**Paper**: Carbonell & Goldstein, 1998

MMR balances relevance and diversity in the final context selection.
At each step, select the candidate that maximises:

$$\text{MMR}(d_i) = \lambda \cdot s(d_i) - (1 - \lambda) \cdot \max_{d_j \in S} J(d_i, d_j)$$

where:
- $s(d_i)$ — hybrid retrieval score (already computed)
- $S$ — set of already selected documents
- $J(A, B) = \dfrac{|T_A \cap T_B|}{|T_A \cup T_B|}$ — Jaccard similarity on token sets
- $\lambda = 0.7$ — relevance/diversity trade-off ($\lambda = 1$: pure relevance, $\lambda = 0$: pure diversity)

**Jaccard vs cosine**: Jaccard on token sets is O(1) per pair (set intersection/union) and captures lexical overlap. Cosine would require an extra embedding call per pair.

---

## 9. HyDE — Hypothetical Document Embedding

**Source**: `src/agent.py` — `_hyde_expand()`  
**Paper**: Gao et al., 2022

Standard HyDE replaces the query vector with the embedding of a LLM-generated hypothetical document. Here we use an augmented-query variant:

$$q_{\text{HyDE}} = q \oplus g_\theta(q)$$

where $\,g_\theta$ is the LLM generating a 2-3 sentence passage that would answer the query, and $\oplus$ is string concatenation with a newline.

**Gating condition** (avoids hurting factual lookups):

$$\text{use HyDE}(q) = \begin{cases} \text{True} & \text{complexity} = \textit{complex} \;\land\; (\text{analytical keyword} \in q \;\lor\; |q|_{\text{words}} \geq 25) \cr \text{False} & \text{otherwise} \end{cases}$$

Analytical keywords: *tại sao, vì sao, giải thích, so sánh, phân tích, why, how, explain, compare, …*

**Why gating matters**: For factual queries ("what is the fee?"), HyDE can generate a passage that does not match the document's table/list format, producing a misleading query vector. For analytical queries, the generated passage approximates the style of a relevant document section.

---

## 10. Self-RAG Context Quality Estimator

**Source**: `src/agent.py` — `_estimate_context_quality()`

A lightweight, LLM-free quality metric based on query-token coverage:

$$q_{\text{quality}} = \frac{|T_q \cap T_c|}{|T_q|}$$

where:
- $\,T_q = \lbrace t : t \in \text{query}, |t| \geq 3 \rbrace$ — set of significant query tokens (unaccented lowercase)
- $\,T_c$ — set of all tokens in the first 3000 characters of the context (unaccented lowercase)

**Thresholds**:
- $\,q_{\text{quality}} < 0.15$ (`self_rag_threshold`): context is poor → re-retrieve with $k' = 2k$, $\text{hops}' = \text{hops}+1$
- $\,q_{\text{quality}} \geq 0.5$ (`critic_skip_threshold`): context is sufficient → skip LLM critic entirely

This avoids paying LLM latency for the critic when the context obviously covers the query (high overlap) or obviously needs re-retrieval (near-zero overlap).

---

## 11. Cross-Encoder Reranking

**Source**: `src/reranker.py`  
**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

A bi-encoder (used in FAISS) encodes query and document independently and scores by inner product:

$$s_{\text{bi}}(q, d) = f_q(q) \cdot f_d(d)$$

A cross-encoder encodes the concatenated pair jointly:

$$s_{\text{CE}}(q, d) = f_\theta([q;\, d])$$

The cross-encoder is slower (no pre-computed document embeddings) but more accurate because it can attend across query and document tokens. We apply it to the top `reranker_top_k = 20` candidates from fusion, replacing their scores with CE scores.

---

## 12. Doc-Type Intent Boost

**Source**: `src/graphrag.py` — `_doc_type_boost()`

Embed natural-language descriptions of each document type:

$$\mathbf{d}_{\text{type}} = \text{Embed}(\text{"học phí, chi phí học tập, ..."}) \quad \text{for type = hoc-phi}$$

Query-to-type similarity:

$$\text{type}^* = \arg\max_k \hat{\mathbf{q}} \cdot \hat{\mathbf{d}}_k, \quad \text{activate if } \hat{\mathbf{q}} \cdot \hat{\mathbf{d}}_{\text{type}^*} \geq 0.40$$

Multiply scores of matching documents:

$$s'(d) = \begin{cases} s(d) \times 1.35 & \text{doc-type}(d) = \text{type}^* \cr s(d) & \text{otherwise} \end{cases}$$

This avoids penalising non-matching documents — it only boosts relevant ones.
