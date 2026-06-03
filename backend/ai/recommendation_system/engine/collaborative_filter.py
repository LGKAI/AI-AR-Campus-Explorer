# engine/collaborative_filter.py
"""
Real Item-Item Collaborative Filtering
=======================================
Thay thế collaborative_filtering() mock cũ (rule-based) bằng CF thực sự
dựa trên co-visitation matrix được xây dựng từ hành vi thực tế của người dùng.

Thuật toán:
  - Item-Item CF (không cần User-Item matrix toàn cục):
      similarity(A, B) = số user đã ghé cả A lẫn B
                         ─────────────────────────────
                         sqrt(freq(A) × freq(B))       ← Cosine trên co-visit vector
  - Cold Start: Trả về [] khi có < MIN_SESSIONS data → caller fallback về content-based.
  - Incremental: update_from_profile() cập nhật model từng user một,
                 không cần rebuild toàn bộ.

So sánh với mock cũ:
  Cũ: _CLUB_RULES hardcode (c++ → lab CNTT, football → nhà thể dục)
  Mới: Học từ hành vi thực tế (nếu nhiều user ghé Tòa B rồi ghé Tòa C thì gợi ý Tòa C
       cho user mới chỉ đến Tòa B)
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

# Cần ít nhất N session để CF có ý nghĩa (tránh cold-start bias)
_MIN_SESSIONS: int = 3
# Seed nodes tối đa dùng để query (top-k node hay ghé nhất)
_MAX_SEED_NODES: int = 6
# CF score threshold: bỏ qua node có similarity quá thấp
_MIN_CF_SCORE: float = 0.01


class ItemItemCF:
    """
    Item-Item Collaborative Filtering dựa trên co-visitation.

    Sử dụng:
        cf = ItemItemCF()
        cf.update_from_all_profiles(all_profiles)   # Nạp từ DB khi startup
        cf.update_from_profile(new_profile)         # Cập nhật incremental
        recs = cf.recommend(seed_nodes, exclude=visited_set, top_k=5)
    """

    def __init__(self) -> None:
        # covisit[a][b] = số user đã ghé cả a lẫn b trong cùng 1 session
        self._covisit: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # item_freq[a] = số user đã ghé a (ít nhất 1 lần)
        self._item_freq: Dict[str, float] = defaultdict(float)
        self._n_sessions: int = 0

    # ------------------------------------------------------------------ #
    # Build / Update                                                       #
    # ------------------------------------------------------------------ #

    def update_from_session(self, visited_nodes: List[str], ratings: Optional[dict] = None) -> None:
        """
        Nạp co-visitation từ một user session (danh sách node đã ghé).
        Mỗi cặp (A, B) trong session tăng covisit[A][B] lên 1 (hoặc tích hợp trọng số ratings).
        """
        # Dedup giữ thứ tự
        unique = list(dict.fromkeys(n for n in visited_nodes if n))
        if len(unique) < 2:
            return

        ratings = ratings or {}
        # Tính trọng số cho từng node dựa trên rating (thang 1-5 sao, trung vị là 3.0)
        # 5 sao -> weight = 2.0 (boost mạnh)
        # 3 sao -> weight = 1.0 (bình thường)
        # 1 sao -> weight = 0.1 (hạ thấp tối đa)
        def _get_weight(n):
            r = ratings.get(n, 3.0)
            if r >= 5.0: return 2.0
            if r >= 4.0: return 1.5
            if r >= 3.0: return 1.0
            if r >= 2.0: return 0.4
            return 0.1

        for i, a in enumerate(unique):
            w_a = _get_weight(a)
            self._item_freq[a] += w_a
            for b in unique[i + 1:]:
                w_b = _get_weight(b)
                self._covisit[a][b] += w_a * w_b
                self._covisit[b][a] += w_a * w_b

        self._n_sessions += 1

    def update_from_profile(self, profile: dict) -> None:
        """
        Extract visited nodes từ profile dict và cập nhật co-visitation,
        tích hợp đánh giá Explicit Ratings làm trọng số (Hybrid CF).
        """
        blog = profile.get("behavior_log", {})
        history = profile.get("visited_history", {})
        ratings = profile.get("ratings", {})

        # Tập hợp tất cả các node có tương tác (đã ghé hoặc đã đánh giá sao)
        all_interacted = set()
        if blog:
            all_interacted.update(n for n, d in blog.items() if d.get("total_visits", 0) >= 1)
        if history:
            all_interacted.update(n for n, cnt in history.items() if cnt > 0)
        all_interacted.update(ratings.keys())

        # Sắp xếp các node theo độ ưu tiên: xếp hạng rating trước, sau đó tới weight_score/visits
        def _get_sort_key(n):
            r = ratings.get(n, 3.0) # default 3.0
            w = blog.get(n, {}).get("weight_score", 0.0) if blog else float(history.get(n, 0))
            return (r, w)

        nodes = sorted(list(all_interacted), key=_get_sort_key, reverse=True)

        if len(nodes) >= 2:
            self.update_from_session(nodes, ratings)

    def update_from_all_profiles(self, all_profiles: Dict[str, dict]) -> None:
        """
        Rebuild toàn bộ CF model từ tất cả profiles.
        Gọi khi startup hoặc định kỳ để đảm bảo model luôn cập nhật.
        """
        self._covisit = defaultdict(lambda: defaultdict(float))
        self._item_freq = defaultdict(float)
        self._n_sessions = 0

        for profile in all_profiles.values():
            self.update_from_profile(profile)

        print(f"✅ [CF] Model built: {self._n_sessions} sessions, "
              f"{len(self._item_freq)} unique items indexed.")

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def similarity(self, a: str, b: str) -> float:
        """
        Cosine similarity giữa 2 item dựa trên co-visitation vector.

        sim(A, B) = covisit(A, B) / sqrt(freq(A) * freq(B))
        """
        co = self._covisit[a].get(b, 0.0)
        if co == 0.0:
            return 0.0
        denom = math.sqrt(
            self._item_freq.get(a, 0.0) * self._item_freq.get(b, 0.0)
        )
        return co / (denom + 1e-8)

    def recommend(
        self,
        seed_nodes: List[str],
        exclude: Optional[Set[str]] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Gợi ý node tương tự với seed_nodes mà user chưa ghé thăm.

        Với mỗi seed node, tổng hợp similarity tới tất cả neighbor
        (aggregated neighborhood: i2i CF chuẩn).

        Args:
            seed_nodes: Các node user đã ghé (dùng làm query)
            exclude:    Tập node không gợi ý (đã ghé, đang đứng gần, v.v.)
            top_k:      Số kết quả trả về tối đa

        Returns:
            List[(node_id, cf_score)] sắp xếp giảm dần
        """
        exclude = exclude or set()
        scores: Dict[str, float] = defaultdict(float)

        for seed in seed_nodes:
            for neighbor in self._covisit.get(seed, {}):
                if neighbor in exclude:
                    continue
                sim = self.similarity(seed, neighbor)
                if sim >= _MIN_CF_SCORE:
                    scores[neighbor] += sim

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def has_data(self) -> bool:
        """Kiểm tra đủ data để dùng CF (tránh cold-start gây noise)."""
        return self._n_sessions >= _MIN_SESSIONS

    @property
    def n_sessions(self) -> int:
        return self._n_sessions

    def debug_stats(self) -> dict:
        return {
            "n_sessions": self._n_sessions,
            "unique_items": len(self._item_freq),
            "total_covisit_pairs": sum(len(v) for v in self._covisit.values()) // 2,
            "has_data": self.has_data(),
        }


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------
_CF_MODEL: Optional[ItemItemCF] = None


def get_cf_model() -> ItemItemCF:
    """Trả về CF model singleton. Khởi tạo rỗng nếu chưa có."""
    global _CF_MODEL
    if _CF_MODEL is None:
        _CF_MODEL = ItemItemCF()
    return _CF_MODEL


def rebuild_cf_model(all_profiles: Dict[str, dict]) -> ItemItemCF:
    """
    Rebuild CF model từ tất cả user profiles trong DB.
    Gọi khi startup và định kỳ (ví dụ mỗi 30 phút).
    """
    global _CF_MODEL
    cf = ItemItemCF()
    cf.update_from_all_profiles(all_profiles)
    _CF_MODEL = cf
    return cf


def increment_cf_model(profile: dict) -> None:
    """
    Cập nhật incremental CF model khi một user có hành vi mới.
    Không cần rebuild toàn bộ → O(k) thay vì O(N).
    """
    get_cf_model().update_from_profile(profile)


# ---------------------------------------------------------------------------
# High-level API cho recommender.py
# ---------------------------------------------------------------------------

def cf_recommend_for_profile(
    G: nx.Graph,
    profile: dict,
    exclude_nodes: Optional[Set[str]] = None,
    top_k: int = 5,
) -> List[dict]:
    """
    Gợi ý CF cho một user profile cụ thể.

    Cold start handling:
      - Nếu CF chưa đủ data (< MIN_SESSIONS) → trả về [] để caller dùng fallback.
      - Nếu user chưa có lịch sử → trả về [].

    Args:
        G:             Campus graph (để validate node tồn tại)
        profile:       User profile dict (visited_history / behavior_log)
        exclude_nodes: Set node không gợi ý
        top_k:         Số kết quả tối đa

    Returns:
        List[dict] mỗi item: {"node", "cf_score", "reason", "source"}
    """
    cf = get_cf_model()
    if not cf.has_data():
        return []  # Cold start → fallback về content-based

    blog = profile.get("behavior_log", {})
    history = profile.get("visited_history", {})

    # Chọn seed nodes
    if blog:
        seed_nodes = sorted(
            [n for n, d in blog.items() if d.get("total_visits", 0) >= 1],
            key=lambda n: blog[n].get("weight_score", 0.0),
            reverse=True,
        )[:_MAX_SEED_NODES]
    else:
        seed_nodes = sorted(
            [n for n, cnt in history.items() if cnt > 0],
            key=lambda n: history.get(n, 0),
            reverse=True,
        )[:_MAX_SEED_NODES]

    if not seed_nodes:
        return []

    exclude = (exclude_nodes or set()) | set(seed_nodes)
    raw_recs = cf.recommend(seed_nodes, exclude=exclude, top_k=top_k)

    result = []
    for node, cf_score in raw_recs:
        if node not in G.nodes:
            continue
        # Giải thích gợi ý bằng seed node tương đồng nhất
        best_seed = max(seed_nodes, key=lambda s: cf.similarity(s, node))
        result.append({
            "node": node,
            "cf_score": round(cf_score, 4),
            "reason": f"Người dùng ghé {best_seed} thường cũng ghé {node}",
            "source": "item_item_cf",
        })

    return result
