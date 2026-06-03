# engine/context_engine.py
"""
ContextEngine — Tính toán Context Signals & Exploration Sampling
=================================================================
Tách từ recommender.py để dễ bảo trì và test riêng biệt.

Giải quyết:
  - Ziczac Bug: detour_distance_score() thay _geo_alignment_score() góc hướng
  - Naive Noise: epsilon_greedy_sample() + gumbel_softmax_rank()
    thay thế random.uniform(0.0, 1.5) không kiểm soát
  - Vector hóa: all_node_distances() tính khoảng cách hàng loạt bằng NumPy
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from engine.utils import haversine
from engine.context_features import (
    get_time_of_day_band,
    indoor_boost,
    is_weekend,
    time_category_boost,
)

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
_DETOUR_ACCEPT_M: float = 60.0   # Ngưỡng quãng đường đi vòng chấp nhận được (m)
_EPSILON_DEFAULT: float = 0.15   # Xác suất khai thác ngẫu nhiên trong ε-greedy
_GUMBEL_TEMP: float     = 0.6    # Temperature của Gumbel-Softmax (thấp = ít ngẫu nhiên)

# Bảng synonym tiếng Việt mở rộng (giải quyết giới hạn TF-IDF lexical-only)
SYNONYM_MAP: Dict[str, List[str]] = {
    "chop mat":   ["nghi ngoi", "ngu trua", "nghi trua", "ngu", "met"],
    "nap nang":   ["nghi ngoi", "nghi trua", "ngu trua"],
    "buon ngu":   ["ngu trua", "nghi ngoi", "met"],
    "met moi":    ["nghi ngoi", "met", "ghe"],
    "sac pin":    ["o cam", "dien", "sac dien thoai", "ban ghe"],
    "nuoc uong":  ["can tin", "cafe", "uong"],
    "giat minh":  ["the thao", "van dong", "the duc"],
    "chay bo":    ["the thao", "san the duc", "the duc"],
    "nong":       ["dieu hoa", "mat me", "may lanh"],
    "lanh":       ["dieu hoa", "may lanh", "mat me"],
    "on ao":      ["yen tinh", "on a", "thu vien"],
    "hoc nhom":   ["phong hop", "ban ghe", "can tin", "khu hoc nhom"],
    "in tai lieu":["may in", "van phong", "thu vien"],
    "rut tien":   ["atm", "may rut tien"],
    "gui xe":     ["nha xe", "bai do xe"],
    "lam bai":    ["thu vien", "tu hoc", "ban ghe", "yen tinh"],
    "kiem tra":   ["lop hoc", "phong thi", "toa b", "toa c"],
}


# ---------------------------------------------------------------------------
# Utility: Vectorized distance computation
# ---------------------------------------------------------------------------
def all_node_distances(
    G: nx.Graph,
    user_lat: float,
    user_lon: float,
) -> Dict[str, float]:
    """
    Tính khoảng cách từ user đến TẤT CẢ node cùng lúc bằng NumPy vectorization.

    Thay thế vòng lặp:
        for node in G.nodes(): dist = haversine(...)

    Tốc độ tăng ~10-50x cho graph lớn.

    Trả về: {node_id: distance_m}
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}

    # Lấy mảng lat/lon của tất cả node
    lats = np.array([G.nodes[n]["gps"][0] for n in nodes], dtype=np.float64)
    lons = np.array([G.nodes[n]["gps"][1] for n in nodes], dtype=np.float64)

    # Haversine vectorized
    R = 6_371_000.0  # bán kính Trái Đất (m)
    phi1  = math.radians(user_lat)
    phi2  = np.radians(lats)
    d_phi = phi2 - phi1
    d_lam = np.radians(lons - user_lon)

    a = (np.sin(d_phi / 2.0) ** 2
         + math.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    distances = R * c  # mảng khoảng cách (m)

    return {nodes[i]: float(distances[i]) for i in range(len(nodes))}


def nearest_node_vectorized(
    G: nx.Graph,
    user_lat: float,
    user_lon: float,
) -> Tuple[str, float]:
    """Trả về (node_id, distance_m) của node gần nhất — dùng NumPy."""
    dists = all_node_distances(G, user_lat, user_lon)
    if not dists:
        return "", float("inf")
    nearest = min(dists, key=lambda n: dists[n])
    return nearest, dists[nearest]


# ---------------------------------------------------------------------------
# Detour Distance (thay _geo_alignment_score góc hướng)
# ---------------------------------------------------------------------------
def detour_distance_m(
    user_lat: float,
    user_lon: float,
    node_lat: float,
    node_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> float:
    """
    Tính "quãng đường đi vòng" (detour) nếu ghé qua node trên đường đến đích (m).

    Công thức:
        detour = dist(user→node) + dist(node→dest) − dist(user→dest)

    - detour ≈ 0   : node nằm chính xác trên đường thẳng → hoàn toàn thuận tiện
    - detour = 50m : phải đi vòng thêm 50m → chấp nhận được
    - detour > 120m: node lệch xa, không nên gợi ý
    """
    d_user_node = haversine(user_lat, user_lon, node_lat, node_lon)
    d_node_dest = haversine(node_lat, node_lon, dest_lat, dest_lon)
    d_user_dest = haversine(user_lat, user_lon, dest_lat, dest_lon)
    return max(0.0, d_user_node + d_node_dest - d_user_dest)


def detour_distance_score(
    user_lat: float,
    user_lon: float,
    node_lat: float,
    node_lon: float,
    dest_lat: float,
    dest_lon: float,
    accept_threshold_m: float = _DETOUR_ACCEPT_M,
) -> float:
    """
    Chuyển đổi detour_distance_m thành điểm số.

    Thay thế _geo_alignment_score() dựa trên góc hướng (bị sai khi đường ziczac).

    Điểm:
    - detour ≤ accept_threshold_m → score = 18.0 × (1 - detour/threshold)
      (linear decay từ 18 về 0 khi detour tăng lên ngưỡng)
    - accept_threshold_m < detour ≤ 2× → score nhỏ dương (vẫn chấp nhận được)
    - detour > 2× threshold → score âm (-5.0)
    """
    detour = detour_distance_m(user_lat, user_lon, node_lat, node_lon, dest_lat, dest_lon)

    if detour <= accept_threshold_m:
        # Linear decay từ 18.0 (detour=0) → 0.0 (detour=threshold)
        return round(18.0 * (1.0 - detour / accept_threshold_m), 2)
    elif detour <= accept_threshold_m * 2.0:
        # Vẫn gợi ý nhưng ít ưu tiên
        overshoot = (detour - accept_threshold_m) / accept_threshold_m
        return round(max(0.0, 5.0 * (1.0 - overshoot)), 2)
    else:
        return -5.0


# ---------------------------------------------------------------------------
# Exploration Sampling: Epsilon-Greedy & Gumbel-Softmax
# ---------------------------------------------------------------------------
def epsilon_greedy_sample(
    candidates: List[Tuple[str, float]],
    epsilon: float = _EPSILON_DEFAULT,
) -> List[Tuple[str, float]]:
    """
    Áp dụng chiến lược ε-greedy để pha trộn Exploitation và Exploration.

    Thay thế: random.uniform(0.0, 1.5) cộng trực tiếp vào điểm số (gây đảo lộn thứ hạng)

    Thuật toán:
    - Với xác suất (1 - ε): giữ nguyên thứ hạng (Exploit top-scored candidates)
    - Với xác suất ε: random chọn 1 node ngoài top-3 và đưa vào cuối danh sách
      (Explore ít nhất 1 item khác biệt mà không phá vỡ thứ hạng cũ)

    Tham số:
        candidates — list of (node_id, score) đã sắp xếp giảm dần
        epsilon    — xác suất explore (mặc định 0.15 = 15%)

    Trả về:
        Danh sách candidates đã được "điều chỉnh nhẹ" theo ε-greedy
    """
    if len(candidates) <= 3 or random.random() > epsilon:
        return candidates  # Exploit: giữ nguyên

    # Explore: swap một item dưới top-3 lên vị trí cuối của top-results
    top3    = candidates[:3]
    rest    = candidates[3:]
    chosen  = random.choice(rest)
    rest.remove(chosen)
    return top3 + [chosen] + rest


def gumbel_softmax_rank(
    candidates: List[Tuple[str, float]],
    temperature: float = _GUMBEL_TEMP,
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Sắp xếp lại candidates theo phân phối Gumbel-Softmax.

    Thay thế: random.uniform noise trực tiếp (phá vỡ phân phối điểm số)

    Thuật toán Gumbel-Softmax:
        g_i = -log(-log(U_i))  với U_i ~ Uniform(0, 1)
        perturbed_score_i = (original_score_i / temperature) + g_i
        Sắp xếp theo perturbed_score giảm dần

    Ưu điểm:
    - Giữ nguyên phân phối xác suất tương đối (item có score cao VẪN có nhiều
      khả năng xếp trên)
    - Temperature cao → ngẫu nhiên hơn; thấp → sát với thứ hạng gốc
    - Không gây đảo lộn vô lý giữa 2 item có score suýt soát nhau

    Tham số:
        candidates  — list of (node_id, score)
        temperature — điều chỉnh độ ngẫu nhiên (0.1=gần gốc, 2.0=rất ngẫu nhiên)
        top_k       — nếu set, chỉ sample từ top_k item đầu

    Trả về:
        Danh sách candidates đã sắp xếp lại theo Gumbel-Softmax
    """
    if not candidates:
        return candidates

    pool = candidates[:top_k] if top_k else candidates
    tail = candidates[top_k:] if top_k else []

    scores     = np.array([s for _, s in pool], dtype=np.float64)
    # Sinh Gumbel noise: g = -log(-log(U))
    u_uniform  = np.random.uniform(1e-8, 1.0, size=len(scores))
    gumbel     = -np.log(-np.log(u_uniform))
    perturbed  = scores / max(temperature, 1e-6) + gumbel

    order = np.argsort(-perturbed)
    reranked = [pool[i] for i in order]

    return reranked + tail


# ---------------------------------------------------------------------------
# ContextEngine
# ---------------------------------------------------------------------------
class ContextEngine:
    """
    Tính toán tất cả context signals cho hệ thống gợi ý.

    Sử dụng:
        ce = ContextEngine(G, current_time_str, weather)
        score = ce.compute_context_score(node, battery=0.1, temp=35.0)
        detour_s = ce.detour_score(user_pos, node_pos, dest_pos)
    """

    def __init__(
        self,
        G: nx.Graph,
        current_time_str: Optional[str] = None,
        weather: str = "normal",
    ):
        self._G          = G
        self._time_str   = current_time_str
        self._weather    = weather
        self._time_band  = get_time_of_day_band(current_time_str)
        self._is_weekend = is_weekend()

    # ------------------------------------------------------------------
    # Context Score (environmental signals)
    # ------------------------------------------------------------------
    def compute_context_score(
        self,
        node: str,
        battery_level: Optional[float] = None,
        temperature:   Optional[float] = None,
        uv_index:      Optional[float] = None,
    ) -> float:
        """Tổng hợp điểm context: thời gian + thời tiết + pin + nhiệt độ + UV."""
        score = 0.0
        score += time_category_boost(node, self._G, self._time_band, self._is_weekend)
        score += indoor_boost(self._G, node, self._weather)

        data      = self._G.nodes.get(node, {})
        features  = data.get("features", {})
        is_indoor = data.get("indoor", False)

        # Pin thấp → ưu tiên nơi có ổ cắm
        if battery_level is not None and battery_level < 0.20:
            if features.get("has_tables") and is_indoor:
                score += 10.0

        # Nhiệt độ cao → ưu tiên nơi có điều hòa
        if temperature is not None and temperature > 33.0:
            if features.get("has_ac"):
                score += 8.0
            elif not is_indoor:
                score -= 12.0

        # UV cao → ưu tiên trong nhà
        if uv_index is not None and uv_index > 5.0:
            if is_indoor:
                score += 5.0
            else:
                score -= 8.0

        return score

    # ------------------------------------------------------------------
    # Detour Distance Score (thay góc hướng)
    # ------------------------------------------------------------------
    def detour_score(
        self,
        user_pos: Tuple[float, float],
        node_pos: Tuple[float, float],
        dest_pos: Tuple[float, float],
        threshold_m: float = _DETOUR_ACCEPT_M,
    ) -> float:
        """Gọi detour_distance_score() với tham số từ ContextEngine."""
        return detour_distance_score(
            user_pos[0], user_pos[1],
            node_pos[0], node_pos[1],
            dest_pos[0], dest_pos[1],
            accept_threshold_m=threshold_m,
        )

    # ------------------------------------------------------------------
    # Synonym Expansion (cải thiện TF-IDF matching)
    # ------------------------------------------------------------------
    @staticmethod
    def expand_query_synonyms(query_normalized: str) -> str:
        """
        Mở rộng câu query bằng bảng synonym tiếng Việt.

        Ví dụ:
            "chop mat" → "chop mat nghi ngoi ngu trua nghi trua ngu met"

        Giúp TF-IDF matching tìm được ngữ nghĩa gần mà không cần Vector DB.
        """
        tokens = query_normalized.split()
        extra: List[str] = []

        # 1-gram match
        for token in tokens:
            if token in SYNONYM_MAP:
                extra.extend(SYNONYM_MAP[token])

        # 2-gram match
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            if bigram in SYNONYM_MAP:
                extra.extend(SYNONYM_MAP[bigram])

        if extra:
            return query_normalized + " " + " ".join(extra)
        return query_normalized

    # ------------------------------------------------------------------
    # Vectorized distances (expose for recommender)
    # ------------------------------------------------------------------
    def compute_all_distances(
        self,
        user_lat: float,
        user_lon: float,
    ) -> Dict[str, float]:
        """Tính khoảng cách hàng loạt bằng NumPy vectorization."""
        return all_node_distances(self._G, user_lat, user_lon)

    # ------------------------------------------------------------------
    # Exploration sampling
    # ------------------------------------------------------------------
    @staticmethod
    def epsilon_greedy(
        candidates: List[Tuple[str, float]],
        epsilon: float = _EPSILON_DEFAULT,
    ) -> List[Tuple[str, float]]:
        """ε-greedy sampling — wrapper."""
        return epsilon_greedy_sample(candidates, epsilon)

    @staticmethod
    def gumbel_rank(
        candidates: List[Tuple[str, float]],
        temperature: float = _GUMBEL_TEMP,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Gumbel-Softmax re-ranking — wrapper."""
        return gumbel_softmax_rank(candidates, temperature, top_k)
