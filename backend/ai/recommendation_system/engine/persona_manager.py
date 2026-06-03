# engine/persona_manager.py
"""
PersonaManager — Quản lý & Cập nhật Hồ sơ Người dùng
======================================================
Tách từ recommender.py để dễ bảo trì và test riêng biệt.

Giải quyết:
  - Popularity Bias: sigmoid_history_score() chặn trên thay vì cộng tuyến tính
  - Filter Bubble: phân loại node thành 2 bucket familiar / discovery
  - Noise Inference: compute_node_weight_score() với Dwell + Intent + Frequency
"""

import math
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
_SIGMOID_MAX_BOOST: float = 8.0   # Điểm lịch sử tối đa (Sigmoid cap)
_SIGMOID_MIDPOINT: float  = 0.5   # Weight score ở 50% = midpoint
_FAMILIAR_THRESHOLD: int  = 1     # Số lần ghé thăm tối thiểu để xếp vào "quen thuộc"
_WEIGHT_ALPHA: float      = 0.5   # Trọng số Dwell Time
_WEIGHT_BETA:  float      = 0.3   # Trọng số Intent Ratio
_WEIGHT_GAMMA: float      = 0.2   # Trọng số Frequency


# ---------------------------------------------------------------------------
# Hàm trợ giúp: tính trọng số
# ---------------------------------------------------------------------------
def calculate_dwell_time_factor(t_mins: float, lambda_val: float = 0.15) -> float:
    """DwellTime_Factor = 1 - e^(-λ·t)  ∈ [0, 1]."""
    return 1.0 - math.exp(-lambda_val * t_mins)


def compute_node_weight_score(
    total_visits: int,
    avg_dwell_time_mins: float,
    intentional_visits: int,
    alpha: float = _WEIGHT_ALPHA,
    beta:  float = _WEIGHT_BETA,
    gamma: float = _WEIGHT_GAMMA,
) -> float:
    """
    Tính trọng số tin cậy động cho một node ∈ [0.0, 1.0].

    Công thức:
        score = α · DwellFactor + β · IntentRatio + γ · FreqFactor

    Tham số:
        total_visits        — Tổng lần ghé thăm (cả chủ đích lẫn vô tình)
        avg_dwell_time_mins — Thời gian lưu trú trung bình (phút)
        intentional_visits  — Số lần ghé có chủ đích (đến đích, tìm kiếm, lịch học)
    """
    if total_visits == 0:
        return 0.0

    dwell_factor  = calculate_dwell_time_factor(avg_dwell_time_mins)
    intent_ratio  = intentional_visits / total_visits
    freq_factor   = min(1.0, math.log1p(total_visits) / math.log1p(30))

    score = (alpha * dwell_factor) + (beta * intent_ratio) + (gamma * freq_factor)
    return round(max(0.0, min(1.0, score)), 3)


def sigmoid_history_score(weight_score: float, max_boost: float = _SIGMOID_MAX_BOOST) -> float:
    """
    Chuyển đổi weight_score ∈ [0,1] thành điểm lịch sử bằng hàm Sigmoid.

    Đặc điểm:
    - Thay thế cộng tuyến tính `weight * 8.0` → tránh Popularity Bias
    - Hàm Sigmoid đảm bảo điểm hội tụ, không tăng vô hạn
    - Tại weight=0.5 → boost ≈ max_boost/2; tại weight→1 → boost → max_boost

    Công thức:
        boost = max_boost × sigmoid(10 × (w - 0.5))
        sigmoid(x) = 1 / (1 + e^(-x))
    """
    # Dịch Sigmoid để tại w=0 → ≈ 0; tại w=1 → ≈ max_boost
    raw_sigmoid = 1.0 / (1.0 + math.exp(-10.0 * (weight_score - 0.5)))
    # Chuẩn hóa để tại w=0 → 0 và w=1 → max_boost
    baseline = 1.0 / (1.0 + math.exp(5.0))   # sigmoid(-5) ≈ 0.0067
    topline  = 1.0 / (1.0 + math.exp(-5.0))  # sigmoid(+5) ≈ 0.9933
    normalized = (raw_sigmoid - baseline) / (topline - baseline)
    return round(max(0.0, normalized) * max_boost, 3)


# ---------------------------------------------------------------------------
# PersonaManager
# ---------------------------------------------------------------------------
class PersonaManager:
    """
    Quản lý hồ sơ người dùng: passive inference, history scoring, bucket phân loại.

    Sử dụng:
        pm = PersonaManager(profile)
        pm.update_passively()                    # suy luận role/interests tự động
        boost = pm.history_boost(node)           # điểm lịch sử an toàn (Sigmoid)
        familiar, discovery = pm.split_buckets() # phân loại node
    """

    def __init__(self, profile: dict):
        self._profile = profile
        self._profile.setdefault("visited_history", {})
        self._profile.setdefault("behavior_log", {})
        self._profile.setdefault("search_history", [])
        self._profile.setdefault("_last_tracking", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def role(self) -> str:
        return self._profile.get("role", "student")

    @property
    def study_style(self) -> str:
        return self._profile.get("study_style", "silent")

    @property
    def interests(self) -> List[str]:
        return self._profile.get("interests", ["hoc_tap"])

    @property
    def visited_history(self) -> Dict[str, float]:
        return self._profile.get("visited_history", {})

    @property
    def behavior_log(self) -> Dict[str, dict]:
        return self._profile.get("behavior_log", {})

    def update_passively(self) -> None:
        """Phân tích behavior_log để tự suy luận role, study_style, interests."""
        history = self._profile.setdefault("visited_history", {})
        blog    = self._profile.setdefault("behavior_log", {})

        if not history and not blog:
            self._profile.setdefault("role", "student")
            self._profile.setdefault("study_style", "silent")
            self._profile.setdefault("interests", ["hoc_tap"])
            return

        def _w(node: str) -> float:
            if node in blog:
                return blog[node].get("weight_score", 0.0)
            count = history.get(node, 0)
            if count > 1.0:
                return count / 100.0
            return min(1.0, count / 5.0) if count > 0 else 0.0

        cntt_w    = _w("Tòa C") + _w("Tòa B")
        sport_w   = _w("Nhà thể dục") + _w("Tòa G")
        food_w    = _w("Tòa D")
        study_w   = _w("Tòa B") + _w("Tòa D") + _w("Tòa F")
        admin_w   = _w("Nhà điều hành")
        visitor_w = _w("Cổng trường") + _w("ATM") + _w("Nhà xe")

        # 1. Interests
        interests: List[str] = []
        if cntt_w  >= 0.3: interests += ["cntt", "hoc_tap"]
        if sport_w >= 0.3: interests.append("the_thao")
        if food_w  >= 0.3: interests.append("an_uong")
        if study_w >= 0.3 and "hoc_tap" not in interests:
            interests.append("hoc_tap")
        if not interests:
            interests = ["hoc_tap"]
        self._profile["interests"] = list(set(interests))

        # 2. Role
        if admin_w > cntt_w + study_w + sport_w:
            self._profile["role"] = "lecturer"
        elif visitor_w > 0.4 and cntt_w + study_w + sport_w < 0.2:
            self._profile["role"] = "visitor"
        else:
            self._profile["role"] = "student"

        # 3. Study style
        silent_score = _w("Tòa B") + _w("Tòa F") + _w("Tòa D") * 0.5
        group_score  = _w("Tòa D") * 0.5 + _w("Tòa G") + _w("Tòa A")
        self._profile["study_style"] = "group" if group_score > silent_score else "silent"

    def history_boost(
        self,
        node: str,
        query_active: bool = False,
        time_boosted_categories: Optional[set] = None,
        node_categories: Optional[set] = None,
    ) -> float:
        """
        Trả về điểm lịch sử cho node, dùng Sigmoid thay vì cộng tuyến tính.

        Giảm điểm khi:
        - query_active=True → người dùng có nhu cầu hiện tại cụ thể (giảm 80%)
        - Loại node không khớp với khung giờ hiện tại (giảm 50%)
        """
        blog    = self.behavior_log
        history = self.visited_history

        boost = 0.0
        if node in blog:
            w_score = blog[node].get("weight_score", 0.0)
            boost = sigmoid_history_score(w_score)
        else:
            visit_count = history.get(node, 0)
            if visit_count > 0:
                # Backward compat: visited_history có thể lưu điểm % (> 1.0)
                w_approx = (visit_count / 100.0) if visit_count > 1.0 else min(1.0, visit_count / 5.0)
                boost = sigmoid_history_score(w_approx)

        if boost <= 0.0:
            return 0.0

        if query_active:
            boost *= 0.2

        if (time_boosted_categories and node_categories is not None
                and not (node_categories & time_boosted_categories)):
            boost *= 0.5

        return boost

    def novelty_boost(self, node: str) -> float:
        """
        Điểm thưởng cho node chưa ghé thăm → khuyến khích khám phá.
        Trả về 4.0 nếu node chưa có trong history, ngược lại trả 0.0.
        """
        blog    = self.behavior_log
        history = self.visited_history
        if node not in blog and history.get(node, 0) == 0:
            return 4.0
        return 0.0

    def is_familiar(self, node: str) -> bool:
        """Node đã được ghé thăm ít nhất _FAMILIAR_THRESHOLD lần."""
        blog    = self.behavior_log
        history = self.visited_history
        if node in blog and blog[node].get("total_visits", 0) >= _FAMILIAR_THRESHOLD:
            return True
        if history.get(node, 0) >= _FAMILIAR_THRESHOLD:
            return True
        return False

    def split_buckets(self, candidate_nodes: List[str]) -> Tuple[List[str], List[str]]:
        """
        Phân loại danh sách node thành 2 bucket:
          - familiar:  đã đi ≥ 1 lần (personalized but may cause filter bubble)
          - discovery: chưa đi lần nào (Exploration / Serendipity)

        Trả về: (familiar_list, discovery_list)
        """
        familiar  = [n for n in candidate_nodes if self.is_familiar(n)]
        discovery = [n for n in candidate_nodes if not self.is_familiar(n)]
        return familiar, discovery

    def record_visit(
        self,
        node: str,
        time_delta_mins: float,
        is_new_visit: bool,
        is_intentional: bool,
    ) -> float:
        """
        Cập nhật behavior_log cho một lần ghé thăm và tính lại weight_score.
        Trả về weight_score mới.
        """
        blog = self._profile.setdefault("behavior_log", {})
        entry = blog.setdefault(node, {
            "total_visits":        0,
            "total_dwell_time_mins": 0.0,
            "intentional_visits":  0,
            "transient_passes":    0,
            "weight_score":        0.0,
            "last_visited":        None,
        })

        entry["total_dwell_time_mins"] += time_delta_mins

        if is_new_visit:
            entry["total_visits"] += 1
            if is_intentional:
                entry["intentional_visits"] += 1
            else:
                entry["transient_passes"]   += 1

        total_v = entry["total_visits"]
        avg_dwell = entry["total_dwell_time_mins"] / max(1, total_v)

        w_score = compute_node_weight_score(
            total_visits=total_v,
            avg_dwell_time_mins=avg_dwell,
            intentional_visits=entry["intentional_visits"],
        )
        entry["weight_score"] = w_score

        hist = self._profile.setdefault("visited_history", {})
        hist[node] = round(w_score * 100, 1)

        return w_score


# ---------------------------------------------------------------------------
# Hàm backward-compat (để các module cũ không bị vỡ import)
# ---------------------------------------------------------------------------
def update_profile_passively(profile: dict) -> dict:
    """Wrapper backward-compat: gọi PersonaManager.update_passively()."""
    pm = PersonaManager(profile)
    pm.update_passively()
    return profile
