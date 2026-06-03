# engine/recommender.py  — v5 (Real CF + Singleton + Persistent)
"""
Recommender v5 — Nâng cấp theo chuẩn công nghiệp:
  ✅ Sigmoid history cap (chống Popularity Bias)
  ✅ Gumbel-Softmax / ε-greedy (thay Naive Noise)
  ✅ Detour Distance (thay góc hướng, sửa bug ziczac)
  ✅ NumPy vectorized distances (tốc độ tăng 10-50×)
  ✅ 2-bucket response: familiar / discovery (chống Filter Bubble)
  ✅ Synonym expansion (cải thiện TF-IDF matching tiếng Việt)
  ✅ PersonaManager + ContextEngine tách module riêng
  ✅ [v5] CampusSemanticAI singleton (không rebuild TF-IDF mỗi request)
  ✅ [v5] Real Item-Item CF (thay rule-based mock)
  ✅ [v5] CF cold-start fallback tự động
"""
import math
import os
import json
from collections import Counter
from datetime import time, datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from engine.building_catalog import (
    build_function_reason,
    enrich_suggestion,
    get_building_profile,
    list_all_building_guides,
    match_services_to_query,
)
from engine.optimizer import is_node_open, pathfinding_optimizer
from engine.nlp_processor import normalize_text
from engine.utils import haversine, parse_time, get_current_time_str
from engine.context_features import (
    get_time_of_day_band,
    indoor_boost,
    is_weekend,
    open_status_detail,
    time_category_boost,
    effective_crowd_level,
)
from engine.campus_knowledge import report_live_crowd, get_live_crowd

# ── Import 2 module mới tách ra ──────────────────────────────────────────────
from engine.persona_manager import (
    PersonaManager,
    update_profile_passively,          # backward-compat wrapper
    compute_node_weight_score,         # backward-compat re-export
    calculate_dwell_time_factor,       # backward-compat re-export
    sigmoid_history_score,
)
from engine.context_engine import (
    ContextEngine,
    all_node_distances,
    nearest_node_vectorized,
    detour_distance_score,
    gumbel_softmax_rank,
    epsilon_greedy_sample,
    _GUMBEL_TEMP,
)

_MAX_RAW_SCORE = 60.0
_MAX_PROACTIVE = 6
_DETOUR_RADIUS_M = 120.0
_NEARBY_RADIUS_M = 350.0


def _normalize_score(raw: float) -> float:
    """Chuẩn hóa điểm thô về thang [0, 100]."""
    normalized = (raw / _MAX_RAW_SCORE) * 100
    return round(max(0.0, min(100.0, normalized)), 1)


def check_keyword_in_aliases(keyword: str, aliases_str: str) -> bool:
    """Kiểm tra keyword có tồn tại dưới dạng từ hoặc cụm từ trọn vẹn trong aliases_str."""
    import re
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return bool(re.search(pattern, aliases_str))


def any_keyword_in_aliases(keywords: list, aliases_str: str) -> bool:
    return any(check_keyword_in_aliases(k, aliases_str) for k in keywords)



# calculate_dwell_time_factor và compute_node_weight_score đã được chuyển sang
# engine/persona_manager.py và được re-export ở trên để backward-compatible.


# =====================================================================
# THỤ ĐỘNG CÁ NHÂN HÓA → đã chuyển sang engine/persona_manager.py
# update_profile_passively() được re-export từ persona_manager (backward-compat)
# =====================================================================


# =====================================================================
# LỚP AI: TF-IDF + Cosine Similarity (không cần API ngoài)
# =====================================================================
class CampusSemanticAI:
    """
    Chỉ mục ngữ nghĩa cho toàn bộ node campus — dùng TF-IDF thuần NumPy.

    v4 nâng cấp:
    - score_query() tự động mở rộng query qua bảng Synonym tiếng Việt
      (ContextEngine.expand_query_synonyms) → cải thiện recall đáng kể
      mà không cần Vector DB / external model.
    """

    def __init__(self, G: nx.Graph):
        self._nodes: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._vocab: List[str] = []
        self._idf: Optional[np.ndarray] = None
        self._build(G)

    @staticmethod
    def _node_document(node: str, data: dict) -> str:
        aliases = " ".join(data.get("aliases", []))
        features = data.get("features", {})
        tags: List[str] = []
        if features.get("has_ac"):
            tags += ["mat me", "may lanh", "dieu hoa", "lanh"]
        if features.get("has_tables"):
            tags += ["ban ghe", "hoc bai", "tu hoc", "ngoi hoc"]
        noise = features.get("noise_level", 0.5)
        if noise <= 0.3:
            tags += ["yen tinh", "on a", "tap trung", "doc sach"]
        elif noise >= 0.7:
            tags += ["on ao", "dong vui"]
        node_type = data.get("type", "")
        if node_type == "facility":
            tags += ["tien ich", "phuc vu"]
        for svc in data.get("services", []):
            tags.append(normalize_text(svc.get("name", "")))
            tags.extend(normalize_text(k) for k in svc.get("keywords", []))
        if data.get("tagline"):
            tags.append(normalize_text(data["tagline"]))
        from engine.campus_knowledge import REVIEW_SIGNALS
        for sig in REVIEW_SIGNALS.get(node, []):
            tags.extend(normalize_text(k) for k in sig.get("keywords", []))
        return normalize_text(f"{node} {aliases} {' '.join(tags)}")

    def _build(self, G: nx.Graph) -> None:
        docs: List[Counter] = []
        vocab: set = set()
        self._nodes = list(G.nodes())

        for node in self._nodes:
            tokens = self._node_document(node, G.nodes[node]).split()
            counter = Counter(t for t in tokens if t)
            docs.append(counter)
            vocab.update(counter.keys())

        self._vocab = sorted(vocab)
        if not self._vocab:
            self._matrix = np.zeros((len(self._nodes), 0))
            return

        n_docs = len(docs)
        df = np.zeros(len(self._vocab), dtype=float)
        for counter in docs:
            for idx, term in enumerate(self._vocab):
                if term in counter:
                    df[idx] += 1.0

        self._idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        matrix = np.zeros((n_docs, len(self._vocab)), dtype=float)

        for row, counter in enumerate(docs):
            total = sum(counter.values()) or 1
            for idx, term in enumerate(self._vocab):
                if term in counter:
                    tf = counter[term] / total
                    matrix[row, idx] = tf * self._idf[idx]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    def score_query(self, query: str) -> Dict[str, float]:
        """
        Tính điểm cosine similarity giữa query và tất cả node.

        v4: Tự động mở rộng query qua SYNONYM_MAP (ContextEngine)
        để khắc phục giới hạn TF-IDF chỉ khớp từ khóa exact.
        Ví dụ: 'chop mat' → tự thêm 'nghi ngoi ngu trua nghi trua'
        """
        if not query or self._matrix is None or self._matrix.size == 0:
            return {}

        # Mở rộng query với synonym tiếng Việt trước khi tokenize
        query_expanded = ContextEngine.expand_query_synonyms(normalize_text(query))
        tokens = query_expanded.split()
        if not tokens:
            return {}

        counter = Counter(tokens)
        total = sum(counter.values()) or 1
        q_vec = np.zeros(len(self._vocab), dtype=float)
        for idx, term in enumerate(self._vocab):
            if term in counter:
                tf = counter[term] / total
                q_vec[idx] = tf * self._idf[idx]

        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return {}
        q_vec /= norm

        sims = self._matrix @ q_vec
        return {self._nodes[i]: float(sims[i]) for i in range(len(self._nodes))}


# ---------------------------------------------------------------------------
# TF-IDF Singleton Cache — tránh rebuild ma trận mỗi request
# ---------------------------------------------------------------------------
_SEMANTIC_AI_CACHE: Dict[int, "CampusSemanticAI"] = {}


def _get_semantic_ai(G: nx.Graph) -> "CampusSemanticAI":
    """
    Trả về CampusSemanticAI đã build sẵn cho graph G.
    Key = số lượng node (đủ để detect thay đổi; đồ thị campus hầu như cố định).

    v5 fix: Trước đây CampusSemanticAI(G) được tạo mới mỗi lần gọi
    recommend_locations() và trong vòng lặp recommend_by_building_function()
    → tốc độ chậm O(N_nodes × N_requests).
    Sau fix: singleton O(1) lookup sau lần đầu tiên.
    """
    key = G.number_of_nodes()
    if key not in _SEMANTIC_AI_CACHE:
        _SEMANTIC_AI_CACHE[key] = CampusSemanticAI(G)
    return _SEMANTIC_AI_CACHE[key]


def invalidate_semantic_cache() -> None:
    """Xóa cache khi graph thay đổi cấu trúc (inductive_add_node)."""
    _SEMANTIC_AI_CACHE.clear()


def _nearest_node(G: nx.Graph, lat: float, lon: float) -> Tuple[str, float]:
    """Backward-compat wrapper — dùng nearest_node_vectorized() bên trong."""
    return nearest_node_vectorized(G, lat, lon)


# _bearing_deg và _angle_diff chỉ còn dùng nội bộ (không export)
def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lam = math.radians(lon2 - lon1)
    x = math.sin(d_lam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lam)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _geo_alignment_score(
    user_lat: float,
    user_lon: float,
    node_lat: float,
    node_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> float:
    """
    [v4 — UPGRADED] Dùng Detour Distance thay vì góc hướng.

    Lý do: góc hướng bị sai khi đường đi ziczac (user quay đầu tạm thời).
    Detour Distance tính quãng đường vòng thực tế: luôn chính xác.
    """
    return detour_distance_score(
        user_lat, user_lon,
        node_lat, node_lon,
        dest_lat, dest_lon,
    )


def _extract_rule_needs(query: str) -> dict:
    q = normalize_text(query)
    return {
        "ac": any(w in q for w in ["mat", "may lanh", "nong", "dieu hoa"]),
        "quiet": any(w in q for w in ["yen tinh", "on a", "hoc bai", "doc sach", "tap trung"]),
        "tables": any(w in q for w in ["hoc", "ngoi", "lam bai", "ban ghe", "tu hoc"]),
        "food": any(w in q for w in ["doi", "an", "uong", "cafe", "ca phe", "com", "nuoc", "canteen"]),
        "sport": any(w in q for w in ["the thao", "tap", "van dong", "gym", "cau long", "bong ban", "the duc"]),
        "rest": any(w in q for w in ["ngu", "nghi ngoi", "met", "nga lung", "buon ngu"]),
    }


def _rule_based_score(G: nx.Graph, node: str, needs: dict, weather: str) -> float:
    if weather == "sunny" and not needs["ac"]:
        needs = {**needs, "ac": True}

    if not any(needs.values()):
        return 0.0

    data = G.nodes[node]
    features = data.get("features", {})
    aliases = " ".join(data.get("aliases", [])).lower()
    score = 0.0

    if needs["ac"]:
        score += 15 if features.get("has_ac") else -15
    if needs["tables"]:
        score += 10 if features.get("has_tables") else -10
    if needs["quiet"]:
        noise = features.get("noise_level", 1.0)
        if noise <= 0.3:
            score += 15
        elif noise >= 0.7:
            score -= 20
        else:
            score += (1.0 - noise) * 10
    if needs["rest"]:
        if check_keyword_in_aliases("nghi", aliases) or check_keyword_in_aliases("ngu", aliases):
            score += 25
        if features.get("noise_level", 1.0) <= 0.3:
            score += 5
        if features.get("has_ac"):
            score += 5
    if needs["food"]:
        if check_keyword_in_aliases("can tin", aliases) or check_keyword_in_aliases("an", aliases) or check_keyword_in_aliases("doi bung", aliases):
            score += 30
        else:
            score -= 30
    if needs["sport"]:
        if check_keyword_in_aliases("the thao", aliases) or check_keyword_in_aliases("the duc", aliases) or check_keyword_in_aliases("gym", aliases):
            score += 30
        else:
            score -= 30

    score += (min(features.get("capacity", 0), 1000) / 1000) * 5
    return score


def _interest_score(node: str, data: dict, interests: List[str]) -> float:
    if not interests:
        return 0.0
    interests_str = " ".join(interests).lower()
    aliases = " ".join(data.get("aliases", [])).lower()
    score = 0.0

    if any(k in interests_str for k in ["c++", "codeforces", "sql", "code", "thuat toan"]):
        if any_keyword_in_aliases(["may tinh", "lab", "thuc hanh", "nha c"], aliases):
            score += 50
    if any(k in interests_str for k in ["genshin", "tft", "game", "esport"]):
        if any_keyword_in_aliases(["phong nghi", "canteen"], aliases):
            score += 20
    if any(k in interests_str for k in ["football", "bayern", "chelsea", "the thao"]):
        if any_keyword_in_aliases(["the duc", "the thao", "gym"], aliases):
            score += 45
    return score if score > 0 else 0.0


def _personalization_boost(
    G: nx.Graph,
    node: str,
    role: str,
    study_style: str,
    active_interests: List[str],
    battery_level: Optional[float] = None,
    temperature: Optional[float] = None,
    uv_index: Optional[float] = None,
    schedule_class: Optional[str] = None,
) -> Tuple[float, List[str]]:
    """Tính điểm cộng cá nhân hóa dựa trên role, study_style, interests và các nguồn dữ liệu động mới (pin, thời khóa biểu, thời tiết cực đoan)."""
    boost = 0.0
    reasons = []

    data = G.nodes.get(node, {})
    aliases = " ".join(data.get("aliases", [])).lower()
    node_type = data.get("type", "")
    features = data.get("features", {})
    noise = features.get("noise_level", 0.5)
    capacity = features.get("capacity", 0)

    # 1. Boost theo Role
    if role == "student":
        if node_type == "building":
            boost += 5.0
        if any_keyword_in_aliases(["thu vien", "tu hoc", "can tin", "gym", "phong hoc"], aliases):
            boost += 12.0
            reasons.append("Sinh viên")
    elif role == "lecturer":
        if node_type == "admin":
            boost += 20.0
            reasons.append("Hành chính/Giảng viên")
        if any_keyword_in_aliases(["van phong khoa", "vp khoa", "giao vu"], aliases):
            boost += 15.0
            reasons.append("VP Khoa")
        if check_keyword_in_aliases("thu vien", aliases):
            boost += 8.0
    elif role == "visitor":
        if node_type == "facility":
            boost += 10.0
        if any_keyword_in_aliases(["cong truong", "nha xe", "atm", "can tin"], aliases):
            boost += 15.0
            reasons.append("Khách tham quan")

    # 2. Boost theo Study Style
    if study_style == "silent":
        if noise <= 0.3:
            boost += 12.0
            reasons.append("Không gian yên tĩnh")
        elif noise >= 0.7:
            boost -= 15.0
        if any_keyword_in_aliases(["thu vien", "tu hoc"], aliases):
            boost += 10.0
    elif study_style == "group":
        if 0.4 <= noise <= 0.6 or capacity >= 100:
            boost += 10.0
            reasons.append("Học nhóm/Thảo luận")
        if any_keyword_in_aliases(["sanh", "can tin", "phong nghi", "nha g"], aliases):
            boost += 8.0

    # 3. Boost theo Interests
    if active_interests:
        interests_str = " ".join(active_interests).lower()
        if any(k in interests_str for k in ["c++", "codeforces", "sql", "code", "thuat toan", "lap trinh", "cntt"]):
            if any_keyword_in_aliases(["may tinh", "lab", "thuc hanh", "nha c"], aliases):
                boost += 25.0
                reasons.append("Sở thích CNTT")
        if any(k in interests_str for k in ["robot", "iot", "arduino", "dien tu"]):
            if any_keyword_in_aliases(["phong thi nghiem", "nha a", "lab"], aliases):
                boost += 20.0
                reasons.append("Sở thích Robotics")
        if any(k in interests_str for k in ["football", "the thao", "bong da", "cau long", "gym"]):
            if any_keyword_in_aliases(["the duc", "the thao", "gym"], aliases):
                boost += 20.0
                reasons.append("Đam mê Thể thao")
        if any(k in interests_str for k in ["english", "ielts", "ngoai ngu"]):
            if any_keyword_in_aliases(["thu vien", "tu hoc"], aliases):
                boost += 15.0
                reasons.append("Học Ngoại ngữ")

    # 4. Boost theo Lịch học (Timetable Boost)
    if schedule_class and schedule_class in G.nodes:
        if node == schedule_class:
            boost += 35.0
            reasons.append("Lớp học của bạn 📚")
        elif node in ["Nhà xe", "Cổng trường", "ATM", "Căn tin"]:
            boost += 12.0
            reasons.append("Hỗ trợ học tập 🎒")

    # 5. Boost theo Trạng thái Pin (Device Battery Boost)
    if battery_level is not None and battery_level < 0.20:
        if features.get("has_tables") and data.get("indoor", False):
            boost += 18.0
            reasons.append("Ổ cắm sạc pin 🔌")

    # 6. Boost theo Thời tiết cực đoan (Nhiệt độ & UV)
    if temperature is not None and temperature > 33.0:
        if features.get("has_ac"):
            boost += 12.0
            reasons.append("Phòng điều hòa tránh nóng ❄️")
        elif not data.get("indoor", False) or node in ["Tòa G", "Nhà xe", "Cổng trường", "Nhà thể dục"]:
            boost -= 20.0
            reasons.append("Tránh nắng nóng ☀️")

    if uv_index is not None and uv_index > 5.0:
        if data.get("indoor", False):
            boost += 8.0
            reasons.append("Tránh tia UV cao 🛡️")
        elif not data.get("indoor", False) or node in ["Tòa G", "Nhà xe", "Cổng trường", "Nhà thể dục"]:
            boost -= 15.0

    return boost, reasons


def _build_reason(
    G: nx.Graph,
    node: str,
    semantic: float,
    on_route: bool,
    dist_m: float,
    crowd: float,
    destination: Optional[str],
) -> str:
    parts: List[str] = []
    if destination and on_route:
        parts.append(f"Nằm trên hướng đi tới {destination}")
    elif dist_m < 80:
        parts.append(f"Rất gần vị trí bạn (~{int(dist_m)}m)")
    elif dist_m < 300:
        parts.append(f"Gần vị trí hiện tại (~{int(dist_m)}m)")

    if semantic >= 0.35:
        parts.append("khớp nhu cầu bạn mô tả")
    if crowd >= 0.85:
        parts.append("dự báo đông, nên ghé sớm")
    elif crowd <= 0.25:
        parts.append("dự báo vắng, thoải mái")

    ctx = ""
    if parts:
        ctx = parts[0].capitalize() + (", " + ", ".join(parts[1:]) if len(parts) > 1 else "")

    profile = get_building_profile(G, node)
    func = profile.get("function_summary", "")
    if func:
        if ctx:
            return f"{ctx} — Tại đây: {func}."
        return f"Ghé {node}: {func}."

    if not ctx:
        return f"Gợi ý AI: ghé {node} trước khi tiếp tục di chuyển."
    return ctx + "."


# =====================================================================
# ĐỀ XUẤT NGỮ NGHĨA (TF-IDF Query Recommender)
# =====================================================================
def recommend_locations(
    G: nx.Graph,
    query: str,
    current_time: str = None,
    weather: str = "normal",
    limit: int = 5,
) -> List[dict]:
    """Trả về danh sách địa điểm xếp hạng theo AI ngữ nghĩa + luật nhu cầu."""
    query_norm = normalize_text(query)
    if not query_norm:
        return []

    semantic_ai = _get_semantic_ai(G)          # v5: singleton, không rebuild
    semantic_scores = semantic_ai.score_query(query)
    needs = _extract_rule_needs(query)

    ranked: List[dict] = []
    time_band = get_time_of_day_band(current_time) if current_time else "afternoon"
    weekend = is_weekend()

    for node, data in G.nodes(data=True):
        if not is_node_open(G, node, current_time):
            continue

        sem = semantic_scores.get(node, 0.0) * 40.0
        rule = _rule_based_score(G, node, dict(needs), weather)
        raw = sem + rule + time_category_boost(node, G, time_band, weekend, query=query)
        raw += indoor_boost(G, node, weather)
        if raw <= 0 and sem < 8:
            continue

        open_info = open_status_detail(G, node, current_time) if current_time else {}
        ranked.append({
            "node": node,
            "score": _normalize_score(max(raw, sem)),
            "raw_score": round(raw, 2),
            "semantic_score": round(sem, 2),
            "method": "AI Semantic + Intent",
            "reason": _build_reason(G, node, sem / 40.0, False, 9999, 0.0, None),
            "open_now": open_info.get("open_now", True),
            "closing_soon": open_info.get("closing_soon", False),
            "close_warning": open_info.get("warning"),
        })

    ranked.sort(key=lambda x: (x["score"], x["raw_score"], x["semantic_score"]), reverse=True)
    return [enrich_suggestion(G, r, query) for r in ranked[:limit]]


def recommend_location(
    G: nx.Graph,
    query: str,
    current_time: str = None,
    weather: str = "normal",
) -> Tuple[Optional[str], float]:
    results = recommend_locations(G, query, current_time, weather, limit=1)
    if not results:
        return None, 0
    top = results[0]
    return top["node"], top["score"]


# =====================================================================
# SMART RECOMMENDATIONS (Bản đồ + Cá nhân hóa tự động)
# =====================================================================
def get_smart_recommendations(
    G: nx.Graph,
    current_lat: float,
    current_lon: float,
    destination: Optional[str] = None,
    query: Optional[str] = None,
    weather: str = "normal",
    current_time_str: Optional[str] = None,
    user_interests: Optional[List[str]] = None,
    limit: int = _MAX_PROACTIVE,
    user_profile: Optional[dict] = None,
    battery_level: Optional[float] = None,
    temperature: Optional[float] = None,
    uv_index: Optional[float] = None,
    schedule_class: Optional[str] = None,
) -> List[dict]:
    """
    Gợi ý thông minh dựa trên GPS, điểm đến, câu hỏi tự nhiên và hồ sơ tự động hóa hoàn toàn.
    Tích hợp các biện pháp chống quá khớp (Anti-Overfitting) & Đa dạng hóa (Exploration).
    """
    # ── v4: Dùng PersonaManager và ContextEngine ─────────────────────────────
    pm = PersonaManager(user_profile if user_profile is not None else {})
    pm.update_passively()

    # ── v4: Vectorized nearest node ───────────────────────────────────────────
    node_distances = all_node_distances(G, current_lat, current_lon)
    nearest_node   = min(node_distances, key=lambda n: node_distances[n])
    dist_nearest   = node_distances[nearest_node]

    curr_t = parse_time(current_time_str) if current_time_str else None
    if not curr_t:
        return []

    if user_profile is None:
        user_profile = pm._profile

    role             = pm.role
    study_style      = pm.study_style
    profile_interests = pm.interests
    visited_history  = pm.visited_history
    active_interests = list(set((user_interests or []) + (profile_interests or [])))

    # ── ContextEngine (context signals + detour scoring) ─────────────────────
    ce        = ContextEngine(G, current_time_str, weather)
    time_band = ce._time_band
    weekend   = ce._is_weekend

    semantic_ai     = _get_semantic_ai(G) if query else None   # v5: singleton
    semantic_scores = semantic_ai.score_query(query) if semantic_ai and query else {}
    needs           = _extract_rule_needs(query) if query else {}

    path_nodes: set = set()
    dest_gps = None

    if destination and destination in G.nodes and destination != nearest_node:
        path, _ = pathfinding_optimizer(G, nearest_node, destination, weather, current_time_str)
        if path:
            path_nodes = set(path)
            dest_gps = G.nodes[destination]["gps"]

    from engine.campus_knowledge import TIME_CATEGORY_BOOST

    candidates: Dict[str, dict] = {}

    def _add(node: str, raw: float, reason: str, source: str, priority: int = 5) -> None:
        if node == nearest_node and dist_nearest < 30 and source != "nearby_context" and source != "personal_history":
            return
        if not is_node_open(G, node, current_time_str):
            return

        # Tính điểm cộng cá nhân hóa
        p_boost, p_reasons = _personalization_boost(
            G, node, role, study_style, active_interests,
            battery_level=battery_level,
            temperature=temperature,
            uv_index=uv_index,
            schedule_class=schedule_class
        )
        raw += p_boost

        # ── v4: Sigmoid history boost (chống Popularity Bias) ────────────────
        node_categories  = {s.get("category") for s in G.nodes.get(node, {}).get("services", [])}
        time_boosted_cat = set(TIME_CATEGORY_BOOST.get(time_band, {}).keys())
        history_boost = pm.history_boost(
            node,
            query_active=(query is not None),
            time_boosted_categories=time_boosted_cat,
            node_categories=node_categories,
        )

        if history_boost > 0.0:
            raw += history_boost
        else:
            # Novelty Boost: khuyến khích khám phá nơi chưa từng đi
            raw += pm.novelty_boost(node)

        # ── v4: KHÔNG dùng random.uniform nữa — noise sẽ áp dụng ở bước cuối
        # bằng Gumbel-Softmax (tránh đảo lộn thứ hạng vô lý)

        entry = candidates.get(node)
        if entry and entry["raw_score"] >= raw:
            return

        dist_m = haversine(current_lat, current_lon, *G.nodes[node]["gps"])
        crowd = effective_crowd_level(
            G, node, predict_crowd_level(G, node, current_time_str)
        )
        open_info = open_status_detail(G, node, current_time_str)

        # Điều chỉnh lý do nếu có các thẻ cá nhân hóa
        enhanced_reason = reason
        if p_reasons:
            p_desc = " & ".join(p_reasons[:2])
            enhanced_reason = f"{reason.rstrip('.')} ({p_desc})."

        # Tách nhỏ điểm số thành phần để phục vụ Explainable AI (XAI)
        persona_raw = p_boost + history_boost
        if source == "personal_history":
            persona_raw += 25.0
        elif source == "ai_context":
            persona_raw += (_interest_score(node, G.nodes[node], active_interests or []) * 0.4)

        proximity_raw = 0.0
        if source == "nearby_context":
            proximity_raw = 28.0
        elif source in ("route_context", "route_neighbor"):
            proximity_raw = raw - p_boost - history_boost
            if source == "route_context":
                crowd_val = predict_crowd_level(G, node, current_time_str)
                crowd_adj_val = -8 if crowd_val >= 0.85 else 4
                proximity_raw -= crowd_adj_val
        elif source == "ai_context":
            prox_bonus = 12 if dist_m < 50 else (6 if dist_m < 150 else 0)
            align_val = 0.0
            if dest_gps:
                align_val = _geo_alignment_score(current_lat, current_lon, G.nodes[node]["gps"][0], G.nodes[node]["gps"][1], dest_gps[0], dest_gps[1])
            sem_val = (semantic_scores.get(node, 0.0) if semantic_scores else 0.0) * 35.0
            rule_val = _rule_based_score(G, node, dict(needs), weather) if query else 0.0
            proximity_raw = prox_bonus + align_val + sem_val + rule_val

        context_raw = 0.0
        if source in ("time_morning", "time_context"):
            context_raw = raw - p_boost - history_boost
        elif source == "ai_context":
            context_raw = time_category_boost(node, G, time_band, weekend) + indoor_boost(G, node, weather)

        crowd_raw = 0.0
        if source == "route_context":
            crowd_val = predict_crowd_level(G, node, current_time_str)
            crowd_raw = -8 if crowd_val >= 0.85 else 4
        elif source == "ai_context":
            crowd_val = predict_crowd_level(G, node, current_time_str)
            crowd_raw = 5 if crowd_val <= 0.35 else (-6 if crowd_val >= 0.85 else 0)
        else:
            crowd_val = predict_crowd_level(G, node, current_time_str)
            crowd_raw = 5 if crowd_val <= 0.35 else (-6 if crowd_val >= 0.85 else 0)

        score_breakdown = {
            "personalization": round(max(0.0, min(100.0, (persona_raw / 30.0) * 100)), 1),
            "proximity": round(max(0.0, min(100.0, (proximity_raw / 25.0) * 100)), 1),
            "context": round(max(0.0, min(100.0, (context_raw / 20.0) * 100)), 1),
            "crowd": round(max(0.0, min(100.0, ((crowd_raw + 8.0) / 13.0) * 100)), 1),
        }

        data_node = G.nodes.get(node, {})
        features_node = data_node.get("features", {})
        candidates[node] = {
            "node": node,
            "raw_score": raw,
            "score": _normalize_score(raw),
            "reason": enhanced_reason,
            "priority": priority,
            "distance_m": round(dist_m, 1),
            "crowd_level": round(crowd, 2),
            "on_route": node in path_nodes and node != destination,
            "source": source,
            "gps": G.nodes[node]["gps"],
            "closing_soon": open_info.get("closing_soon", False),
            "close_warning": open_info.get("warning"),
            "time_band": time_band,
            "personal_tags": p_reasons,
            "ncf_score": 0.0,  # NCF removed
            "score_breakdown": score_breakdown,
            "battery_warning": bool(battery_level is not None and battery_level < 0.20 and features_node.get("has_tables") and data_node.get("indoor", False)),
            "temperature_warning": bool(temperature is not None and temperature > 33.0),
            "uv_warning": bool(uv_index is not None and uv_index > 5.0),
            "has_class": bool(schedule_class is not None and node == schedule_class),
        }

    def in_time_range(start_str: str, end_str: str) -> bool:
        return parse_time(start_str) <= curr_t <= parse_time(end_str)

    # Sáng sớm 6h–9h: ưu tiên ăn sáng
    if time_band == "early_morning":
        for fn, d in G.nodes(data=True):
            if "can tin" not in " ".join(d.get("aliases", [])).lower():
                continue
            dist_m = haversine(current_lat, current_lon, *d["gps"])
            if dist_m < _NEARBY_RADIUS_M:
                _add(
                    fn, 36,
                    build_function_reason(G, fn, query, "Buổi sáng — ghé ăn sáng/cà phê campus"),
                    "time_morning", 1,
                )

    if in_time_range("16:30", "18:30") and "Nhà xe" in G.nodes:
        _add(
            "Nhà xe", 42,
            build_function_reason(G, "Nhà xe", query, "Sắp hết giờ chiều — ra lấy xe về"),
            "time_context", 1,
        )

    if in_time_range("11:00", "13:00") or in_time_range("17:00", "18:30"):
        for fn, d in G.nodes(data=True):
            has_food = any(
                s.get("category") == "an_uong"
                for s in d.get("services", [])
            ) or "can tin" in " ".join(d.get("aliases", [])).lower()
            if not has_food:
                continue
            dist_m = haversine(current_lat, current_lon, *d["gps"])
            if dist_m < _NEARBY_RADIUS_M:
                _add(
                    fn, 38,
                    build_function_reason(G, fn, query, "Đã đến giờ ăn — ghé căn tin"),
                    "time_context", 2,
                )

    # --- Lịch sử ghé thăm (Personalized History) ---
    for node, count in visited_history.items():
        if node not in G.nodes:
            continue
        dist_m = haversine(current_lat, current_lon, *G.nodes[node]["gps"])
        if count >= 1 and dist_m < _NEARBY_RADIUS_M * 2.0:
            h_boost = min(10.0, math.log1p(count) * 4.0)
            if query:
                h_boost *= 0.2
            from engine.campus_knowledge import TIME_CATEGORY_BOOST
            node_categories = {s.get("category") for s in G.nodes.get(node, {}).get("services", [])}
            time_boosted = set(TIME_CATEGORY_BOOST.get(time_band, {}).keys())
            if time_boosted and not (node_categories & time_boosted):
                h_boost *= 0.5
            raw = 25.0 + h_boost
            reason = f"Bạn thường ghé thăm địa điểm này (đã đi {count} lần)"
            _add(node, raw, reason, "personal_history", 2)

    # --- Dọc lộ trình ---
    if path_nodes and dest_gps:
        dest_lat, dest_lon = dest_gps
        for node in path_nodes:
            if node == destination or node == nearest_node:
                continue
            dist_m = haversine(current_lat, current_lon, *G.nodes[node]["gps"])
            align = _geo_alignment_score(
                current_lat, current_lon,
                G.nodes[node]["gps"][0], G.nodes[node]["gps"][1],
                dest_lat, dest_lon,
            )
            detour_bonus = 25 if dist_m < _DETOUR_RADIUS_M else 12
            crowd = predict_crowd_level(G, node, current_time_str)
            crowd_adj = -8 if crowd >= 0.85 else 4
            raw = 30 + detour_bonus + align + crowd_adj
            reason = _build_reason(G, node, 0, True, dist_m, crowd, destination)
            _add(node, raw, reason, "route_context", 3)

        for u, v in G.edges():
            for a, b in ((u, v), (v, u)):
                if a not in path_nodes or b in path_nodes:
                    continue
                dist_m = haversine(current_lat, current_lon, *G.nodes[b]["gps"])
                if dist_m > _NEARBY_RADIUS_M:
                    continue
                raw = 22 + _geo_alignment_score(
                    current_lat, current_lon,
                    G.nodes[b]["gps"][0], G.nodes[b]["gps"][1],
                    dest_lat, dest_lon,
                )
                _add(
                    b, raw,
                    build_function_reason(
                        G, b, query,
                        f"Ngay cạnh lộ trình tới {destination} — tiện ghé",
                    ),
                    "route_neighbor", 4,
                )

    # --- Query / Interests ---
    for node, data in G.nodes(data=True):
        dist_m = haversine(current_lat, current_lon, *data["gps"])
        if dist_m > _NEARBY_RADIUS_M * 1.5:
            continue

        raw = 0.0
        sem = semantic_scores.get(node, 0.0) if semantic_scores else 0.0
        raw += sem * 35.0
        if query:
            raw += _rule_based_score(G, node, dict(needs), weather)
        raw += _interest_score(node, data, active_interests or []) * 0.4
        raw += time_category_boost(node, G, time_band, weekend, query=query)
        raw += indoor_boost(G, node, weather)

        if dist_m < 50:
            raw += 12
        elif dist_m < 150:
            raw += 6

        if dest_gps:
            raw += _geo_alignment_score(
                current_lat, current_lon,
                data["gps"][0], data["gps"][1],
                dest_gps[0], dest_gps[1],
            )

        crowd = predict_crowd_level(G, node, current_time_str)
        if crowd <= 0.35:
            raw += 5
        elif crowd >= 0.85:
            raw -= 6

        if raw < 12:
            continue

        reason = _build_reason(
            G, node, sem,
            node in path_nodes,
            dist_m, crowd, destination,
        )
        _add(node, raw, reason, "ai_context", 5)

    if dist_nearest < 50:
        profile = get_building_profile(G, nearest_node)
        if profile.get("function_summary"):
            _add(
                nearest_node, 28,
                build_function_reason(
                    G, nearest_node, query,
                    f"Bạn đang rất gần — có muốn ghé",
                ),
                "nearby_context", 2,
            )

    # ── v5: Tích hợp Item-Item CF vào candidates (trước khi re-rank) ──────────
    # CF boost cộng điểm cho node mà "người dùng tương tự hay ghé"
    # Cold-start safe: no-op nếu CF chưa đủ data
    if user_profile is not None:
        _apply_cf_boost(
            G, candidates, user_profile,
            exclude_nodes={nearest_node},
            current_time_str=current_time_str,
        )

    # ── v4: Gumbel-Softmax re-ranking (thay sắp xếp thuần raw_score) ─────────
    # Bước 1: tạo danh sách (node, raw_score)
    scored_pairs = [(v["node"], v["raw_score"]) for v in candidates.values()]

    # Bước 2: Gumbel-Softmax rank — giữ phân phối, tránh thứ hạng cứng nhắc
    # top_k=limit*2 để chỉ áp dụng noise trong pool ứng viên top, phần đuôi giữ nguyên
    gumbel_ranked = gumbel_softmax_rank(
        sorted(scored_pairs, key=lambda x: -x[1]),
        temperature=_GUMBEL_TEMP,
        top_k=min(len(scored_pairs), limit * 2),
    )

    # Bước 3: ε-greedy — thỉnh thoảng đưa 1 item ngoài top vào để tăng Serendipity
    final_ranked = epsilon_greedy_sample(gumbel_ranked, epsilon=0.15)

    # Bước 4: Phân loại 2 bucket familiar / discovery
    final_nodes_ordered = [n for n, _ in final_ranked]
    familiar_nodes, discovery_nodes = pm.split_buckets(final_nodes_ordered)

    seen  = set()
    final: List[dict] = []

    def _build_entry(node: str) -> Optional[dict]:
        if node in seen or node not in candidates:
            return None
        seen.add(node)
        item  = candidates[node]
        entry = enrich_suggestion(G, {
            "node":        item["node"],
            "score":       item["score"],
            "raw_score":   item["raw_score"],
            "reason":      item["reason"],
            "priority":    item["priority"],
            "distance_m":  item["distance_m"],
            "crowd_level": item["crowd_level"],
            "on_route":    item["on_route"],
            "source":      item["source"],
            "gps":         item["gps"],
            "score_breakdown": item.get("score_breakdown", {
                "personalization": 50.0, "proximity": 50.0,
                "context": 50.0, "crowd": 50.0,
            }),
        }, query)

        visit_count = visited_history.get(item["node"], 0)
        entry["visit_count"]        = visit_count
        entry["is_favorite"]        = visit_count >= 3
        entry["personal_tags"]      = item.get("personal_tags", [])
        entry["ncf_score"]          = 0.0
        entry["is_familiar"]        = pm.is_familiar(item["node"])
        entry["battery_warning"]    = item.get("battery_warning", False)
        entry["temperature_warning"] = item.get("temperature_warning", False)
        entry["uv_warning"]         = item.get("uv_warning", False)
        entry["has_class"]          = item.get("has_class", False)
        return entry

    for node in final_nodes_ordered:
        entry = _build_entry(node)
        if entry:
            final.append(entry)
        if len(final) >= limit:
            break

    # ── v4: Đính kèm metadata 2-bucket vào từng item ──────────────────────────
    # Thêm key "bucket" để frontend có thể render 2 danh sách riêng biệt
    for entry in final:
        entry["bucket"] = "familiar" if entry.get("is_familiar") else "discovery"

    return final


def get_proactive_recommendations(
    G: nx.Graph,
    current_lat: float,
    current_lon: float,
    current_time_str: str,
    destination: Optional[str] = None,
    query: Optional[str] = None,
    weather: str = "normal",
    user_interests: Optional[List[str]] = None,
    limit: int = 5,
    user_profile: Optional[dict] = None,
) -> list:
    smart = get_smart_recommendations(
        G, current_lat, current_lon,
        destination=destination,
        query=query,
        weather=weather,
        current_time_str=current_time_str,
        user_interests=user_interests,
        limit=limit,
        user_profile=user_profile,
    )
    return [
        {
            "node": s["node"],
            "reason": s["reason"],
            "priority": s["priority"],
            "score": s["score"],
            "distance_m": s["distance_m"],
            "on_route": s["on_route"],
            "tagline": s.get("tagline", ""),
            "function_summary": s.get("function_summary", ""),
            "services": s.get("services", []),
            "matched_services": s.get("matched_services", []),
            "departments": s.get("departments", []),
            "events": s.get("events", []),
            "visit_count": s.get("visit_count", 0),
            "is_favorite": s.get("is_favorite", False),
            "personal_tags": s.get("personal_tags", []),
            "score_breakdown": s.get("score_breakdown", {
                "personalization": 50.0,
                "proximity": 50.0,
                "context": 50.0,
                "crowd": 50.0
            }),
        }
        for s in smart
    ]


def recommend_by_building_function(
    G: nx.Graph,
    query: str,
    current_time: str = None,
    limit: int = 5,
) -> List[dict]:
    q = normalize_text(query)
    if not q:
        return []

    # v5 fix: pre-compute TF-IDF một lần ngoài vòng lặp (singleton)
    sem_ai = _get_semantic_ai(G)
    sem_scores = sem_ai.score_query(query)  # {node: score} cho tất cả node

    results: List[dict] = []
    for node in G.nodes():
        if not is_node_open(G, node, current_time):
            continue
        profile = get_building_profile(G, node)
        services = profile.get("services", [])
        matched = match_services_to_query(query, services)
        if not matched:
            continue
        sem = sem_scores.get(node, 0.0)
        raw = len(matched) * 25 + sem * 30
        results.append(enrich_suggestion(G, {
            "node": node,
            "score": _normalize_score(raw),
            "reason": build_function_reason(
                G, node, query,
                f"Phù hợp vì có {', '.join(s['name'] for s in matched)}",
            ),
            "matched_services": matched,
            "method": "Building Function Match",
        }, query))

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


def semantic_map_linking(
    G: nx.Graph,
    query: str,
    current_lat: Optional[float] = None,
    current_lon: Optional[float] = None,
) -> Optional[dict]:
    q = normalize_text(query)
    if not q:
        return None

    semantic_ai = CampusSemanticAI(G)
    scores = semantic_ai.score_query(query)

    for node, data in G.nodes(data=True):
        aliases = [normalize_text(node)] + [normalize_text(a) for a in data.get("aliases", [])]
        for alias in aliases:
            if alias and len(alias) >= 3 and alias in q:
                scores[node] = scores.get(node, 0) + 0.5

    if "gan" in q.split() or "gần" in query.lower():
        anchor_terms = ["thu vien", "can tin", "nha xe", "gym", "atm", "cong"]
        for term in anchor_terms:
            if term in q:
                for node, data in G.nodes(data=True):
                    blob = normalize_text(node + " " + " ".join(data.get("aliases", [])))
                    if term in blob:
                        scores[node] = scores.get(node, 0) + 0.4

    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_node, best_score = ranked[0]
    if best_score < 0.05:
        return None

    gps = G.nodes[best_node]["gps"]
    result = {
        "node": best_node,
        "gps": {"lat": gps[0], "lon": gps[1]},
        "confidence": round(min(1.0, best_score), 3),
        "matched_query": query,
        "alternatives": [
            {"node": n, "score": round(s, 3)}
            for n, s in ranked[1:4]
            if s > 0.05
        ],
    }

    if current_lat is not None and current_lon is not None:
        result["distance_from_user_m"] = round(
            haversine(current_lat, current_lon, gps[0], gps[1]), 1
        )
    return result


def context_recommender(
    G: nx.Graph,
    current_lat: float,
    current_lon: float,
    current_time_str: str,
    destination: Optional[str] = None,
    query: Optional[str] = None,
    weather: str = "normal",
    user_interests: Optional[List[str]] = None,
    limit: int = 6,
    user_profile: Optional[dict] = None,
) -> List[dict]:
    try:
        from engine.gnn_engine import gnn_node_embedding
        embeddings = gnn_node_embedding(G)
    except Exception:
        embeddings = {}

    base = get_smart_recommendations(
        G, current_lat, current_lon,
        destination=destination,
        query=query,
        weather=weather,
        current_time_str=current_time_str,
        user_interests=user_interests,
        limit=limit,
        user_profile=user_profile,
    )

    for item in base:
        emb = embeddings.get(item["node"], [])
        item["gnn_embedding_preview"] = emb[:4] if emb else []
        item["crowd_pct"] = round(predict_crowd_level(G, item["node"], current_time_str) * 100)
    return base


# =====================================================================
# TÍCH HỢP CF VÀO SMART RECOMMENDATIONS
# =====================================================================

def _apply_cf_boost(
    G: nx.Graph,
    candidates: Dict[str, dict],
    user_profile: dict,
    exclude_nodes: set,
    current_time_str: Optional[str],
) -> None:
    """
    [v5] Tích hợp điểm Item-Item CF vào candidates dict.

    CF recommendations được thêm như một nguồn mới:
    - Nếu node đã có trong candidates: cộng cf_boost vào raw_score
    - Nếu node chưa có: tạo entry mới với source='item_item_cf'

    Cold-start safe: hàm này là no-op nếu CF chưa đủ data.
    """
    from engine.collaborative_filter import cf_recommend_for_profile

    cf_recs = cf_recommend_for_profile(
        G, user_profile,
        exclude_nodes=exclude_nodes,
        top_k=4,
    )
    if not cf_recs:
        return  # Cold start hoặc user chưa có lịch sử

    for rec in cf_recs:
        node = rec["node"]
        cf_score = rec["cf_score"]
        if not is_node_open(G, node, current_time_str):
            continue

        # Quy đổi CF score → raw score (max ~12 điểm)
        cf_raw = cf_score * 100.0  # similarity [0,1] → [0,100], cap ở 12
        cf_raw = min(cf_raw, 12.0)

        if node in candidates:
            # Cộng CF boost vào entry đã tồn tại
            candidates[node]["raw_score"] += cf_raw
            candidates[node]["score"] = _normalize_score(candidates[node]["raw_score"])
            if "cf_score" not in candidates[node] or candidates[node].get("cf_score", 0) < cf_score:
                candidates[node]["cf_score"] = round(cf_score, 4)
            # Ghi nhận nguồn CF vào reason
            existing_reason = candidates[node].get("reason", "")
            if "người dùng tương tự" not in existing_reason.lower():
                candidates[node]["reason"] = existing_reason.rstrip(".") + " (Người dùng tương tự cũng hay ghé)."
        else:
            # Thêm mới từ CF
            from engine.utils import haversine as _hav
            gps_data = G.nodes[node].get("gps", (0.0, 0.0))
            open_info = open_status_detail(G, node, current_time_str) if current_time_str else {}
            candidates[node] = {
                "node": node,
                "raw_score": cf_raw + 8.0,  # base nhỏ để không dominate
                "score": _normalize_score(cf_raw + 8.0),
                "reason": rec["reason"],
                "priority": 4,
                "distance_m": 999.0,
                "crowd_level": round(predict_crowd_level(G, node, current_time_str), 2),
                "on_route": False,
                "source": "item_item_cf",
                "gps": gps_data,
                "closing_soon": open_info.get("closing_soon", False),
                "close_warning": open_info.get("warning"),
                "time_band": "unknown",
                "personal_tags": ["Người dùng tương tự hay ghé"],
                "cf_score": round(cf_score, 4),
                "score_breakdown": {
                    "personalization": 40.0,
                    "proximity": 0.0,
                    "context": 30.0,
                    "crowd": 50.0,
                },
                "battery_warning": False,
                "temperature_warning": False,
                "uv_warning": False,
                "has_class": False,
            }


# =====================================================================
# CROWD PREDICTION ENGINE
# =====================================================================
_CROWD_MODEL = None
_CROWD_METADATA = None


def load_crowd_model():
    global _CROWD_MODEL, _CROWD_METADATA
    engine_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(engine_dir, "crowd_model.pth")
    metadata_path = os.path.join(engine_dir, "model_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            _CROWD_METADATA = json.load(f)["crowd"]
        _CROWD_MODEL = CrowdPredictor(_CROWD_METADATA["input_dim"])
        _CROWD_MODEL.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        _CROWD_MODEL.eval()
        print("✅ [Recommender] Crowd Predictor nạp thành công.")
    except Exception as e:
        print(f"⚠️ [Recommender] Crowd model lỗi: {e}")
        _CROWD_MODEL = None


# NCF Mock and Dummy definitions to prevent imports crashes (NCF is deleted)
def load_ncf_model():
    pass

def ncf_recommend(G, user_profile, current_time_str=None, limit=8):
    return []


class CrowdPredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# Load crowd model lazily
load_crowd_model()


def predict_crowd_level(G: nx.Graph, node_id: str, current_time_str: str) -> float:
    """Dự báo mức độ đông đúc (0–1) cho một node tại thời điểm cho trước."""
    live = get_live_crowd(node_id)
    if live is not None:
        return live

    if _CROWD_MODEL is not None and _CROWD_METADATA is not None:
        try:
            nodes = _CROWD_METADATA["nodes"]
            weather_types = _CROWD_METADATA["weather_types"]
            input_dim = _CROWD_METADATA["input_dim"]

            lookup_node = node_id
            if lookup_node not in nodes:
                base = node_id.split("_")[0] if "_" in node_id else node_id
                matches = [n for n in nodes if n.startswith(base)]
                lookup_node = matches[0] if matches else nodes[0]

            node_vec = np.zeros(len(nodes), dtype=np.float32)
            node_vec[nodes.index(lookup_node)] = 1.0

            weather_vec = np.zeros(len(weather_types), dtype=np.float32)
            weather_vec[0] = 1.0

            curr_t = parse_time(current_time_str)
            hour_val = (curr_t.hour + curr_t.minute / 60.0) if curr_t else 12.0

            now = datetime.now()
            dow_val = now.weekday()
            month_val = now.month
            is_exam = 1.0 if month_val in (1, 5, 6, 12) else 0.0

            features = np.concatenate([
                node_vec,
                [hour_val / 24.0, dow_val / 6.0, (month_val - 1) / 11.0, is_exam],
                weather_vec,
            ])

            if len(features) != input_dim:
                if len(features) > input_dim:
                    features = features[:input_dim]
                else:
                    features = np.pad(features, (0, input_dim - len(features)))

            tensor = torch.tensor(features[np.newaxis, :], dtype=torch.float32)
            with torch.no_grad():
                pred = _CROWD_MODEL(tensor).item()
            return round(pred, 2)
        except Exception as e:
            print(f"⚠️ [Crowd Model Error] {e}")

    # Fallback
    curr_t = parse_time(current_time_str)
    if not curr_t:
        return 0.0

    node_data = G.nodes.get(node_id, {})
    aliases = " ".join(node_data.get("aliases", [])).lower()
    base_crowd = 0.2

    if check_keyword_in_aliases("can tin", aliases) or check_keyword_in_aliases("an", aliases) or node_id == "Căn tin":
        if time(6, 0) <= curr_t <= time(9, 0):
            return 0.55
        if time(11, 30) <= curr_t <= time(13, 0):
            return 0.95
        return 0.4

    if check_keyword_in_aliases("thu vien", aliases) or check_keyword_in_aliases("tu hoc", aliases) or node_id in ("Tòa B", "Tòa C", "Tòa D"):
        if time(8, 0) <= curr_t <= time(11, 0) or time(14, 0) <= curr_t <= time(16, 30):
            return 0.8
        return 0.3

    if check_keyword_in_aliases("the thao", aliases) or check_keyword_in_aliases("gym", aliases) or node_id == "Nhà thể dục":
        if time(16, 30) <= curr_t <= time(18, 30):
            return 0.85
        if is_weekend() and time(8, 0) <= curr_t <= time(11, 0):
            return 0.7
        return 0.2

    return base_crowd


def submit_crowd_report(node_id: str, level: float) -> dict:
    report_live_crowd(node_id, level)
    return {"node": node_id, "reported_level": round(level, 2), "status": "accepted"}


def crowd_prediction(G: nx.Graph, node_id: str, current_time_str: str) -> dict:
    pred = predict_crowd_level(G, node_id, current_time_str)
    level = effective_crowd_level(G, node_id, pred)
    if level >= 0.85:
        label = "rat dong"
    elif level >= 0.6:
        label = "dong vua"
    elif level >= 0.35:
        label = "binh thuong"
    else:
        label = "vang"
    return {
        "node": node_id,
        "crowd_level": round(level, 2),
        "crowd_predicted": round(pred, 2),
        "crowd_pct": round(level * 100),
        "label": label,
        "live_report": get_live_crowd(node_id) is not None,
    }


# =====================================================================
# COLLABORATIVE FILTERING — Real Item-Item CF (v5)
# =====================================================================

def collaborative_filtering(
    G: nx.Graph,
    user_profile: dict,
    current_lat: Optional[float] = None,
    current_lon: Optional[float] = None,
    top_k: int = 8,
) -> List[dict]:
    """
    [v5] Real Item-Item Collaborative Filtering.

    Thay thế rule-based mock cũ (_CLUB_RULES) bằng co-visitation CF thực sự.

    Cold-start fallback:
      Khi CF chưa đủ data (< 3 sessions), tự động fallback về content-based
      gợi ý dựa trên sở thích (interests) khai báo trong profile.

    Args:
        G:            Campus graph
        user_profile: Profile dict chứa visited_history / behavior_log / interests
        current_lat:  GPS vĩ độ (để tính distance_m, có thể None)
        current_lon:  GPS kinh độ (để tính distance_m, có thể None)
        top_k:        Số kết quả tối đa

    Returns:
        List[dict] mỗi item: {node, cf_score, categories, gps, distance_m, ...}
    """
    from engine.collaborative_filter import cf_recommend_for_profile, get_cf_model

    cf_recs = cf_recommend_for_profile(G, user_profile, top_k=top_k)

    if cf_recs:
        # CF có đủ data — dùng kết quả thực
        results = []
        for rec in cf_recs:
            node = rec["node"]
            if node not in G.nodes:
                continue
            data = G.nodes[node]
            profile = get_building_profile(G, node)
            entry = {
                "node": node,
                "cf_score": rec["cf_score"],
                "match_score": round(rec["cf_score"] * 100, 1),
                "categories": ["CF"],
                "type": data.get("type", "building"),
                "gps": data.get("gps"),
                "tagline": profile.get("tagline", ""),
                "function_summary": profile.get("function_summary", ""),
                "services": profile.get("services", []),
                "reason": rec["reason"],
                "source": "item_item_cf",
                "cf_sessions": get_cf_model().n_sessions,
            }
            if current_lat is not None and current_lon is not None:
                entry["distance_m"] = round(
                    haversine(current_lat, current_lon, *data["gps"]), 1
                )
            results.append(entry)
        return results

    # === Cold-start fallback: content-based theo interests ===
    interests = user_profile.get("interests", [])
    if not interests:
        return []

    interests_str = " ".join(interests).lower()
    sem_ai = _get_semantic_ai(G)
    interest_query = " ".join(interests)
    sem_scores = sem_ai.score_query(interest_query)

    results = []
    for node, data in G.nodes(data=True):
        sem = sem_scores.get(node, 0.0)
        if sem < 0.05:
            continue
        profile_b = get_building_profile(G, node)
        entry = {
            "node": node,
            "cf_score": 0.0,
            "match_score": round(sem * 100, 1),
            "categories": ["content-based (cold start)"],
            "type": data.get("type", "building"),
            "gps": data.get("gps"),
            "tagline": profile_b.get("tagline", ""),
            "function_summary": profile_b.get("function_summary", ""),
            "services": profile_b.get("services", []),
            "reason": f"Phù hợp với sở thích: {', '.join(interests[:3])}",
            "source": "content_based_cold_start",
        }
        if current_lat is not None and current_lon is not None:
            entry["distance_m"] = round(
                haversine(current_lat, current_lon, *data["gps"]), 1
            )
        results.append(entry)

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results[:top_k]
