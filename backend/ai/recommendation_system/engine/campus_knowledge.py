# engine/campus_knowledge.py
"""
Tri thức campus KHTN — cơ sở Linh Trung (ĐHQG-HCM).
Dùng cho Knowledge Graph, POI clustering và metadata AR.
"""
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Thông tin tổng quan cơ sở Linh Trung (thu thập từ nguồn công khai)
# ---------------------------------------------------------------------------
CAMPUS_LINH_TRUNG = {
    "id": "hcmus_linh_trung",
    "name": "Trường Đại học Khoa học Tự nhiên — Cơ sở Linh Trung",
    "short_name": "KHTN Linh Trung",
    "address": "Khu đô thị ĐHQG-HCM, phường Linh Trung, TP. Thủ Đức",
    "role": "Sinh viên đại cương / giai đoạn nền tảng",
    "center_gps": (10.8766, 106.8007),
    "highlights": [
        "Dãy tòa A–G nối hành lang có mái che",
        "Tòa D: thư viện, quầy giáo trình · Căn tin: ăn uống",
        "Nhà thể dục: gym, CLB thể thao",
        "WiFi phủ sóng toàn khu; bãi xe riêng gần cổng",
    ],
    "typical_hours": {
        "weekday_buildings": "06:00-18:00",
        "canteen_peak": "11:00-13:00",
        "gym_peak": "16:30-18:30",
    },
}

# Khung giờ sinh hoạt (ánh xạ sang category dịch vụ campus)
TIME_OF_DAY_BANDS = {
    "early_morning": {"start": "06:00", "end": "09:00", "label": "Sáng sớm"},
    "morning": {"start": "09:00", "end": "11:30", "label": "Buổi sáng"},
    "lunch": {"start": "11:00", "end": "13:30", "label": "Trưa"},
    "afternoon": {"start": "13:30", "end": "16:30", "label": "Chiều"},
    "evening": {"start": "16:30", "end": "19:00", "label": "Chiều tối"},
    "night": {"start": "19:00", "end": "23:59", "label": "Tối/đêm"},
}

# Ưu tiên category theo khung giờ (campus: cafe/pho → căn tin sáng, ăn trưa, gym tối)
TIME_CATEGORY_BOOST = {
    "early_morning": {"an_uong": 35, "hoc_tap": 15, "tien_ich": 10},
    "morning": {"hoc_tap": 30, "cntt": 25, "an_uong": 10},
    "lunch": {"an_uong": 40, "nghi_ngoi": 15},
    "afternoon": {"hoc_tap": 35, "cntt": 20, "the_thao": 10},
    "evening": {"the_thao": 35, "an_uong": 15, "giai_tri": 20},
    "night": {"tien_ich": 20},
}

# Cuối tuần & mùa vụ
WEEKEND_CATEGORY_BOOST = {"giai_tri": 30, "the_thao": 25, "an_uong": 15}
INDOOR_WEATHER_BOOST = 25  # rainy | winter

# Cụm POI — “phố chuyên doanh” trong campus
POI_CLUSTERS: Dict[str, dict] = {
    "day_academic": {
        "label": "Dãy học tập A–G",
        "nodes": ["Tòa A", "Tòa B", "Tòa C", "Tòa D", "Tòa E", "Tòa F", "Tòa G", "Căn tin"],
        "explore_mode": True,
        "description": "Hành lang tòa nhà — lab, tự học, thư viện, căn tin",
    },
    "sports_zone": {
        "label": "Khu thể thao",
        "nodes": ["Nhà thể dục", "Tòa G"],
        "explore_mode": True,
        "description": "Gym, sân sự kiện, CLB thể thao",
    },
    "services_hub": {
        "label": "Khu tiện ích & ra vào",
        "nodes": ["Cổng trường", "Nhà xe", "ATM", "Nhà điều hành"],
        "explore_mode": False,
        "description": "Cổng, bãi xe, ATM, hành chính",
    },
}

# Knowledge Graph — quan hệ ngữ nghĩa (đồ thị tri thức)
# relation: co_occurs_with | similar_to | contrasts_with | busy_at
KNOWLEDGE_GRAPH_EDGES: List[dict] = [
    {"from": "Tòa D", "to": "Tòa B", "relation": "similar_to",
     "tags": ["hoc tap", "yen tinh"], "weight": 0.7},
    {"from": "Tòa D", "to": "Nhà thể dục", "relation": "contrasts_with",
     "tags": ["on ao", "the thao"], "weight": 0.5},
    {"from": "Tòa D", "to": "Tòa F", "relation": "similar_to",
     "tags": ["nghi ngoi"], "weight": 0.6},
    {"from": "Nhà thể dục", "to": "Tòa G", "relation": "co_occurs_with",
     "tags": ["su kien", "the thao"], "weight": 0.8},
    {"from": "Căn tin", "to": "self", "relation": "busy_at",
     "tags": ["dong", "an", "trua"], "time_band": "lunch", "weight": 0.9},
    {"from": "Căn tin", "to": "Tòa D", "relation": "co_occurs_with",
     "tags": ["can tin", "thu vien"], "weight": 0.9},
    {"from": "Tòa B", "to": "self", "relation": "busy_at",
     "tags": ["dong", "thi"], "time_band": "morning", "weight": 0.75},
    {"from": "Nhà thể dục", "to": "self", "relation": "busy_at",
     "tags": ["dong", "tap"], "time_band": "evening", "weight": 0.85},
]

# Tín hiệu từ “đánh giá” (NLP reviews) — mô phỏng trích xuất từ review
REVIEW_SIGNALS: Dict[str, List[dict]] = {
    "Tòa D": [
        {"phrase": "thu vien yen tinh sang som", "keywords": ["yen tinh", "sang"], "time_band": "morning"},
    ],
    "Căn tin": [
        {"phrase": "can tin dong trua", "keywords": ["can tin", "dong", "trua"], "time_band": "lunch"},
    ],
    "Tòa B": [
        {"phrase": "phong tu hoc rat yen", "keywords": ["yen tinh", "tu hoc"], "time_band": "morning"},
        {"phrase": "on cho hoc nhom chieu", "keywords": ["hoc nhom"], "time_band": "afternoon"},
    ],
    "Nhà thể dục": [
        {"phrase": "gym dong sau 16h", "keywords": ["gym", "dong", "chieu"], "time_band": "evening"},
    ],
    "Tòa G": [
        {"phrase": "san su kien cuoi tuan vui", "keywords": ["su kien", "cuoi tuan"], "day_type": "weekend"},
    ],
}

# Bán kính gợi ý theo phương thức di chuyển (mét)
MOBILITY_RADIUS_M = {
    "walk": (80, 1000),
    "bike": (150, 2000),
    "drive": (500, 8000),
}

# Vùng indoor (WiFi / beacon — mô phỏng)
INDOOR_ZONES: Dict[str, dict] = {
    "Tòa D": {
        "floor_hints": ["Tầng trệt: Quầy Giáo trình & Sách", "Tầng 2: Thư viện Trung tâm"],
        "beacon_ids": ["LT-D-02"],
    },
    "Căn tin": {
        "floor_hints": ["Tầng trệt: Căn tin & Khu ẩm thực sinh viên"],
        "beacon_ids": ["LT-CAN-TIN", "LT-D-00"],
    },
    "Tòa C": {"floor_hints": ["Phòng máy CNTT"], "beacon_ids": ["LT-C-LAB"]},
}

# Đám đông thời gian thực (crowdsourcing ẩn danh — mô phỏng in-memory)
_live_crowd_reports: Dict[str, float] = {}


def report_live_crowd(node_id: str, level: float) -> None:
    """Ghi nhận mức đông thực tế từ client (0–1)."""
    _live_crowd_reports[node_id] = max(0.0, min(1.0, level))


def get_live_crowd(node_id: str) -> Optional[float]:
    return _live_crowd_reports.get(node_id)


def get_cluster_for_node(node_id: str) -> Optional[str]:
    for cid, c in POI_CLUSTERS.items():
        if node_id in c["nodes"]:
            return cid
    return None


def get_cluster_at_location(node_ids_in_radius: List[str]) -> Optional[str]:
    """Cụm chiếm ưu thế khi user đứng trong vùng."""
    counts: Dict[str, int] = {}
    for n in node_ids_in_radius:
        cid = get_cluster_for_node(n)
        if cid:
            counts[cid] = counts.get(cid, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda x: x[1])[0]


def knowledge_neighbors(node_id: str, relation: Optional[str] = None) -> List[dict]:
    out = []
    for e in KNOWLEDGE_GRAPH_EDGES:
        if e["from"] != node_id and e.get("to") != node_id:
            continue
        if relation and e.get("relation") != relation:
            continue
        target = e["to"] if e["from"] == node_id else e["from"]
        if target == "self":
            target = node_id
        out.append({**e, "target": target})
    return out
