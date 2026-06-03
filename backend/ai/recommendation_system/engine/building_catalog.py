# engine/building_catalog.py
"""
Danh mục chức năng / dịch vụ từng tòa — dùng cho gợi ý và hiển thị AR.
"""
from typing import Dict, List, Optional

import networkx as nx

from engine.nlp_processor import normalize_text

# Mỗi service: id, tên hiển thị, icon, category, từ khóa tìm kiếm
_BUILDING_PROFILES: Dict[str, dict] = {
    "Tòa A": {
        "image_url": "/static/images/toa_a.jpg",
        "tagline": "Khu thực nghiệm & Phòng Lab kỹ thuật",
        "departments": [
            "Văn phòng khoa Vật lý - Vật lý kỹ thuật",
            "Phòng thí nghiệm Cơ học ứng dụng",
            "Phòng thực nghiệm Điện tử hạt nhân",
            "Phòng lab AI & Robot"
        ],
        "events": [
            "Triển lãm Robotics Sinh viên (08:00 - 11:30)",
            "Hội thảo Công nghệ bán dẫn (13:30 - 16:30)"
        ],
        "services": [
            {"id": "lab", "name": "Phòng thí nghiệm", "icon": "🔬", "category": "hoc_tap",
             "keywords": ["thuc nghiem", "lab", "kỹ thuật"]},
            {"id": "group_study", "name": "Học nhóm / ôn tập", "icon": "👥", "category": "hoc_tap",
             "keywords": ["hoc nhom", "on tap"]},
        ],
    },
    "Tòa B": {
        "image_url": "/static/images/toa_b.jpg",
        "tagline": "Giảng đường & Phòng tự học",
        "departments": [
            "Phòng tự học B201 (Yên tĩnh)",
            "Văn phòng Đoàn - Hội Sinh viên",
            "Phòng máy B301 (Thực hành)",
            "Trung tâm Khảo thí"
        ],
        "events": [
            "Tư vấn hướng nghiệp (09:00 - 11:00)",
            "Hoạt động sinh hoạt Câu lạc bộ (17:00 - 19:00)"
        ],
        "services": [
            {"id": "self_study", "name": "Phòng tự học", "icon": "📖", "category": "hoc_tap",
             "keywords": ["tu hoc", "yen tinh", "tap trung"]},
            {"id": "group_quiet", "name": "Học nhóm nhỏ", "icon": "🤫", "category": "hoc_tap",
             "keywords": ["hoc nhom"]},
        ],
    },
    "Tòa C": {
        "image_url": "/static/images/toa_c.jpg",
        "tagline": "Thư viện & Phòng Lab kỹ thuật",
        "departments": [
            "Thư viện Trung tâm KHTN (Tầng 2)",
            "Phòng lab Kỹ thuật phần mềm",
            "Phòng Y tế Học đường"
        ],
        "events": [
            "Hội sách cũ USSH - KHTN (Cả ngày)"
        ],
        "services": [
            {"id": "library", "name": "Thư viện / đọc sách", "icon": "📚", "category": "hoc_tap",
             "keywords": ["thu vien", "doc sach", "muon sach"]},
            {"id": "practice", "name": "Thực hành môn học", "icon": "⌨️", "category": "hoc_tap",
             "keywords": ["thuc hanh", "lab"]},
        ],
    },
    "Tòa D": {
        "image_url": "/static/images/toa_d.jpg",
        "tagline": "Phòng Lab máy tính & Quầy giáo trình",
        "departments": [
            "Khoa Công nghệ Thông tin",
            "Văn phòng Giáo vụ khoa CNTT",
            "Phòng máy tính 202 (Thực hành)",
            "Quầy Giáo trình & Sách"
        ],
        "events": [
            "Seminar Khoa học Dữ liệu & AI (14:00 - 16:00)"
        ],
        "services": [
            {"id": "computer_lab", "name": "Phòng máy / Lab CNTT", "icon": "💻", "category": "cntt",
             "keywords": ["may tinh", "lab", "lap trinh", "code"]},
            {"id": "bookstore", "name": "Quầy giáo trình / sách", "icon": "📕", "category": "hoc_tap",
             "keywords": ["giao trinh", "mua sach"]},
        ],
    },
    "Canteen": {
        "image_url": "/static/images/canteen.jpg",
        "tagline": "Khu ăn uống - tán gẫu",
        "departments": [
            "Căn tin trường (Tầng trệt)"
        ],
        "events": [],
        "services": [
            {"id": "canteen", "name": "Căn tin / ăn uống", "icon": "🍽️", "category": "an_uong",
             "keywords": ["can tin", "an trua", "an", "doi bung", "com", "an vat", "tra sua"]},
        ],
    },
    "Tòa E": {
        "image_url": "/static/images/toa_e.jpg",
        "tagline": "Phòng học lý thuyết & Phòng nghỉ trưa",
        "departments": [
            "Văn phòng bộ môn Toán học",
            "Phòng học Lý thuyết E101-E104",
            "Khu vực nghỉ trưa tự do"
        ],
        "events": [
            "Lớp chuyên đề Toán ứng dụng (08:00 - 10:00)"
        ],
        "services": [
            {"id": "lecture", "name": "Phòng học lý thuyết", "icon": "🎓", "category": "hoc_tap",
             "keywords": ["ly thuyet", "bai giang"]},
            {"id": "nap", "name": "Khu nghỉ trưa", "icon": "😴", "category": "nghi_ngoi",
             "keywords": ["nghi trua", "ngu trua"]},
        ],
    },
    "Tòa F": {
        "image_url": "/static/images/toa_f.jpg",
        "tagline": "Phòng học lý thuyết & Phòng nghỉ trưa",
        "departments": [
            "Phòng nghỉ Sinh viên F102 (Có điều hòa)",
            "Phòng sinh hoạt chung CLB tiếng Anh",
            "Khu tự học F201"
        ],
        "events": [
            "Workshop kỹ năng mềm (15:00 - 17:00)"
        ],
        "services": [
            {"id": "rest", "name": "Phòng nghỉ sinh viên", "icon": "🛋️", "category": "nghi_ngoi",
             "keywords": ["nghi", "ngu", "met", "buon ngu"]},
        ],
    },
    "Tòa G": {
        "image_url": "/static/images/toa_g.jpg",
        "tagline": "Phòng Lab Hoá - Sinh",
        "departments": [
            "Sân sự kiện trung tâm",
            "Văn phòng Đoàn khoa Hóa học",
            "Khoa học Vật liệu"
        ],
        "events": [
            "Nhạc hội chào đón Tân sinh viên (18:00 - 21:00)"
        ],
        "services": [
            {"id": "event", "name": "Sân / khu sự kiện", "icon": "🎪", "category": "giai_tri",
             "keywords": ["su kien", "hoat dong"]},
        ],
    },
    "Nhà thể dục": {
        "image_url": "/static/images/nha_the_duc.jpg",
        "tagline": "Nơi tập luyện thể dục - thể thao & Nhà thi đấu",
        "departments": [
            "Phòng GYM & Fitness",
            "Sân cầu lông",
            "Sân bóng bàn",
            "Văn phòng Bộ môn Giáo dục Thể chất"
        ],
        "events": [
            "Giải cầu lông truyền thống (14:30 - 17:30)"
        ],
        "services": [
            {"id": "gym", "name": "Tập gym / fitness", "icon": "🏋️", "category": "the_thao",
             "keywords": ["gym", "tap", "the luc"]},
            {"id": "sports", "name": "Cầu lông, bóng bàn, CLB thể thao", "icon": "🏸", "category": "the_thao",
             "keywords": ["cau long", "bong ban", "the thao", "clb"]},
        ],
    },
    "Nhà xe": {
        "image_url": "/static/images/nha_xe.jpg",
        "tagline": "Nơi giữ xe sinh viên - giảng viên",
        "departments": [
            "Khu vực gửi xe máy Sinh viên",
            "Khu vực gửi xe đạp điện",
            "Trạm sạc xe điện"
        ],
        "events": [],
        "services": [
            {"id": "parking", "name": "Gửi / lấy xe máy", "icon": "🛵", "category": "tien_ich",
             "keywords": ["gui xe", "lay xe", "xe may", "parking"]},
            {"id": "exit", "name": "Ra về cuối ngày", "icon": "🚪", "category": "tien_ich",
             "keywords": ["ra ve", "ve nha"]},
        ],
    },
    "Cây ATM": {
        "image_url": "/static/images/cay_atm.jpg",
        "tagline": "Nơi thực hiện các giao dịch ngân hàng",
        "departments": [
            "Trụ ATM Vietcombank",
            "Trụ ATM BIDV"
        ],
        "events": [],
        "services": [
            {"id": "atm", "name": "Cây ATM", "icon": "🏧", "category": "tien_ich",
             "keywords": ["atm", "rut tien", "tien mat"]},
        ],
    },
    "Nhà điều hành": {
        "image_url": "/static/images/nha_dieu_hanh.jpg",
        "tagline": "Văn phòng Ban Giám hiệu & Phòng nghỉ trưa",
        "departments": [
            "Phòng Đào tạo",
            "Phòng Công tác Sinh viên",
            "Phòng Kế hoạch Tài chính",
            "Ban Giám hiệu"
        ],
        "events": [
            "Họp giao ban Quản lý Đào tạo (10:00 - 11:30)"
        ],
        "services": [
            {"id": "admin", "name": "Phòng ban / giao vụ", "icon": "🏛️", "category": "hanh_chinh",
             "keywords": ["giao vu", "giay to", "hanh chinh"]},
            {"id": "tuition", "name": "Đóng học phí", "icon": "💳", "category": "hanh_chinh",
             "keywords": ["hoc phi", "dong tien"]},
        ],
    },
    "Cổng trường": {
        "image_url": "/static/images/cong_truong.jpg",
        "tagline": "Nơi ra - vào trường",
        "departments": [
            "Bốt Bảo vệ chính",
            "Trạm xe buýt nội khu ĐHQG"
        ],
        "events": [],
        "services": [
            {"id": "entrance", "name": "Check-in / vào campus", "icon": "🚧", "category": "tien_ich",
             "keywords": ["cong", "vao truong"]},
        ],
    },
}

_CATEGORY_LABELS = {
    "hoc_tap": "Học tập",
    "an_uong": "Ăn uống",
    "nghi_ngoi": "Nghỉ ngơi",
    "the_thao": "Thể thao",
    "cntt": "CNTT / Lab",
    "tien_ich": "Tiện ích",
    "hanh_chinh": "Hành chính",
    "giai_tri": "Giải trí",
}


def get_building_profile(G: nx.Graph, node_id: str) -> dict:
    """Hồ sơ đầy đủ chức năng của một tòa."""
    if node_id not in G.nodes:
        return {}

    data = G.nodes[node_id]
    catalog = _BUILDING_PROFILES.get(node_id) or {}
    services = list(data.get("services") or catalog.get("services", []))
    tagline = data.get("tagline") or catalog.get("tagline", "")
    departments = catalog.get("departments", [])
    events = catalog.get("events", [])

    features = data.get("features", {})
    amenity_tags = []
    if features.get("has_ac"):
        amenity_tags.append("Điều hòa")
    if features.get("has_tables"):
        amenity_tags.append("Bàn ghế học tập")
    if features.get("noise_level", 1) <= 0.3:
        amenity_tags.append("Yên tĩnh")
    elif features.get("noise_level", 0) >= 0.7:
        amenity_tags.append("Sôi động")

    service_names = [s["name"] for s in services]
    function_summary = " · ".join(service_names) if service_names else tagline

    return {
        "node": node_id,
        "tagline": tagline,
        "function_summary": function_summary,
        "services": services,
        "amenities": amenity_tags,
        "departments": departments,
        "events": events,
        "type": data.get("type", "building"),
        "restricted": bool(data.get("restricted", False)),
        "open_time": data.get("open_time"),
        "close_time": data.get("close_time"),
        "gps": data.get("gps"),
        "image_url": catalog.get("image_url", ""),
    }


def match_services_to_query(query: str, services: List[dict]) -> List[dict]:
    """Lọc dịch vụ khớp câu hỏi người dùng."""
    q = normalize_text(query)
    if not q:
        return []
    matched = []
    for svc in services:
        keys = [normalize_text(svc.get("name", ""))] + [normalize_text(k) for k in svc.get("keywords", [])]
        if any(k and k in q for k in keys) or any(k and len(k) >= 4 and k in q for k in keys):
            matched.append(svc)
    return matched


def build_function_reason(
    G: nx.Graph,
    node_id: str,
    query: Optional[str] = None,
    context_hint: str = "",
) -> str:
    """Câu gợi ý có nêu rõ chức năng tòa."""
    profile = get_building_profile(G, node_id)
    if not profile:
        return context_hint or f"Ghé {node_id}."

    summary = profile["function_summary"]
    tagline = profile["tagline"]

    if query:
        matched = match_services_to_query(query, profile["services"])
        if matched:
            names = ", ".join(s["name"] for s in matched[:3])
            base = f"{node_id} có {names}"
        else:
            base = f"{node_id}: {summary}"
    else:
        base = f"{node_id} — {tagline}" if tagline else f"{node_id}: {summary}"

    if context_hint:
        return f"{context_hint} ({base})."
    return f"{base}."


def enrich_suggestion(G: nx.Graph, item: dict, query: Optional[str] = None) -> dict:
    """Bổ sung chức năng tòa vào object gợi ý."""
    node = item.get("node")
    if not node:
        return item

    profile = get_building_profile(G, node)
    item["tagline"] = profile.get("tagline", "")
    item["function_summary"] = profile.get("function_summary", "")
    item["services"] = profile.get("services", [])
    item["amenities"] = profile.get("amenities", [])
    item["departments"] = profile.get("departments", [])
    item["events"] = profile.get("events", [])

    if query and profile.get("services"):
        item["matched_services"] = match_services_to_query(query, profile["services"])
    else:
        item["matched_services"] = []

    if (
        profile.get("function_summary")
        and item.get("reason")
        and profile["function_summary"] not in item["reason"]
        and "Bạn có thể" not in item["reason"]
    ):
        item["reason"] = (
            f"{item['reason'].rstrip('.')} — "
            f"Bạn có thể: {profile['function_summary']}."
        )
    return item


def list_all_building_guides(G: nx.Graph) -> List[dict]:
    """Danh sách chức năng toàn campus — cho màn hình tra cứu."""
    guides = []
    for node in sorted(G.nodes()):
        p = get_building_profile(G, node)
        if p:
            guides.append(p)
    return guides
