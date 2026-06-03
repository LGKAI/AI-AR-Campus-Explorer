# engine/context_features.py
"""
Tính năng ngữ cảnh thời gian & không gian cho recommender campus.
"""
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import networkx as nx

from engine.building_catalog import get_building_profile, match_services_to_query
from engine.campus_knowledge import (
    CAMPUS_LINH_TRUNG,
    INDOOR_WEATHER_BOOST,
    INDOOR_ZONES,
    MOBILITY_RADIUS_M,
    POI_CLUSTERS,
    REVIEW_SIGNALS,
    TIME_CATEGORY_BOOST,
    TIME_OF_DAY_BANDS,
    WEEKEND_CATEGORY_BOOST,
    get_cluster_at_location,
    get_cluster_for_node,
    get_live_crowd,
    knowledge_neighbors,
)
from engine.nlp_processor import normalize_text
from engine.optimizer import is_node_open
from engine.utils import haversine, parse_time


def get_time_of_day_band(current_time_str: str) -> str:
    curr = parse_time(current_time_str)
    if not curr:
        return "afternoon"
    for band_id, band in TIME_OF_DAY_BANDS.items():
        start = parse_time(band["start"])
        end = parse_time(band["end"])
        if start and end and start <= curr <= end:
            return band_id
    return "afternoon"


def is_weekend(dt: Optional[datetime] = None) -> bool:
    dt = dt or datetime.now()
    return dt.weekday() >= 5


def minutes_until_close(
    G: nx.Graph,
    node_id: str,
    current_time_str: str,
) -> Optional[int]:
    """Số phút còn lại trước khi đóng cửa. None nếu không xác định."""
    data = G.nodes.get(node_id, {})
    close_str = data.get("close_time")
    if not close_str or close_str == "23:59":
        return None

    curr = parse_time(current_time_str)
    close_t = parse_time(close_str)
    if not curr or not close_t:
        return None

    now_dt = datetime.combine(datetime.today(), curr)
    close_dt = datetime.combine(datetime.today(), close_t)
    if close_dt < now_dt:
        close_dt += timedelta(days=1)
    delta = close_dt - now_dt
    return int(delta.total_seconds() // 60)


def open_status_detail(
    G: nx.Graph,
    node_id: str,
    current_time_str: str,
    closing_warn_mins: int = 30,
) -> dict:
    """Open Now + cảnh báo sắp đóng."""
    open_now = is_node_open(G, node_id, current_time_str)
    mins_left = minutes_until_close(G, node_id, current_time_str)
    closing_soon = (
        open_now
        and mins_left is not None
        and 0 < mins_left <= closing_warn_mins
    )
    warn_msg = None
    if closing_soon:
        warn_msg = f"Sắp đóng cửa trong ~{mins_left} phút"
    elif open_now is False:
        warn_msg = "Địa điểm hiện đang đóng cửa"

    return {
        "node": node_id,
        "open_now": open_now,
        "minutes_until_close": mins_left,
        "closing_soon": closing_soon,
        "warning": warn_msg,
        "hours": f"{G.nodes[node_id].get('open_time', '')}-{G.nodes[node_id].get('close_time', '')}",
    }


def time_category_boost(node_id: str, G: nx.Graph, time_band: str, weekend: bool, query: Optional[str] = None) -> float:
    """Điểm cộng theo khung giờ + category dịch vụ."""
    profile = get_building_profile(G, node_id)
    boosts = dict(TIME_CATEGORY_BOOST.get(time_band, {}))
    if weekend:
        for k, v in WEEKEND_CATEGORY_BOOST.items():
            boosts[k] = boosts.get(k, 0) + v

    score = 0.0
    has_specific_intent = False
    active_cats = set()
    if query:
        q = normalize_text(query)
        if any(w in q for w in ["doi", "an", "uong", "cafe", "ca phe", "com", "nuoc", "canteen"]):
            active_cats.add("an_uong")
            has_specific_intent = True
        if any(w in q for w in ["the thao", "tap", "van dong", "gym", "cau long", "bong ban", "the duc"]):
            active_cats.add("the_thao")
            has_specific_intent = True
        if any(w in q for w in ["ngu", "nghi ngoi", "met", "nga lung", "buon ngu"]):
            active_cats.add("nghi_ngoi")
            has_specific_intent = True
        if any(w in q for w in ["hoc", "ngoi", "lam bai", "ban ghe", "tu hoc", "yen tinh", "on a", "hoc bai", "doc sach", "tap trung"]):
            active_cats.add("hoc_tap")
            active_cats.add("cntt")
            has_specific_intent = True

    for svc in profile.get("services", []):
        cat = svc.get("category", "")
        if has_specific_intent and cat not in active_cats:
            continue
        score += boosts.get(cat, 0)
        
    # Explicit boosts specified in refactoring specs
    if time_band == "lunch" and node_id == "Căn tin":
        if not has_specific_intent or "an_uong" in active_cats:
            score += 30.0  # Boost for food/canteen during lunch hours
    elif time_band == "evening" and node_id == "Nhà thể dục":
        if not has_specific_intent or "the_thao" in active_cats:
            score += 30.0  # Boost for sports/gym during evening hours
        
    return score


def indoor_boost(G: nx.Graph, node_id: str, weather: str) -> float:
    if weather not in ("rainy", "winter"):
        return 0.0
    if G.nodes[node_id].get("indoor", False):
        return float(INDOOR_WEATHER_BOOST)
    features = G.nodes[node_id].get("features", {})
    if features.get("has_ac"):
        return float(INDOOR_WEATHER_BOOST) * 0.7
    return -10.0


def review_nlp_boost(node_id: str, query: Optional[str], time_band: str, weekend: bool) -> Tuple[float, List[str]]:
    """Trích xuất tín hiệu từ review mô phỏng + query."""
    signals = REVIEW_SIGNALS.get(node_id, [])
    if not signals:
        return 0.0, []

    q = normalize_text(query or "")
    matched_phrases: List[str] = []
    score = 0.0

    for sig in signals:
        keys = sig.get("keywords", [])
        band_ok = sig.get("time_band") == time_band or not sig.get("time_band")
        weekend_ok = sig.get("day_type") != "weekend" or weekend
        keyword_hit = any(k in q for k in keys) if q else False
        contextual = band_ok and weekend_ok

        if contextual or keyword_hit:
            score += 12.0 if keyword_hit else 8.0
            matched_phrases.append(sig["phrase"])

    return score, matched_phrases


def knowledge_graph_score(
    G: nx.Graph,
    node_id: str,
    user_prefs: Optional[List[str]] = None,
    time_band: Optional[str] = None,
) -> float:
    """Đối chiếu KG với sở thích (ồn ào, yên tĩnh, đông...)."""
    prefs = " ".join(user_prefs or []).lower()
    score = 0.0
    for edge in knowledge_neighbors(node_id):
        tags = " ".join(edge.get("tags", []))
        rel = edge.get("relation", "")
        if edge.get("time_band") and edge["time_band"] != time_band:
            continue

        if "yen tinh" in prefs and "yen tinh" in tags and rel == "similar_to":
            score += 15 * edge.get("weight", 0.5)
        if "on ao" in prefs and ("on ao" in tags or "dong" in tags):
            if rel in ("busy_at", "contrasts_with"):
                score -= 10
            if rel == "co_occurs_with" and "the thao" in prefs:
                score += 12
        if "an" in prefs and "can tin" in tags:
            score += 18 * edge.get("weight", 0.5)
    return score


def proximity_score(dist_m: float, mobility: str = "walk") -> float:
    """Location bias — gần hơn = điểm cao hơn."""
    min_r, max_r = MOBILITY_RADIUS_M.get(mobility, MOBILITY_RADIUS_M["walk"])
    if dist_m > max_r:
        return -20.0
    if dist_m <= min_r:
        return 22.0
    if dist_m <= 200:
        return 18.0 - (dist_m / 200) * 4
    if dist_m <= 500:
        return 12.0 - (dist_m - 200) / 300 * 6
    return max(0.0, 8.0 - (dist_m - 500) / 500 * 8)


def mobility_radius_ok(dist_m: float, mobility: str) -> bool:
    _, max_r = MOBILITY_RADIUS_M.get(mobility, MOBILITY_RADIUS_M["walk"])
    return dist_m <= max_r


def detect_explore_cluster(
    G: nx.Graph,
    lat: float,
    lon: float,
    radius_m: float = 120.0,
) -> Optional[dict]:
    """POI clustering — chế độ Explore khi đứng trong cụm."""
    nearby = []
    for n, d in G.nodes(data=True):
        if haversine(lat, lon, *d["gps"]) <= radius_m:
            nearby.append(n)
    cluster_id = get_cluster_at_location(nearby)
    if not cluster_id:
        return None
    cluster = POI_CLUSTERS[cluster_id]
    if not cluster.get("explore_mode"):
        return None
    members = [n for n in cluster["nodes"] if n in G.nodes]
    return {
        "cluster_id": cluster_id,
        "label": cluster["label"],
        "description": cluster["description"],
        "member_nodes": members,
        "nearby_nodes": nearby,
    }


def indoor_positioning_hint(node_id: str) -> Optional[dict]:
    return INDOOR_ZONES.get(node_id)


def effective_crowd_level(
    G: nx.Graph,
    node_id: str,
    predicted: float,
) -> float:
    """Kết hợp dự báo + crowdsourcing thời gian thực."""
    live = get_live_crowd(node_id)
    if live is None:
        return predicted
    return round(0.4 * predicted + 0.6 * live, 2)


def find_less_busy_alternative(
    G: nx.Graph,
    busy_node: str,
    current_lat: float,
    current_lon: float,
    crowd_fn,
    current_time_str: str,
    max_dist_m: float = 400.0,
) -> Optional[dict]:
    """Nếu A quá tải → gợi ý B tương tự gần đó."""
    if busy_node not in G.nodes:
        return None

    busy_crowd = effective_crowd_level(
        G, busy_node, crowd_fn(G, busy_node, current_time_str)
    )
    if busy_crowd < 0.8:
        return None

    profile = get_building_profile(G, busy_node)
    busy_cats = {s.get("category") for s in profile.get("services", [])}
    busy_aliases = set(G.nodes[busy_node].get("aliases", []))

    best = None
    best_score = -1.0

    for node, data in G.nodes(data=True):
        if node == busy_node or not is_node_open(G, node, current_time_str):
            continue
        dist = haversine(current_lat, current_lon, *data["gps"])
        if dist > max_dist_m:
            continue
        crowd = effective_crowd_level(G, node, crowd_fn(G, node, current_time_str))
        if crowd >= 0.65:
            continue

        p = get_building_profile(G, node)
        cats = {s.get("category") for s in p.get("services", [])}
        alias_overlap = len(set(data.get("aliases", [])) & busy_aliases)

        sim = len(cats & busy_cats) * 20 + alias_overlap * 5
        for kn in knowledge_neighbors(busy_node, "similar_to"):
            if kn.get("target") == node:
                sim += 25

        sim += proximity_score(dist) * 0.5
        if sim > best_score:
            best_score = sim
            best = {
                "node": node,
                "reason": f"Thay thế {busy_node} — ít đông hơn (~{int(dist)}m)",
                "crowd_level": round(crowd, 2),
                "distance_m": round(dist, 1),
                "similarity_score": round(sim, 1),
            }
    return best


def compute_context_scores(
    G: nx.Graph,
    node_id: str,
    current_lat: float,
    current_lon: float,
    current_time_str: str,
    weather: str = "normal",
    mobility: str = "walk",
    query: Optional[str] = None,
    user_prefs: Optional[List[str]] = None,
    crowd_fn=None,
) -> dict:
    """
    Tổng hợp điểm ngữ cảnh cho một node — dùng trong enhanced recommender.
    """
    from engine.recommender import predict_crowd_level

    crowd_fn = crowd_fn or predict_crowd_level
    time_band = get_time_of_day_band(current_time_str)
    weekend = is_weekend()
    dist_m = haversine(current_lat, current_lon, *G.nodes[node_id]["gps"])

    open_info = open_status_detail(G, node_id, current_time_str)
    if not open_info["open_now"]:
        return {"node": node_id, "total": -999, "filtered": "closed", **open_info}

    if not mobility_radius_ok(dist_m, mobility):
        return {"node": node_id, "total": -999, "filtered": "out_of_radius", "distance_m": dist_m}

    pred_crowd = crowd_fn(G, node_id, current_time_str)
    crowd = effective_crowd_level(G, node_id, pred_crowd)
    rev_score, rev_hits = review_nlp_boost(node_id, query, time_band, weekend)

    components = {
        "time_of_day": time_category_boost(node_id, G, time_band, weekend, query=query),
        "proximity": proximity_score(dist_m, mobility),
        "indoor_weather": indoor_boost(G, node_id, weather),
        "review_nlp": rev_score,
        "knowledge_graph": knowledge_graph_score(G, node_id, user_prefs, time_band),
        "crowd_penalty": -15 if crowd >= 0.85 else (5 if crowd <= 0.35 else 0),
    }
    total = sum(components.values())

    return {
        "node": node_id,
        "total": round(total, 2),
        "distance_m": round(dist_m, 1),
        "time_band": time_band,
        "time_band_label": TIME_OF_DAY_BANDS[time_band]["label"],
        "crowd_level": crowd,
        "crowd_predicted": round(pred_crowd, 2),
        "review_matches": rev_hits,
        "components": components,
        "open_status": open_info,
        "indoor": G.nodes[node_id].get("indoor", False),
        "poi_cluster": get_cluster_for_node(node_id),
        "indoor_hint": indoor_positioning_hint(node_id),
    }


def enhanced_recommendations(
    G: nx.Graph,
    current_lat: float,
    current_lon: float,
    current_time_str: str,
    destination: Optional[str] = None,
    query: Optional[str] = None,
    weather: str = "normal",
    mobility: str = "walk",
    user_interests: Optional[List[str]] = None,
    limit: int = 8,
    crowd_fn=None,
) -> dict:
    """
    API payload đầy đủ: gợi ý + explore cluster + thay thế khi đông.
    """
    from engine.recommender import get_smart_recommendations, predict_crowd_level

    crowd_fn = crowd_fn or predict_crowd_level
    time_band = get_time_of_day_band(current_time_str)
    explore = detect_explore_cluster(G, current_lat, current_lon)

    base = get_smart_recommendations(
        G, current_lat, current_lon,
        destination=destination,
        query=query,
        weather=weather,
        current_time_str=current_time_str,
        user_interests=user_interests,
        limit=limit,
    )

    enriched = []
    for item in base:
        node = item["node"]
        ctx = compute_context_scores(
            G, node, current_lat, current_lon, current_time_str,
            weather, mobility, query, user_interests, crowd_fn,
        )
        if ctx.get("filtered"):
            continue
        open_info = ctx["open_status"]
        alt = find_less_busy_alternative(
            G, node, current_lat, current_lon, crowd_fn, current_time_str,
        )
        merged_score = item.get("raw_score", item.get("score", 0)) + ctx["total"] * 0.35
        enriched.append({
            **item,
            "context_score": ctx["total"],
            "combined_raw": round(merged_score, 2),
            "time_band": ctx["time_band"],
            "time_band_label": ctx["time_band_label"],
            "open_now": open_info["open_now"],
            "closing_soon": open_info["closing_soon"],
            "close_warning": open_info.get("warning"),
            "crowd_level": ctx["crowd_level"],
            "review_signals": ctx["review_matches"],
            "poi_cluster": ctx["poi_cluster"],
            "indoor_hint": ctx["indoor_hint"],
            "less_busy_alternative": alt,
            "mobility_mode": mobility,
        })

    enriched.sort(key=lambda x: -x["combined_raw"])

    # Explore mode: thêm điểm trong cụm nếu chưa có
    if explore:
        existing = {e["node"] for e in enriched}
        for member in explore["member_nodes"]:
            if member in existing or not is_node_open(G, member, current_time_str):
                continue
            ctx = compute_context_scores(
                G, member, current_lat, current_lon, current_time_str,
                weather, mobility, query, user_interests, crowd_fn,
            )
            if ctx.get("filtered"):
                continue
            profile = get_building_profile(G, member)
            enriched.append({
                "node": member,
                "score": min(100, max(40, 50 + ctx["total"])),
                "reason": f"Chế độ Khám phá — {explore['label']}: {profile.get('tagline', '')}",
                "source": "explore_cluster",
                "context_score": ctx["total"],
                "combined_raw": ctx["total"] + 30,
                "time_band": time_band,
                "explore_mode": True,
                "open_now": ctx["open_status"]["open_now"],
                "crowd_level": ctx["crowd_level"],
                "gps": G.nodes[member]["gps"],
            })

    enriched = enriched[:limit]

    return {
        "campus": CAMPUS_LINH_TRUNG,
        "context": {
            "current_time": current_time_str,
            "time_band": time_band,
            "time_band_label": TIME_OF_DAY_BANDS[time_band]["label"],
            "is_weekend": is_weekend(),
            "weather": weather,
            "mobility": mobility,
            "explore_cluster": explore,
        },
        "recommendations": enriched,
    }
