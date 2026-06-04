"""
AI AR Campus API — Pathfinding & Smart Recommender
===================================================
Chỉ giữ lại các nhóm chức năng cốt lõi:
  1. Tìm đường (Pathfinding)  — A* + GNN, đa điểm
  2. Đề xuất thời gian thực  — GPS + crowd + cá nhân hóa tự động
  3. Quản lý vị trí GPS và cảnh báo geofence tự động
"""

import sys
import os
from datetime import datetime
from typing import Optional

# --- THỦ THUẬT ÉP ĐƯỜNG DẪN CHO CODE CỦA KHOA ---
# Lấy đường dẫn trỏ tới thư mục 'backend/ai/recommendation_system'
current_router_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_router_dir)
khoa_root_dir = os.path.join(backend_dir, "ai", "recommendation_system")

# Thêm nó vào bộ nhớ đệm của Python
if khoa_root_dir not in sys.path:
    sys.path.insert(0, khoa_root_dir)

from fastapi import APIRouter, HTTPException, Query, Depends, status
from fastapi.responses import HTMLResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# Import trực tiếp từ module không gian địa hình của Khoa
from ai.recommendation_system.engine.graph_builder_v2 import build_campus_graph, get_canvas_bounds
from ai.recommendation_system.engine.optimizer import (
    pathfinding_optimizer,
    multi_stop_routing,
    is_node_open,
    calc_remaining_distance,
    dynamic_edge_update,
    geofencing_logic,
    restricted_zone_alert
)
from ai.recommendation_system.engine.nlp_processor import find_node_by_keyword
from ai.recommendation_system.engine.building_catalog import get_building_profile
from ai.recommendation_system.engine.recommender import (
    recommend_locations,
    recommend_by_building_function,
    get_smart_recommendations,
    semantic_map_linking,
    crowd_prediction,
    predict_crowd_level,
    submit_crowd_report,
    update_profile_passively,
    load_crowd_model
)
from ai.recommendation_system.engine.utils import haversine, get_current_time_str, WALKING_SPEED_MPM

# ---------------------------------------------------------------------------
# Khởi tạo
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/map", tags=["Recommendation & Map Engine"])

G          = build_campus_graph()
list_nodes = sorted(G.nodes())
_bounds    = get_canvas_bounds(G)

# Quản lý hồ sơ người dùng đa phiên bản (Multi-session)
USER_PROFILES_DB: dict = {}

def get_session_profile(session_id: Optional[str] = None) -> dict:
    """Lấy hoặc khởi tạo hồ sơ người dùng tương ứng với session_id và cập nhật thụ động."""
    if not session_id or session_id == "null" or session_id == "undefined":
        session_id = "default"
    if session_id not in USER_PROFILES_DB:
        USER_PROFILES_DB[session_id] = {
            "role":            "student",   # Tự động suy luận
            "study_style":     "silent",    # Tự động suy luận
            "interests":       ["hoc_tap"],  # Tự động suy luận
            "visited_history": {},          # {node_id: visit_count}
        }
    profile = USER_PROFILES_DB[session_id]
    update_profile_passively(profile)
    return profile

# Khởi tạo GNN
_gnn_ready = False
try:
    from ai.recommendation_system.engine.gnn_engine import gnn_node_embedding
    gnn_node_embedding(G)
    _gnn_ready = True
except Exception as _e:
    print(f"[WARN] GNN init: {_e}")

# ---- Danh sách các API Endpoints hệ thống ----

# ===========================================================================
# USER PROFILE (Cá nhân hóa thụ động)
# ===========================================================================

@router.get("/api_user_profile", tags=["Profile"])
def get_user_profile(session_id: Optional[str] = Query(None, description="Session ID của người dùng")):
    """Lấy hồ sơ người dùng hiện tại (được cập nhật thụ động dựa trên lịch sử ghé thăm)."""
    profile = get_session_profile(session_id)
    return {"status": "success", "user_profile": profile}


@router.post("/api_user_profile", tags=["Profile"])
def update_user_profile(
    reset_history: bool = Query(False,       description="Xóa lịch sử ghé thăm"),
    session_id:    Optional[str] = Query(None, description="Session ID của người dùng"),
):
    """
    Cập nhật hoặc đặt lại hồ sơ người dùng.
    Mọi tham số thiết lập thủ công đã được loại bỏ, cá nhân hóa diễn ra thụ động.
    """
    profile = get_session_profile(session_id)
    if reset_history:
        profile["visited_history"] = {}
        profile["role"] = "student"
        profile["study_style"] = "silent"
        profile["interests"] = ["hoc_tap"]
    else:
        update_profile_passively(profile)
    return {"status": "success", "user_profile": profile}


# ===========================================================================
# PATHFINDING — Tìm đường
# ===========================================================================

@router.get("/api_get_graph", tags=["Pathfinding"])
def get_graph():
    """Trả về toàn bộ nodes + edges + bounds để frontend vẽ bản đồ."""
    now   = get_current_time_str()
    nodes = []
    for n in G.nodes():
        profile = get_building_profile(G, n)
        nodes.append({
            "id":               n,
            "x":                G.nodes[n]["pos"][0],
            "y":                G.nodes[n]["pos"][1],
            "gps":              G.nodes[n]["gps"],
            "type":             G.nodes[n]["type"],
            "is_open":          is_node_open(G, n, now),
            "features":         G.nodes[n].get("features", {}),
            "hours":            f"{G.nodes[n].get('open_time','N/A')} - {G.nodes[n].get('close_time','N/A')}",
            "tagline":          profile.get("tagline", ""),
            "function_summary": profile.get("function_summary", ""),
            "image_url":        profile.get("image_url", ""),
        })
    edges = [
        {
            "source":   u,
            "target":   v,
            "status":   d["status"],
            "has_roof": d["has_roof"],
            "weight_m": d.get("weight", 0),
            "edge_type":d.get("edge_type", "walkway"),
        }
        for u, v, d in G.edges(data=True)
    ]
    return {
        "nodes":        nodes,
        "edges":        edges,
        "current_time": now,
        "bounds":       _bounds,
        "routing_engine": "A* + GNN-GAT" if _gnn_ready else "A*",
    }


@router.get("/api_get_route", tags=["Pathfinding"])
def get_route(
    waypoints:  str = Query(..., description="Danh sách node cách nhau bởi dấu phẩy"),
    weather:    str = Query("normal",  description="normal | sunny | rainy"),
    preference: str = Query("fastest", description="fastest | covered | wheelchair"),
):
    """
    Lập lộ trình đa điểm dùng A* + GNN attention.
    """
    pts = [p.strip() for p in waypoints.split(",") if p.strip()]
    if len(pts) < 2:
        raise HTTPException(status_code=400, detail="Cần ít nhất 2 điểm (start, end).")

    invalid = [p for p in pts if p not in G.nodes]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: {invalid}")

    now  = get_current_time_str()
    path, all_open = multi_stop_routing(G, pts, weather, now, preference=preference)
    if not path:
        raise HTTPException(status_code=404, detail="Không tìm thấy lộ trình.")

    route_coords = []
    for i, n in enumerate(path):
        lat, lon = G.nodes[n]["gps"]
        seg = {
            "node":    n,
            "gps":     [lat, lon],
            "order":   i,
            "is_open": is_node_open(G, n, now),
            "crowd":   predict_crowd_level(G, n, now),
        }
        if i < len(path) - 1:
            ed = G[n][path[i + 1]]
            seg["next_bearing"] = _bearing_hint(lat, lon, *G.nodes[path[i + 1]]["gps"])
            seg["edge"] = {
                "has_roof":  ed.get("has_roof"),
                "weight_m":  ed.get("weight"),
                "status":    ed.get("status"),
                "edge_type": ed.get("edge_type"),
            }
        route_coords.append(seg)

    total_m = sum(G[path[i]][path[i+1]].get("weight", 0) for i in range(len(path)-1))
    return {
        "status":           "success",
        "path":             path,
        "coordinates":      route_coords,
        "ar_waypoints":     [{"node": c["node"], "gps": c["gps"]} for c in route_coords],
        "total_distance_m": round(total_m, 2),
        "estimated_mins":   round(total_m / WALKING_SPEED_MPM, 1),
        "all_open":         all_open,
        "routing_engine":   "A* + GNN-GAT" if _gnn_ready else "A*",
    }


@router.get("/api_update_edge", tags=["Pathfinding"])
def update_edge(u: str, v: str, status: str = Query(..., description="open | repairing | closed")):
    """Cập nhật trạng thái đường đi theo thời gian thực (sự cố, sửa chữa)."""
    if status not in ("open", "repairing", "closed"):
        raise HTTPException(status_code=400, detail="status phải là open | repairing | closed")
    ok = dynamic_edge_update(G, u, v, status)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy cạnh {u} — {v}")
    return {"status": "success", "message": f"Cạnh {u} — {v} → {status}"}


# ===========================================================================
# REAL-TIME TRACKING — Theo dõi vị trí + tìm đường động
# ===========================================================================

@router.get("/api_realtime_tracking", tags=["Tracking"])
def realtime_tracking(
    current_lat: float = Query(..., description="Vĩ độ GPS hiện tại"),
    current_lon: float = Query(..., description="Kinh độ GPS hiện tại"),
    end:         str   = Query(..., description="Node đích"),
    weather:     str   = Query("normal"),
    preference:  str   = Query("fastest"),
    session_id:  Optional[str] = Query(None, description="Session ID của người dùng"),
):
    """
    Tracking GPS thời gian thực:
      1. Snap vị trí GPS → node gần nhất (cấp Tòa)
      2. A* từ node đó đến đích
      3. Cảnh báo geofencing và hiển thị pop-up tự động
      4. Gợi ý AI dọc đường (thích ứng thụ động)
      5. Ghi nhận lịch sử ghé thăm (bán kính < 25m) và tự động suy luận hồ sơ
    """
    if end not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node đích không tồn tại: '{end}'")

    nearest = min(G.nodes(), key=lambda n: haversine(current_lat, current_lon, *G.nodes[n]["gps"]))
    dist_to_nearest = haversine(current_lat, current_lon, *G.nodes[nearest]["gps"])

    # Geofencing
    alerts = geofencing_logic(G, current_lat, current_lon, radius=25.0)

    # Lấy profile phiên tương ứng
    profile = get_session_profile(session_id)

    # Ghi lịch sử ghé thăm và tự động suy luận lại hồ sơ
    if dist_to_nearest < 25.0:
        ntype = G.nodes[nearest].get("type", "")
        if ntype in ("building", "admin") and nearest not in ("ATM", "Nhà điều hành"):
            hist = profile.setdefault("visited_history", {})
            hist[nearest] = hist.get(nearest, 0) + 1
            update_profile_passively(profile)

    # Đã đến nơi
    if nearest == end and dist_to_nearest < 5:
        return {
            "status":          "arrived",
            "message":         "Bạn đã đến nơi!",
            "geofence_alerts": alerts,
        }

    now  = get_current_time_str()
    path, dest_open = pathfinding_optimizer(G, nearest, end, weather, now, preference=preference)
    if not path:
        raise HTTPException(status_code=404, detail="Không tìm thấy đường đi!")

    total_remaining = calc_remaining_distance(G, path, dist_to_nearest)
    path_coords     = [{"node": n, "gps": G.nodes[n]["gps"]} for n in path]

    # Gợi ý AI dọc đường
    route_suggestions = get_smart_recommendations(
        G, current_lat, current_lon,
        destination=end,
        weather=weather,
        current_time_str=now,
        limit=5,
        user_profile=profile,
    )

    return {
        "status":                  "tracking",
        "snroutered_node":            nearest,
        "dist_to_node_m":          round(dist_to_nearest, 2),
        "total_remaining_meters":  round(total_remaining, 2),
        "estimated_mins":          round(total_remaining / WALKING_SPEED_MPM, 1),
        "path":                    path,
        "path_coords":             path_coords,
        "dest_open":               dest_open,
        "geofence_alerts":         alerts,
        "route_suggestions":       route_suggestions,
        "routing_engine":          "A* + GNN-GAT" if _gnn_ready else "A*",
    }


# ===========================================================================
# RECOMMENDATION — Đề xuất thông minh
# ===========================================================================

@router.get("/api_recommend", tags=["Recommendation"])
def smart_recommend(
    current_lat: float          = Query(..., description="Vĩ độ GPS"),
    current_lon: float          = Query(..., description="Kinh độ GPS"),
    destination: Optional[str]  = Query(None, description="Node đích (nếu đang di chuyển)"),
    query:       Optional[str]  = Query(None, description="Nhu cầu tự nhiên: 'an trua', 'hoc bai'..."),
    weather:     str            = Query("normal", description="normal | sunny | rainy"),
    interests:   Optional[str]  = Query(None, description="Sở thích bổ sung, phân tách bởi dấu phẩy"),
    limit:       int            = Query(6, ge=1, le=12),
    session_id:  Optional[str]  = Query(None, description="Session ID của người dùng"),
):
    """
    Đề xuất địa điểm thông minh kết hợp crowd + GPS + khung giờ + cá nhân hóa thụ động.
    NCF đã bị gỡ bỏ hoàn toàn.
    """
    if destination and destination not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node đích không tồn tại: '{destination}'")

    now            = get_current_time_str()
    user_interests = [i.strip() for i in interests.split(",") if i.strip()] if interests else None

    profile = get_session_profile(session_id)
    items = get_smart_recommendations(
        G, current_lat, current_lon,
        destination=destination,
        query=query,
        weather=weather,
        current_time_str=now,
        user_interests=user_interests,
        limit=limit,
        user_profile=profile,
    )
    for it in items:
        b_profile = get_building_profile(G, it["node"])
        it["image_url"] = b_profile.get("image_url", "")
        
    return {
        "status":          "success",
        "current_time":    now,
        "user_profile":    {k: v for k, v in profile.items() if k != "visited_history"},
        "recommendations": items,
        "ncf_active":      False,
    }


@router.get("/api_search", tags=["Recommendation"])
def search_semantic(
    query:   str = Query(..., min_length=1, description="Câu hỏi tìm kiếm tự nhiên"),
    weather: str = Query("normal"),
):
    """
    Tìm kiếm địa điểm bằng ngôn ngữ tự nhiên
    """
    now = get_current_time_str()

    node = find_node_by_keyword(G, query)
    if node:
        gps = G.nodes[node]["gps"]
        return {
            "status": "success", "matched_node": node,
            "is_open": is_node_open(G, node, now),
            "method": "Keyword Match",
            "gps": {"lat": gps[0], "lon": gps[1]},
        }

    linked = semantic_map_linking(G, query)
    if linked:
        n = linked["node"]
        return {
            "status": "success", "matched_node": n,
            "is_open": is_node_open(G, n, now),
            "score": linked["confidence"] * 100,
            "method": "Semantic Map Linking",
            "gps": linked["gps"],
            "alternatives": linked.get("alternatives", []),
        }

    by_func = recommend_by_building_function(G, query, now, limit=5)
    if by_func:
        top = by_func[0]
        return {
            "status": "success", "matched_node": top["node"],
            "is_open": True, "score": top["score"],
            "method": "Building Function Match",
            "gps": {"lat": G.nodes[top["node"]]["gps"][0], "lon": G.nodes[top["node"]]["gps"][1]},
            "recommendations": by_func,
        }

    ranked = recommend_locations(G, query, now, weather, limit=5)
    if ranked:
        top = ranked[0]
        return {
            "status": "success", "matched_node": top["node"],
            "is_open": True, "score": top["score"],
            "method": f"AI Semantic ({top['score']}/100)",
            "recommendations": ranked,
        }

    return {"status": "error", "message": "Không tìm thấy địa điểm phù hợp."}


@router.get("/api_crowd", tags=["Recommendation"])
def api_crowd(node: str = Query(...)):
    """Dự báo mật độ đám đông tại một địa điểm."""
    if node not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node}'")
    now = get_current_time_str()
    return {"status": "success", **crowd_prediction(G, node, now)}


@router.post("/api_crowd_report", tags=["Recommendation"])
def api_crowd_report(
    node:  str   = Query(...),
    level: float = Query(..., ge=0.0, le=1.0, description="0=vắng, 1=rất đông"),
):
    """Crowdsourcing: người dùng báo cáo mật độ thực tế."""
    if node not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node}'")
    return {"status": "success", **submit_crowd_report(node, level)}


# ===========================================================================
# TRAINING — Huấn luyện lại mô hình AI
# ===========================================================================

@router.post("/api_train_models", tags=["Training"])
def api_train_models(
    session_id: Optional[str] = Query(None, description="Session ID của người dùng"),
):
    """
    Huấn luyện lại các mô hình AI:
      1. IntentClassifier
      2. CrowdPredictor
    """
    try:
        from ai.recommendation_system.engine.train_models import train_all
        from ai.recommendation_system.engine.nlp_processor import load_intent_model
        profile = get_session_profile(session_id)
        stats = train_all(profile)
        load_intent_model()
        load_crowd_model()
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================================================
# RATINGS & COMMENTS
# ===========================================================================

from ai.recommendation_system.engine import storage

@router.post("/api_submit_rating", tags=["Recommendation"])
def submit_rating(
    node_id:    str = Query(..., description="Tên địa điểm cần đánh giá"),
    rating:     float = Query(..., ge=1.0, le=5.0, description="Đánh giá từ 1 đến 5 sao"),
    session_id: Optional[str] = Query(None, description="Session ID của người dùng"),
):
    """
    Nhận đánh giá sao (explicit rating).
    """
    if node_id not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node_id}'")

    profile = get_session_profile(session_id)
    ratings = profile.setdefault("ratings", {})
    ratings[node_id] = float(rating)
    
    return {
        "status": "success",
        "message": f"Đã ghi nhận đánh giá {rating} sao cho {node_id}.",
        "ratings": ratings
    }

@router.get("/api_get_comments", tags=["Recommendation"])
def api_get_comments(node_id: str = Query(..., description="Tên địa điểm")):
    """Lấy danh sách bình luận của một địa điểm."""
    if node_id not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node_id}'")
    storage.init_db()
    comments = storage.get_comments(node_id)
    return {
        "status": "success", 
        "comments": [
            {"session_id": c["session_id"], "content": c["content"], "timestamp": c["created_at"]} 
            for c in comments
        ]
    }

@router.post("/api_submit_comment", tags=["Recommendation"])
def api_submit_comment(
    node_id: str = Query(..., description="Tên địa điểm"),
    content: str = Query(..., max_length=180, description="Nội dung bình luận"),
    session_id: Optional[str] = Query(None, description="Session ID của người dùng"),
):
    """Gửi bình luận mới cho một địa điểm."""
    if node_id not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node_id}'")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Bình luận không được để trống")
    if not session_id or session_id in ("null", "undefined", ""):
        session_id = "default"
        
    storage.init_db()
    storage.save_comment(node_id, session_id, content.strip())
    return {"status": "success", "message": f"Đã gửi bình luận cho {node_id}."}


@router.get("/api_get_node_stats", tags=["Recommendation"])
def api_get_node_stats(
    node_id: str = Query(..., description="Tên địa điểm"),
    session_id: Optional[str] = Query(None, description="Session ID của người dùng")
):
    if node_id not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node_id}'")
        
    storage.init_db()
    # Lấy comment
    comments = storage.get_comments(node_id)
    comments_count = len(comments)
    
    # Lấy like
    likes_count, has_liked = storage.get_likes_info(node_id, session_id)
    
    # Lấy ratings
    profiles = storage.get_all_profiles()
    ratings = [p["ratings"][node_id] for p in profiles.values() if "ratings" in p and node_id in p["ratings"]]
    ratings_count = len(ratings)
    
    return {
        "status": "success",
        "comments_count": comments_count,
        "likes_count": likes_count,
        "has_liked": has_liked,
        "ratings_count": ratings_count
    }


@router.post("/api_toggle_like", tags=["Recommendation"])
def api_toggle_like(
    node_id: str = Query(..., description="Tên địa điểm"),
    session_id: Optional[str] = Query(None, description="Session ID của người dùng")
):
    if node_id not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node_id}'")
    if not session_id or session_id in ("null", "undefined", ""):
        session_id = "default"
        
    storage.init_db()
    has_liked = storage.toggle_like(node_id, session_id)
    
    return {
        "status": "success",
        "has_liked": has_liked
    }



# ===========================================================================
# WEB UI — Giao diện bản đồ cá nhân hóa AI
# ===========================================================================

def _opts_html() -> str:
    return "\n".join(
        f'<option value="{n}">{n.replace("_", " ")}</option>'
        for n in list_nodes
    )


def _bearing_hint(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    import math
    d = math.radians(lon2 - lon1)
    y = math.sin(d) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(d))
    deg  = (math.degrees(math.atan2(y, x)) + 360) % 360
    dirs = ["Bắc", "Đông Bắc", "Đông", "Đông Nam", "Nam", "Tây Nam", "Tây", "Tây Bắc"]
    return dirs[int((deg + 22.5) / 45) % 8]


@router.get("/", response_class=HTMLResponse, tags=["UI"])
def web_ui():
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    opts = _opts_html()
    b = _bounds
    
    html_content = html_content.replace("__OPTS__", opts)
    html_content = html_content.replace("__MIN_X__", str(b["min_x"]))
    html_content = html_content.replace("__MAX_X__", str(b["max_x"]))
    html_content = html_content.replace("__MIN_Y__", str(b["min_y"]))
    html_content = html_content.replace("__MAX_Y__", str(b["max_y"]))
    
    return html_content

@router.get("/api_get_nodes")
async def api_get_nodes():
    """API trả về danh sách toàn bộ các địa điểm (Nodes) trên bản đồ cho Frontend"""
    if not list_nodes:
        raise HTTPException(status_code=500, detail="Chưa khởi tạo được đồ thị")
    return {"nodes": list_nodes}