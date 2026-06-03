# api/main.py  — v5 (SQLite + Real CF + No Admin)
"""
AI AR Campus API — Pathfinding & Smart Recommender  v5
=======================================================
Nâng cấp v5:
  ✅ SQLite persistent storage — profile không mất khi restart
  ✅ Session TTL 48h + background cleanup định kỳ
  ✅ Real Item-Item CF — tích hợp từ engine/collaborative_filter.py
  ✅ CF model rebuild khi startup + incremental update sau mỗi visit
  ✅ Xóa các endpoint admin-only:
       - /api_calibrate_map    (định vị lại tọa độ tòa nhà)
       - /api_update_building_events  (quản lý sự kiện)
       - /api_update_edge      (đóng/mở đường đi)
       - /api_train_models     (train lại AI)
  ✅ Input validation cải thiện (max_length, sanitize)
  ✅ async def + ThreadPoolExecutor cho các route CPU-heavy
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Optional, List

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from engine.graph_builder_v2 import build_campus_graph, get_canvas_bounds
from engine.optimizer import (
    pathfinding_optimizer,
    multi_stop_routing,
    is_node_open,
    calc_remaining_distance,
    geofencing_logic,
    restricted_zone_alert,
)
from engine.nlp_processor import find_node_by_keyword
from engine.building_catalog import get_building_profile
from engine.recommender import (
    recommend_locations,
    recommend_by_building_function,
    get_smart_recommendations,
    semantic_map_linking,
    crowd_prediction,
    predict_crowd_level,
    submit_crowd_report,
    update_profile_passively,
    load_crowd_model,
    collaborative_filtering,
)
from engine.utils import haversine, get_current_time_str, WALKING_SPEED_MPM
from engine.persona_manager import PersonaManager
from engine.context_engine import _GUMBEL_TEMP
import engine.storage as storage
from engine.collaborative_filter import rebuild_cf_model, increment_cf_model

# ---------------------------------------------------------------------------
# ThreadPoolExecutor — xử lý các tác vụ CPU-bound
# ---------------------------------------------------------------------------
_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# Khởi tạo
# ---------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="AI AR Campus — Pathfinding & Smart Recommender",
    description=(
        "Tìm đường A* + GNN | Hệ thống gợi ý AI cá nhân hóa theo thời gian thực\n\n"
        "**v5**: SQLite persistent profiles · Real Item-Item CF · No admin endpoints"
    ),
    version="5.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

G          = build_campus_graph()
list_nodes = sorted(G.nodes())
_bounds    = get_canvas_bounds(G)


# ---------------------------------------------------------------------------
# Startup: khởi tạo DB + rebuild CF model
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    """Khởi tạo SQLite DB và rebuild CF model từ dữ liệu đã lưu."""
    loop = asyncio.get_event_loop()

    # 1. Khởi tạo SQLite
    await loop.run_in_executor(_EXECUTOR, storage.init_db)
    await loop.run_in_executor(_EXECUTOR, storage.delete_old_comments)

    # 2. Rebuild CF model từ tất cả profiles đã lưu
    all_profiles = await loop.run_in_executor(_EXECUTOR, storage.get_all_profiles)
    if all_profiles:
        await loop.run_in_executor(_EXECUTOR, partial(rebuild_cf_model, all_profiles))

    # 3. Khởi tạo GNN
    try:
        from engine.gnn_engine import gnn_node_embedding
        await loop.run_in_executor(_EXECUTOR, partial(gnn_node_embedding, G))
        print("✅ [API] GNN engine sẵn sàng.")
    except Exception as e:
        print(f"[WARN] GNN init: {e}")

    print(f"✅ [API] Server v5 sẵn sàng — {len(list_nodes)} nodes, "
          f"{len(all_profiles)} profiles loaded.")


# ---------------------------------------------------------------------------
# Background: cleanup session định kỳ (mỗi 1 giờ)
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def schedule_cleanup():
    """Dọn session hết TTL mỗi 1 giờ trong background."""
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(3600)  # 1 giờ
            try:
                loop = asyncio.get_event_loop()
                deleted = await loop.run_in_executor(
                    _EXECUTOR, storage.delete_expired_sessions
                )
                await loop.run_in_executor(_EXECUTOR, storage.delete_old_comments)
                if deleted:
                    # Sau cleanup, rebuild CF với data còn lại
                    all_profiles = await loop.run_in_executor(
                        _EXECUTOR, storage.get_all_profiles
                    )
                    await loop.run_in_executor(
                        _EXECUTOR, partial(rebuild_cf_model, all_profiles)
                    )
            except Exception as e:
                print(f"[WARN] Cleanup error: {e}")

    asyncio.create_task(_cleanup_loop())


# ---------------------------------------------------------------------------
# GNN readiness flag
# ---------------------------------------------------------------------------
_gnn_ready = False
try:
    from engine.gnn_engine import gnn_node_embedding
    gnn_node_embedding(G)
    _gnn_ready = True
except Exception:
    pass


# ===========================================================================
# HELPERS
# ===========================================================================

def _default_profile() -> dict:
    return {
        "role":            "student",
        "study_style":     "silent",
        "interests":       ["hoc_tap"],
        "visited_history": {},
        "schedule_class":  "Tòa B",
        "behavior_log":    {},
        "search_history":  [],
        "ratings":         {},
        "_last_tracking":  {},
    }


def get_session_profile(session_id: Optional[str] = None) -> dict:
    """
    Lấy profile từ SQLite (+ RAM cache).
    Tạo profile mặc định nếu chưa có.
    Cập nhật thụ động (role, interests inference) mỗi lần load.
    """
    if not session_id or session_id in ("null", "undefined", ""):
        session_id = "default"

    profile = storage.load_profile(session_id)
    if profile is None:
        profile = _default_profile()

    # Đảm bảo tất cả fields tồn tại
    for key, default_val in _default_profile().items():
        profile.setdefault(key, default_val)

    update_profile_passively(profile)
    storage.save_profile(session_id, profile)
    return profile


def _save_session(session_id: str, profile: dict) -> None:
    """Lưu profile vào SQLite và cập nhật CF model incremental."""
    if not session_id or session_id in ("null", "undefined", ""):
        session_id = "default"
    storage.save_profile(session_id, profile)
    # Cập nhật CF incremental (không cần rebuild toàn bộ)
    increment_cf_model(profile)


def _sanitize_query(q: str, max_length: int = 200) -> str:
    """Giới hạn độ dài và loại bỏ ký tự nguy hiểm khỏi query string."""
    q = q[:max_length].strip()
    # Loại bỏ các ký tự đặc biệt có thể gây injection
    q = q.replace("<", "").replace(">", "").replace("\"", "").replace("'", "")
    return q


def _bearing_hint(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    import math
    d = math.radians(lon2 - lon1)
    y = math.sin(d) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(d))
    deg  = (math.degrees(math.atan2(y, x)) + 360) % 360
    dirs = ["Bắc", "Đông Bắc", "Đông", "Đông Nam", "Nam", "Tây Nam", "Tây", "Tây Bắc"]
    return dirs[int((deg + 22.5) / 45) % 8]


def _opts_html() -> str:
    return "\n".join(
        f'<option value="{n}">{n.replace("_", " ")}</option>'
        for n in list_nodes
    )


# ===========================================================================
# USER PROFILE (Cá nhân hóa thụ động)
# ===========================================================================

@app.get("/api_user_profile", tags=["Profile"])
def get_user_profile(
    session_id: Optional[str] = Query(None, description="Session ID của người dùng"),
):
    """
    Lấy hồ sơ người dùng (được cập nhật thụ động dựa trên lịch sử ghé thăm).
    Profile được lưu persistent — không mất khi server restart.
    """
    profile = get_session_profile(session_id)
    # Ẩn _last_tracking khỏi response (internal field)
    visible = {k: v for k, v in profile.items() if not k.startswith("_")}
    return {"status": "success", "user_profile": visible}


@app.post("/api_user_profile", tags=["Profile"])
def update_user_profile(
    reset_history: bool = Query(False, description="Xóa lịch sử ghé thăm"),
    session_id:    Optional[str] = Query(None, description="Session ID của người dùng"),
):
    """
    Đặt lại hồ sơ người dùng.
    Cá nhân hóa diễn ra thụ động — không cần cài đặt thủ công.
    """
    if not session_id or session_id in ("null", "undefined", ""):
        session_id = "default"

    if reset_history:
        profile = _default_profile()
        storage.save_profile(session_id, profile)
    else:
        profile = get_session_profile(session_id)
        update_profile_passively(profile)
        storage.save_profile(session_id, profile)

    visible = {k: v for k, v in profile.items() if not k.startswith("_")}
    return {"status": "success", "user_profile": visible}


@app.post("/api_submit_rating", tags=["Profile"])
def submit_rating(
    node_id:    str = Query(..., description="Tên địa điểm cần đánh giá"),
    rating:     float = Query(..., ge=1.0, le=5.0, description="Đánh giá từ 1 đến 5 sao"),
    session_id: Optional[str] = Query(None, description="Session ID của người dùng"),
):
    """
    Nhận đánh giá sao (explicit rating) từ giao diện TikTok Feed của người dùng.
    Ghi nhận vào SQLite và cập nhật tăng dần Collaborative Filtering.
    """
    if node_id not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node_id}'")

    if not session_id or session_id in ("null", "undefined", ""):
        session_id = "default"

    profile = get_session_profile(session_id)
    ratings = profile.setdefault("ratings", {})
    ratings[node_id] = float(rating)

    # Lưu profile và cập nhật CF model
    _save_session(session_id, profile)

    return {
        "status": "success",
        "message": f"Đã ghi nhận đánh giá {rating} sao cho {node_id}.",
        "ratings": ratings
    }


@app.get("/api_get_comments", tags=["Comments"])
def api_get_comments(node_id: str = Query(..., description="Tên địa điểm")):
    """Lấy danh sách bình luận của một địa điểm."""
    if node_id not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node_id}'")
    comments = storage.get_comments(node_id)
    return {"status": "success", "comments": comments}


@app.post("/api_submit_comment", tags=["Comments"])
def api_submit_comment(
    node_id: str = Query(..., description="Tên địa điểm"),
    content: str = Query(..., max_length=180, description="Nội dung bình luận"),
    session_id: Optional[str] = Query(None, description="Session ID của người dùng"),
):
    """Gửi bình luận mới cho một địa điểm."""
    if node_id not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node_id}'")
    content = _sanitize_query(content, 180)
    if not content.strip():
        raise HTTPException(status_code=400, detail="Bình luận không được để trống")
    if not session_id or session_id in ("null", "undefined", ""):
        session_id = "default"
        
    storage.save_comment(node_id, session_id, content.strip())
    return {"status": "success", "message": f"Đã gửi bình luận cho {node_id}."}


@app.get("/api_trending_locations", tags=["Recommender"])
def api_trending_locations():
    """Tính toán và trả về danh sách 13 địa điểm sắp xếp theo lượt đánh giá cao (Trending)."""
    profiles = storage.get_all_profiles()
    nodes = list(G.nodes)
    
    # Khởi tạo điểm số rating cơ sở (default fallback)
    default_ratings = {
        "Căn tin": 4.5,
        "Thư viện Tòa D": 4.2,
        "Tòa B": 4.0,
        "Tòa A": 3.8,
        "Tòa E": 3.7,
        "Tòa C": 3.6,
        "Tòa F": 3.5,
        "Nhà thể dục": 3.4,
        "ATM": 3.3,
        "Nhà điều hành": 3.2,
        "Cổng trường": 3.1,
        "Tòa G": 3.0,
        "Nhà xe": 2.8
    }
    
    rating_sums = {n: 0.0 for n in nodes}
    rating_counts = {n: 0.0 for n in nodes}
    
    for n, r in default_ratings.items():
        if n in rating_sums:
            rating_sums[n] += r * 5.0  # Giả lập 5 lượt rate ban đầu để tránh cold-start
            rating_counts[n] += 5.0
            
    for pid, profile in profiles.items():
        ratings = profile.get("ratings", {})
        for node, val in ratings.items():
            if node in rating_sums:
                rating_sums[node] += float(val)
                rating_counts[node] += 1.0
                
    trending = []
    for n in nodes:
        avg = rating_sums[n] / rating_counts[n] if rating_counts[n] > 0 else 3.0
        trending.append({
            "node": n,
            "avg_rating": round(avg, 2),
            "total_ratings": int(rating_counts[n])
        })
        
    trending.sort(key=lambda x: (x["avg_rating"], x["total_ratings"]), reverse=True)
    return {"status": "success", "trending": trending}


# ===========================================================================
# BUILDING CATALOG
# ===========================================================================

@app.get("/api_get_building_profile", tags=["Building Catalog"])
def api_get_building_profile(
    building_name: str = Query(..., description="Tên tòa nhà"),
):
    """Lấy hồ sơ chi tiết (sự kiện, phòng ban, dịch vụ) của một tòa nhà."""
    if building_name not in G.nodes:
        raise HTTPException(status_code=404, detail="Tòa nhà không tồn tại")
    profile = get_building_profile(G, building_name)
    return {"status": "success", "profile": profile}


# ===========================================================================
# PATHFINDING — Tìm đường
# ===========================================================================

@app.get("/api_get_graph", tags=["Pathfinding"])
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
        })
    edges = [
        {
            "source":    u,
            "target":    v,
            "status":    d["status"],
            "has_roof":  d["has_roof"],
            "weight_m":  d.get("weight", 0),
            "edge_type": d.get("edge_type", "walkway"),
        }
        for u, v, d in G.edges(data=True)
    ]
    return {
        "nodes":          nodes,
        "edges":          edges,
        "current_time":   now,
        "bounds":         _bounds,
        "routing_engine": "A* + GNN-GAT" if _gnn_ready else "A*",
    }


@app.get("/api_get_route", tags=["Pathfinding"])
def get_route(
    waypoints:  str = Query(..., description="Danh sách node cách nhau bởi dấu phẩy"),
    weather:    str = Query("normal",  description="normal | sunny | rainy"),
    preference: str = Query("fastest", description="fastest | covered | wheelchair"),
):
    """
    Lập lộ trình đa điểm dùng A* + GNN attention.
    Hỗ trợ 3 chế độ: nhanh nhất, có mái che, xe lăn.
    """
    pts = [p.strip() for p in waypoints.split(",") if p.strip()]
    if len(pts) < 2:
        raise HTTPException(status_code=400, detail="Cần ít nhất 2 điểm (start, end).")
    if len(pts) > 10:
        raise HTTPException(status_code=400, detail="Tối đa 10 điểm dừng.")

    invalid = [p for p in pts if p not in G.nodes]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: {invalid}")

    if weather not in ("normal", "sunny", "rainy"):
        raise HTTPException(status_code=400, detail="weather phải là normal | sunny | rainy")
    if preference not in ("fastest", "covered", "wheelchair"):
        raise HTTPException(status_code=400, detail="preference phải là fastest | covered | wheelchair")

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


# ===========================================================================
# REAL-TIME TRACKING — Theo dõi vị trí + tìm đường động
# ===========================================================================

@app.get("/api_realtime_tracking", tags=["Tracking"])
async def realtime_tracking(
    current_lat: float = Query(..., description="Vĩ độ GPS hiện tại"),
    current_lon: float = Query(..., description="Kinh độ GPS hiện tại"),
    end:         str   = Query(..., description="Node đích"),
    weather:     str   = Query("normal"),
    preference:  str   = Query("fastest"),
    query:       Optional[str] = Query(None, max_length=200, description="Nhu cầu từ khóa"),
    session_id:  Optional[str] = Query(None, description="Session ID của người dùng"),
    battery_level: Optional[float] = Query(None, ge=0.0, le=1.0, description="Mức pin (0-1)"),
    temperature: Optional[float] = Query(None, description="Nhiệt độ môi trường (°C)"),
    uv_index:    Optional[float] = Query(None, ge=0.0, description="Chỉ số UV"),
    schedule_class: Optional[str] = Query(None, description="Tên tòa nhà lớp học tiếp theo"),
):
    """
    Tracking GPS thời gian thực (async — v5):
      1. Snap GPS → node gần nhất
      2. A* từ node đó đến đích
      3. Geofencing + pop-up tự động
      4. Gợi ý AI dọc đường (thích ứng + CF)
      5. Ghi lịch sử qua PersonaManager → cập nhật CF incremental
      6. Lưu profile vào SQLite sau mỗi tracking update
    """
    if end not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node đích không tồn tại: '{end}'")
    if weather not in ("normal", "sunny", "rainy"):
        raise HTTPException(status_code=400, detail="weather phải là normal | sunny | rainy")

    loop = asyncio.get_event_loop()
    if not session_id or session_id in ("null", "undefined", ""):
        session_id = "default"

    if query:
        query = _sanitize_query(query)

    # ── Vectorized nearest node ────────────────────────────────────────────
    from engine.context_engine import all_node_distances
    node_dists = await loop.run_in_executor(
        _EXECUTOR,
        partial(all_node_distances, G, current_lat, current_lon)
    )
    nearest         = min(node_dists, key=lambda n: node_dists[n])
    dist_to_nearest = node_dists[nearest]

    # Geofencing
    alerts = geofencing_logic(G, current_lat, current_lon, radius=25.0)

    # Lấy profile từ SQLite
    profile = get_session_profile(session_id)

    # ── Ghi lịch sử ghé thăm qua PersonaManager ──────────────────────────
    import time
    pm       = PersonaManager(profile)
    now_ts   = time.time()
    last_tk  = profile.setdefault("_last_tracking", {})

    is_same_node    = (last_tk.get("node") == nearest)
    time_delta_mins = 0.0
    if is_same_node and "timestamp" in last_tk:
        delta_sec = now_ts - last_tk["timestamp"]
        if delta_sec < 300.0:
            time_delta_mins = delta_sec / 60.0

    last_tk["node"]      = nearest
    last_tk["timestamp"] = now_ts

    profile_updated = False
    if dist_to_nearest < 25.0:
        ntype = G.nodes[nearest].get("type", "")
        if ntype in ("building", "admin") and nearest not in ("ATM", "Nhà điều hành"):
            is_intentional = (
                nearest == end
                or (schedule_class and nearest == schedule_class)
                or any(
                    nearest in sh.get("query", "") or sh.get("destination_node") == nearest
                    for sh in profile.get("search_history", [])[-3:]
                )
            )
            pm.record_visit(
                node=nearest,
                time_delta_mins=time_delta_mins,
                is_new_visit=(not is_same_node),
                is_intentional=is_intentional,
            )
            pm.update_passively()
            profile_updated = True

    # Lưu profile sau khi cập nhật
    if profile_updated:
        await loop.run_in_executor(_EXECUTOR, partial(_save_session, session_id, profile))

    # Đã đến nơi
    if nearest == end and dist_to_nearest < 5:
        return {
            "status":          "arrived",
            "message":         "Bạn đã đến nơi!",
            "geofence_alerts": alerts,
        }

    now = get_current_time_str()

    # ── A* pathfinding ────────────────────────────────────────────────────
    path, dest_open = await loop.run_in_executor(
        _EXECUTOR,
        partial(pathfinding_optimizer, G, nearest, end, weather, now, preference)
    )
    if not path:
        raise HTTPException(status_code=404, detail="Không tìm thấy đường đi!")

    total_remaining = calc_remaining_distance(G, path, dist_to_nearest)
    path_coords     = [{"node": n, "gps": G.nodes[n]["gps"]} for n in path]

    # ── Gợi ý AI dọc đường (Smart + CF) ─────────────────────────────────
    route_suggestions = await loop.run_in_executor(
        _EXECUTOR,
        partial(
            get_smart_recommendations,
            G, current_lat, current_lon,
            end, query, weather, now, None, 5,
            profile, battery_level, temperature, uv_index,
            schedule_class or profile.get("schedule_class"),
        )
    )

    return {
        "status":                 "tracking",
        "snapped_node":           nearest,
        "dist_to_node_m":         round(dist_to_nearest, 2),
        "total_remaining_meters": round(total_remaining, 2),
        "estimated_mins":         round(total_remaining / WALKING_SPEED_MPM, 1),
        "path":                   path,
        "path_coords":            path_coords,
        "dest_open":              dest_open,
        "geofence_alerts":        alerts,
        "route_suggestions":      route_suggestions,
        "routing_engine":         "A* + GNN-GAT" if _gnn_ready else "A*",
    }


# ===========================================================================
# RECOMMENDATION — Đề xuất thông minh
# ===========================================================================

@app.get("/api_recommend", tags=["Recommendation"])
async def smart_recommend(
    current_lat: float         = Query(..., description="Vĩ độ GPS"),
    current_lon: float         = Query(..., description="Kinh độ GPS"),
    destination: Optional[str] = Query(None, description="Node đích (nếu đang di chuyển)"),
    query:       Optional[str] = Query(None, max_length=200, description="Nhu cầu tự nhiên: 'an trua', 'hoc bai'..."),
    weather:     str           = Query("normal", description="normal | sunny | rainy"),
    interests:   Optional[str] = Query(None, description="Sở thích bổ sung, phân tách bởi dấu phẩy"),
    limit:       int           = Query(6, ge=1, le=12),
    session_id:  Optional[str] = Query(None, description="Session ID của người dùng"),
    battery_level: Optional[float] = Query(None, ge=0.0, le=1.0, description="Mức pin (0-1)"),
    temperature: Optional[float] = Query(None, description="Nhiệt độ môi trường (°C)"),
    uv_index:    Optional[float] = Query(None, ge=0.0, description="Chỉ số UV"),
    schedule_class: Optional[str] = Query(None, description="Tên tòa nhà lớp học tiếp theo"),
):
    """
    Đề xuất địa điểm thông minh (async — v5).

    Tích hợp đa nguồn:
      - TF-IDF semantic matching (query)
      - Rule-based (thời tiết, thời gian, pin, UV)
      - Personalization (role, study_style, interests)
      - Item-Item CF (hành vi người dùng tương tự)
      - GNN attention (cấu trúc đồ thị campus)
      - Gumbel-Softmax + ε-greedy (exploration)

    Response chứa 2 bucket:
      - familiar_recommendations: địa điểm quen thuộc
      - discovery_recommendations: địa điểm mới (Serendipity)
    """
    if destination and destination not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node đích không tồn tại: '{destination}'")
    if weather not in ("normal", "sunny", "rainy"):
        raise HTTPException(status_code=400, detail="weather phải là normal | sunny | rainy")

    loop           = asyncio.get_event_loop()
    now            = get_current_time_str()
    user_interests = [i.strip() for i in interests.split(",") if i.strip()] if interests else None

    # Sanitize query
    if query:
        query = _sanitize_query(query)

    profile = get_session_profile(session_id)

    # ── Smart Recommendations (CPU-bound) ──────────────────────────────────
    items = await loop.run_in_executor(
        _EXECUTOR,
        partial(
            get_smart_recommendations,
            G, current_lat, current_lon,
            destination, query, weather, now, user_interests, limit,
            profile, battery_level, temperature, uv_index,
            schedule_class or profile.get("schedule_class"),
        )
    )

    # ── Tách 2 bucket ─────────────────────────────────────────────────────
    familiar  = [it for it in items if it.get("bucket") == "familiar"]
    discovery = [it for it in items if it.get("bucket") == "discovery"]

    # Lưu profile sau khi cập nhật thụ động
    if not session_id or session_id in ("null", "undefined", ""):
        session_id = "default"
    await loop.run_in_executor(_EXECUTOR, partial(storage.save_profile, session_id, profile))

    return {
        "status":                    "success",
        "current_time":              now,
        "user_profile":              {k: v for k, v in profile.items()
                                      if not k.startswith("_")},
        "recommendations":           items,
        "familiar_recommendations":  familiar,
        "discovery_recommendations": discovery,
        "cf_active":                 True,
        "engine_version":            "v5",
    }


@app.get("/api_collaborative_filtering", tags=["Recommendation"])
async def api_collaborative_filtering(
    session_id:  Optional[str] = Query(None, description="Session ID"),
    current_lat: Optional[float] = Query(None, description="Vĩ độ GPS (tuỳ chọn)"),
    current_lon: Optional[float] = Query(None, description="Kinh độ GPS (tuỳ chọn)"),
    top_k:       int = Query(8, ge=1, le=20, description="Số kết quả"),
):
    """
    Gợi ý thuần CF (Item-Item Collaborative Filtering).

    Trả về các địa điểm mà người dùng có hành vi tương tự hay ghé thăm.
    Cold-start fallback: nếu chưa đủ data CF, trả về content-based theo interests.

    Hữu ích cho trang 'Khám phá địa điểm mới' hoặc onboarding.
    """
    loop    = asyncio.get_event_loop()
    profile = get_session_profile(session_id)

    from engine.collaborative_filter import get_cf_model
    cf = get_cf_model()

    results = await loop.run_in_executor(
        _EXECUTOR,
        partial(collaborative_filtering, G, profile, current_lat, current_lon, top_k)
    )

    return {
        "status":          "success",
        "cf_mode":         "item_item_cf" if cf.has_data() else "content_based_cold_start",
        "cf_sessions":     cf.n_sessions,
        "recommendations": results,
        "engine_version":  "v5",
    }


@app.get("/api_search", tags=["Recommendation"])
async def search_semantic(
    query:      str = Query(..., min_length=1, max_length=200, description="Câu hỏi tìm kiếm tự nhiên"),
    weather:    str = Query("normal"),
    session_id: Optional[str] = Query(None, description="Session ID của người dùng"),
):
    """
    Tìm kiếm địa điểm bằng ngôn ngữ tự nhiên (async — v5).
    Query tự động mở rộng qua bảng Synonym tiếng Việt (ContextEngine).
    Lịch sử tìm kiếm được lưu persistent vào SQLite.
    """
    query = _sanitize_query(query)
    if not query:
        raise HTTPException(status_code=400, detail="Query không hợp lệ.")

    loop = asyncio.get_event_loop()
    now  = await loop.run_in_executor(_EXECUTOR, get_current_time_str)
    node = find_node_by_keyword(G, query)

    # Ghi lịch sử tìm kiếm
    if not session_id or session_id in ("null", "undefined", ""):
        session_id = "default"
    profile     = get_session_profile(session_id)
    search_hist = profile.setdefault("search_history", [])

    resolved_node = node
    if not resolved_node:
        linked = semantic_map_linking(G, query)
        if linked:
            resolved_node = linked["node"]
        else:
            by_func = await loop.run_in_executor(
                _EXECUTOR,
                partial(recommend_by_building_function, G, query, now, 1)
            )
            if by_func:
                resolved_node = by_func[0]["node"]

    search_hist.append({
        "query":            query,
        "timestamp":        datetime.now().isoformat(),
        "destination_node": resolved_node,
    })
    if len(search_hist) > 20:  # v5: tăng từ 10 → 20
        search_hist.pop(0)

    await loop.run_in_executor(
        _EXECUTOR, partial(storage.save_profile, session_id, profile)
    )

    # Trả kết quả theo thứ tự ưu tiên: exact → semantic → function → TF-IDF
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

    by_func = await loop.run_in_executor(
        _EXECUTOR,
        partial(recommend_by_building_function, G, query, now, 5)
    )
    if by_func:
        top = by_func[0]
        return {
            "status": "success", "matched_node": top["node"],
            "is_open": True, "score": top["score"],
            "method": "Building Function Match",
            "gps": {"lat": G.nodes[top["node"]]["gps"][0], "lon": G.nodes[top["node"]]["gps"][1]},
            "recommendations": by_func,
        }

    ranked = await loop.run_in_executor(
        _EXECUTOR,
        partial(recommend_locations, G, query, now, weather, 5)
    )
    if ranked:
        top = ranked[0]
        return {
            "status": "success", "matched_node": top["node"],
            "is_open": True, "score": top["score"],
            "method": f"AI Semantic ({top['score']}/100)",
            "recommendations": ranked,
        }

    return {"status": "error", "message": "Không tìm thấy địa điểm phù hợp."}


@app.get("/api_crowd", tags=["Recommendation"])
def api_crowd(node: str = Query(...)):
    """Dự báo mật độ đám đông tại một địa điểm."""
    if node not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node}'")
    now = get_current_time_str()
    return {"status": "success", **crowd_prediction(G, node, now)}


@app.post("/api_crowd_report", tags=["Recommendation"])
def api_crowd_report(
    node:  str   = Query(...),
    level: float = Query(..., ge=0.0, le=1.0, description="0=vắng, 1=rất đông"),
):
    """
    Crowdsourcing: người dùng báo cáo mật độ thực tế.
    Dữ liệu live override sẽ ưu tiên hơn dự báo của model.
    """
    if node not in G.nodes:
        raise HTTPException(status_code=400, detail=f"Node không tồn tại: '{node}'")
    return {"status": "success", **submit_crowd_report(node, level)}


# ===========================================================================
# CF MODEL STATUS — Debug/Monitoring
# ===========================================================================

@app.get("/api_cf_stats", tags=["Recommendation"])
def api_cf_stats():
    """
    Trạng thái CF model: số session đã indexed, số item pairs, v.v.
    Hữu ích để monitor khi nào CF có đủ data (>= 3 sessions).
    """
    from engine.collaborative_filter import get_cf_model
    cf = get_cf_model()
    return {
        "status": "success",
        "cf_stats": cf.debug_stats(),
        "engine_version": "v5",
    }


# ===========================================================================
# WEB UI — Giao diện bản đồ
# ===========================================================================

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def web_ui():
    import os
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    opts = _opts_html()
    b    = _bounds

    html_content = html_content.replace("__OPTS__",  opts)
    html_content = html_content.replace("__MIN_X__", str(b["min_x"]))
    html_content = html_content.replace("__MAX_X__", str(b["max_x"]))
    html_content = html_content.replace("__MIN_Y__", str(b["min_y"]))
    html_content = html_content.replace("__MAX_Y__", str(b["max_y"]))

    return html_content
