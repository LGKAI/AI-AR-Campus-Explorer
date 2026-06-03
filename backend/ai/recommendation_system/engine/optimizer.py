# engine/optimizer.py
import networkx as nx
import math
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from engine.utils import parse_time, get_current_time_str


def is_node_open(G: nx.Graph, node_id: str, current_time_str: str = None) -> bool:
    """
    Kiểm tra Node có đang trong giờ mở cửa không.
    Dùng datetime.time để so sánh an toàn, tránh lỗi chuỗi qua nửa đêm.

    Args:
        current_time_str: Định dạng "HH:MM". Nếu None sẽ lấy giờ hệ thống.
    """
    node_data = G.nodes[node_id]
    open_t_str  = node_data.get("open_time")
    close_t_str = node_data.get("close_time")

    # Không có giờ quy định -> mặc định mở
    if not open_t_str or not close_t_str:
        return True

    if current_time_str is None:
        current_time_str = get_current_time_str()

    open_t    = parse_time(open_t_str)
    close_t   = parse_time(close_t_str)
    current_t = parse_time(current_time_str)

    # Nếu parse thất bại -> mặc định coi là mở
    if open_t is None or close_t is None or current_t is None:
        return True

    # Xử lý trường hợp qua nửa đêm (VD: 22:00 -> 06:00)
    if open_t <= close_t:
        return open_t <= current_t <= close_t
    else:
        return current_t >= open_t or current_t <= close_t


def heuristic(node1: str, node2: str, pos: dict) -> float:
    """
    Hàm h(n): Khoảng cách Euclid giữa 2 điểm — dùng làm heuristic cho A*.
    Giúp A* có "trực giác" đi về hướng đích thay vì tìm kiếm mù.
    """
    x1, y1 = pos[node1]
    x2, y2 = pos[node2]
    return math.hypot(x2 - x1, y2 - y1)


def pathfinding_optimizer(
    G: nx.Graph,
    start_node: str,
    end_node: str,
    weather: str = "normal",
    current_time: str = None,
    use_gnn: bool = True,
    preference: str = "fastest",
) -> Tuple[Optional[List], bool]:
    """
    A* tìm đường tối ưu — tích hợp GNN attention, tránh nắng/mưa, né khu đông.
    Hỗ trợ lộ trình ưu tiên: nhanh nhất, mái che, xe lăn.
    """
    for node in (start_node, end_node):
        if node not in G.nodes:
            return None, False

    if current_time is None:
        current_time = get_current_time_str()

    pos = nx.get_node_attributes(G, "pos")
    dest_open = is_node_open(G, end_node, current_time)

    crowd_fn = None
    if use_gnn:
        try:
            from engine.gnn_engine import gnn_edge_cost
            from engine.recommender import predict_crowd_level
            crowd_fn = predict_crowd_level
        except ImportError:
            use_gnn = False

    def custom_weight(u, v, edge_data):
        if edge_data.get("status") in ("repairing", "closed"):
            return 999_999

        # Lấy trọng số vật lý
        base = edge_data.get("weight", 1.0)
        edge_type = edge_data.get("edge_type", "walkway")

        # 1. Ưu tiên thời tiết (sunny / rainy)
        weather_penalty = 1.0
        if weather in ("sunny", "rainy") and not edge_data.get("has_roof"):
            weather_penalty = 3.0

        # 2. Xử lý tùy chọn lộ trình (preference)
        pref_cost = base
        if preference == "wheelchair":
            if edge_type == "stairs":
                return 999_999  # Xe lăn không đi được cầu thang bộ
            elif edge_type == "elevator":
                pref_cost = base * 0.4  # Ưu tiên đi thang máy
            elif edge_type == "bridge":
                pref_cost = base * 0.8  # Cầu nối bằng phẳng ổn
        elif preference == "covered":
            if not edge_data.get("has_roof"):
                pref_cost = base * 15.0  # Phạt cực nặng nếu đi đường ngoài trời
        elif preference == "fastest":
            if edge_type == "stairs":
                pref_cost = base * 1.4  # Đi thang bộ chậm và mệt hơn
            elif edge_type == "elevator":
                pref_cost = base * 0.8  # Thang máy nhanh hơn

        # 3. Kết hợp độ đông đúc (GNN hoặc fallback)
        crowd_penalty = 1.0
        if use_gnn and crowd_fn:
            try:
                gnn_w = gnn_edge_cost(G, u, v, weather, current_time, crowd_fn)
                crowd_penalty = gnn_w / max(1.0, base)
            except Exception:
                pass
        else:
            try:
                from engine.recommender import predict_crowd_level
                cr_u = predict_crowd_level(G, u, current_time)
                cr_v = predict_crowd_level(G, v, current_time)
                avg_crowd = (cr_u + cr_v) / 2.0
                if avg_crowd > 0.75:
                    crowd_penalty = 2.0
            except Exception:
                pass

        return pref_cost * weather_penalty * crowd_penalty

    try:
        path = nx.astar_path(
            G,
            source=start_node,
            target=end_node,
            heuristic=lambda u, v: heuristic(u, v, pos),
            weight=custom_weight,
        )
        return path, dest_open
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, False


def calc_remaining_distance(G: nx.Graph, path: list, user_dist_to_first_node: float) -> float:
    """
    Tính khoảng cách còn lại từ vị trí hiện tại (gần nearest_node) đến đích.

    Args:
        path: Danh sách node từ nearest_node đến đích (kết quả của A*).
        user_dist_to_first_node: Khoảng cách Haversine từ GPS user đến node đầu path.

    Returns:
        Tổng khoảng cách còn lại (mét).
    """
    edge_dist = sum(
        G[path[i]][path[i + 1]].get("weight", 0)
        for i in range(len(path) - 1)
    )
    return user_dist_to_first_node + edge_dist


def visualize_path(G: nx.Graph, path: list, title: str = "Đường đi tối ưu"):
    """Vẽ đường đi (màu xanh lá) lên bản đồ — dùng để debug local."""
    plt.figure(figsize=(10, 6))
    pos = nx.get_node_attributes(G, "pos")

    open_edges   = [(u, v) for u, v, d in G.edges(data=True) if d["status"] == "open"]
    repair_edges = [(u, v) for u, v, d in G.edges(data=True) if d["status"] == "repairing"]

    nx.draw_networkx_edges(G, pos, edgelist=open_edges,   width=2, edge_color="#E0E0E0")
    nx.draw_networkx_edges(G, pos, edgelist=repair_edges, width=2, edge_color="red",
                           style="dashed", alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightgray", edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=10)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4, edge_color="green")
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=800,
                               node_color="lightgreen", edgecolors="black")
        nx.draw_networkx_labels(G, pos, labels={n: n for n in path},
                                font_size=10, font_weight="bold")

    plt.title(title)
    plt.axis("off")
    plt.show()


def multi_stop_routing(
    G: nx.Graph,
    waypoints: List[str],
    weather: str = "normal",
    current_time: str = None,
    preference: str = "fastest",
) -> Tuple[Optional[List], bool]:
    """
    Lập lộ trình qua nhiều điểm dừng (Waypoints) có kèm theo ưu tiên đường đi.

    Returns:
        (full_path, all_open): Đường đi đầy đủ và trạng thái mở cửa của tất cả điểm.
        Trả về (None, False) nếu bất kỳ chặng nào không có đường.
    """
    if len(waypoints) < 2:
        return None, True

    # Validate tất cả waypoints tồn tại trước khi chạy
    invalid = [w for w in waypoints if w not in G.nodes]
    if invalid:
        return None, False

    full_path = []
    all_open  = True

    for i in range(len(waypoints) - 1):
        segment_path, is_open = pathfinding_optimizer(
            G, waypoints[i], waypoints[i + 1], weather, current_time, preference=preference
        )

        if not segment_path:
            return None, False

        if not is_open:
            all_open = False

        # Nối các chặng: chặng đầu lấy toàn bộ, các chặng sau bỏ node đầu (tránh trùng)
        full_path.extend(segment_path if i == 0 else segment_path[1:])

    return full_path, all_open

# ===========================================================================
# NHÓM 1: GEOFENCING & DYNAMIC SPACE LOGIC
# ===========================================================================
from engine.utils import haversine

def dynamic_edge_update(G, u: str, v: str, new_status: str) -> bool:
    """
    Cập nhật trạng thái đường đi theo thời gian thực (sửa chữa / đóng / mở).
    new_status: 'open', 'repairing', 'closed'
    """
    updated = False
    if G.has_edge(u, v):
        G[u][v]["status"] = new_status
        updated = True
    if G.has_edge(v, u):
        G[v][u]["status"] = new_status
        updated = True
    if updated:
        try:
            from engine.gnn_engine import get_campus_engine
            get_campus_engine(G).refresh()
        except Exception:
            pass
    return updated


def restricted_zone_alert(
    G: nx.Graph,
    current_lat: float,
    current_lon: float,
    radius: float = 30.0,
) -> List[dict]:
    """Cảnh báo AR nếu sinh viên đi vào khu vực không phận sự."""
    alerts = []
    for node, data in G.nodes(data=True):
        if "gps" not in data:
            continue
        dist = haversine(current_lat, current_lon, *data["gps"])
        if dist > radius:
            continue

        restricted = data.get("restricted", False) or data.get("type") == "admin"
        aliases = " ".join(data.get("aliases", [])).lower()
        if restricted or "cam" in aliases or "khong phan su" in aliases:
            alerts.append({
                "level": "danger",
                "node": node,
                "distance_m": round(dist, 1),
                "type": "restricted_zone",
                "msg": (
                    f"🚨 CẢNH BÁO: Bạn đang trong khu vực hạn chế ({node}) — "
                    "không phận sự sinh viên. Vui lòng rời khỏi ngay."
                ),
            })
    return alerts


def geofencing_logic(G, current_lat: float, current_lon: float, radius: float = 25.0) -> list:
    """
    Kích hoạt thông báo AR và tự động pop-up khi bước vào vùng (bán kính < 25m).
    """
    alerts = list(restricted_zone_alert(G, current_lat, current_lon, radius=radius))
    restricted_nodes = {a["node"] for a in alerts}

    from engine.building_catalog import get_building_profile
    from engine.recommender import predict_crowd_level
    from engine.utils import get_current_time_str

    for node, data in G.nodes(data=True):
        if "gps" not in data or node in restricted_nodes:
            continue

        dist = haversine(current_lat, current_lon, *data["gps"])
        if dist >= radius:
            continue

        # Lấy thông tin tòa nhà
        profile = get_building_profile(G, node)
        current_time = get_current_time_str()
        crowd_val = predict_crowd_level(G, node, current_time)
        
        # Mật độ đám đông
        if crowd_val >= 0.85:
            crowd_status = "Rất đông"
        elif crowd_val >= 0.6:
            crowd_status = "Đông vừa"
        elif crowd_val >= 0.35:
            crowd_status = "Bình thường"
        else:
            crowd_status = "Vắng"

        aliases = " ".join(data.get("aliases", [])).lower()
        if "cong" in aliases or "cổng" in aliases:
            alerts.append({
                "level": "success",
                "node": node,
                "distance_m": round(dist, 1),
                "type": "welcome",
                "msg": f"👋 Chào mừng bạn đã đến {node}. Chúc một ngày học tập hiệu quả!",
            })
        else:
            # Thêm thông báo pop-up tự động
            alerts.append({
                "level": "info",
                "node": node,
                "distance_m": round(dist, 1),
                "type": "popup",
                "tagline": profile.get("tagline", ""),
                "function_summary": profile.get("function_summary", ""),
                "services": profile.get("services", []),
                "departments": profile.get("departments", []),
                "events": profile.get("events", []),
                "crowd_level": round(crowd_val, 2),
                "crowd_status": crowd_status,
                "msg": f"📍 Bạn đang ở gần {node}. {profile.get('tagline', '')} ({crowd_status})",
            })
    return alerts

# ---------------------------------------------------------------------------
# Chạy thử trực tiếp
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from engine.graph_builder_v2 import build_flat_campus_graph

    campus_graph = build_flat_campus_graph()
    waypoints    = ["Nhà xe", "ATM", "Tòa F"]
    test_time    = "14:00"

    print(f"--- Đang lập lộ trình: {' ➔ '.join(waypoints)} ---")
    print(f"Giờ hiện tại (Giả lập): {test_time}")

    lo_trinh, tat_ca_deu_mo = multi_stop_routing(
        campus_graph, waypoints, weather="normal", current_time=test_time
    )

    if lo_trinh:
        print(f"✅ Lộ trình: {lo_trinh}")
        if not tat_ca_deu_mo:
            print(f"⚠️  CẢNH BÁO: Ít nhất một điểm đang ĐÓNG CỬA lúc {test_time}!")
        visualize_path(campus_graph, lo_trinh, "Lộ trình: Nhà xe → ATM → Tòa F")
    else:
        print("❌ Không tìm được lộ trình qua tất cả các điểm này.")
