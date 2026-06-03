# engine/graph_builder_v2.py
import networkx as nx
import matplotlib.pyplot as plt
from engine.building_catalog import _BUILDING_PROFILES
from engine.campus_knowledge import CAMPUS_LINH_TRUNG, get_cluster_for_node
from engine.utils import haversine


def build_campus_graph() -> nx.Graph:
    """
    Xây dựng đồ thị campus (Node = tòa nhà/tiện ích, Edge = đường đi).
    Alias chính thức theo spec dự án.
    """
    return build_flat_campus_graph()


def build_flat_campus_graph() -> nx.Graph:
    G = nx.Graph()

    # Khung giờ mặc định
    DEFAULT_OPEN  = "06:00"
    DEFAULT_CLOSE = "18:00"
    REST_OPEN     = "11:30"
    REST_CLOSE    = "12:30"

    # ---------------------------------------------------------
    # 1. ĐỊNH NGHĨA NODE FLAT BUILDING-LEVEL
    # ---------------------------------------------------------
    nodes_data = {
        "Tòa A": {
            "gps": (10.877500, 106.797500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.53, "capacity": 300},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa a", "nha a", "sanh toa a", "tang 1 toa a", "phong thi nghiem a201", "lab a201", "thuc nghiem a201", "phong thi nghiem a301", "lab a301", "thuc nghiem a301"]
        },
        "Tòa B": {
            "gps": (10.877500, 106.798000), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.33, "capacity": 240},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa b", "nha b", "sanh toa b", "phong tu hoc b201", "tu hoc b201", "hoc nhom b201", "tu hoc yen tinh", "phong may b301", "lab b301", "thuc hanh b301", "phong lab b301"]
        },
        "Tòa C": {
            "gps": (10.877500, 106.798500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.37, "capacity": 200},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa c", "nha c", "sanh toa c", "lab may tinh 202", "phong may 202", "lab cntt", "may tinh", "phong thuc hanh may tinh", "van phong khoa", "vp khoa", "giao vu khoa"]
        },
        "Tòa D": {
            "gps": (10.878000, 106.798750), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.37, "capacity": 380},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa d", "nha d", "thu vien", "doc sach", "muon sach", "thu vien khtn", "cho doc sach", "quay giao trinh", "mua sach", "tiem sach"]
        },
        "Canteen": {
            "gps": (10.878050, 106.798700), "type": "facility",
            "features": {"has_ac": 0, "has_tables": 1, "noise_level": 0.65, "capacity": 300},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["can tin", "canteen", "an trua", "com can tin", "doi bung", "an uong", "tra sua", "an vat"]
        },
        "Tòa E": {
            "gps": (10.877500, 106.799000), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.3, "capacity": 170},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa e", "nha e", "phong hoc 101", "ly thuyet", "phong nghi trua", "cho ngu trua", "nghi trua"]
        },
        "Tòa F": {
            "gps": (10.877500, 106.799500), "type": "building",
            "features": {"has_ac": 1, "has_tables": 1, "noise_level": 0.2, "capacity": 100},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa f", "nha f", "phong nghi 102", "cho nga lung", "buon ngu", "met qua", "phong tu hoc f201", "tu hoc f201"]
        },
        "Tòa G": {
            "gps": (10.877500, 106.800000), "type": "building",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.5, "capacity": 200},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["toa g", "nha g", "san toa g"]
        },
        "Nhà thể dục": {
            "gps": (10.878700, 106.799250), "type": "building",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.8, "capacity": 1000},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["nha the duc", "gym", "the thao", "clb", "tap gym", "cau long", "bong ban"]
        },
        "Nhà xe": {
            "gps": (10.876300, 106.797500), "type": "facility",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.9, "capacity": 1000},
            "open_time": DEFAULT_OPEN, "close_time": DEFAULT_CLOSE,
            "aliases": ["bai giu xe", "parking", "gui xe", "lay xe", "cat xe", "xe may", "nha de xe"]
        },
        "Cây ATM": {
            "gps": (10.876800, 106.799000), "type": "facility",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.5, "capacity": 5},
            "open_time": "00:00", "close_time": "23:59",
            "aliases": ["cay atm", "rut tien", "het tien", "ngan hang", "tien mat"]
        },
        "Nhà điều hành": {
            "gps": (10.876100, 106.799200), "type": "admin",
            "features": {"has_ac": 1, "has_tables": 0, "noise_level": 0.1, "capacity": 100},
            "open_time": REST_OPEN, "close_time": REST_CLOSE,
            "restricted": True,
            "aliases": ["phong ban", "giao vu", "hanh chinh", "giay to", "dong hoc phi", "staff only"]
        },
        "Cổng trường": {
            "gps": (10.876000, 106.798500), "type": "facility",
            "features": {"has_ac": 0, "has_tables": 0, "noise_level": 0.6, "capacity": 200},
            "open_time": "00:00", "close_time": "23:59",
            "aliases": ["cong truong", "cong chinh", "cong", "entrance", "main gate"]
        },
    }

    # Thêm Node vào đồ thị
    for name, data in nodes_data.items():
        catalog = _BUILDING_PROFILES.get(name, {})
        indoor = data["type"] in ("building", "admin", "facility") and name not in (
            "Nhà xe", "Cổng trường", "Tòa G",
        )
        G.add_node(
            name,
            pos=(data["gps"][1], data["gps"][0]),  # X=Longitude, Y=Latitude
            gps=data["gps"],
            type=data["type"],
            features=data["features"],
            open_time=data["open_time"],
            close_time=data["close_time"],
            aliases=data["aliases"],
            restricted=data.get("restricted", data.get("type") == "admin"),
            tagline=catalog.get("tagline", ""),
            services=catalog.get("services", []),
            indoor=indoor,
            poi_cluster=get_cluster_for_node(name),
            campus_id=CAMPUS_LINH_TRUNG["id"],
        )

    # ---------------------------------------------------------
    # 2. ĐỊNH NGHĨA EDGE FLAT BUILDING-LEVEL
    # ---------------------------------------------------------
    edges = [
        # Đường hành lang có mái che (Horizontal line: A -> B -> C -> E -> F -> G)
        ("Tòa A",        "Tòa B",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa B",        "Tòa C",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa C",        "Tòa E",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa E",        "Tòa F",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa F",        "Tòa G",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        
        # Tam giác Canteen (C -> D -> E) - Có mái che
        ("Tòa C",        "Tòa D",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        ("Tòa D",        "Tòa E",          {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
        
        # Nhà thi đấu (Nhà thể dục) nối E và F (Không mái che)
        ("Tòa E",        "Nhà thể dục",    {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Tòa F",        "Nhà thể dục",    {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        
        # Cổng trường đến Nhà xe và Nhà điều hành (Không mái che)
        ("Cổng trường",  "Nhà xe",         {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Cổng trường",  "Nhà điều hành",  {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        
        # Nhà điều hành nối E và F (Không mái che)
        ("Nhà điều hành", "Tòa E",          {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Nhà điều hành", "Tòa F",          {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        
        # Liên kết Cây ATM (Gần Tòa C, D, Nhà xe)
        ("Tòa C",        "Cây ATM",            {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Tòa D",        "Cây ATM",            {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Nhà xe",       "Cây ATM",            {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Cây ATM",      "Cổng trường",        {"has_roof": False, "status": "open", "edge_type": "walkway"}),
        ("Tòa D",        "Canteen",            {"has_roof": True,  "status": "open", "edge_type": "corridor"}),
    ]

    for u, v, attr in edges:
        lat1, lon1 = G.nodes[u]["gps"]
        lat2, lon2 = G.nodes[v]["gps"]
        dist = round(haversine(lat1, lon1, lat2, lon2), 2)
        attr["weight"] = dist
        G.add_edge(u, v, **attr)

    return G


def get_canvas_bounds(G) -> dict:
    """
    Tính min/max tọa độ của đồ thị để frontend có thể scale động.
    """
    xs = [d["pos"][0] for _, d in G.nodes(data=True)]
    ys = [d["pos"][1] for _, d in G.nodes(data=True)]
    return {
        "min_x": min(xs), "max_x": max(xs),
        "min_y": min(ys), "max_y": max(ys),
    }


def visualize_flat_graph(G: nx.Graph):
    """Vẽ đồ thị dựa trên tọa độ GPS (dùng cho debug local)."""
    plt.figure(figsize=(10, 8))
    pos = nx.get_node_attributes(G, "pos")
    edges = G.edges(data=True)

    open_edges   = [(u, v) for u, v, d in edges if d["status"] == "open"]
    repair_edges = [(u, v) for u, v, d in edges if d["status"] == "repairing"]

    nx.draw_networkx_edges(G, pos, edgelist=open_edges,   width=2, edge_color="#888888")
    nx.draw_networkx_edges(G, pos, edgelist=repair_edges, width=2, edge_color="red", style="dashed")

    node_colors = []
    for _, data in G.nodes(data=True):
        if   data.get("type") == "facility": node_colors.append("lightgreen")
        elif data.get("type") == "admin":    node_colors.append("lightgrey")
        else:                                node_colors.append("skyblue")

    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, edgecolors="black")
    edge_labels = {(u, v): f"{d['weight']}m" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="red")
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

    plt.title("Bản đồ 2D AI AR Campus (GPS Thực tế)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    campus_graph = build_campus_graph()
    visualize_flat_graph(campus_graph)
