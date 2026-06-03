# engine/gnn_engine.py
"""
Campus Graph Neural Network — embedding, attention, inductive learning, RAG context.
Dùng PyTorch Geometric (GAT) trên đồ thị campus; không cần train offline nặng.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv

from engine.nlp_processor import normalize_text
from engine.utils import haversine

# ---------------------------------------------------------------------------
# Kiến trúc GNN
# ---------------------------------------------------------------------------
class GraphAttentionLayer(torch.nn.Module):
    """GAT layer — gán trọng số ưu tiên cho từng cạnh (tuyến huyết mạch)."""

    def __init__(self, in_channels: int, out_channels: int, heads: int = 2):
        super().__init__()
        self.gat = GATConv(
            in_channels, out_channels, heads=heads, concat=False, dropout=0.0,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.gat(x, edge_index)


class CampusGNN(torch.nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 16, embed_dim: int = 12):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=2, concat=True, dropout=0.0)
        self.gat2 = GATConv(hidden * 2, embed_dim, heads=1, concat=False, dropout=0.0)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x


# ---------------------------------------------------------------------------
# Engine singleton — giữ embedding & attention theo đồ thị hiện tại
# ---------------------------------------------------------------------------
class CampusAIEngine:
    """Quản lý GNN, embedding node, attention cạnh, học inductive."""

    EMBED_DIM = 12

    def __init__(self, G: nx.Graph):
        self.G = G
        self._sync_nodes()
        self.model = CampusGNN(in_dim=8, hidden=16, embed_dim=self.EMBED_DIM)
        self.model.eval()
        self._embeddings: np.ndarray = np.zeros((len(self.node_list), self.EMBED_DIM))
        self.edge_attention: Dict[Tuple[str, str], float] = {}
        self._recompute()

    def _sync_nodes(self) -> None:
        self.node_list = sorted(self.G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}

    def _node_feature_vector(self, node: str) -> List[float]:
        d = self.G.nodes[node]
        f = d.get("features", {})
        t = d.get("type", "building")
        type_map = {"building": 0, "facility": 1, "admin": 2}
        return [
            float(f.get("has_ac", 0)),
            float(f.get("has_tables", 0)),
            float(f.get("noise_level", 0.5)),
            min(float(f.get("capacity", 0)) / 1000.0, 1.0),
            1.0 if t == "building" else 0.0,
            1.0 if t == "facility" else 0.0,
            1.0 if t == "admin" else 0.0,
            float(d.get("restricted", False) or type_map.get(t) == 2),
        ]

    def _to_pyg(self) -> Tuple[Tensor, Tensor]:
        x = torch.tensor(
            [self._node_feature_vector(n) for n in self.node_list],
            dtype=torch.float32,
        )
        edges: List[List[int]] = []
        for u, v in self.G.edges():
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            edges.append([i, j])
            edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return x, edge_index

    def _recompute(self) -> None:
        if not self.node_list:
            return
        x, edge_index = self._to_pyg()
        with torch.no_grad():
            emb = self.model(x, edge_index).numpy()
        self._embeddings = emb
        self._compute_edge_attention()

    def _compute_edge_attention(self) -> None:
        """Trọng số attention cho mỗi cạnh — cao = tuyến huyết mạch."""
        self.edge_attention.clear()
        for u, v, data in self.G.edges(data=True):
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            ei, ej = self._embeddings[i], self._embeddings[j]
            norm = (np.linalg.norm(ei) * np.linalg.norm(ej)) or 1e-8
            sim = float(np.dot(ei, ej) / norm)
            backbone = 1.35 if data.get("has_roof") and data.get("status") == "open" else 1.0
            score = max(0.15, (sim + 1.0) / 2.0 * backbone)
            self.edge_attention[(u, v)] = round(score, 4)
            self.edge_attention[(v, u)] = round(score, 4)

    def refresh(self) -> None:
        """Gọi sau dynamic_edge_update hoặc inductive_learning."""
        self._sync_nodes()
        self._recompute()

    def get_embedding(self, node: str) -> List[float]:
        idx = self.node_to_idx.get(node)
        if idx is None:
            return [0.0] * self.EMBED_DIM
        return self._embeddings[idx].tolist()

    def inductive_add_node(
        self,
        node_id: str,
        gps: Tuple[float, float],
        *,
        node_type: str = "building",
        features: Optional[dict] = None,
        aliases: Optional[List[str]] = None,
        open_time: str = "06:00",
        close_time: str = "18:00",
        restricted: bool = False,
        connect_to: Optional[List[str]] = None,
    ) -> bool:
        """
        Thêm địa điểm mới — embedding = trung bình láng giềng (không train lại GNN).
        """
        if node_id in self.G:
            return False

        features = features or {
            "has_ac": 1, "has_tables": 0, "noise_level": 0.5, "capacity": 50,
        }
        aliases = aliases or [normalize_text(node_id)]

        self.G.add_node(
            node_id,
            pos=(gps[1], gps[0]),
            gps=gps,
            type=node_type,
            features=features,
            open_time=open_time,
            close_time=close_time,
            aliases=aliases,
            restricted=restricted,
        )

        if connect_to:
            for nb in connect_to:
                if nb in self.G and nb != node_id:
                    lat1, lon1 = self.G.nodes[node_id]["gps"]
                    lat2, lon2 = self.G.nodes[nb]["gps"]
                    w = round(haversine(lat1, lon1, lat2, lon2), 2)
                    self.G.add_edge(
                        node_id, nb,
                        weight=w, has_roof=False, status="open",
                    )

        self._sync_nodes()
        self._recompute()
        return True


# Global engine — khởi tạo lazy
_engine: Optional[CampusAIEngine] = None


def get_campus_engine(G: nx.Graph) -> CampusAIEngine:
    global _engine
    if _engine is None or _engine.G is not G:
        _engine = CampusAIEngine(G)
    return _engine


def invalidate_engine() -> None:
    global _engine
    _engine = None


# ---------------------------------------------------------------------------
# Public API — đúng tên task spec
# ---------------------------------------------------------------------------
def gnn_node_embedding(G: nx.Graph) -> Dict[str, List[float]]:
    """Tạo vector đặc trưng GNN cho từng địa điểm."""
    engine = get_campus_engine(G)
    return {n: engine.get_embedding(n) for n in G.nodes()}


def graph_attention_layer(G: nx.Graph) -> Dict[str, float]:
    """
    Gán trọng số ưu tiên cho các tuyến đường (cạnh).
    Trả về dạng 'u|v': score.
    """
    engine = get_campus_engine(G)
    return {f"{u}|{v}": w for (u, v), w in engine.edge_attention.items() if u < v}


def inductive_learning(
    G: nx.Graph,
    node_id: str,
    gps: Tuple[float, float],
    connect_to: List[str],
    **kwargs,
) -> dict:
    """Cập nhật địa điểm mới vào đồ thị mà không cần train lại toàn bộ."""
    engine = get_campus_engine(G)
    ok = engine.inductive_add_node(node_id, gps, connect_to=connect_to, **kwargs)
    emb = gnn_node_embedding(G).get(node_id, [])
    return {
        "success": ok,
        "node": node_id,
        "embedding_dim": len(emb),
        "embedding_preview": emb[:4],
        "message": "Đã thêm node (inductive)" if ok else "Node đã tồn tại",
    }


def gnn_edge_cost(
    G: nx.Graph,
    u: str,
    v: str,
    weather: str,
    current_time: str,
    crowd_fn,
) -> float:
    """Chi phí cạnh cho pathfinding — GNN attention + tránh nắng/đông."""
    data = G[u][v]
    if data.get("status") in ("repairing", "closed"):
        return 999_999.0

    base = float(data.get("weight", 1.0))
    engine = get_campus_engine(G)
    attn = engine.edge_attention.get((u, v), 1.0)
    attn = max(attn, 0.2)

    crowd = (crowd_fn(G, u, current_time) + crowd_fn(G, v, current_time)) / 2.0
    crowd_mul = 1.0 + crowd * 2.5

    weather_mul = 5.0 if weather in ("sunny", "rainy") and not data.get("has_roof") else 1.0

    return base * weather_mul * crowd_mul / attn


def graph_rag_context(
    G: nx.Graph,
    current_time: str,
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
    crowd_fn=None,
) -> dict:
    """
    Ngữ cảnh cấu trúc campus cho Member 2 (RAG / LLM downstream).
    """
    from engine.recommender import predict_crowd_level

    crowd_fn = crowd_fn or predict_crowd_level
    engine = get_campus_engine(G)
    embeddings = gnn_node_embedding(G)
    attention = graph_attention_layer(G)

    nodes_ctx = []
    for n, d in G.nodes(data=True):
        crowd = crowd_fn(G, n, current_time)
        entry = {
            "id": n,
            "type": d.get("type"),
            "gps": d.get("gps"),
            "aliases": d.get("aliases", []),
            "features": d.get("features", {}),
            "open_hours": f"{d.get('open_time')}-{d.get('close_time')}",
            "restricted": bool(d.get("restricted") or d.get("type") == "admin"),
            "crowd_level": round(crowd, 2),
            "embedding_dim": len(embeddings.get(n, [])),
        }
        if user_lat is not None and user_lon is not None:
            entry["distance_m"] = round(haversine(user_lat, user_lon, *d["gps"]), 1)
        nodes_ctx.append(entry)

    edges_ctx = [
        {
            "from": u, "to": v,
            "weight_m": d.get("weight"),
            "has_roof": d.get("has_roof"),
            "status": d.get("status"),
            "attention": attention.get(f"{u}|{v}") or attention.get(f"{min(u,v)}|{max(u,v)}"),
        }
        for u, v, d in G.edges(data=True)
        if u < v
    ]

    backbone = sorted(
        [(k, v) for k, v in attention.items()],
        key=lambda x: -x[1],
    )[:8]

    return {
        "campus_summary": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "current_time": current_time,
        },
        "nodes": nodes_ctx,
        "edges": edges_ctx,
        "backbone_routes": [{"edge": k, "priority": v} for k, v in backbone],
        "gnn_embedding_sample": {
            n: embeddings[n][:4] for n in list(G.nodes())[:3]
        },
    }
