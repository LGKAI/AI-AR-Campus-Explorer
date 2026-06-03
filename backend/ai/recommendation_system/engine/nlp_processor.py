# engine/nlp_processor.py
"""
Bộ phân tích NLP và nhận dạng ý định bằng PyTorch.
Hỗ trợ phân loại ý định người dùng và trích xuất tham số chỉ đường.
"""
import re
import os
import json
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Optional, Dict

from engine.utils import MIN_KEYWORD_LENGTH

# ---------------------------------------------------------------------------
# Bảng ánh xạ ký tự tiếng Việt có dấu -> không dấu
# ---------------------------------------------------------------------------
_ACCENTED = (
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệđìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂễỆĐÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ"
)
_PLAIN = (
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyy"
    "AAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYY"
)
_TRANS_TABLE = str.maketrans(_ACCENTED, _PLAIN)


def remove_accents(text: str) -> str:
    """Chuyển tiếng Việt có dấu thành không dấu."""
    return text.translate(_TRANS_TABLE)


def normalize_text(text: str) -> str:
    """Chuẩn hóa chuỗi: chữ thường, bỏ dấu, bỏ ký tự đặc biệt."""
    text = text.lower().strip()
    text = remove_accents(text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def find_node_by_keyword(graph: nx.Graph, query: str) -> Optional[str]:
    """
    Tìm node phù hợp nhất từ câu hỏi của người dùng.
    """
    if not query:
        return None

    norm_query = normalize_text(query)

    # 1. Khớp tên node trực tiếp
    for node in graph.nodes():
        norm_node = normalize_text(node)
        if norm_node and norm_node in norm_query:
            return node

    # 2. Khớp qua alias
    for node, data in graph.nodes(data=True):
        for alias in data.get("aliases", []):
            norm_alias = normalize_text(alias)

            if norm_alias and norm_alias in norm_query:
                return node

            if (
                len(norm_query) >= MIN_KEYWORD_LENGTH
                and norm_query in norm_alias
            ):
                return node

    return None


# ---------------------------------------------------------------------------
# Cấu trúc mạng nơ-ron nhận dạng ý định (PyTorch)
# ---------------------------------------------------------------------------
class IntentClassifier(nn.Module):
    """Khớp với kiến trúc MLP 3 lớp trong train_models.py."""
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


_MODEL: Optional[IntentClassifier] = None
_VOCAB: Optional[list] = None
_LABELS: Optional[dict] = None


def load_intent_model():
    """Nạp trọng số và metadata của mô hình phân loại ý định."""
    global _MODEL, _VOCAB, _LABELS
    engine_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(engine_dir, "intent_model.pth")
    metadata_path = os.path.join(engine_dir, "model_metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print("⚠️ [NLP Processor] Không tìm thấy tệp mô hình PyTorch, sử dụng chế độ Rule-based.")
        return
        
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            _VOCAB = metadata["intent"]["vocab"]
            raw_labels = metadata["intent"]["labels"]
            _LABELS = {int(k): v for k, v in raw_labels.items()}
            
        vocab_size = len(_VOCAB)
        num_classes = len(_LABELS)
        
        _MODEL = IntentClassifier(vocab_size, num_classes)
        _MODEL.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        _MODEL.eval()
        print("✅ [NLP Processor] Đã nạp thành công mô hình PyTorch Intent Classifier.")
    except Exception as e:
        print(f"⚠️ [NLP Processor] Lỗi khi nạp mô hình: {e}. Sử dụng chế độ Rule-based.")
        _MODEL = None


def text_to_bow(text: str, vocab: list) -> np.ndarray:
    words = text.split()
    vector = np.zeros(len(vocab), dtype=np.float32)
    for w in words:
        if w in vocab:
            vector[vocab.index(w)] += 1.0
    return vector


def classify_intent_rule_based(query_norm: str) -> str:
    """Fallback phân loại bằng luật nếu chưa train mô hình."""
    if any(k in query_norm for k in ["di", "duong", "chi duong", "lo trinh", "huong dan", "toi", "den"]):
        return "route_search"
    if any(k in query_norm for k in ["trong", "may tinh", "lab", "tu hoc", "phong hoc", "ranh"]):
        return "search_empty_lab"
    if any(k in query_norm for k in ["an", "uong", "can tin", "canteen", "com", "doi", "vang"]):
        return "search_food_low_crowd"
    if any(k in query_norm for k in ["hoi thao", "su kien", "clb", "seminar"]):
        return "event_recommend"
    return "general_chat"


def classify_query(query: str) -> dict:
    """
    Phân tích câu hỏi của sinh viên bằng mô hình nơ-ron hoặc luật fallback.
    Trích xuất ý định (Intent), độ tin cậy (Confidence) và các tham số lộ trình đi kèm.
    """
    if not query:
        return {"intent": "general_chat", "confidence": 1.0, "preference": "fastest"}
        
    query_norm = normalize_text(query)
    intent_label = None
    confidence = 0.5
    
    # Suy luận bằng nơ-ron PyTorch
    if _MODEL is not None and _VOCAB is not None and _LABELS is not None:
        try:
            bow = text_to_bow(query_norm, _VOCAB)
            tensor = torch.tensor(np.array([bow]), dtype=torch.float32)
            with torch.no_grad():
                outputs = _MODEL(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                intent_label = _LABELS[pred_idx]
                confidence = round(probs[pred_idx].item(), 3)
        except Exception as e:
            print(f"⚠️ [NLP Inference Error] {e}")
            intent_label = None
            
    # Sử dụng luật nếu suy luận lỗi hoặc không có mô hình
    if not intent_label:
        intent_label = classify_intent_rule_based(query_norm)
        confidence = 0.8
        
    # Trích xuất tùy chọn đường đi (Preference)
    preference = "fastest"
    wheelchair_keys = ["xe lan", "thang may", "khong leo thang", "thang bo hong"]
    covered_keys = ["mai che", "mua", "nang", "tranh mua", "tranh nang", "co mai"]
    
    if any(k in query_norm for k in wheelchair_keys):
        preference = "wheelchair"
    elif any(k in query_norm for k in covered_keys):
        preference = "covered"
        
    return {
        "intent": intent_label,
        "confidence": confidence,
        "preference": preference,
        "query": query
    }

# Tự động nạp khi import
load_intent_model()
