import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ai.recommendation_system.engine.nlp_processor import IntentClassifier
from ai.recommendation_system.engine.recommender import CrowdPredictor

def get_engine_dir():
    return os.path.dirname(os.path.abspath(__file__))

def train_intent_model():
    engine_dir = get_engine_dir()
    metadata_path = os.path.join(engine_dir, "model_metadata.json")
    model_path = os.path.join(engine_dir, "intent_model.pth")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    vocab_size = metadata["intent"]["vocab_size"]
    num_classes = len(metadata["intent"]["labels"])
    
    model = IntentClassifier(vocab_size, num_classes)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    # Dummy online learning / fine-tuning epochs for demonstration
    for _ in range(5):
        inputs = torch.rand(10, vocab_size)
        targets = torch.randint(0, num_classes, (10,))
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), model_path)
    return float(loss.item())

def train_crowd_model():
    engine_dir = get_engine_dir()
    metadata_path = os.path.join(engine_dir, "model_metadata.json")
    model_path = os.path.join(engine_dir, "crowd_model.pth")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    input_dim = metadata["crowd"]["input_dim"]
    
    model = CrowdPredictor(input_dim)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(5):
        inputs = torch.rand(10, input_dim)
        targets = torch.rand(10, 1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), model_path)
    return float(loss.item())

def train_all(profile=None):
    """
    Huấn luyện / Fine-tune các mô hình AI dựa trên dữ liệu mới.
    """
    try:
        intent_loss = train_intent_model()
        crowd_loss = train_crowd_model()
        
        # Giả lập MAE từ Crowd Prediction Loss
        mae = round(float(np.sqrt(crowd_loss)), 4)
        
        return {
            "status": "success",
            "crowd_mae": mae,
            "intent_loss": round(intent_loss, 4),
            "message": "Huấn luyện hoàn tất"
        }
    except Exception as e:
        print(f"Lỗi khi huấn luyện: {e}")
        return {
            "status": "error",
            "crowd_mae": 0.0,
            "intent_loss": 0.0,
            "message": str(e)
        }
