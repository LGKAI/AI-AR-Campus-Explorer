import sys
import os
from typing import Optional

# Lấy đường dẫn trỏ tới thư mục 'backend/ai/information_chatbot'
current_router_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_router_dir)
kin_root_dir = os.path.join(backend_dir, "ai", "information_chatbot")
kin_src_dir = os.path.join(kin_root_dir, "src")

# Thêm vào sys.path
if kin_src_dir not in sys.path:
    sys.path.insert(0, kin_src_dir)

from fastapi import APIRouter, HTTPException, Query
from ai.information_chatbot.src.agent import Agent

router = APIRouter(prefix="/chat", tags=["Information Chatbot"])

# Khởi tạo một global agent (Singleton pattern)
_agent_instance = None

def get_agent():
    global _agent_instance
    if _agent_instance is None:
        try:
            _agent_instance = Agent()
        except Exception as e:
            print(f"[WARN] Lỗi khởi tạo RAG Agent: {e}")
            return None
    return _agent_instance

@router.get("/query", tags=["Chat"])
def ask_chatbot(q: str = Query(..., description="Câu hỏi dành cho chatbot")):
    """
    Trả lời câu hỏi bằng STELLAR-RAG.
    """
    agent = get_agent()
    if not agent:
        raise HTTPException(status_code=500, detail="Chatbot AI đang khởi động hoặc gặp lỗi.")
    
    # Dual mode hay single mode tuỳ vào hàm answer. Ở đây dùng answer() thông thường
    try:
        ans, turn_id = agent.answer(q)
        return {
            "status": "success",
            "answer": ans,
            "turn_id": turn_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
