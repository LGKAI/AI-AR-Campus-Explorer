from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

@router.websocket("/ws/ar-stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Bóc tách Data Flow: Xác định lệnh là Chat (RAG) hay Vector đồ họa (GNN/DUSt3R)
            action = data.get("action")
            
            if action == "chat":
                # Chuyển payload sang module Chatbot của Bảo Kin
                response = {"type": "chat_reply", "message": f"AI Assistant đang xử lý: {data.get('payload')}"}
                await websocket.send_json(response)
                
            elif action == "ar_nav":
                # Chuyển toạ độ sang GNN (Chấn Khoa) và CV (Trung Hiếu)
                response = {"type": "ar_vector", "waypoints": []} 
                await websocket.send_json(response)
                
            else:
                await websocket.send_json({"error": "Hành động không hợp lệ"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)