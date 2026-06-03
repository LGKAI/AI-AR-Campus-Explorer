import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from core.config import db
from routers.users import get_current_user
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/history", tags=["History"])

class LocationEntry(BaseModel):
    start_node: str
    end_node: str

class SearchEntry(BaseModel):
    query: str
    matched_node: Optional[str] = None

@router.post("/location")
async def save_location_history(entry: LocationEntry, current_user: dict = Depends(get_current_user)):
    """Lưu lịch sử chỉ đường vào Firebase"""
    try:
        user_id = current_user.get("UserID")
        doc_ref = db.collection("LOCATION_HISTORY").document()
        doc_ref.set({
            "HistoryID": doc_ref.id,
            "UserID": user_id,
            "StartNode": entry.start_node,
            "EndNode": entry.end_node,
            "Timestamp": datetime.utcnow().isoformat()
        })
        return {"status": "success", "message": "Đã lưu lịch sử di chuyển"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def save_search_history(entry: SearchEntry, current_user: dict = Depends(get_current_user)):
    """Lưu lịch sử tìm kiếm vào Firebase"""
    try:
        user_id = current_user.get("UserID")
        doc_ref = db.collection("SEARCH_HISTORY").document()
        doc_ref.set({
            "HistoryID": doc_ref.id,
            "UserID": user_id,
            "Query": entry.query,
            "MatchedNode": entry.matched_node,
            "Timestamp": datetime.utcnow().isoformat()
        })
        return {"status": "success", "message": "Đã lưu lịch sử tìm kiếm"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user")
async def get_user_history(current_user: dict = Depends(get_current_user)):
    """Lấy toàn bộ lịch sử (location & search) của user"""
    try:
        user_id = current_user.get("UserID")
        
        # Lấy lịch sử di chuyển
        loc_query = db.collection("LOCATION_HISTORY").where("UserID", "==", user_id).get()
        locations = [doc.to_dict() for doc in loc_query]
        
        # Lấy lịch sử tìm kiếm
        search_query = db.collection("SEARCH_HISTORY").where("UserID", "==", user_id).get()
        searches = [doc.to_dict() for doc in search_query]
        
        # Sắp xếp theo Timestamp mới nhất
        locations.sort(key=lambda x: x.get("Timestamp", ""), reverse=True)
        searches.sort(key=lambda x: x.get("Timestamp", ""), reverse=True)
        
        return {
            "status": "success",
            "locations": locations[:20], # Giới hạn 20 lịch sử gần nhất
            "searches": searches[:20]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
