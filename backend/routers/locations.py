from fastapi import APIRouter
from core.config import db
from models.schemas import Location
from typing import List

router = APIRouter(prefix="/locations", tags=["Locations"])

@router.get("/", response_model=List[Location])
async def get_locations():
    locations_ref = db.collection("LOCATION")
    docs = locations_ref.get()
    return [Location(**doc.to_dict()) for doc in docs]