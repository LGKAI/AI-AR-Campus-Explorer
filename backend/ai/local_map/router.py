from fastapi import APIRouter, HTTPException, Request
import json
import os
import traceback

router = APIRouter(prefix="/local_map", tags=["Local Map AR"])

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BUILDINGS_FILE = os.path.join(DATA_DIR, "buildings.json")
GRAPH_FILE = os.path.join(DATA_DIR, "nav_graph.json")

def load_json(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@router.get("/api/buildings")
async def get_buildings():
    try:
        data = load_json(BUILDINGS_FILE)
        return {"success": True, "buildings": data.get("buildings", [])}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@router.post("/api/buildings")
async def save_buildings(request: Request):
    try:
        data = await request.json()
        buildings = data.get("buildings", [])
        save_json(BUILDINGS_FILE, {"buildings": buildings})
        return {"success": True}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@router.get("/api/graph")
async def get_graph():
    try:
        data = load_json(GRAPH_FILE)
        return data
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}
