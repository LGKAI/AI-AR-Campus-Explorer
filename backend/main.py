from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import users, locations, ws, recommendation, chat, history
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="AI AR Campus Explorer Gateway")

# Cho phép Web App (Frontend) gọi API chéo tên miền
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình phục vụ file tĩnh cho Recommendation System (Hình ảnh các địa điểm)
static_path = os.path.join(os.path.dirname(__file__), "ai", "recommendation_system", "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

app.include_router(users.router)
app.include_router(locations.router)
app.include_router(ws.router)
app.include_router(recommendation.router)
app.include_router(chat.router)
app.include_router(history.router)

@app.get("/")
async def root():
    return {"status": "ok", "message": "FastAPI Gateway is running smoothly"}