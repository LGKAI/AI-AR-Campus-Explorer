import cv2
import numpy as np
import base64
import os
from .detector import FaceGuard
from .database import FaceDB

# Đảm bảo đường dẫn tới file pkl luôn đúng dù chạy lệnh uvicorn từ đâu
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(CURRENT_DIR, "face_db.pkl")

# Khởi tạo mô hình (Chỉ load 1 lần vào RAM khi khởi động server)
fg = FaceGuard()
db = FaceDB(db_path=DB_PATH)

def decode_base64_image(base64_str: str):
    """Chuyển đổi chuỗi Base64 từ Web thành ảnh OpenCV (numpy array)"""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def register_face(base64_str: str, user_id: str) -> bool:
    """Trích xuất và lưu vector khuôn mặt vào FAISS DB"""
    try:
        img = decode_base64_image(base64_str)
        emb = fg.get_embedding(img)
        if emb is not None:
            db.add(emb, user_id)  # Lưu vector kèm ID của user
            return True
        return False
    except Exception as e:
        print(f"Lỗi xử lý ảnh: {e}")
        return False

def verify_face(base64_str: str) -> str:
    """So khớp khuôn mặt, trả về user_id hoặc UNKNOWN"""
    try:
        img = decode_base64_image(base64_str)
        emb = fg.get_embedding(img)
        if emb is not None:
            # Code của Khanh để ngưỡng threshold=1.2
            match_id, dist = db.search(emb, threshold=1.2)
            return match_id
        return "NO_FACE"
    except Exception:
        return "ERROR"