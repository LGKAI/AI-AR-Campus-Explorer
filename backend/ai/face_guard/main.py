import cv2
import time
from detector import FaceGuard
from database import FaceDB

fg = FaceGuard()
db = FaceDB()

cap = cv2.VideoCapture(0)

last_time = 0
INTERVAL = 4  # seconds

print("🚀 System started... (Ctrl+C to stop)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    # giảm lag
    frame = cv2.resize(frame, (640, 480))

    current_time = time.time()

    if current_time - last_time >= INTERVAL:
        last_time = current_time

        emb = fg.get_embedding(frame)

        if emb is not None:
            name, dist = db.search(emb, threshold=1.2)

            print(f"[RESULT] {name} | distance = {dist:.3f}")
        else:
            print("[RESULT] NO FACE DETECTED")

cap.release()