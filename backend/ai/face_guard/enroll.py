import cv2
from detector import FaceGuard
from database import FaceDB

fg = FaceGuard()
db = FaceDB()

name = input("Enter name: ")

cap = cv2.VideoCapture(0)

print("Press 's' to capture")

cv2.namedWindow("Enroll")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    cv2.imshow("Enroll", frame)

    key = cv2.waitKey(10) & 0xFF

    if key == 27:  # ESC
        break

    if key == ord('s'):  # 🔥 ổn định hơn SPACE
        emb = fg.get_embedding(frame)

        if emb is not None:
            db.add(emb, name)
            print(f"✅ Saved {name}")
        else:
            print("❌ No face detected")

        break

cap.release()
cv2.destroyAllWindows()