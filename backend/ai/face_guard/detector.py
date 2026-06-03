from ultralytics import YOLO
from insightface.app import FaceAnalysis
import numpy as np

class FaceGuard:
    def __init__(self):
        # Face detection (YOLO - nhanh)
        self.detector = YOLO("yolov8n.pt")

        # Face embedding (model nhẹ)
        self.embedder = FaceAnalysis(name="buffalo_s")
        self.embedder.prepare(ctx_id=0)

    def get_embedding(self, image):
        faces = self.embedder.get(image)

        if len(faces) == 0:
            return None

        emb = faces[0].embedding

        # 🔥 CỰC KỲ QUAN TRỌNG
        emb = emb / np.linalg.norm(emb)

        return emb