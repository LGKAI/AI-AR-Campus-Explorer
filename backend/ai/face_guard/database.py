import faiss
import numpy as np
import pickle
import os

class FaceDB:
    def __init__(self, db_path="face_db.pkl"):
        self.db_path = db_path
        self.dim = 512

        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                data = pickle.load(f)
                self.index = data["index"]
                self.labels = data["labels"]
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.labels = []

    def add(self, embedding, name):
        embedding = embedding / np.linalg.norm(embedding)
        self.index.add(np.array([embedding]).astype('float32'))
        self.labels.append(name)
        self.save()

    def search(self, embedding, threshold=1.2):
        if self.index.ntotal == 0:
            return "UNKNOWN", None

        embedding = embedding / np.linalg.norm(embedding)

        D, I = self.index.search(
            np.array([embedding]).astype('float32'), k=1
        )

        dist = D[0][0]

        if dist < threshold:
            return self.labels[I[0][0]], dist
        else:
            return "UNKNOWN", dist

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump({
                "index": self.index,
                "labels": self.labels
            }, f)