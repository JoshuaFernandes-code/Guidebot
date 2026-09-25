import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from config import CONFIG

class FaceRecognizer:
    def __init__(self):
        print("[Face] Loading InsightFace...")
        self.app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("[Face] InsightFace loaded")

        self.known_faces = {}
        self._load_known_faces()

    def _load_known_faces(self):
        folder = "data/known_faces"
        if not os.path.exists(folder):
            print(f"[Face] No known_faces folder found at {folder}")
            return

        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(filename)[0]
                path = os.path.join(folder, filename)
                img = cv2.imread(path)

                if img is None:
                    print(f"[Face] Could not read {filename}")
                    continue

                faces = self.app.get(img)
                if len(faces) == 0:
                    print(f"[Face] No face found in {filename}")
                    continue

                self.known_faces[name] = faces[0].embedding
                print(f"[Face] Loaded known face: {name}")

        print(f"[Face] Total known faces loaded: {len(self.known_faces)}")

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def identify(self, frame):
        faces = self.app.get(frame)

        if len(faces) == 0:
            return "no face detected"

        results = []
        for face in faces:
            embedding = face.embedding
            best_match = None
            best_score = 0

            for name, known_embedding in self.known_faces.items():
                score = self._cosine_similarity(embedding, known_embedding)
                if score > best_score:
                    best_score = score
                    best_match = name

            if best_match and best_score >= CONFIG["face_similarity_threshold"]:
                results.append(best_match)
            else:
                results.append("unknown person")

        return ", ".join(results)