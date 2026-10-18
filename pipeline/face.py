import cv2
import numpy as np
import os
import threading
from insightface.app import FaceAnalysis
from config import CONFIG
from pipeline.faces_db import KNOWN_FACES_DIR, get_name_map


class FaceRecognizer:
    def __init__(self):
        print("[Face] Loading InsightFace...")
        self.app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("[Face] InsightFace loaded")

        # {display_name: [embedding, embedding, …]}  — one person can have multiple images
        self.known_faces: dict[str, list] = {}
        self._lock = threading.Lock()
        self._load_known_faces()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_known_faces(self):
        """Scan data/known_faces/, resolve display names via DB, build embedding map."""
        if not os.path.exists(KNOWN_FACES_DIR):
            print(f"[Face] No known_faces folder found at {KNOWN_FACES_DIR}")
            return

        # DB name map: {filename: display_name}
        # Falls back to filename stem when a file isn't registered in the DB.
        name_map = get_name_map()

        new_faces: dict[str, list] = {}
        for filename in os.listdir(KNOWN_FACES_DIR):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue

            display_name = name_map.get(filename) or _stem_to_display(filename)
            path = os.path.join(KNOWN_FACES_DIR, filename)
            img  = cv2.imread(path)

            if img is None:
                print(f"[Face] Could not read {filename}")
                continue

            faces = self.app.get(img)
            if not faces:
                print(f"[Face] No face found in {filename}")
                continue

            embedding = faces[0].embedding
            new_faces.setdefault(display_name, []).append(embedding)
            print(f"[Face] Loaded: {filename}  → '{display_name}'")

        with self._lock:
            self.known_faces = new_faces

        total = sum(len(v) for v in new_faces.values())
        print(f"[Face] {total} embedding(s) loaded for {len(new_faces)} person(s)")

    def reload(self):
        """Hot-reload known faces without restarting the server."""
        print("[Face] Reloading known faces...")
        self._load_known_faces()

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def identify(self, frame) -> str:
        faces = self.app.get(frame)
        if not faces:
            return "no face detected"

        threshold = CONFIG["face_similarity_threshold"]
        results   = []

        with self._lock:
            snapshot = {name: list(embeds) for name, embeds in self.known_faces.items()}

        for face in faces:
            emb        = face.embedding
            best_name  = None
            best_score = 0.0

            for name, embeddings in snapshot.items():
                # Score against every embedding for this person, take the best
                for known_emb in embeddings:
                    score = self._cosine(emb, known_emb)
                    if score > best_score:
                        best_score = score
                        best_name  = name

            if best_name and best_score >= threshold:
                results.append(best_name)
            else:
                results.append("unknown person")

        return ", ".join(results)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stem_to_display(filename: str) -> str:
    """Convert a bare filename stem to a display name as fallback.

    'nathan_pereira_1.png' → 'Nathan Pereira'   (strips trailing _<digits>)
    'john.png'             → 'John'
    """
    stem = os.path.splitext(filename)[0]
    # Strip trailing _<number> suffix added by faces_db
    stem = stem.rstrip("0123456789").rstrip("_")
    return stem.replace("_", " ").title()