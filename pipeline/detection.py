from ultralytics import YOLO
from config import CONFIG

class Detector:
    def __init__(self):
        print(f"[Detector] Loading {CONFIG['yolo_model']}...")
        self.model = YOLO(CONFIG["yolo_model"])
        self.model.to("cuda")
        print("[Detector] Model loaded on GPU")

    def detect(self, frame):
        results = self.model(
            frame,
            conf=CONFIG["confidence_threshold"],
            verbose=False
        )
        return results[0]