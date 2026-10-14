from ultralytics import YOLO
import torch
from config import CONFIG

class Detector:
    def __init__(self):
        print(f"[Detector] Loading {CONFIG['yolo_model']}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(CONFIG["yolo_model"])
        self.model.to(self.device)
        print(f"[Detector] Model loaded on {self.device.upper()}")

    def detect(self, frame):
        results = self.model(
            frame,
            conf=CONFIG["confidence_threshold"],
            verbose=False
        )
        return results[0]