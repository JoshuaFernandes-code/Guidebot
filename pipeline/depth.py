import torch
import cv2
import numpy as np
from config import CONFIG

class DepthEstimator:
    def __init__(self):
        print(f"[Depth] Loading {CONFIG['midas_model']}...")
        self.model = torch.hub.load("intel-isl/MiDaS", CONFIG["midas_model"])
        self.model.to("cuda")
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

        print("[Depth] Model loaded on GPU")

    def estimate(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to("cuda")

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

    def get_distance_at_box(self, depth_map, x1, y1, x2, y2):
        cx1 = int(x1 + (x2 - x1) * 0.25)
        cx2 = int(x1 + (x2 - x1) * 0.75)
        cy1 = int(y1 + (y2 - y1) * 0.25)
        cy2 = int(y1 + (y2 - y1) * 0.75)

        region = depth_map[cy1:cy2, cx1:cx2]
        if region.size == 0:
            return "Unknown"

        raw_value = np.median(region)

        if raw_value > 400:
            return "Close"
        elif raw_value > 200:
            return "Medium"
        else:
            return "Far"