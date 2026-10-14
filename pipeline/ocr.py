import easyocr
import cv2
from config import CONFIG

class OCRReader:
    def __init__(self):
        print("[OCR] Loading EasyOCR...")
        self.reader = easyocr.Reader(CONFIG["ocr_languages"], gpu=True)
        print("[OCR] EasyOCR loaded")

    def read_frame(self, frame):
        # Camera feed is mirrored — flip before reading so text orientation is correct
        flipped = cv2.flip(frame, 1)

        results = self.reader.readtext(flipped)

        boxes = []
        for (bbox, text, confidence) in results:
            text = text.strip()
            if confidence >= CONFIG["ocr_min_confidence"] and text:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                boxes.append({
                    "text": text,
                    "x_min": min(xs),
                    "x_max": max(xs),
                    "y_center": sum(ys) / len(ys),
                    "height": max(ys) - min(ys),
                })

        if not boxes:
            return "no text found"

        # Group detections into lines based on vertical position
        boxes.sort(key=lambda b: b["y_center"])
        lines = []
        current_line = [boxes[0]]

        for box in boxes[1:]:
            avg_height = sum(b["height"] for b in current_line) / len(current_line)
            if abs(box["y_center"] - current_line[-1]["y_center"]) <= avg_height * 0.6:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
        lines.append(current_line)

        # Within each line, sort left-to-right and merge based on horizontal gap
        # Small gap = same word, no space. Larger gap = new word, add space.
        final_lines = []
        for line in lines:
            line.sort(key=lambda b: b["x_min"])
            merged = line[0]["text"]
            for prev, curr in zip(line, line[1:]):
                gap = curr["x_min"] - prev["x_max"]
                avg_height = (prev["height"] + curr["height"]) / 2
                if gap > avg_height * 0.4:
                    merged += " "
                merged += curr["text"]
            final_lines.append(merged)

        return ". ".join(final_lines)