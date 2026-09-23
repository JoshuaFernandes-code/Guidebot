import easyocr
from config import CONFIG

class OCRReader:
    def __init__(self):
        print("[OCR] Loading EasyOCR...")
        self.reader = easyocr.Reader(CONFIG["ocr_languages"], gpu=True)
        print("[OCR] EasyOCR loaded")

    def read_frame(self, frame):
        results = self.reader.readtext(frame)

        texts = []
        for (bbox, text, confidence) in results:
            if confidence >= CONFIG["ocr_min_confidence"]:
                cleaned = text.strip()
                if cleaned:
                    texts.append(cleaned)

        if not texts:
            return "no text found"

        return ", ".join(texts)