import cv2
import urllib.request
import numpy as np
from config import CONFIG

class Stream:
    def __init__(self):
        self.url = CONFIG["stream_url"]
        self.stream = urllib.request.urlopen(self.url, timeout=CONFIG["stream_timeout"])
        self.bytes_buffer = b""
        print(f"[Stream] Connected to {self.url}")

    def get_frame(self):
        try:
            self.bytes_buffer += self.stream.read(4096)
        except Exception as e:
            print(f"[Stream] Read error: {e}")
            return None

        start = self.bytes_buffer.find(b'\xff\xd8')
        end = self.bytes_buffer.find(b'\xff\xd9')

        if start != -1 and end != -1 and end > start:
            jpg = self.bytes_buffer[start:end+2]
            self.bytes_buffer = self.bytes_buffer[end+2:]

            if len(jpg) == 0:
                return None

            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            return frame

        # Prevent buffer from growing unbounded if corrupted
        if len(self.bytes_buffer) > 100000:
            self.bytes_buffer = b""

        return None

    def release(self):
        try:
            self.stream.close()
        except:
            pass
        print("[Stream] Disconnected")