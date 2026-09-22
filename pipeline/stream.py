import cv2
import threading
import time
from config import CONFIG

class Stream:
    def __init__(self):
        self.url = CONFIG["stream_url"]
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.cap = None

        self._connect()

        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _connect(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = cv2.VideoCapture(self.url)
        # Keep internal buffer small so we always get the freshest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            print(f"[Stream] Connected to {self.url}")
        else:
            print(f"[Stream] Failed to open {self.url}")

    def _read_loop(self):
        fail_count = 0
        while self.running:
            if not self.cap or not self.cap.isOpened():
                print("[Stream] Not connected — retrying...")
                time.sleep(2)
                self._connect()
                continue

            ret, frame = self.cap.read()

            if not ret or frame is None:
                fail_count += 1
                if fail_count > 10:
                    print("[Stream] Too many failed reads — reconnecting...")
                    time.sleep(1)
                    self._connect()
                    fail_count = 0
                continue

            fail_count = 0
            with self.lock:
                self.latest_frame = frame

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def release(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass
        print("[Stream] Disconnected")