import cv2
import threading
import time
import numpy as np
from config import CONFIG

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException, status
except ImportError:
    uvicorn = None
    FastAPI = None

class Stream:
    def __init__(self):
        self.mode = CONFIG.get("stream_mode", "push_receiver")
        self.latest_frame = None
        self.latest_timestamp = 0
        self.frame_count = 0
        self.lock = threading.Lock()
        self.running = True

        self.auth_token = CONFIG.get("stream_auth_token", "")
        self.host = CONFIG.get("receiver_host", "0.0.0.0")
        self.port = int(CONFIG.get("receiver_port", 8000))
        self.url = CONFIG.get("stream_url", "http://10.254.76.220/stream")

        self.cap = None
        self.thread = None
        self.server = None
        self.server_thread = None

        if self.mode == "push_receiver":
            self._start_push_receiver()
        else:
            self._start_pull_client()

    # -------------------------------------------------------------
    # PUSH RECEIVER (Cloudflare Tunnel / Ingestion Server)
    # -------------------------------------------------------------
    def _validate_token(self, provided_token: str | None) -> bool:
        if not self.auth_token:
            return True
        return provided_token == self.auth_token

    def _process_frame_bytes(self, raw_bytes: bytes) -> bool:
        if not raw_bytes or len(raw_bytes) == 0:
            return False
        try:
            np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                return False

            with self.lock:
                self.latest_frame = frame
                self.latest_timestamp = time.time()
                self.frame_count += 1
            return True
        except Exception as e:
            print(f"[Stream Receiver] Decode error: {e}")
            return False

    def _start_push_receiver(self):
        if FastAPI is None or uvicorn is None:
            raise RuntimeError(
                "[Stream] fastapi and uvicorn are required for push_receiver mode. "
                "Please run: pip install fastapi uvicorn"
            )

        app = FastAPI(title="GuideBot Stream Ingestion Server", docs_url=None, redoc_url=None)

        @app.get("/")
        @app.get("/health")
        def health():
            with self.lock:
                has_frame = self.latest_frame is not None
                last_seen = round(time.time() - self.latest_timestamp, 2) if has_frame else None
            return {
                "status": "online",
                "service": "GuideBot Stream Receiver",
                "frames_received": self.frame_count,
                "has_active_frame": has_frame,
                "seconds_since_last_frame": last_seen,
            }

        @app.post("/frame")
        async def receive_frame(request: Request):
            token = request.headers.get("X-Auth-Token") or request.query_params.get("token")
            if not self._validate_token(token):
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth token")

            body = await request.body()
            if not self._process_frame_bytes(body):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image payload")

            return Response(content='{"status":"ok"}', media_type="application/json")

        @app.websocket("/ws/stream")
        async def websocket_stream(websocket: WebSocket):
            token = websocket.query_params.get("token") or websocket.headers.get("X-Auth-Token")
            if not self._validate_token(token):
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            await websocket.accept()
            print("[Stream Receiver] ESP32 connected via WebSocket.")

            # Immediately tell the ESP32 whether to force high-res or use adaptive mode
            force_high_res = CONFIG.get("force_high_res", False)
            cmd = "FORCE_HIGH_RES:1" if force_high_res else "FORCE_HIGH_RES:0"
            await websocket.send_text(cmd)
            print(f"[Stream Receiver] Sent resolution command to ESP32: {cmd}")

            try:
                while self.running:
                    data = await websocket.receive_bytes()
                    self._process_frame_bytes(data)
            except WebSocketDisconnect:
                print("[Stream Receiver] ESP32 WebSocket disconnected.")
            except Exception as e:
                print(f"[Stream Receiver] WebSocket connection error: {e}")

        server_config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self.server = uvicorn.Server(server_config)
        self.server_thread = threading.Thread(target=self.server.run, daemon=True)
        self.server_thread.start()

        print(f"[Stream] Push receiver listening on http://{self.host}:{self.port}")
        print(f"         WebSocket endpoint: ws://{self.host}:{self.port}/ws/stream")
        print(f"         HTTP POST endpoint: http://{self.host}:{self.port}/frame")
        print(f"[Stream] Waiting for ESP32 to connect and send frames...")

        # Background thread: warn periodically if no frames have arrived
        def _no_frame_watcher():
            warned_intervals = 0
            while self.running:
                time.sleep(5)
                with self.lock:
                    count = self.frame_count
                if count == 0:
                    warned_intervals += 1
                    print(f"[Stream] WARNING: No frames received yet after {warned_intervals * 5}s — "
                          f"check ESP32 Serial Monitor and Cloudflare Tunnel status.")
                else:
                    break  # Frames are flowing, stop watching

        threading.Thread(target=_no_frame_watcher, daemon=True).start()

    # -------------------------------------------------------------
    # PULL CLIENT (Legacy LAN HTTP GET)
    # -------------------------------------------------------------
    def _start_pull_client(self):
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
                self.latest_timestamp = time.time()
                self.frame_count += 1

    # -------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------
    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def is_alive(self, timeout: float = 3.0) -> bool:
        with self.lock:
            if self.latest_frame is None or self.latest_timestamp == 0:
                return False
            return (time.time() - self.latest_timestamp) < timeout

    def release(self):
        self.running = False

        if self.server:
            try:
                self.server.should_exit = True
            except Exception:
                pass

        if self.server_thread and self.server_thread.is_alive():
            try:
                self.server_thread.join(timeout=2)
            except Exception:
                pass

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

        print("[Stream] Disconnected / Stopped")