import cv2
import threading
import time
import json
import asyncio
import numpy as np
from config import CONFIG

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException, status
    from fastapi.responses import HTMLResponse
except ImportError:
    uvicorn = None
    FastAPI = None

class Stream:
    def __init__(self, voice_input=None):
        self.mode = CONFIG.get("stream_mode", "push_receiver")
        self.voice_input = voice_input
        self.latest_frame = None
        self.latest_timestamp = 0
        self.frame_count = 0
        self.lock = threading.Lock()
        self.running = True

        # ESP WebSocket management & Dashboard subscribers
        self.esp_ws = None
        self.esp_ws_loop = None
        self.esp_current_ssid = None
        self.dash_clients = set()
        self.pending_scan_future = None

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
        self.app = app

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

        async def broadcast_esp_status(connected: bool):
            msg = json.dumps({"type": "esp_status", "connected": connected})
            dead_clients = set()
            for client in list(self.dash_clients):
                try:
                    await client.send_text(msg)
                except Exception:
                    dead_clients.add(client)
            self.dash_clients.difference_update(dead_clients)

        @app.websocket("/ws/stream")
        async def websocket_stream(websocket: WebSocket):
            token = websocket.query_params.get("token") or websocket.headers.get("X-Auth-Token")
            if not self._validate_token(token):
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            await websocket.accept()
            print("[Stream Receiver] ESP32 connected via WebSocket.")

            # Set connection references
            self.esp_ws = websocket
            self.esp_ws_loop = asyncio.get_running_loop()

            # Broadcast connected status to all /esp dashboard WebSocket clients
            await broadcast_esp_status(True)

            # Immediately tell the ESP32 whether to force high-res or use adaptive mode
            force_high_res = CONFIG.get("force_high_res", False)
            cmd = "FORCE_HIGH_RES:1" if force_high_res else "FORCE_HIGH_RES:0"
            await websocket.send_text(cmd)
            print(f"[Stream Receiver] Sent resolution command to ESP32: {cmd}")

            try:
                while self.running:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        break
                    if "bytes" in message and message["bytes"]:
                        self._process_frame_bytes(message["bytes"])
                    elif "text" in message and message["text"]:
                        txt = message["text"].strip()
                        # Check for scan results response from ESP
                        if txt.startswith("SCAN_RESULTS:"):
                            try:
                                json_part = txt[len("SCAN_RESULTS:"):]
                                networks = json.loads(json_part)
                                if self.pending_scan_future and not self.pending_scan_future.done():
                                    self.pending_scan_future.set_result(networks)
                            except Exception as parse_err:
                                print(f"[Stream Receiver] Error parsing scan results from ESP: {parse_err}")
                                if self.pending_scan_future and not self.pending_scan_future.done():
                                    self.pending_scan_future.set_result([])
                        elif txt.startswith("DEVICE_INFO:"):
                            try:
                                info = json.loads(txt[len("DEVICE_INFO:"):])
                                self.esp_current_ssid = info.get("ssid", "")
                                print(f"[Stream Receiver] ESP device info: SSID='{self.esp_current_ssid}'")
                            except Exception as parse_err:
                                print(f"[Stream Receiver] Error parsing DEVICE_INFO: {parse_err}")
                        else:
                            print(f"[Stream Receiver] Received text message from ESP: {txt}")
            except WebSocketDisconnect:
                print("[Stream Receiver] ESP32 WebSocket disconnected.")
            except Exception as e:
                print(f"[Stream Receiver] WebSocket connection error: {e}")
            finally:
                if self.esp_ws is websocket:
                    self.esp_ws = None
                    self.esp_ws_loop = None
                    self.esp_current_ssid = None
                    if self.pending_scan_future and not self.pending_scan_future.done():
                        self.pending_scan_future.set_result([])
                    await broadcast_esp_status(False)

        # -------------------------------------------------------------
        # CONDITIONAL WEB MIC UI (/audio)
        # Only active if USE_WEB_MIC=true in .env / config
        # If false or unset, the /audio route does NOT exist (returns 404)
        # -------------------------------------------------------------
        if CONFIG.get("use_web_mic", False):
            import tempfile
            import os

            @app.get("/audio", response_class=HTMLResponse)
            def audio_control_panel():
                html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GuideBot Voice Input</title>
  <style>
    :root {
      --bg: #090a0f;
      --surface: #13151c;
      --border: #222634;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --danger: #ef4444;
      --danger-hover: #dc2626;
      --success: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 24px 16px; display: flex; justify-content: center; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 28px; width: 100%; max-width: 440px; box-shadow: 0 16px 40px rgba(0,0,0,0.5); }
    h1 { font-size: 20px; font-weight: 700; text-align: center; margin-bottom: 4px; }
    p.sub { font-size: 13px; color: var(--text-muted); text-align: center; margin-bottom: 24px; }
    .rec-container { text-align: center; margin-bottom: 28px; }
    .rec-btn {
      width: 100px; height: 100px; border-radius: 50%; border: 3px solid var(--border);
      background: #1a1d27; color: #fff; font-size: 13px; font-weight: 600; cursor: pointer;
      display: inline-flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px;
      transition: all 0.2s; outline: none; box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .rec-btn:hover { border-color: var(--primary); }
    .rec-btn.recording {
      background: var(--danger); border-color: #fca5a5;
      animation: pulse 1.2s infinite;
    }
    .mic-icon { font-size: 24px; }
    @keyframes pulse {
      0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
      70% { transform: scale(1.05); box-shadow: 0 0 0 16px rgba(239, 68, 68, 0); }
      100% { transform: scale(1); }
    }
    .status-msg { font-size: 13px; color: var(--text-muted); margin-top: 10px; min-height: 20px; }
    .divider { height: 1px; background: var(--border); margin: 24px 0; }
    h2 { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); margin-bottom: 12px; }
    .quick-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px; }
    .q-btn {
      padding: 12px; border-radius: 10px; background: #1a1d27; border: 1px solid var(--border);
      color: var(--text); font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s; text-align: left;
    }
    .q-btn:hover { border-color: var(--primary); background: #222635; color: #fff; }
    .input-box { display: flex; gap: 8px; }
    input[type='text'] {
      flex: 1; padding: 11px 14px; background: #1a1d27; border: 1px solid var(--border);
      border-radius: 10px; color: #fff; font-size: 14px; outline: none;
    }
    input[type='text']:focus { border-color: var(--primary); }
    .send-btn {
      padding: 11px 18px; border-radius: 10px; background: var(--primary); color: #fff;
      font-size: 14px; font-weight: 600; border: none; cursor: pointer; transition: background 0.15s;
    }
    .send-btn:hover { background: var(--primary-hover); }
    .feedback { margin-top: 16px; padding: 12px; border-radius: 10px; background: #132b1a; border: 1px solid #059669; color: #a7f3d0; font-size: 13px; display: none; }
  </style>
</head>
<body>
  <div class="card">
    <h1>GuideBot Voice Input</h1>
    <p class="sub">Speak or trigger commands remotely</p>

    <!-- Microphone Recording -->
    <div class="rec-container">
      <button id="recBtn" class="rec-btn" onclick="toggleRecording()">
        <span class="mic-icon">🎤</span>
        <span id="recText">Hold/Click</span>
      </button>
      <div id="statusMsg" class="status-msg">Click to record speech</div>
    </div>

    <div class="divider"></div>

    <!-- Quick Action Buttons -->
    <h2>Quick Voice Commands</h2>
    <div class="quick-grid">
      <button class="q-btn" onclick="sendCommand('what is ahead')">Ahead?</button>
      <button class="q-btn" onclick="sendCommand('what is on my left')">Left?</button>
      <button class="q-btn" onclick="sendCommand('what is on my right')">Right?</button>
      <button class="q-btn" onclick="sendCommand('read')">Read Text</button>
      <button class="q-btn" onclick="sendCommand('who is this')">Who is this?</button>
      <button class="q-btn" onclick="sendCommand('status')">Status</button>
    </div>

    <!-- Custom Text Command Input -->
    <h2>Custom Command</h2>
    <div class="input-box">
      <input type="text" id="cmdInput" placeholder="e.g. read, ahead, who..." onkeydown="if(event.key==='Enter') sendCustom()">
      <button class="send-btn" onclick="sendCustom()">Send</button>
    </div>

    <div id="feedback" class="feedback"></div>
  </div>

  <script>
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    async function toggleRecording() {
      if (!isRecording) {
        startRecording();
      } else {
        stopRecording();
      }
    }

    async function startRecording() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.onstop = uploadAudio;

        mediaRecorder.start();
        isRecording = true;
        document.getElementById('recBtn').classList.add('recording');
        document.getElementById('recText').innerText = "Stop";
        document.getElementById('statusMsg').innerText = "Recording... speak now";
        document.getElementById('statusMsg').style.color = "#ef4444";
      } catch (err) {
        showFeedback("Microphone access denied: " + err.message, true);
      }
    }

    function stopRecording() {
      if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(t => t.stop());
        isRecording = false;
        document.getElementById('recBtn').classList.remove('recording');
        document.getElementById('recText').innerText = "Hold/Click";
        document.getElementById('statusMsg').innerText = "Transcribing with Whisper...";
        document.getElementById('statusMsg').style.color = "#94a3b8";
      }
    }

    async function uploadAudio() {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      try {
        const res = await fetch("/audio/upload", {
          method: "POST",
          headers: { "Content-Type": "audio/webm" },
          body: audioBlob
        });
        const data = await res.json();
        if (data.status === "ok") {
          showFeedback("Heard: '" + data.transcription + "'");
          document.getElementById('statusMsg').innerText = "Heard: '" + data.transcription + "'";
        } else {
          showFeedback(data.detail || "Transcription failed", true);
          document.getElementById('statusMsg').innerText = "Could not transcribe audio";
        }
      } catch (err) {
        showFeedback("Upload failed: " + err.message, true);
      }
    }

    async function sendCommand(text) {
      try {
        const res = await fetch("/audio/command", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ command: text })
        });
        const data = await res.json();
        showFeedback("Sent: '" + text + "'");
      } catch (err) {
        showFeedback("Failed to send command", true);
      }
    }

    function sendCustom() {
      const inp = document.getElementById('cmdInput');
      const val = inp.value.trim();
      if (val) {
        sendCommand(val);
        inp.value = "";
      }
    }

    function showFeedback(msg, isErr=false) {
      const fb = document.getElementById('feedback');
      fb.style.display = 'block';
      fb.innerText = msg;
      fb.style.background = isErr ? '#451a1a' : '#132b1a';
      fb.style.borderColor = isErr ? '#ef4444' : '#059669';
      fb.style.color = isErr ? '#fca5a5' : '#a7f3d0';
      setTimeout(() => { fb.style.display = 'none'; }, 4000);
    }
  </script>
</body>
</html>"""
                return HTMLResponse(content=html)

            @app.post("/audio/command")
            async def submit_audio_command(request: Request):
                body = await request.json()
                command = body.get("command", "").strip()
                if not command:
                    raise HTTPException(status_code=400, detail="Empty command")
                if self.voice_input:
                    self.voice_input.submit_command(command)
                return {"status": "ok", "command": command}

            @app.post("/audio/upload")
            async def upload_audio_recording(request: Request):
                if not self.voice_input:
                    raise HTTPException(status_code=500, detail="Voice input not initialized")

                content = await request.body()
                if not content:
                    raise HTTPException(status_code=400, detail="Empty audio content")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                try:
                    text = self.voice_input.transcribe_audio_file(tmp_path)
                    return {"status": "ok", "transcription": text}
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        # -------------------------------------------------------------
        # ESP MANAGEMENT & REMOTE CONTROL DASHBOARD (/esp)
        # -------------------------------------------------------------
        @app.get("/esp", response_class=HTMLResponse)
        def esp_control_panel():
            current_ssid = self.esp_current_ssid or ""
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GuideBot ESP Control Panel</title>
  <style>
    :root {{
      --bg: #090a0f;
      --surface: #13151c;
      --border: #222634;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --danger: #ef4444;
      --danger-hover: #dc2626;
      --success: #10b981;
      --warning: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); padding: 28px 16px; display: flex; justify-content: center; -webkit-font-smoothing: antialiased; }}
    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 28px; width: 100%; max-width: 460px; box-shadow: 0 16px 40px rgba(0,0,0,0.5); }}
    .header {{ text-align: center; margin-bottom: 24px; }}
    h1 {{ font-size: 20px; font-weight: 700; color: #fff; margin-bottom: 4px; }}
    p.sub {{ font-size: 13px; color: var(--text-muted); }}

    .status-badge {{
      display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px;
      border-radius: 999px; font-size: 13px; font-weight: 600; margin-top: 12px;
      transition: all 0.3s ease;
    }}
    .status-badge.online {{ background: rgba(16, 185, 129, 0.15); border: 1px solid #059669; color: #a7f3d0; }}
    .status-badge.offline {{ background: rgba(239, 68, 68, 0.15); border: 1px solid #dc2626; color: #fca5a5; }}
    .dot {{ width: 8px; height: 8px; border-radius: 50%; }}
    .status-badge.online .dot {{ background: #10b981; box-shadow: 0 0 8px #10b981; }}
    .status-badge.offline .dot {{ background: #ef4444; }}

    .offline-banner {{
      background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25);
      border-radius: 12px; padding: 12px; font-size: 12px; color: #fca5a5;
      text-align: center; margin-bottom: 20px; display: none; line-height: 1.5;
    }}

    .section-title {{ font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); margin-bottom: 12px; }}
    .field-group {{ margin-bottom: 16px; }}
    label {{ font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.6px; display: block; margin-bottom: 7px; }}
    select, input {{
      width: 100%; padding: 11px 14px; background: #1a1d27; border: 1px solid var(--border);
      border-radius: 10px; color: #fff; font-size: 14px; outline: none; transition: border-color 0.15s;
    }}
    select:focus, input:focus {{ border-color: var(--primary); }}

    .pw-wrap {{ position: relative; display: flex; }}
    .pw-wrap input {{ flex: 1; padding-right: 42px; }}
    .eye-btn {{
      position: absolute; right: 0; top: 0; height: 100%; width: 40px; background: none;
      border: none; cursor: pointer; font-size: 16px; color: var(--text-muted); opacity: 0.45;
      display: flex; align-items: center; justify-content: center; transition: opacity 0.15s;
    }}
    .eye-btn:hover {{ opacity: 0.8; }}

    .scan-row {{ display: flex; gap: 8px; margin-bottom: 10px; }}
    .btn-scan {{
      padding: 10px 14px; border-radius: 10px; background: #222634; border: 1px solid var(--border);
      color: var(--text); font-size: 13px; font-weight: 600; cursor: pointer; white-space: nowrap;
      transition: all 0.15s;
    }}
    .btn-scan:hover:not(:disabled) {{ border-color: var(--primary); color: #fff; }}

    .btn {{
      width: 100%; padding: 12px 16px; border-radius: 10px; font-size: 14px; font-weight: 600;
      cursor: pointer; border: none; transition: all 0.15s; display: flex; align-items: center; justify-content: center; gap: 8px;
    }}
    .btn-primary {{ background: var(--primary); color: #fff; }}
    .btn-primary:hover:not(:disabled) {{ background: var(--primary-hover); }}
    .btn-warning {{ background: transparent; border: 1px solid #78350f; color: #fde68a; margin-top: 12px; }}
    .btn-warning:hover:not(:disabled) {{ background: rgba(245, 158, 11, 0.1); border-color: var(--warning); color: #fff; }}
    .btn-danger {{ background: transparent; border: 1px solid #7f1d1d; color: #fca5a5; margin-top: 12px; }}
    .btn-danger:hover:not(:disabled) {{ background: rgba(239, 68, 68, 0.12); border-color: var(--danger); color: #fff; }}

    .btn:disabled, .btn-scan:disabled, select:disabled, input:disabled {{
      opacity: 0.4; cursor: not-allowed; pointer-events: none;
    }}
    .helper {{ font-size: 12px; color: var(--text-muted); margin-top: 6px; line-height: 1.4; }}

    .divider {{ height: 1px; background: var(--border); margin: 24px 0; }}
    .feedback {{ margin-top: 16px; padding: 12px; border-radius: 10px; font-size: 13px; display: none; line-height: 1.4; }}

    .modal-overlay {{
      position: fixed; inset: 0; background: rgba(0,0,0,0.75); backdrop-filter: blur(4px);
      display: none; align-items: center; justify-content: center; padding: 16px; z-index: 99;
    }}
    .modal {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
      padding: 24px; max-width: 380px; width: 100%; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.6);
    }}
    .modal h3 {{ font-size: 17px; font-weight: 700; color: #fff; margin-bottom: 8px; }}
    .modal p {{ font-size: 13px; color: var(--text-muted); line-height: 1.5; margin-bottom: 20px; }}
    .modal-actions {{ display: flex; gap: 10px; }}
    .modal-actions button {{ flex: 1; padding: 11px; font-size: 13px; font-weight: 600; border-radius: 9px; border: none; cursor: pointer; }}
    .btn-modal-cancel {{ background: #222634; color: var(--text); }}
    .btn-modal-confirm {{ background: var(--danger); color: #fff; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <h1>GuideBot ESP Control Panel</h1>
      <p class="sub">Remote Wi-Fi & Device Management</p>
      <div id="statusBadge" class="status-badge offline">
        <span class="dot"></span>
        <span id="statusText">Checking status...</span>
      </div>
    </div>

    <div id="offlineNotice" class="offline-banner">
      ESP32 is currently disconnected. Remote action buttons are disabled until the device connects to the server.
    </div>

    <!-- Add / Switch Wi-Fi Section -->
    <div class="section-title">Add / Switch Wi-Fi Network</div>
    
    <div class="field-group">
      <div class="scan-row">
        <button id="scanBtn" class="btn-scan" onclick="scanNetworks()">🔍 Scan Wi-Fi</button>
        <select id="networkSelect" onchange="onSelectNetwork(this)" style="flex:1;">
          <option value="">-- Click Scan or Enter Manually --</option>
        </select>
      </div>
    </div>

    <div class="field-group">
      <label>Network Name (SSID)</label>
      <input type="text" id="ssidInput" placeholder="e.g. MyHomeNetwork">
    </div>

    <div id="passGroup" class="field-group">
      <label>Password</label>
      <div class="pw-wrap">
        <input type="password" id="passInput" placeholder="Enter password">
        <button type="button" class="eye-btn" onclick="togglePw('passInput', this)" aria-label="Show/hide password">&#128065;</button>
      </div>
      <div id="passHint" class="helper" style="display:none;"></div>
    </div>

    <button id="switchBtn" class="btn btn-primary" onclick="submitWifiChange()">Save & Switch Wi-Fi</button>

    <div class="divider"></div>

    <!-- Device Actions -->
    <div class="section-title">Device Operations</div>
    <button id="rebootBtn" class="btn btn-warning" onclick="openModal('reboot', 'Reboot ESP32?', 'The ESP32 will restart immediately and reconnect using its saved configuration.')">Restart ESP32</button>
    <button id="eraseBtn" class="btn btn-danger" onclick="openModal('erase', 'Erase Wi-Fi & Launch Hotspot?', 'This will wipe all stored Wi-Fi credentials from ESP flash and restart it in GuideBot-Setup hotspot mode.')">Erase Wi-Fi & Launch Hotspot</button>

    <div id="feedback" class="feedback"></div>
  </div>

  <!-- Custom Confirmation Modal -->
  <div id="modal" class="modal-overlay">
    <div class="modal">
      <h3 id="mTitle">Confirm Action</h3>
      <p id="mDesc">Are you sure you want to proceed?</p>
      <div class="modal-actions">
        <button class="btn-modal-cancel" onclick="closeModal()">Cancel</button>
        <button id="mConfirmBtn" class="btn-modal-confirm" onclick="executeModalAction()">Confirm</button>
      </div>
    </div>
  </div>

  <script>
    let isConnected = false;
    let pendingAction = null;
    let ws = null;
    const ESP_CURRENT_SSID = {json.dumps(current_ssid)};

    function initWebSocket() {{
      const loc = window.location;
      const wsProto = loc.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = wsProto + "//" + loc.host + "/ws/esp-status";
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {{ console.log("[WS] Connected to status stream"); }};
      ws.onmessage = (event) => {{
        try {{
          const data = JSON.parse(event.data);
          if (data.type === "esp_status") {{
            updateStatusUI(data.connected);
          }}
        }} catch (e) {{ }}
      }};
      ws.onclose = () => {{
        setTimeout(initWebSocket, 2000);
      }};
    }}

    async function checkStatusDirect() {{
      try {{
        const res = await fetch("/esp/status");
        const data = await res.json();
        updateStatusUI(data.connected);
      }} catch (err) {{
        updateStatusUI(false);
      }}
    }}

    function updateStatusUI(connected) {{
      isConnected = connected;
      const badge = document.getElementById('statusBadge');
      const text = document.getElementById('statusText');
      const notice = document.getElementById('offlineNotice');

      if (connected) {{
        badge.className = 'status-badge online';
        text.innerText = 'ESP Connected';
        notice.style.display = 'none';
      }} else {{
        badge.className = 'status-badge offline';
        text.innerText = 'ESP Disconnected';
        notice.style.display = 'block';
      }}

      // Toggle form & action buttons
      document.getElementById('scanBtn').disabled = !connected;
      document.getElementById('networkSelect').disabled = !connected;
      document.getElementById('ssidInput').disabled = !connected;
      document.getElementById('passInput').disabled = !connected;
      document.getElementById('switchBtn').disabled = !connected;
      document.getElementById('rebootBtn').disabled = !connected;
      document.getElementById('eraseBtn').disabled = !connected;
    }}

    async function scanNetworks() {{
      if (!isConnected) return;
      const scanBtn = document.getElementById('scanBtn');
      const select = document.getElementById('networkSelect');
      scanBtn.disabled = true;
      scanBtn.innerText = "Scanning...";
      showFeedback("Scanning nearby Wi-Fi networks via ESP32...");

      try {{
        const res = await fetch("/esp/scan", {{ method: "POST" }});
        const data = await res.json();
        if (data.status === "ok") {{
          select.innerHTML = '<option value="">-- Select a Scanned Network --</option>';
          if (data.networks && data.networks.length > 0) {{
            data.networks.forEach(net => {{
              const opt = document.createElement('option');
              opt.value = net.ssid;
              const icon = net.open ? '🔓 ' : '🔒 ';
              const sec = net.open ? 'Open' : `${{net.rssi}} dBm`;
              const savedTag = net.saved ? ' · Saved' : '';
              opt.innerText = `${{icon}}${{net.ssid}} (${{sec}}${{savedTag}})`;
              opt.dataset.open = net.open ? '1' : '0';
              opt.dataset.saved = net.saved ? '1' : '0';
              select.appendChild(opt);
            }});
            showFeedback(`Found ${{data.networks.length}} networks nearby.`);
          }} else {{
            select.innerHTML = '<option value="">No networks detected (type manually)</option>';
            showFeedback("No networks detected nearby.", true);
          }}
        }} else {{
          showFeedback(data.detail || "Scan request failed.", true);
        }}
      }} catch (err) {{
        showFeedback("Error requesting scan: " + err.message, true);
      }} finally {{
        scanBtn.disabled = false;
        scanBtn.innerText = "🔍 Scan Wi-Fi";
      }}
    }}

    function onSelectNetwork(sel) {{
      if (!sel.value) return;
      document.getElementById('ssidInput').value = sel.value;
      const opt = sel.options[sel.selectedIndex];
      const isOpen = (opt && opt.dataset.open === '1');
      const isSaved = (opt && opt.dataset.saved === '1') || sel.value === ESP_CURRENT_SSID;
      const passGroup = document.getElementById('passGroup');
      const passInp = document.getElementById('passInput');
      const passHint = document.getElementById('passHint');

      if (isOpen) {{
        passGroup.style.display = 'none';
        passInp.value = '';
        passHint.style.display = 'none';
      }} else {{
        passGroup.style.display = 'block';
        passInp.value = '';
        if (isSaved) {{
          passInp.placeholder = 'Leave blank to keep saved password';
          passHint.innerText = 'Previously saved network — leave blank to reuse stored password.';
          passHint.style.display = 'block';
        }} else {{
          passInp.placeholder = 'Enter password';
          passHint.style.display = 'none';
        }}
      }}
    }}

    function togglePw(id, btn) {{
      const el = document.getElementById(id);
      el.type = (el.type === 'password') ? 'text' : 'password';
      btn.style.opacity = (el.type === 'text') ? '1' : '0.45';
    }}

    async function submitWifiChange() {{
      if (!isConnected) return;
      const ssid = document.getElementById('ssidInput').value.trim();
      const password = document.getElementById('passInput').value;

      if (!ssid) {{
        showFeedback("Please specify an SSID network name.", true);
        return;
      }}

      // Determine if open or saved network
      const select = document.getElementById('networkSelect');
      const opt = select.options[select.selectedIndex];
      const isOpen = (opt && opt.dataset && opt.dataset.open === '1');
      const isSaved = (opt && opt.dataset && opt.dataset.saved === '1') || ssid === ESP_CURRENT_SSID;

      // Block: locked network + not previously saved + no password
      if (!isOpen && !isSaved && !password.trim()) {{
        showFeedback("Password is required for this secured network.", true);
        return;
      }}

      showFeedback("Sending new Wi-Fi credentials to ESP32...");
      try {{
        const res = await fetch("/esp/wifi", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ ssid, password }})
        }});
        const data = await res.json();
        if (data.status === "ok") {{
          showFeedback(`Saved! ESP32 is switching to '${{ssid}}' and reconnecting.`);
        }} else {{
          showFeedback(data.detail || "Failed to update Wi-Fi.", true);
        }}
      }} catch (err) {{
        showFeedback("Failed to send command: " + err.message, true);
      }}
    }}

    function openModal(action, title, desc) {{
      if (!isConnected) return;
      pendingAction = action;
      document.getElementById('mTitle').innerText = title;
      document.getElementById('mDesc').innerText = desc;
      const confirmBtn = document.getElementById('mConfirmBtn');
      if (action === 'erase') {{
        confirmBtn.className = 'btn-modal-confirm';
        confirmBtn.innerText = 'Erase & Reboot';
      }} else {{
        confirmBtn.className = 'btn-modal-confirm';
        confirmBtn.style.background = '#d97706';
        confirmBtn.innerText = 'Restart';
      }}
      document.getElementById('modal').style.display = 'flex';
    }}

    function closeModal() {{
      document.getElementById('modal').style.display = 'none';
      pendingAction = null;
    }}

    async function executeModalAction() {{
      const action = pendingAction;
      closeModal();
      if (!action || !isConnected) return;

      if (action === 'reboot') {{
        showFeedback("Sending soft reboot command to ESP32...");
        try {{
          const res = await fetch("/esp/reboot", {{ method: "POST" }});
          const data = await res.json();
          if (data.status === "ok") showFeedback("Reboot command sent! ESP32 is restarting.");
          else showFeedback(data.detail || "Command failed.", true);
        }} catch (err) {{
          showFeedback("Error: " + err.message, true);
        }}
      }} else if (action === 'erase') {{
        showFeedback("Sending wipe command to ESP32...");
        try {{
          const res = await fetch("/esp/erase", {{ method: "POST" }});
          const data = await res.json();
          if (data.status === "ok") showFeedback("Wipe command sent! ESP32 is wiping credentials and launching GuideBot-Setup.");
          else showFeedback(data.detail || "Command failed.", true);
        }} catch (err) {{
          showFeedback("Error: " + err.message, true);
        }}
      }}
    }}

    function showFeedback(msg, isErr = false) {{
      const fb = document.getElementById('feedback');
      fb.style.display = 'block';
      fb.innerText = msg;
      fb.style.background = isErr ? '#451a1a' : '#132b1a';
      fb.style.border = isErr ? '1px solid #ef4444' : '1px solid #059669';
      fb.style.color = isErr ? '#fca5a5' : '#a7f3d0';
    }}

    // Startup
    checkStatusDirect();
    initWebSocket();
  </script>
</body>
</html>"""
            return HTMLResponse(content=html)

        @app.get("/esp/status")
        def esp_status():
            connected = self.esp_ws is not None
            return {"status": "ok", "connected": connected}

        @app.websocket("/ws/esp-status")
        async def websocket_esp_status(websocket: WebSocket):
            await websocket.accept()
            self.dash_clients.add(websocket)
            # Immediately send initial state
            await websocket.send_text(json.dumps({"type": "esp_status", "connected": self.esp_ws is not None}))
            try:
                while self.running:
                    # Keep client alive until disconnect
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                self.dash_clients.discard(websocket)

        @app.post("/esp/scan")
        async def esp_scan():
            if not self.esp_ws:
                raise HTTPException(status_code=400, detail="ESP32 is not connected via WebSocket")

            loop = asyncio.get_running_loop()
            self.pending_scan_future = loop.create_future()

            try:
                await self.esp_ws.send_text("SCAN_WIFI")
                print("[Stream Receiver] Sent SCAN_WIFI request to ESP32.")
            except Exception as e:
                self.pending_scan_future = None
                self.esp_ws = None
                await broadcast_esp_status(False)
                raise HTTPException(status_code=500, detail=f"Failed to deliver scan request to ESP32: {e}")

            try:
                results = await asyncio.wait_for(self.pending_scan_future, timeout=12.0)
                return {"status": "ok", "networks": results}
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="ESP32 Wi-Fi scan timed out")
            finally:
                self.pending_scan_future = None

        @app.post("/esp/wifi")
        async def esp_switch_wifi(request: Request):
            if not self.esp_ws:
                raise HTTPException(status_code=400, detail="ESP32 is not connected via WebSocket")

            body = await request.json()
            ssid = body.get("ssid", "").strip()
            password = body.get("password", "")

            if not ssid:
                raise HTTPException(status_code=400, detail="SSID is required")

            payload = json.dumps({"ssid": ssid, "password": password})
            cmd = f"SET_WIFI:{payload}"
            try:
                await self.esp_ws.send_text(cmd)
                print(f"[Stream Receiver] Sent SET_WIFI to ESP32 for network '{ssid}'.")
                return {"status": "ok", "message": f"Command sent to switch Wi-Fi to '{ssid}'"}
            except Exception as e:
                self.esp_ws = None
                await broadcast_esp_status(False)
                raise HTTPException(status_code=500, detail=f"Failed to deliver command to ESP32: {e}")

        @app.post("/esp/reboot")
        async def esp_reboot():
            if not self.esp_ws:
                raise HTTPException(status_code=400, detail="ESP32 is not connected via WebSocket")

            try:
                await self.esp_ws.send_text("REBOOT")
                print("[Stream Receiver] Sent REBOOT command to ESP32.")
                return {"status": "ok", "message": "Reboot command sent to ESP32"}
            except Exception as e:
                self.esp_ws = None
                await broadcast_esp_status(False)
                raise HTTPException(status_code=500, detail=f"Failed to deliver command to ESP32: {e}")

        @app.post("/esp/erase")
        async def esp_erase():
            if not self.esp_ws:
                raise HTTPException(status_code=400, detail="ESP32 is not connected via WebSocket")

            try:
                await self.esp_ws.send_text("ERASE_WIFI")
                print("[Stream Receiver] Sent ERASE_WIFI command to ESP32.")
                return {"status": "ok", "message": "Erase credentials command sent to ESP32"}
            except Exception as e:
                self.esp_ws = None
                await broadcast_esp_status(False)
                raise HTTPException(status_code=500, detail=f"Failed to deliver command to ESP32: {e}")

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