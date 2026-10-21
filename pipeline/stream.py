import cv2
import threading
import time
import json
import asyncio
import os
import base64
import queue
import secrets
from datetime import datetime, timedelta, timezone
import numpy as np
from config import CONFIG
from pipeline import faces_db, emails_db
from pipeline.networking import discover_stream_lan_host

try:
    import uvicorn
    from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException, status
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
except ImportError:
    uvicorn = None
    FastAPI = None

class Stream:
    def __init__(self, voice_input=None, face_recognizer=None, tts=None):
        self.mode = CONFIG.get("stream_mode", "push_receiver")
        self.voice_input = voice_input
        self.face_recognizer = face_recognizer  # FaceRecognizer ref for hot-reload after face add/delete
        self.tts = tts                          # TTS ref for Acoustic Echo Cancellation (speaking gate)
        self._face_reload_lock = threading.Lock()
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

        # ESP32 mic utterance buffering: the firmware streams ~100 ms PCM chunks.
        # Voiced chunks are accumulated here with DC bias removed and flushed
        # to a dedicated background Whisper worker when speech pauses.
        self._esp_audio_chunks: list = []
        self._esp_audio_last_voiced = 0.0
        self._esp_audio_log_ts = 0.0
        self._esp_audio_gap_s = 0.45   # halved from 0.9s for 2x faster end-of-speech response
        self._esp_audio_max_s = 6.0    # cap on buffered utterance length (s)
        self._esp_audio_min_s = 0.25   # shorter buffers are noise — skip them

        # Dedicated background transcription worker queue (decoupled from audio ingest)
        self._transcribe_queue = queue.Queue()
        self._transcribe_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self._transcribe_thread.start()

        # TTS WebSocket broadcast subscribers
        self.tts_clients: set = set()
        self._tts_loop = None  # asyncio event loop, captured at first /ws/tts-feed connect

        self.auth_token = CONFIG.get("stream_auth_token", "")
        self.session_timeout = timedelta(hours=24)
        self.host = CONFIG.get("receiver_host", "0.0.0.0")
        self.port = int(CONFIG.get("receiver_port", 8000))
        self.stream_lan_host = discover_stream_lan_host(CONFIG.get("stream_lan_host", ""))
        self.url = CONFIG.get("stream_url", "http://10.254.76.220/stream")

        self.cap = None
        self.thread = None
        self.server = None
        self.server_thread = None

        if self.mode == "push_receiver":
            print("[Stream] Building push-receiver routes...")
            self._start_push_receiver()
        else:
            print(f"[Stream] Starting pull client for {self.url}...")
            self._start_pull_client()

    # -------------------------------------------------------------
    # PUSH RECEIVER (Cloudflare Tunnel / Ingestion Server)
    # -------------------------------------------------------------
    def _validate_token(self, provided_token: str | None) -> bool:
        return bool(self.auth_token) and provided_token == self.auth_token

    # -------------------------------------------------------------
    # DASHBOARD SESSION MANAGEMENT
    # -------------------------------------------------------------
    def _create_session(self, email: str) -> str:
        """Create a new session token for the given email. Returns the token string."""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + self.session_timeout
        pw_ts = emails_db.get_password_updated_at(email) or ""
        emails_db.create_dashboard_session(token, email, expires_at, pw_ts)
        return token

    def _validate_session(self, token: str | None) -> str | None:
        """Validate a session token. Returns the email if valid, None otherwise."""
        if not token:
            return None
        session = emails_db.get_dashboard_session(token)
        if session is None:
            return None
        if datetime.now(timezone.utc) > datetime.fromisoformat(session["expires_at"]):
            emails_db.delete_dashboard_session(token)
            return None
        email = session["email"]
        # Drop sessions created before THAT user's last password change.
        if (session.get("pw_ts") or "") != (emails_db.get_password_updated_at(email) or ""):
            emails_db.delete_dashboard_session(token)
            return None
        return email

    def _clear_session(self, token: str | None) -> None:
        """Remove a session token."""
        if token:
            emails_db.delete_dashboard_session(token)

    def _require_dashboard_auth(self, request: Request) -> str:
        """Check dashboard authentication.

        Returns the authenticated email or raises HTTPException if login is
        not configured or the session is unauthenticated.
        """
        if not emails_db.list_emails():
            raise HTTPException(
                status_code=503,
                detail="Dashboard login is not configured. Register an account on the server.",
            )

        token = request.cookies.get("guidebot_session")
        email = self._validate_session(token)
        if email is None or not emails_db.is_allowed(email):
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return email

    def _require_dashboard_auth_html(self, request: Request) -> str | RedirectResponse:
        """Like _require_dashboard_auth but redirects to /login for HTML clients.

        Returns:
            str: authenticated email if valid
            RedirectResponse: if unauthenticated, redirect to /login
        """
        if not emails_db.list_emails():
            return RedirectResponse(url="/login?setup=required", status_code=303)

        token = request.cookies.get("guidebot_session")
        email = self._validate_session(token)
        if email is not None and emails_db.is_allowed(email):
            return email

        # Redirect to login with the original path as 'next'
        next_url = request.url.path
        return RedirectResponse(url=f"/login?next={next_url}", status_code=307)

    def _reload_faces(self):
        """Refresh face embeddings without holding up a dashboard request."""
        if self.face_recognizer is None:
            return
        with self._face_reload_lock:
            try:
                self.face_recognizer.reload()
            except Exception as reload_err:
                print(f"[Faces] Warning: hot-reload failed: {reload_err}")

    def _process_frame_bytes(self, raw_bytes: bytes) -> bool:
        if not raw_bytes or len(raw_bytes) == 0:
            return False
        try:
            np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                return False

            frame = cv2.flip(frame, 1)  # always un-mirror the wearable camera

            with self.lock:
                self.latest_frame = frame
                self.latest_timestamp = time.time()
                self.frame_count += 1
            return True
        except Exception as e:
            print(f"[Stream Receiver] Decode error: {e}")
            return False

    def _process_esp32_audio(self, audio_b64: str):
        """Decode an ESP32 mic chunk and accumulate it into an utterance buffer.

        The firmware streams ~100 ms PCM chunks at a fixed cadence. Whisper
        cannot transcribe 100 ms clips — it returns empty text — so voiced
        chunks are buffered here and flushed to Whisper as one utterance when
        speech pauses (_esp_audio_gap_s without a voiced chunk) or the window
        cap is reached.

        All three audio sources (host PC mic, Web UI /audio, ESP32 mic) converge
        into voice_input.command_queue, so main() needs no changes.
        """
        if not CONFIG.get("use_esp32_mic", False):
            return
        if self.voice_input is None:
            print("[ESP32 Mic] Warning: voice_input not wired to Stream; dropping audio.")
            return

        # 1. Acoustic Echo Cancellation (AEC): Drop audio chunks while GuideBot is speaking
        if self.tts and self.tts.is_active():
            return

        try:
            import base64
            raw = base64.b64decode(audio_b64)
            if not raw or len(raw) == 0:
                return

            # ESP32 sends 16-bit PCM little-endian; convert to float32 [-1, 1]
            pcm_int16 = np.frombuffer(raw, dtype="<i2")
            if pcm_int16.size == 0:
                return
            audio = pcm_int16.astype(np.float32) / 32768.0

            # 2. Remove PDM MEMS DC bias: raw PDM decimation filter introduces a
            # static DC bias (~1800 counts = ~0.05 float). Removing the mean brings
            # ambient silence down to ~0.001-0.003 RMS, solving continuous false VAD triggers!
            audio_ac = audio - np.mean(audio)

            sample_rate = int(CONFIG.get("esp32_mic_sample_rate", 16000))
            vad_threshold = float(CONFIG.get("esp32_mic_vad_threshold", 0.015))
            rms = float(np.sqrt(np.mean(audio_ac ** 2))) if audio_ac.size else 0.0
            now = time.time()

            # Chunks arrive continuously even during silence, so a gap since the
            # last voiced chunk means the utterance ended — flush it to background worker.
            if self._esp_audio_chunks and (now - self._esp_audio_last_voiced) >= self._esp_audio_gap_s:
                self._flush_esp32_utterance("speech gap")

            # Throttled diagnostic (~every 2 s): shows whether audio is arriving
            # and whether RMS clears the VAD gate.
            if now - self._esp_audio_log_ts >= 2.0:
                self._esp_audio_log_ts = now
                buffered = sum(c.size for c in self._esp_audio_chunks) / sample_rate
                state = f"voiced, buffered={buffered:.1f}s" if rms >= vad_threshold else "below VAD"
                print(f"[ESP32 Mic] Chunk RMS={rms:.4f} (VAD={vad_threshold:.4f}) -> {state}")

            if rms < vad_threshold:
                return

            self._esp_audio_chunks.append(audio_ac)
            self._esp_audio_last_voiced = now
            buffered = sum(c.size for c in self._esp_audio_chunks) / sample_rate
            if buffered >= self._esp_audio_max_s:
                self._flush_esp32_utterance("max window")
        except Exception as e:
            print(f"[ESP32 Mic] Error processing ESP32 audio: {e}")

    def _flush_esp32_utterance(self, reason: str):
        """Dispatches the buffered ESP32 utterance to the background transcription worker queue.
        Non-blocking: returns in <0.1ms so incoming frames and audio are never stalled."""
        if not self._esp_audio_chunks:
            return
        chunks, self._esp_audio_chunks = self._esp_audio_chunks, []
        audio = np.concatenate(chunks)
        sample_rate = int(CONFIG.get("esp32_mic_sample_rate", 16000))
        seconds = audio.size / sample_rate
        if seconds < self._esp_audio_min_s:
            return  # too short to be real speech — discard noise blip

        # Non-blocking handoff to background worker
        self._transcribe_queue.put((audio, seconds, reason))

    def _transcription_worker(self):
        """Dedicated background thread for Whisper transcription.
        Decoupled from audio ingestion to prevent freezing the event loop or camera ingest."""
        while self.running:
            try:
                item = self._transcribe_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            audio, seconds, reason = item
            if self.voice_input is None or getattr(self.voice_input, "model", None) is None:
                continue

            try:
                t0 = time.time()
                # Run Whisper transcription
                result = self.voice_input.model.transcribe(audio, fp16=False, language="en")
                dur = time.time() - t0
                raw_text = result.get("text", "").strip()
                clean_text = raw_text.lower().rstrip(".?!,")

                # Noise filter: Discard empty, single-character, or common silence hallucination artifacts
                NOISE_ARTIFACTS = {
                    "", ".", "..", "...", "you", "thank you", "thank you.", "subtitles by",
                    "subtitles", "thanks for watching", "bye", "amara.org", "the", "a"
                }
                if clean_text in NOISE_ARTIFACTS or len(clean_text) < 2:
                    continue

                # Echo rejection: Discard text if it matches what GuideBot spoke recently
                if self.tts and self.tts.is_echo(clean_text):
                    print(f"[ESP32 Mic] Discarded self-echo: '{raw_text}'")
                    continue

                print(f"[ESP32 Mic] Heard in {dur:.2f}s ({reason}): '{raw_text}'")
                self.voice_input.command_queue.put(clean_text)
            except Exception as e:
                print(f"[ESP32 Mic] Transcription worker error: {e}")

    def _start_push_receiver(self):
        if FastAPI is None or uvicorn is None:
            raise RuntimeError(
                "[Stream] fastapi and uvicorn are required for push_receiver mode. "
                "Please run: pip install fastapi uvicorn"
            )

        app = FastAPI(title="GuideBot Stream Ingestion Server", docs_url=None, redoc_url=None)
        self.app = app

        @app.middleware("http")
        async def require_dashboard_login(request: Request, call_next):
            if request.url.path in {"/login", "/frame"}:
                return await call_next(request)

            if not emails_db.list_emails():
                if request.method in {"GET", "HEAD"} and "text/html" in request.headers.get("accept", ""):
                    return RedirectResponse(url="/login?setup=required", status_code=303)
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Dashboard login is not configured. Register an account on the server."},
                )

            token = request.cookies.get("guidebot_session")
            email = self._validate_session(token)
            if email is None or not emails_db.is_allowed(email):
                if request.method in {"GET", "HEAD"} and "text/html" in request.headers.get("accept", ""):
                    return RedirectResponse(
                        url=f"/login?next={request.url.path}", status_code=303
                    )
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required"},
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return await call_next(request)

        def render_nav_tabs(active_tab: str) -> str:
            use_web_mic = CONFIG.get("use_web_mic", False)
            tabs = [
                ("home", "/", "🏠 Home"),
                ("esp", "/esp", "📡 ESP Control"),
                ("faces", "/faces", "🧑‍🤝‍🧑 Faces"),
            ]
            if use_web_mic:
                tabs.append(("audio", "/audio", "🎤 Voice Input"))

            def tab(tid: str, path: str, label: str) -> str:
                cls = "app-tab active" if tid == active_tab else "app-tab"
                return f'<a href="{path}" class="{cls}">{label}</a>'

            def logout_control(class_name: str) -> str:
                return (
                    f'<form action="/logout" method="post" class="{class_name}" style="display:inline;margin:0">'
                    '<button type="submit" class="app-tab" style="background:none;border:0;cursor:pointer" '
                    'title="Sign out">🚪 Logout</button></form>'
                )

            # Row 1: Home; Logout moves to the viewport edge on wide screens.
            top_row = tab("home", "/", "🏠 Home") + logout_control("logout-control")
            # Row 2: the remaining nav buttons (ESP, Faces, Voice Input)
            bottom_row = "".join(tab(tid, path, label) for tid, path, label in tabs if tid != "home")

            return (
                '<nav class="app-nav-bar">'
                f'<div class="nav-row top">{top_row}</div>'
                f'<div class="nav-row bottom">{bottom_row}</div>'
                "</nav>"
                + logout_control("logout-control-wide")
            )

        @app.get("/")
        def home(request: Request):
            accept = request.headers.get("accept", "")
            # If request is from browser requesting HTML, render portal page
            if "text/html" in accept:
                # Dashboard auth guard (HTML clients redirect to /login)
                auth = self._require_dashboard_auth_html(request)
                if isinstance(auth, RedirectResponse):
                    return auth
            else:
                # JSON health probe — health() applies the auth guard itself
                return health(request)

            use_web_mic = CONFIG.get("use_web_mic", False)
            audio_link = '<a href="/audio" class="nav-card">🎤<span>Voice Input</span><small>Web Audio recording</small></a>' if use_web_mic else \
                         '<div class="nav-card disabled">🎤<span>Voice Input</span><small>USE_WEB_MIC=false</small></div>'
            nav_html = render_nav_tabs("home")
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GuideBot</title>
  <style>
    :root {{
      --bg: #090a0f; --surface: #13151c; --border: #222634;
      --text: #f1f5f9; --muted: #94a3b8; --primary: #3b82f6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif; }}
    html, body {{ width: 100%; max-width: 100%; overflow-x: clip; }}
    body {{ background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding: calc(16px + env(safe-area-inset-top, 0px)) 16px 32px; -webkit-font-smoothing: antialiased; }}
    .dashboard-kicker, .dashboard-jumps, .dashboard-grid {{ display: none; }}
    
    .app-nav-bar {{
      position: sticky; top: 8px; z-index: 1000; margin: 0 auto 24px;
      display: flex; flex-direction: column; align-items: center; gap: 8px;
      background: rgba(19, 21, 28, 0.92); border: 1px solid var(--border);
      border-radius: 14px; padding: 10px 8px; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      box-shadow: 0 8px 30px rgba(0,0,0,0.55);
      width: min(100%, 560px); max-width: 100%;
    }}
    .nav-row {{
      display: flex; align-items: center; gap: 6px; width: 100%;
    }}
    .nav-row.top  {{ justify-content: space-between; }}
    .nav-row.bottom {{ justify-content: center; flex-wrap: wrap; }}
    .app-tab {{
      padding: 8px 14px; border-radius: 999px; text-decoration: none; color: #94a3b8;
      font-size: 13px; font-weight: 600; transition: all 0.15s; white-space: nowrap;
      flex: 0 0 auto;
    }}
    .app-tab:hover {{ color: #fff; background: rgba(255,255,255,0.08); }}
    .app-tab.active {{ background: var(--primary); color: #fff; box-shadow: 0 2px 10px rgba(59,130,246,0.45); }}

    /* Desktop: keep all tabs in a single centered row */
    @media (min-width: 640px) {{
      .app-nav-bar {{ flex-direction: row; flex-wrap: nowrap; justify-content: center; }}
      .nav-row {{ display: contents; }}
    }}
    .logout-control-wide {{ display: none !important; }}
    @media (min-width: 768px) {{
      body {{ align-items: stretch; padding: 28px clamp(24px, 4vw, 64px) 40px; }}
      .app-nav-bar {{ display: none; }}
      .logout-control {{ display: none !important; }}
      .logout-control-wide {{ display: inline !important; position: fixed; top: 18px; right: 24px; z-index: 1001; }}
      h1 {{ font-size: clamp(32px, 4vw, 54px); letter-spacing: -0.04em; }}
      p.sub {{ margin-bottom: 0; max-width: 660px; line-height: 1.6; }}
      .dashboard-main {{ width: 100%; max-width: 1560px; margin: 0 auto; }}
      .dashboard-header {{ padding: 34px 0 26px; }}
      .dashboard-kicker {{ display: block; color: #60a5fa; font-size: 11px; font-weight: 800; letter-spacing: .16em; margin-bottom: 10px; }}
      .dashboard-jumps {{ display: flex; flex-wrap: wrap; gap: 10px; padding: 0 0 24px; }}
      .dashboard-jumps a {{ color: #cbd5e1; text-decoration: none; font-size: 13px; font-weight: 650; padding: 9px 14px; border: 1px solid var(--border); border-radius: 999px; background: rgba(19,21,28,.8); }}
      .dashboard-jumps a:hover {{ color: #fff; border-color: var(--primary); }}
      .grid {{ display: none !important; }}
      .dashboard-grid {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; }}
      .dashboard-panel {{ min-width: 0; overflow: hidden; border: 1px solid var(--border); border-radius: 20px; background: linear-gradient(145deg, rgba(19,21,28,.96), rgba(13,15,22,.96)); box-shadow: 0 18px 48px rgba(0,0,0,.24); scroll-margin-top: 24px; }}
      .dashboard-panel-header {{ display: flex; align-items: center; gap: 12px; padding: 18px 22px 14px; }}
      .dashboard-panel-icon {{ display: grid; place-items: center; width: 38px; height: 38px; flex: 0 0 38px; border: 1px solid #334155; border-radius: 12px; background: #1a2232; font-size: 19px; }}
      .dashboard-panel-header h2 {{ font-size: 16px; font-weight: 750; }}
      .dashboard-panel-header p {{ margin-top: 3px; color: var(--muted); font-size: 12px; }}
      .dashboard-panel iframe {{ display: block; width: 100%; min-height: 260px; border: 0; }}
      footer {{ margin-top: 28px; align-self: center; }}
    }}
    @media (min-width: 1080px) {{
      .dashboard-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: start; gap: 22px; }}
      #faces-section {{ grid-column: 1 / -1; }}
    }}

    h1 {{ width: 100%; max-width: 640px; font-size: 28px; font-weight: 800; margin-bottom: 6px; }}
    p.sub {{ width: 100%; max-width: 640px; color: var(--muted); font-size: 14px; margin-bottom: 40px; }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; width: 100%; max-width: 640px; }}
    .nav-card {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
      min-width: 0; padding: 16px 18px; display: flex; flex-direction: row; align-items: center; gap: 12px;
      text-decoration: none; color: var(--text); font-size: 15px; font-weight: 600;
      transition: all 0.15s; cursor: pointer;
    }}
    .nav-card:hover {{ border-color: var(--primary); background: #181c2a; }}
    .nav-card.disabled {{ opacity: 0.4; cursor: not-allowed; pointer-events: none; }}
    .nav-card > :first-child {{ font-size: 28px; }}
    .nav-card small {{ margin-left: auto; text-align: right; }}
    small {{ font-size: 11px; color: var(--muted); font-weight: 400; }}
    .status-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block; margin-right: 6px; }}
    footer {{ margin-top: 40px; font-size: 12px; color: var(--muted); }}
  </style>
</head>
<body>
  {nav_html}
  <main class="dashboard-main">
    <header class="dashboard-header">
      <div class="dashboard-kicker">GUIDEBOT · ACCESSIBILITY CONSOLE</div>
      <h1>🦯 GuideBot</h1>
      <p class="sub">Your navigation assistant, device controls, voice tools, and known faces — together in one workspace.</p>
    </header>
    <nav class="dashboard-jumps" aria-label="Dashboard sections">
      <a href="#esp-section">📡 Device</a>
      {f'<a href="#voice-section">🎤 Voice</a>' if use_web_mic else ''}
      <a href="#faces-section">🧑‍🤝‍🧑 Known faces</a>
    </nav>
    <div class="grid">
      <a href="/esp" class="nav-card">📡<span>ESP Control</span><small>Wi-Fi &amp; device</small></a>
      <a href="/faces" class="nav-card">🧑‍🤝‍🧑<span>Known Faces</span><small>Add &amp; manage</small></a>
      {audio_link}
    </div>
    <div class="dashboard-grid">
      <section class="dashboard-panel" id="esp-section">
        <div class="dashboard-panel-header"><span class="dashboard-panel-icon">📡</span><div><h2>Device &amp; Wi-Fi</h2><p>Connection status, network setup, and ESP microphone</p></div></div>
        <iframe title="ESP device and Wi-Fi controls" src="/esp?dashboard_embed=1" loading="lazy" allow="camera; microphone"></iframe>
      </section>
      {'''<section class="dashboard-panel" id="voice-section">
        <div class="dashboard-panel-header"><span class="dashboard-panel-icon">🎤</span><div><h2>Voice tools</h2><p>Talk to GuideBot, send commands, and manage voice output</p></div></div>
        <iframe title="Voice input and output controls" src="/audio?dashboard_embed=1" loading="lazy" allow="microphone"></iframe>
      </section>''' if use_web_mic else ''}
      <section class="dashboard-panel" id="faces-section">
        <div class="dashboard-panel-header"><span class="dashboard-panel-icon">🧑‍🤝‍🧑</span><div><h2>Known faces</h2><p>Add, identify, and manage people recognized by GuideBot</p></div></div>
        <iframe title="Known face management" src="/faces?dashboard_embed=1" loading="lazy" allow="camera"></iframe>
      </section>
    </div>
    <footer>GuideBot Stream Receiver &mdash; port {self.port}</footer>
  </main>
  <script>
    const embeddedFrames = document.querySelectorAll('.dashboard-panel iframe');
    const embedStyles = `
      html, body {{ height: auto !important; min-height: 0 !important; overflow: hidden !important; background: transparent !important; }}
      body {{ display: block !important; padding: 0 !important; margin: 0 !important; }}
      .app-nav-bar, .logout-control-wide, footer, .back-link {{ display: none !important; }}
      .card {{ width: 100% !important; max-width: none !important; margin: 0 !important; padding: clamp(18px, 2.4vw, 30px) !important; border-radius: 0 !important; box-shadow: none !important; }}
    `;
    embeddedFrames.forEach(frame => frame.addEventListener('load', () => {{
      const doc = frame.contentDocument;
      const style = doc.createElement('style');
      style.textContent = embedStyles;
      if (frame.closest('#faces-section')) {{
        style.textContent += '#videoWrap, #previewImg {{ max-width: 440px; margin-left: auto; margin-right: auto; }}';
      }}
      doc.head.appendChild(style);
      const resize = () => {{
        const height = Math.max(doc.body.scrollHeight, doc.documentElement.scrollHeight);
        if (Math.abs(frame.getBoundingClientRect().height - height) > 2) frame.style.height = height + 'px';
      }};
      const observer = new ResizeObserver(resize);
      observer.observe(doc.body);
      observer.observe(doc.documentElement);
      resize();
    }}));
  </script>
</body>
</html>"""
            return HTMLResponse(content=html)

        @app.get("/health")
        def health(request: Request):
            # Dashboard auth guard (JSON API)
            self._require_dashboard_auth(request)
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

        @app.websocket("/ws/frame")
        async def websocket_frame(websocket: WebSocket):
            token = websocket.query_params.get("token") or websocket.headers.get("X-Auth-Token")
            if not self._validate_token(token):
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            await websocket.accept()
            print("[Stream Receiver] ESP32 connected directly over LAN for video.")
            try:
                while self.running:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        print(
                            "[Stream Receiver] Direct LAN video disconnected "
                            f"(code={message.get('code')}, reason={message.get('reason', '')!r})."
                        )
                        break
                    if "bytes" in message and message["bytes"]:
                        if not await asyncio.to_thread(
                            self._process_frame_bytes, message["bytes"]
                        ):
                            print("[Stream Receiver] Dropped invalid LAN frame payload.")
                    elif message.get("text", "").strip() == "PING":
                        await websocket.send_text("PONG")
            except WebSocketDisconnect as exc:
                print(
                    "[Stream Receiver] Direct LAN video disconnected "
                    f"(code={exc.code}, reason={exc.reason!r})."
                )

        async def broadcast_esp_status(connected: bool):
            msg = json.dumps({"type": "esp_status", "connected": connected})
            dead_clients = set()
            for client in list(self.dash_clients):
                try:
                    await client.send_text(msg)
                except Exception:
                    dead_clients.add(client)
            self.dash_clients.difference_update(dead_clients)

        def broadcast_tts_text(text: str):
            """Called from TTS worker thread — schedules async broadcast to all
            /ws/tts-feed WebSocket clients via the captured asyncio event loop."""
            if not self.tts_clients or self._tts_loop is None:
                return

            async def _send():
                dead = set()
                msg = json.dumps({"type": "tts", "text": text})
                for ws in list(self.tts_clients):
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.add(ws)
                self.tts_clients.difference_update(dead)

            asyncio.run_coroutine_threadsafe(_send(), self._tts_loop)

        # Store reference so main.py can wire it to TTS
        self.broadcast_tts_text = broadcast_tts_text

        @app.websocket("/ws/tts-feed")
        async def tts_feed(websocket: WebSocket):
            token = websocket.cookies.get("guidebot_session")
            email = self._validate_session(token)
            if email is None or not emails_db.is_allowed(email):
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            await websocket.accept()
            self._tts_loop = asyncio.get_running_loop()
            self.tts_clients.add(websocket)
            try:
                while True:
                    # Keep alive; client sends nothing meaningful
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                self.tts_clients.discard(websocket)

        @app.websocket("/ws/stream")
        async def websocket_stream(websocket: WebSocket):
            token = websocket.query_params.get("token") or websocket.headers.get("X-Auth-Token")
            if not self._validate_token(token):
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return

            await websocket.accept()
            print("[Stream Receiver] ESP32 connected via WebSocket.")

            if self.stream_lan_host:
                await websocket.send_text(
                    f"STREAM_TARGET:{self.stream_lan_host}:{self.port}"
                )

            # Set connection references
            self.esp_ws = websocket
            self.esp_ws_loop = asyncio.get_running_loop()

            # Broadcast connected status to all /esp dashboard WebSocket clients
            await broadcast_esp_status(True)

            # Tell the ESP32 whether it may downgrade to QVGA on slow networks.
            # allow_low_res=False (default) → force VGA always (FORCE_HIGH_RES:1)
            # allow_low_res=True           → let ESP32 adapt     (FORCE_HIGH_RES:0)
            allow_low_res = CONFIG.get("allow_low_res", False)
            cmd = "FORCE_HIGH_RES:0" if allow_low_res else "FORCE_HIGH_RES:1"
            await websocket.send_text(cmd)
            print(f"[Stream Receiver] Sent resolution command to ESP32: {cmd}")

            # If the server is configured to use the ESP32 I2S mic, enable the
            # device's mic streaming on connect. The ESP32 persists its own mic
            # state and defaults to OFF, so without this USE_ESP32_MIC=true
            # would never actually receive any AUDIO_DATA.
            if CONFIG.get("use_esp32_mic", False):
                await websocket.send_text("TOGGLE_MIC:1")
                print("[Stream Receiver] USE_ESP32_MIC=true -> sent TOGGLE_MIC:1 to ESP32.")

            try:
                while self.running:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        break
                    if "bytes" in message and message["bytes"]:
                        await asyncio.to_thread(
                            self._process_frame_bytes, message["bytes"]
                        )
                    elif "text" in message and message["text"]:
                        txt = message["text"].strip()
                        if txt == "PING":
                            await websocket.send_text("PONG")
                        elif txt == "GET_STREAM_TARGET":
                            self.stream_lan_host = discover_stream_lan_host(
                                CONFIG.get("stream_lan_host", "")
                            )
                            if self.stream_lan_host:
                                await websocket.send_text(
                                    f"STREAM_TARGET:{self.stream_lan_host}:{self.port}"
                                )
                                print(
                                    f"[Stream Receiver] Refreshed LAN video target: "
                                    f"{self.stream_lan_host}:{self.port}"
                                )
                            else:
                                await websocket.send_text("STREAM_TARGET_UNAVAILABLE")
                        elif txt.startswith("SCAN_RESULTS:"):
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
                        elif txt.startswith("AUDIO_DATA:"):
                            audio_b64 = txt[len("AUDIO_DATA:"):]
                            # Offload decode + VAD + Whisper to a worker thread.
                            # Transcription can take seconds; running it inline would
                            # freeze the asyncio event loop, stalling video frame
                            # ingest and every other HTTP/WS client.
                            await asyncio.to_thread(self._process_esp32_audio, audio_b64)
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

            @app.get("/audio", response_class=HTMLResponse)
            def audio_control_panel(request: Request):
                # Dashboard auth guard
                auth = self._require_dashboard_auth_html(request)
                if isinstance(auth, RedirectResponse):
                    return auth

                nav_html = render_nav_tabs("audio")
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
    body { background: var(--bg); color: var(--text); padding: calc(16px + env(safe-area-inset-top, 0px)) 16px 32px; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; min-height: 100vh; -webkit-font-smoothing: antialiased; }
    
    .app-nav-bar {
      position: sticky; top: 8px; z-index: 1000; margin: 0 auto 24px;
      display: flex; flex-direction: column; align-items: center; gap: 8px;
      background: rgba(19, 21, 28, 0.92); border: 1px solid var(--border);
      border-radius: 14px; padding: 10px 8px; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      box-shadow: 0 8px 30px rgba(0,0,0,0.55);
      width: 100%; max-width: 560px;
    }
    .nav-row {
      display: flex; align-items: center; gap: 6px; width: 100%;
    }
    .nav-row.top  { justify-content: space-between; }
    .nav-row.bottom { justify-content: center; flex-wrap: wrap; }
    .app-tab {
      padding: 8px 14px; border-radius: 999px; text-decoration: none; color: #94a3b8;
      font-size: 13px; font-weight: 600; transition: all 0.15s; white-space: nowrap;
      flex: 0 0 auto;
    }
    .app-tab:hover { color: #fff; background: rgba(255,255,255,0.08); }
    .app-tab.active { background: var(--primary); color: #fff; box-shadow: 0 2px 10px rgba(59,130,246,0.45); }

    /* Desktop: keep all tabs in a single centered row */
    @media (min-width: 640px) {
      .app-nav-bar { flex-direction: row; flex-wrap: nowrap; justify-content: center; }
      .nav-row { display: contents; }
    }
    .logout-control-wide { display: none !important; }
    @media (min-width: 900px) {
      .logout-control { display: none !important; }
      .logout-control-wide { display: inline !important; position: fixed; top: 16px; right: 24px; z-index: 1001; }
    }

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
    .tts-toggle-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
    .tts-toggle-btn {
      position: relative; width: 46px; height: 26px; border-radius: 999px; border: none; cursor: pointer;
      background: #222634; transition: background 0.2s; flex-shrink: 0;
    }
    .tts-toggle-btn.on { background: var(--success); }
    .tts-toggle-btn::after {
      content: ''; position: absolute; top: 3px; left: 3px; width: 20px; height: 20px;
      border-radius: 50%; background: #fff; transition: transform 0.2s;
    }
    .tts-toggle-btn.on::after { transform: translateX(20px); }
    .tts-log {
      max-height: 180px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px;
      padding: 4px 0;
    }
    .tts-log-item {
      background: #1a1d27; border: 1px solid var(--border); border-radius: 10px;
      padding: 10px 12px; font-size: 13px; line-height: 1.4; color: var(--text);
      display: flex; gap: 10px; align-items: flex-start;
    }
    .tts-log-time { font-size: 11px; color: var(--text-muted); white-space: nowrap; margin-top: 1px; }
    .tts-empty { text-align: center; color: var(--text-muted); font-size: 13px; padding: 12px 0; }
    .tts-speaking-indicator { display: none; font-size: 12px; color: var(--success); margin-left: 8px; }
    .tts-speaking-indicator.active { display: inline; }
  </style>
</head>
<body>
  {{NAV_BAR}}
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

    <div class="divider"></div>

    <!-- Voice Output Section -->
    <h2>Voice Output <span id="ttsSpeaking" class="tts-speaking-indicator">&#128266; speaking...</span></h2>
    <div class="tts-toggle-row">
      <span style="font-size:13px;color:var(--text-muted);">Play GuideBot speech in browser</span>
      <button id="ttsToggle" class="tts-toggle-btn" onclick="toggleTtsOutput()" title="Enable/disable voice output"></button>
    </div>
    <div id="ttsLog" class="tts-log">
      <div class="tts-empty">Voice output is disabled. Enable above to hear GuideBot.</div>
    </div>
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
        if (!res.ok) {
          throw new Error(data.detail || "Server rejected the command");
        }
        showFeedback("Sent: '" + text + "'");
      } catch (err) {
        showFeedback("Failed to send command: " + err.message, true);
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

    // ── TTS Output ─────────────────────────────────────────────
    let ttsEnabled = false;
    let ttsWs      = null;
    let ttsLog     = [];

    function toggleTtsOutput() {
      ttsEnabled = !ttsEnabled;
      const btn = document.getElementById('ttsToggle');
      btn.classList.toggle('on', ttsEnabled);
      if (ttsEnabled) {
        connectTtsFeed();
      } else {
        if (ttsWs) { ttsWs.close(); ttsWs = null; }
        window.speechSynthesis.cancel();
      }
      updateTtsLog();
    }

    function connectTtsFeed() {
      if (ttsWs) return;
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ttsWs = new WebSocket(proto + '://' + location.host + '/ws/tts-feed');
      ttsWs.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.type === 'tts' && data.text) onTtsMessage(data.text);
        } catch(e) {}
      };
      ttsWs.onclose = () => {
        ttsWs = null;
        if (ttsEnabled) setTimeout(connectTtsFeed, 2000);
      };
    }

    function onTtsMessage(text) {
      const ts = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
      ttsLog.unshift({ text, ts });
      if (ttsLog.length > 30) ttsLog.pop();
      updateTtsLog();
      if (!ttsEnabled) return;
      window.speechSynthesis.cancel();
      const utt = new SpeechSynthesisUtterance(text);
      utt.rate = 1.1;
      utt.onstart = () => document.getElementById('ttsSpeaking').classList.add('active');
      utt.onend   = () => document.getElementById('ttsSpeaking').classList.remove('active');
      window.speechSynthesis.speak(utt);
    }

    function updateTtsLog() {
      const logEl = document.getElementById('ttsLog');
      if (!ttsEnabled || ttsLog.length === 0) {
        logEl.innerHTML = '<div class="tts-empty">' +
          (ttsEnabled ? 'Waiting for GuideBot to speak...' : 'Voice output is disabled. Enable above to hear GuideBot.') +
          '</div>';
        return;
      }
      logEl.innerHTML = ttsLog.map(e =>
        '<div class="tts-log-item"><span class="tts-log-time">' + e.ts + '</span>' + escHtml(e.text) + '</div>'
      ).join('');
    }

    function escHtml(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
  </script>
</body>
</html>"""
                return HTMLResponse(content=html.replace("{{NAV_BAR}}", nav_html))

            @app.post("/audio/command")
            async def submit_audio_command(request: Request):
                client = request.client.host if request.client else "unknown"
                print(f"[Voice / Web] Command request received from {client}.")
                # Dashboard auth guard (JSON API)
                self._require_dashboard_auth(request)
                body = await request.json()
                command = body.get("command", "").strip()
                if not command:
                    raise HTTPException(status_code=400, detail="Empty command")
                if self.voice_input:
                    self.voice_input.submit_command(command)
                else:
                    print("[Voice / Web] Command dropped: voice input is unavailable.")
                return {"status": "ok", "command": command}

            @app.post("/audio/upload")
            async def upload_audio_recording(request: Request):
                client = request.client.host if request.client else "unknown"
                print(f"[Voice / Web] Audio upload received from {client}.")
                # Dashboard auth guard (JSON API)
                self._require_dashboard_auth(request)
                if not self.voice_input:
                    raise HTTPException(status_code=500, detail="Voice input not initialized")

                content = await request.body()
                if not content:
                    raise HTTPException(status_code=400, detail="Empty audio content")
                print(f"[Voice / Web] Processing {len(content)} audio bytes with Whisper.")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                try:
                    text = self.voice_input.transcribe_audio_file(tmp_path)
                    if not text:
                        print("[Voice / Web] Whisper produced no usable command.")
                    return {"status": "ok", "transcription": text}
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        # -------------------------------------------------------------
        # ESP MANAGEMENT & REMOTE CONTROL DASHBOARD (/esp)
        # -------------------------------------------------------------
        @app.get("/esp", response_class=HTMLResponse)
        def esp_control_panel(request: Request):
            # Dashboard auth guard (HTML clients redirect to /login).
            # NOTE: the ESP transport itself (/frame, /ws/stream) stays
            # STREAM_AUTH_TOKEN-only so the ESP32 (which has no dashboard
            # session) keeps working.
            auth = self._require_dashboard_auth_html(request)
            if isinstance(auth, RedirectResponse):
                return auth
            current_ssid = self.esp_current_ssid or ""
            nav_html = render_nav_tabs("esp")
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
    body {{ background: var(--bg); color: var(--text); padding: calc(16px + env(safe-area-inset-top, 0px)) 16px 32px; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; min-height: 100vh; -webkit-font-smoothing: antialiased; }}
    
    .app-nav-bar {{
      position: sticky; top: 8px; z-index: 1000; margin: 0 auto 24px;
      display: flex; flex-direction: column; align-items: center; gap: 8px;
      background: rgba(19, 21, 28, 0.92); border: 1px solid var(--border);
      border-radius: 14px; padding: 10px 8px; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      box-shadow: 0 8px 30px rgba(0,0,0,0.55);
      width: 100%; max-width: 560px;
    }}
    .nav-row {{
      display: flex; align-items: center; gap: 6px; width: 100%;
    }}
    .nav-row.top  {{ justify-content: space-between; }}
    .nav-row.bottom {{ justify-content: center; flex-wrap: wrap; }}
    .app-tab {{
      padding: 8px 14px; border-radius: 999px; text-decoration: none; color: #94a3b8;
      font-size: 13px; font-weight: 600; transition: all 0.15s; white-space: nowrap;
      flex: 0 0 auto;
    }}
    .app-tab:hover {{ color: #fff; background: rgba(255,255,255,0.08); }}
    .app-tab.active {{ background: var(--primary); color: #fff; box-shadow: 0 2px 10px rgba(59,130,246,0.45); }}

    /* Desktop: keep all tabs in a single centered row */
    @media (min-width: 640px) {{
      .app-nav-bar {{ flex-direction: row; flex-wrap: nowrap; justify-content: center; }}
      .nav-row {{ display: contents; }}
    }}
    .logout-control-wide {{ display: none !important; }}
    @media (min-width: 900px) {{
      .logout-control {{ display: none !important; }}
      .logout-control-wide {{ display: inline !important; position: fixed; top: 16px; right: 24px; z-index: 1001; }}
    }}

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

    .switch {{ position: relative; display: inline-block; width: 44px; height: 24px; }}
    .switch input {{ opacity: 0; width: 0; height: 0; }}
    .slider {{ position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
      background: #222634; transition: .4s; border-radius: 24px; }}
    .slider:before {{ position: absolute; content: ""; height: 18px; width: 18px;
      left: 3px; bottom: 3px; background: #fff; transition: .4s; border-radius: 50%; }}
    .switch input:checked + .slider {{ background: var(--success); }}
    .switch input:checked + .slider:before {{ transform: translateX(20px); }}

    .helper {{ font-size: 12px; color: var(--text-muted); margin: 0; }}
  </style>
</head>
<body>
  {nav_html}
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
    <div class="field-group" style="margin-top:12px;">
      <label style="display:flex; justify-content:space-between; align-items:center;">
        <span>ESP32 Microphone</span>
        <label class="switch" style="margin:0;">
          <input type="checkbox" id="micToggle" onchange="toggleMic(this)">
          <span class="slider"></span>
        </label>
      </label>
      <p class="helper" style="font-size:12px; margin-top:4px; color:var(--text-muted);">
        Enable/disable I2S microphone audio streaming from ESP32-S3 Sense.
      </p>
    </div>

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
        const contentType = res.headers.get("content-type") || "";
        if (!contentType.includes("application/json")) {{
          const message = res.status === 503
            ? "Scan service is temporarily unavailable (HTTP 503). Check the server/tunnel and try again."
            : `Scan request failed (HTTP ${{res.status}}): server returned an unexpected response.`;
          throw new Error(message);
        }}
        const data = await res.json();
        if (!res.ok) {{
          showFeedback(data.detail || `Scan request failed (HTTP ${{res.status}}).`, true);
          return;
        }}
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

    async function toggleMic() {{
      if (!isConnected) return;
      const btn = document.getElementById('micToggleBtn');
      const currentlyOn = btn.classList.contains('btn-warning');
      const newState = !currentlyOn;
      showFeedback(`Turning ESP32 mic ${{newState ? 'ON' : 'OFF'}}...`);
      try {{
        const res = await fetch("/esp/toggle_mic", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ enabled: newState }})
        }});
        const data = await res.json();
        if (data.status === "ok") {{
          btn.classList.toggle('btn-warning', newState);
          btn.classList.toggle('btn-info', !newState);
          btn.innerText = newState ? "🎤 ESP32 Mic: ON" : "🎤 Toggle ESP32 Mic";
          showFeedback(`ESP32 mic ${{data.message}}`);
        }} else {{
          showFeedback(data.detail || "Failed to toggle mic.", true);
        }}
      }} catch (err) {{
        showFeedback("Failed to toggle mic: " + err.message, true);
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

    async function toggleMic(checkbox) {{
      const enabled = checkbox.checked;
      try {{
        const res = await fetch("/esp/mic", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ enabled: enabled }})
        }});
        if (!res.ok) {{
          showFeedback("Failed to toggle ESP32 microphone", true);
          checkbox.checked = !enabled;
          return;
        }}
        const data = await res.json();
        showFeedback(data.mic_enabled ? "ESP32 microphone enabled" : "ESP32 microphone disabled");
      }} catch (e) {{
        showFeedback("Error toggling microphone: " + e, true);
        checkbox.checked = !enabled;
      }}
    }}
  </script>
</body>
</html>"""
            return HTMLResponse(content=html)

        @app.get("/esp/status")
        def esp_status(request: Request):
            # Dashboard auth guard (JSON API). Only the raw ESP transport
            # (/frame, /ws/stream) stays STREAM_AUTH_TOKEN-only.
            self._require_dashboard_auth(request)
            connected = self.esp_ws is not None
            return {"status": "ok", "connected": connected}

        @app.websocket("/ws/esp-status")
        async def websocket_esp_status(websocket: WebSocket):
            # Dashboard socket — requires a valid login session.
            # (Distinct from /ws/stream, which is the ESP's STREAM_AUTH_TOKEN channel.)
            token = websocket.cookies.get("guidebot_session")
            email = self._validate_session(token)
            if email is None or not emails_db.is_allowed(email):
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
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
        async def esp_scan(request: Request):
            # Dashboard auth guard (JSON API)
            self._require_dashboard_auth(request)
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
            # Dashboard auth guard (JSON API)
            self._require_dashboard_auth(request)
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
        async def esp_reboot(request: Request):
            # Dashboard auth guard (JSON API)
            self._require_dashboard_auth(request)
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
        async def esp_erase(request: Request):
            # Dashboard auth guard (JSON API)
            self._require_dashboard_auth(request)
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

        @app.post("/esp/toggle_mic")
        async def esp_toggle_mic(request: Request):
            # Dashboard auth guard (JSON API)
            self._require_dashboard_auth(request)
            if not self.esp_ws:
                raise HTTPException(status_code=400, detail="ESP32 is not connected via WebSocket")

            body = await request.json()
            enabled = bool(body.get("enabled", False))
            cmd = f"TOGGLE_MIC:{1 if enabled else 0}"
            try:
                await self.esp_ws.send_text(cmd)
                print(f"[Stream Receiver] Sent TOGGLE_MIC:{1 if enabled else 0} to ESP32.")
                return {"status": "ok", "message": f"ESP32 mic {'enabled' if enabled else 'disabled'}"}
            except Exception as e:
                self.esp_ws = None
                await broadcast_esp_status(False)
                raise HTTPException(status_code=500, detail=f"Failed to deliver command to ESP32: {e}")

        @app.post("/esp/mic")
        async def esp_mic_toggle(request: Request):
            # Dashboard auth guard (JSON API)
            self._require_dashboard_auth(request)
            if not self.esp_ws:
                raise HTTPException(status_code=400, detail="ESP32 is not connected via WebSocket")

            body = await request.json()
            enabled = bool(body.get("enabled", False))

            try:
                cmd = f"TOGGLE_MIC:{1 if enabled else 0}"
                await self.esp_ws.send_text(cmd)
                print(f"[Stream Receiver] Sent {cmd} to ESP32.")
                return {"status": "ok", "mic_enabled": enabled}
            except Exception as e:
                self.esp_ws = None
                await broadcast_esp_status(False)
                raise HTTPException(status_code=500, detail=f"Failed to deliver mic command to ESP32: {e}")

        # -------------------------------------------------------------
        # KNOWN FACES MANAGEMENT DASHBOARD (/faces)
        # -------------------------------------------------------------
        @app.get("/faces", response_class=HTMLResponse)
        def faces_panel(request: Request):
            # Dashboard auth guard
            auth = self._require_dashboard_auth_html(request)
            if isinstance(auth, RedirectResponse):
                return auth

            nav_html = render_nav_tabs("faces")
            html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GuideBot — Known Faces</title>
  <style>
    :root {
      --bg: #090a0f; --surface: #13151c; --border: #222634;
      --text: #f1f5f9; --muted: #94a3b8; --primary: #3b82f6;
      --primary-hover: #2563eb; --success: #10b981; --danger: #ef4444;
      --danger-hover: #dc2626; --warning: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: calc(16px + env(safe-area-inset-top, 0px)) 16px 32px; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; min-height: 100vh; -webkit-font-smoothing: antialiased; }
    
    .app-nav-bar {
      position: sticky; top: 8px; z-index: 1000; margin: 0 auto 24px;
      display: flex; flex-direction: column; align-items: center; gap: 8px;
      background: rgba(19, 21, 28, 0.92); border: 1px solid var(--border);
      border-radius: 14px; padding: 10px 8px; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      box-shadow: 0 8px 30px rgba(0,0,0,0.55);
      width: 100%; max-width: 560px;
    }
    .nav-row {
      display: flex; align-items: center; gap: 6px; width: 100%;
    }
    .nav-row.top  { justify-content: space-between; }
    .nav-row.bottom { justify-content: center; flex-wrap: wrap; }
    .app-tab {
      padding: 8px 14px; border-radius: 999px; text-decoration: none; color: #94a3b8;
      font-size: 13px; font-weight: 600; transition: all 0.15s; white-space: nowrap;
      flex: 0 0 auto;
    }
    .app-tab:hover { color: #fff; background: rgba(255,255,255,0.08); }
    .app-tab.active { background: var(--primary); color: #fff; box-shadow: 0 2px 10px rgba(59,130,246,0.45); }

    /* Desktop: keep all tabs in a single centered row */
    @media (min-width: 640px) {
      .app-nav-bar { flex-direction: row; flex-wrap: nowrap; justify-content: center; }
      .nav-row { display: contents; }
    }
    .logout-control-wide { display: none !important; }
    @media (min-width: 900px) {
      .logout-control { display: none !important; }
      .logout-control-wide { display: inline !important; position: fixed; top: 16px; right: 24px; z-index: 1001; }
    }

    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 28px; width: 100%; max-width: 520px; box-shadow: 0 16px 40px rgba(0,0,0,0.5); }
    .header { text-align: center; margin-bottom: 28px; }
    h1 { font-size: 21px; font-weight: 800; margin-bottom: 4px; }
    .sub { font-size: 13px; color: var(--muted); }
    .back-link { display: inline-block; margin-bottom: 18px; font-size: 13px; color: var(--primary); text-decoration: none; }
    .back-link:hover { text-decoration: underline; }

    .section-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: var(--muted); margin-bottom: 14px; }
    .field-group { margin-bottom: 16px; }
    label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--muted); }
    input[type=text] {
      width: 100%; padding: 11px 14px; background: #1a1d27; border: 1px solid var(--border);
      border-radius: 10px; color: #fff; font-size: 14px; outline: none;
    }
    input[type=text]:focus { border-color: var(--primary); }

    /* Tabs */
    .tabs { display: flex; gap: 0; margin-bottom: 20px; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
    .tab-btn {
      flex: 1; padding: 10px; background: transparent; border: none; color: var(--muted);
      font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s;
    }
    .tab-btn.active { background: #1e2232; color: #fff; }
    .tab-btn:hover:not(.active) { background: #161a25; }
    .tab-pane { display: none; }
    .tab-pane.active { display: block; }

    /* Webcam */
    #videoWrap { position: relative; border-radius: 12px; overflow: hidden; background: #000; margin-bottom: 12px; aspect-ratio: 4/3; }
    #videoEl { width: 100%; height: 100%; object-fit: cover; display: block; }
    #snapCanvas { display: none; }
    #previewImg { width: 100%; border-radius: 12px; margin-bottom: 12px; display: none; }
    .cam-flip-btn {
      position: absolute; bottom: 10px; right: 10px; z-index: 10;
      width: 36px; height: 36px; border-radius: 50%; border: none; cursor: pointer;
      background: rgba(0,0,0,0.55); color: #fff; font-size: 18px;
      display: flex; align-items: center; justify-content: center;
      backdrop-filter: blur(6px); transition: background 0.15s, transform 0.2s;
    }
    .cam-flip-btn:hover { background: rgba(59,130,246,0.7); }
    .cam-flip-btn:active { transform: rotate(180deg); }

    /* File drop */
    .drop-zone {
      border: 2px dashed var(--border); border-radius: 12px; padding: 32px 20px;
      text-align: center; cursor: pointer; transition: all 0.15s; margin-bottom: 12px;
    }
    .drop-zone:hover, .drop-zone.dragover { border-color: var(--primary); background: #111828; }
    .drop-zone .icon { font-size: 32px; margin-bottom: 8px; }
    .drop-zone p { font-size: 13px; color: var(--muted); }
    #fileInput { display: none; }
    #filePreview { width: 100%; border-radius: 12px; margin-bottom: 12px; display: none; }

    /* Buttons */
    .btn {
      width: 100%; padding: 12px 16px; border-radius: 10px; font-size: 14px; font-weight: 600;
      cursor: pointer; border: none; transition: all 0.15s; display: flex; align-items: center; justify-content: center; gap: 8px;
    }
    .btn-primary { background: var(--primary); color: #fff; }
    .btn-primary:hover:not(:disabled) { background: var(--primary-hover); }
    .btn-secondary { background: #222634; border: 1px solid var(--border); color: var(--text); }
    .btn-secondary:hover:not(:disabled) { border-color: var(--primary); color: #fff; }
    .btn-danger-sm {
      padding: 6px 12px; border-radius: 8px; font-size: 12px; font-weight: 600;
      background: transparent; border: 1px solid #7f1d1d; color: #fca5a5; cursor: pointer; transition: all 0.15s;
    }
    .btn-danger-sm:hover { background: rgba(239,68,68,0.12); border-color: var(--danger); color: #fff; }
    .btn-edit-sm {
      padding: 6px 12px; border-radius: 8px; font-size: 12px; font-weight: 600;
      background: transparent; border: 1px solid var(--primary); color: var(--primary); cursor: pointer; transition: all 0.15s;
    }
    .btn-edit-sm:hover { background: rgba(59,130,246,0.12); border-color: var(--primary-hover); color: #fff; }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }

    .divider { height: 1px; background: var(--border); margin: 24px 0; }
    .helper { font-size: 12px; color: var(--muted); margin-top: 6px; }

    /* Faces list */
    .faces-list { display: flex; flex-direction: column; gap: 10px; margin-top: 12px; }
    .face-row {
      display: flex; align-items: center; gap: 14px; background: #1a1d27;
      border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px;
    }
    .face-thumb { width: 52px; height: 52px; border-radius: 8px; object-fit: cover; flex-shrink: 0; background: #222634; }
    .face-info { flex: 1; min-width: 0; }
    .face-name { font-size: 14px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .face-meta { font-size: 11px; color: var(--muted); margin-top: 2px; }
    .empty-state { text-align: center; color: var(--muted); font-size: 13px; padding: 24px 0; }

    /* Feedback */
    .feedback {
      margin-top: 16px; padding: 12px 14px; border-radius: 10px; font-size: 13px;
      display: none; line-height: 1.4;
    }
    .feedback.success { background: #0d2218; border: 1px solid #059669; color: #a7f3d0; }
    .feedback.error   { background: #1f0e0e; border: 1px solid var(--danger); color: #fca5a5; }

    /* Modal */
    .modal-overlay {
      position: fixed; inset: 0; background: rgba(0,0,0,0.75); backdrop-filter: blur(4px);
      display: none; align-items: center; justify-content: center; padding: 16px; z-index: 99;
    }
    .modal {
      background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
      padding: 24px; max-width: 340px; width: 100%; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.6);
    }
    .modal h3 { font-size: 17px; font-weight: 700; margin-bottom: 8px; }
    .modal p  { font-size: 13px; color: var(--muted); margin-bottom: 20px; }
    .modal-actions { display: flex; gap: 10px; }
    .modal-actions button { flex: 1; padding: 11px; font-size: 13px; font-weight: 600; border-radius: 9px; border: none; cursor: pointer; }
    .btn-modal-cancel  { background: #222634; color: var(--text); }
    .btn-modal-confirm { background: var(--danger); color: #fff; }
  </style>
</head>
<body>
  {{NAV_BAR}}
  <div class="card">
    <a href="/" class="back-link">← Back to Dashboard</a>
    <div class="header">
      <h1>🧑‍🤝‍🧑 Known Faces</h1>
      <p class="sub">Add &amp; manage faces for recognition</p>
    </div>

    <!-- Add Face Section -->
    <div class="section-title">Add New Face</div>

    <div class="field-group">
      <label>Person's Name</label>
      <input type="text" id="nameInput" placeholder="e.g. Nathan Pereira" maxlength="80">
      <div class="helper">Used as the display label during recognition. Spaces allowed.</div>
    </div>

    <!-- Source Tabs -->
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('webcam', this)">📷 Webcam</button>
      <button class="tab-btn"       onclick="switchTab('file', this)">📁 Upload File</button>
    </div>

    <!-- Webcam pane -->
    <div id="tab-webcam" class="tab-pane active">
      <div id="videoWrap">
        <video id="videoEl" autoplay playsinline muted></video>
        <button id="flipBtn" class="cam-flip-btn" onclick="flipCamera()" title="Switch camera" style="display:none;">🔄</button>
      </div>
      <canvas id="snapCanvas"></canvas>
      <img id="previewImg" alt="Captured photo">
      <div style="display:flex;gap:8px;margin-bottom:12px;">
        <button class="btn btn-secondary" id="camBtn" onclick="startCamera()" style="flex:1;">Start Camera</button>
        <button class="btn btn-secondary" id="snapBtn" onclick="snapPhoto()" style="flex:1;" disabled>📸 Capture</button>
        <button class="btn btn-secondary" id="retakeBtn" onclick="retakePhoto()" style="flex:1;display:none;">↺ Retake</button>
      </div>
      <button class="btn btn-primary" id="submitWebcamBtn" onclick="submitFace('webcam')" disabled>Save Face</button>
    </div>

    <!-- File upload pane -->
    <div id="tab-file" class="tab-pane">
      <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()"
           ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)" ondrop="handleDrop(event)">
        <div class="icon">🖼️</div>
        <p>Click to choose an image or drag &amp; drop here</p>
        <p style="margin-top:6px;font-size:11px;">JPG, PNG, WebP accepted</p>
      </div>
      <input type="file" id="fileInput" accept=".jpg,.jpeg,.png,.webp" onchange="onFileChosen(event)">
      <img id="filePreview" alt="Preview">
      <button class="btn btn-primary" id="submitFileBtn" onclick="submitFace('file')" disabled>Save Face</button>
    </div>

    <div id="feedback" class="feedback"></div>

    <div class="divider"></div>

    <!-- Faces List -->
    <div class="section-title">Registered Faces <span id="faceCount" style="color:var(--text);font-size:12px;"></span></div>
    <div id="facesList" class="faces-list">
      <div class="empty-state">Loading…</div>
    </div>
  </div>

  <!-- Delete Confirm Modal -->
  <div id="modal" class="modal-overlay">
    <div class="modal">
      <h3 id="mTitle">Delete Face?</h3>
      <p id="mDesc">This will remove the image and registration permanently.</p>
      <div class="modal-actions">
        <button class="btn-modal-cancel"  onclick="closeModal()">Cancel</button>
        <button class="btn-modal-confirm" id="mConfirm" onclick="confirmDelete()">Delete</button>
      </div>
    </div>
  </div>

  <!-- Edit Name Modal -->
  <div id="editModal" class="modal-overlay">
    <div class="modal">
      <h3 id="eTitle">Edit Name</h3>
      <p class="sub" style="margin-bottom:20px;">Update the display label for this face.</p>
      <div class="field-group" style="margin-bottom:20px;">
        <label>New Name</label>
        <input type="text" id="eNameInput" placeholder="Enter new name" maxlength="80">
      </div>
      <div class="modal-actions">
        <button class="btn-modal-cancel" onclick="closeEditModal()">Cancel</button>
        <button class="btn-modal-confirm" id="eConfirm" onclick="confirmEdit()">Save</button>
      </div>
    </div>
  </div>

  <script>
    // ── State ──────────────────────────────────────────────────────────
    let stream       = null;   // MediaStream
    let capturedBlob = null;   // Blob from webcam snapshot
    let fileBlob     = null;   // Blob from file picker
    let pendingDeleteId = null;
    let pendingEditId   = null;
    let facingMode   = 'user'; // 'user' = front, 'environment' = back

    // ── Tab switching ──────────────────────────────────────────────────
    function switchTab(name, btn) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-' + name).classList.add('active');
      stopCamera();
      clearCapture();
    }

    // ── Webcam ─────────────────────────────────────────────────────────
    async function startCamera(facing) {
      if (facing) facingMode = facing;
      try {
        // Stop any existing stream first
        if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }

        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: facingMode },
          audio: false
        });
        const vid = document.getElementById('videoEl');
        vid.srcObject = stream;
        document.getElementById('camBtn').disabled  = true;
        document.getElementById('snapBtn').disabled = false;
        document.getElementById('previewImg').style.display = 'none';
        document.getElementById('videoWrap').style.display  = 'block';

        // Show flip button only when camera is live; check if device has multiple cameras
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoCams = devices.filter(d => d.kind === 'videoinput');
        const flipBtn = document.getElementById('flipBtn');
        flipBtn.style.display = videoCams.length > 1 ? 'flex' : 'none';
        flipBtn.title = facingMode === 'user' ? 'Switch to back camera' : 'Switch to front camera';
      } catch (e) {
        showFeedback('Camera access denied: ' + e.message, true);
      }
    }

    async function flipCamera() {
      facingMode = facingMode === 'user' ? 'environment' : 'user';
      await startCamera(facingMode);
    }

    function stopCamera() {
      if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
      const vid = document.getElementById('videoEl');
      vid.srcObject = null;
      document.getElementById('camBtn').disabled  = false;
      document.getElementById('snapBtn').disabled = true;
      document.getElementById('flipBtn').style.display = 'none';
    }

    function snapPhoto() {
      const vid = document.getElementById('videoEl');
      const canvas = document.getElementById('snapCanvas');
      canvas.width  = vid.videoWidth  || 640;
      canvas.height = vid.videoHeight || 480;
      canvas.getContext('2d').drawImage(vid, 0, 0);
      canvas.toBlob(blob => {
        capturedBlob = blob;
        const url = URL.createObjectURL(blob);
        const preview = document.getElementById('previewImg');
        preview.src = url;
        preview.style.display = 'block';
        document.getElementById('videoWrap').style.display  = 'none';
        document.getElementById('snapBtn').style.display    = 'none';
        document.getElementById('retakeBtn').style.display  = 'flex';
        document.getElementById('submitWebcamBtn').disabled = false;
        stopCamera();
      }, 'image/png');
    }

    function retakePhoto() {
      clearCapture();
      document.getElementById('videoWrap').style.display  = 'block';
      document.getElementById('snapBtn').style.display    = 'flex';
      document.getElementById('retakeBtn').style.display  = 'none';
      document.getElementById('submitWebcamBtn').disabled = true;
      startCamera(); // re-uses last facingMode
    }

    function clearCapture() {
      capturedBlob = null;
      fileBlob     = null;
      document.getElementById('previewImg').style.display  = 'none';
      document.getElementById('filePreview').style.display = 'none';
      document.getElementById('submitWebcamBtn').disabled  = true;
      document.getElementById('submitFileBtn').disabled    = true;
    }

    // ── File upload ────────────────────────────────────────────────────
    function onFileChosen(e) {
      const file = e.target.files[0];
      if (!file) return;
      loadFileBlob(file);
    }

    function handleDragOver(e)  { e.preventDefault(); document.getElementById('dropZone').classList.add('dragover'); }
    function handleDragLeave(e) { document.getElementById('dropZone').classList.remove('dragover'); }
    function handleDrop(e) {
      e.preventDefault();
      document.getElementById('dropZone').classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file) loadFileBlob(file);
    }

    function loadFileBlob(file) {
      fileBlob = file;
      const url = URL.createObjectURL(file);
      const preview = document.getElementById('filePreview');
      preview.src = url;
      preview.style.display = 'block';
      document.getElementById('submitFileBtn').disabled = false;
    }

    // ── Submit ─────────────────────────────────────────────────────────
    async function submitFace(source) {
      const name = document.getElementById('nameInput').value.trim();
      if (!name) { showFeedback('Please enter a name first.', true); return; }

      const blob = source === 'webcam' ? capturedBlob : fileBlob;
      if (!blob) { showFeedback('No image selected.', true); return; }

      const ext  = blob.type === 'image/png' ? '.png'
                 : blob.type === 'image/webp' ? '.webp' : '.jpg';
      const btn  = source === 'webcam'
                   ? document.getElementById('submitWebcamBtn')
                   : document.getElementById('submitFileBtn');

      btn.disabled    = true;
      btn.textContent = 'Saving…';

      try {
        const reader = new FileReader();
        const base64Data = await new Promise((resolve, reject) => {
          reader.onload = () => resolve(reader.result.split(',')[1]);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });

        const res = await fetch('/faces/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            display_name: name,
            image_b64: base64Data,
            ext: ext
          })
        });
        const data = await res.json();

        if (res.ok) {
          showFeedback('✅ ' + name + ' saved as ' + data.filename + '. Recognition is updating.', false);
          document.getElementById('nameInput').value = '';
          clearCapture();
          stopCamera();
          document.getElementById('snapBtn').style.display   = 'flex';
          document.getElementById('retakeBtn').style.display = 'none';
          document.getElementById('videoWrap').style.display = 'block';
          loadFacesList();
        } else {
          showFeedback('Error: ' + (data.detail || 'Upload failed'), true);
        }
      } catch (e) {
        showFeedback('Network error: ' + e.message, true);
      } finally {
        btn.disabled    = false;
        btn.textContent = 'Save Face';
      }
    }

    // ── Faces list ─────────────────────────────────────────────────────
    async function loadFacesList() {
      const list = document.getElementById('facesList');
      try {
        const res  = await fetch('/faces/list');
        const data = await res.json();
        const faces = data.faces || [];
        document.getElementById('faceCount').textContent = faces.length ? '(' + faces.length + ')' : '';

        if (!faces.length) {
          list.innerHTML = '<div class="empty-state">No faces registered yet.</div>';
          return;
        }

        list.innerHTML = faces.map(f => `
          <div class="face-row" id="row-${f.id}">
            <img class="face-thumb" src="/faces/thumb/${encodeURIComponent(f.filename)}"
                 onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 52 52%22><rect fill=%22%23222634%22 width=%2252%22 height=%2252%22/><text x=%2226%22 y=%2234%22 text-anchor=%22middle%22 font-size=%2226%22>👤</text></svg>'"
                 alt="${f.display_name}">
            <div class="face-info">
              <div class="face-name">${escHtml(f.display_name)}</div>
              <div class="face-meta">${escHtml(f.filename)} &middot; ${formatDate(f.added_at)}</div>
            </div>
            <button class="btn-edit-sm"   onclick="askEdit(${f.id}, '${escHtml(f.display_name)}')">Edit</button>
            <button class="btn-danger-sm" onclick="askDelete(${f.id}, '${escHtml(f.display_name)}')">Delete</button>
          </div>`).join('');
      } catch (e) {
        list.innerHTML = '<div class="empty-state" style="color:#fca5a5;">Failed to load faces.</div>';
      }
    }

    // ── Delete ─────────────────────────────────────────────────────────
    function askDelete(id, name) {
      pendingDeleteId = id;
      document.getElementById('mTitle').textContent = 'Delete "' + name + '"?';
      document.getElementById('mDesc').textContent  = 'This will permanently remove the image and its registration.';
      document.getElementById('modal').style.display = 'flex';
    }

    function closeModal() {
      document.getElementById('modal').style.display = 'none';
      pendingDeleteId = null;
    }

    async function confirmDelete() {
      if (!pendingDeleteId) return;
      const faceId = pendingDeleteId;
      closeModal();
      try {
        const res  = await fetch('/faces/' + faceId, { method: 'DELETE' });
        const data = await res.json();
        if (res.ok) {
          showFeedback('Face removed successfully.', false);
          loadFacesList();
        } else {
          showFeedback('Error: ' + (data.detail || 'Delete failed'), true);
        }
      } catch (e) {
        showFeedback('Network error: ' + e.message, true);
      }
    }


    // ── Utilities ──────────────────────────────────────────────────────
    // Edit Name functions
    function askEdit(id, currentName) {
      pendingEditId = id;
      document.getElementById('eTitle').textContent = 'Edit "' + currentName + '"';
      document.getElementById('eNameInput').value = currentName;
      document.getElementById('eNameInput').focus();
      document.getElementById('editModal').style.display = 'flex';
    }

    function closeEditModal() {
      document.getElementById('editModal').style.display = 'none';
      pendingEditId = null;
      document.getElementById('eNameInput').value = '';
    }

    async function confirmEdit() {
      if (!pendingEditId) return;
      const faceId = pendingEditId;
      const newName = document.getElementById('eNameInput').value.trim();
      if (!newName) {
        showFeedback('Please enter a name.', true);
        return;
      }
      closeEditModal();
      try {
        const res  = await fetch('/faces/' + faceId, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ display_name: newName })
        });
        const data = await res.json();
        if (res.ok) {
          showFeedback('Name updated to "' + newName + '".', false);
          loadFacesList();
        } else {
          showFeedback('Error: ' + (data.detail || 'Update failed'), true);
        }
      } catch (e) {
        showFeedback('Network error: ' + e.message, true);
      }
    }

    function showFeedback(msg, isErr) {
      const fb = document.getElementById('feedback');
      fb.textContent  = msg;
      fb.className    = 'feedback ' + (isErr ? 'error' : 'success');
      fb.style.display = 'block';
      setTimeout(() => { fb.style.display = 'none'; }, 5000);
    }

    function escHtml(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }

    function formatDate(iso) {
      try { return new Date(iso).toLocaleString(undefined, { dateStyle:'medium', timeStyle:'short' }); }
      catch { return iso; }
    }

    // Close modal on overlay click
    document.getElementById('modal').addEventListener('click', e => {
      if (e.target === document.getElementById('modal')) closeModal();
    });

    document.getElementById('editModal').addEventListener('click', e => {
      if (e.target === document.getElementById('editModal')) closeEditModal();
    });

    // Init
    loadFacesList();
  </script>
</body>
</html>"""
            return HTMLResponse(content=html.replace("{{NAV_BAR}}", nav_html))

        @app.get("/faces/list")
        def faces_list(request: Request):
            # Dashboard auth guard (JSON API)
            self._require_dashboard_auth(request)
            return JSONResponse({"faces": faces_db.list_faces()})

        @app.get("/faces/thumb/{filename}")
        def faces_thumb(filename: str, request: Request):
            """Serve a face image by filename (basename only, no path traversal)."""
            # Dashboard auth guard (JSON API)
            self._require_dashboard_auth(request)
            import re as _re
            # Sanitise: only allow safe basenames
            if not _re.match(r'^[\w\-. ]+$', filename) or '..' in filename:
                raise HTTPException(status_code=400, detail="Invalid filename")
            path = os.path.join(faces_db.KNOWN_FACES_DIR, filename)
            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail="Image not found")
            ext = os.path.splitext(filename)[1].lower()
            mime = {"jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
            with open(path, "rb") as fh:
                data = fh.read()
            return Response(content=data, media_type=mime)

        @app.post("/faces/add")
        async def faces_add(request: Request, background_tasks: BackgroundTasks):
            # Dashboard auth guard (JSON API)
            self._require_dashboard_auth(request)
            content_type = request.headers.get("content-type", "")
            name = ""
            raw_bytes = b""
            ext = ".png"

            if "application/json" in content_type:
                try:
                    body = await request.json()
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid JSON payload")

                name = (body.get("display_name") or "").strip()
                b64_str = body.get("image_b64") or ""
                ext = (body.get("ext") or ".png").strip().lower()

                if not b64_str:
                    raise HTTPException(status_code=400, detail="image file is required")

                try:
                    raw_bytes = base64.b64decode(b64_str)
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid base64 image data")
            else:
                # Fallback: attempt form parsing if multipart is present
                try:
                    form = await request.form()
                    name = (form.get("display_name") or "").strip()
                    upload = form.get("image")
                    if upload is not None:
                        raw_bytes = await upload.read()
                        orig = getattr(upload, "filename", "") or ""
                        ext = os.path.splitext(orig)[1].lower() or ".png"
                except Exception as form_err:
                    raise HTTPException(status_code=400, detail=f"Failed to parse form: {form_err}")

            if not name:
                raise HTTPException(status_code=400, detail="display_name is required")
            if not raw_bytes:
                raise HTTPException(status_code=400, detail="image file is required")

            if not ext.startswith("."):
                ext = "." + ext

            try:
                result = faces_db.add_face(name, raw_bytes, ext)
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))

            background_tasks.add_task(self._reload_faces)

            print(f"[Faces] Added '{name}' → {result['filename']}")
            return JSONResponse(result, status_code=201)

        @app.delete("/faces/{face_id}")
        async def faces_delete(face_id: int, request: Request, background_tasks: BackgroundTasks):
            # Dashboard auth guard (JSON API)
            self._require_dashboard_auth(request)
            deleted = faces_db.delete_face(face_id, delete_file=True)
            if not deleted:
                raise HTTPException(status_code=404, detail="Face not found")

            background_tasks.add_task(self._reload_faces)

            print(f"[Faces] Deleted face id={face_id}")
            return {"status": "ok", "deleted_id": face_id}

        @app.patch("/faces/{face_id}")
        async def faces_rename(face_id: int, request: Request, background_tasks: BackgroundTasks):
            # Dashboard auth guard (JSON API)
            self._require_dashboard_auth(request)
            body = await request.json()
            new_name = (body.get("display_name") or "").strip()

            if not new_name:
                raise HTTPException(status_code=400, detail="display_name is required")

            try:
                updated = faces_db.update_face_name(face_id, new_name)
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))

            if not updated:
                raise HTTPException(status_code=404, detail="Face not found")

            background_tasks.add_task(self._reload_faces)

            print(f"[Faces] Renamed face id={face_id} to '{new_name}'")
            return {"status": "ok", "face_id": face_id, "display_name": new_name}

        # -------------------------------------------------------------
        # DASHBOARD LOGIN / LOGOUT
        # -------------------------------------------------------------
        LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GuideBot — Login</title>
  <style>
    :root { --bg: #090a0f; --surface: #13151c; --border: #222634; --text: #f1f5f9; --muted: #94a3b8; --primary: #3b82f6; --primary-hover: #2563eb; --danger: #ef4444; }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: 20px; padding: 40px; width: 100%; max-width: 400px; box-shadow: 0 16px 40px rgba(0,0,0,0.5); }
    h1 { font-size: 20px; font-weight: 700; margin-bottom: 8px; text-align: center; }
    p.sub { font-size: 13px; color: var(--muted); text-align: center; margin-bottom: 28px; }
    .field-group { margin-bottom: 18px; }
    label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--muted); }
    input[type=email], input[type=password] { width: 100%; padding: 11px 14px; background: #1a1d27; border: 1px solid var(--border); border-radius: 10px; color: #fff; font-size: 14px; outline: none; }
    input[type=email]:focus, input[type=password]:focus { border-color: var(--primary); }
    .btn { width: 100%; padding: 11px; border-radius: 10px; border: none; font-size: 14px; font-weight: 600; cursor: pointer; }
    .btn-primary { background: var(--primary); color: #fff; }
    .btn-primary:hover { background: var(--primary-hover); }
    .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
    .feedback { margin-top: 16px; font-size: 13px; text-align: center; }
    .feedback.error { color: #fca5a5; }
    .feedback.success { color: #a7f3d0; }
    .hint { font-size: 12px; color: var(--muted); margin-top: 20px; text-align: center; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Sign In</h1>
    <p class="sub">Enter your email and password to access the GuideBot dashboard</p>
    <form id="loginForm" novalidate>
      <div class="field-group">
        <label for="email">Email Address</label>
        <input type="email" id="email" placeholder="you@example.com" required autocomplete="email">
      </div>
      <div class="field-group">
        <label for="password">Password</label>
        <input type="password" id="password" placeholder="Your password" required autocomplete="current-password">
      </div>
      <button class="btn btn-primary" type="submit">Sign In</button>
    </form>
    <div id="feedback" class="feedback"></div>
    <p class="hint">Accounts are managed by the administrator — contact them to be added.</p>
  </div>
  <script>
    document.getElementById('loginForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const email = document.getElementById('email').value.trim();
      const password = document.getElementById('password').value;
      const feedback = document.getElementById('feedback');
      feedback.textContent = '';
      feedback.className = 'feedback';

      if (!email || !password) {
        feedback.className = 'feedback error';
        feedback.textContent = 'Email and password are required.';
        return;
      }

      try {
        const res = await fetch('/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email, password })
        });
        const data = await res.json();

        if (res.ok) {
          feedback.className = 'feedback success';
          feedback.textContent = 'Logged in. Redirecting...';
          const next = new URLSearchParams(window.location.search).get('next') || '/';
          setTimeout(() => { window.location.href = next; }, 600);
        } else {
          feedback.className = 'feedback error';
          feedback.textContent = data.detail || 'Login failed.';
        }
      } catch (err) {
        feedback.className = 'feedback error';
        feedback.textContent = 'Network error: ' + err.message;
      }
    });
  </script>
</body>
</html>"""

        @app.get("/login", response_class=HTMLResponse)
        async def login_page(request: Request):
            allowed_emails = emails_db.list_emails()
            if allowed_emails:
                token = request.cookies.get("guidebot_session")
                email = self._validate_session(token)
                if email is not None and emails_db.is_allowed(email):
                    next_url = request.query_params.get("next", "/")
                    if not next_url.startswith("/") or next_url.startswith("//"):
                        next_url = "/"
                    return RedirectResponse(url=next_url, status_code=303)
                return HTMLResponse(content=LOGIN_HTML)

            setup_notice = (
                '<p class="hint setup-notice">No dashboard account is configured. '
                'On the server, run <code>python update_email.py add your-email@example.com</code>, '
                'then <code>python update_email.py set-password your-email@example.com</code>.</p>'
            )
            setup_page = LOGIN_HTML.replace(
                '<form id="loginForm" novalidate>',
                setup_notice + '<form id="loginForm" novalidate hidden>',
            )
            return HTMLResponse(content=setup_page)

        @app.post("/login")
        async def login_submit(request: Request):
            if not emails_db.list_emails():
                raise HTTPException(
                    status_code=503,
                    detail="Dashboard login is not configured. Register an account on the server.",
                )

            body = await request.json()
            email = (body.get("email") or "").strip().lower()
            password = body.get("password") or ""

            if not email or not password:
                raise HTTPException(status_code=400, detail="Email and password are required")

            # Evaluate both checks before failing so timing does not reveal
            # whether the email itself exists (generic error below).
            allowed = emails_db.is_allowed(email)
            pw_ok = emails_db.verify_password(email, password)
            if not allowed:
                raise HTTPException(status_code=401, detail="Invalid email or password")
            if not emails_db.has_password(email):
                # This user was registered but has no password yet.
                raise HTTPException(
                    status_code=409,
                    detail="No password set for this account. Ask the admin to run: "
                           f"python update_email.py set-password {email}",
                )
            if not pw_ok:
                raise HTTPException(status_code=401, detail="Invalid email or password")

            token = self._create_session(email)
            next_url = request.query_params.get("next", "/")
            if not next_url.startswith("/") or next_url.startswith("//"):
                next_url = "/"
            response = JSONResponse({"status": "ok", "email": email, "redirect": next_url})
            response.set_cookie(
                key="guidebot_session",
                value=token,
                httponly=True,
                secure=False,
                samesite="lax",
                max_age=int(self.session_timeout.total_seconds()),
                path="/",
            )
            return response

        @app.post("/logout")
        async def logout(request: Request):
            token = request.cookies.get("guidebot_session")
            self._clear_session(token)
            response = RedirectResponse(url="/login", status_code=303)
            response.delete_cookie(key="guidebot_session", path="/")
            return response

        print("[Stream] Routes ready; starting receiver thread...")
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
            frame = cv2.flip(frame, 1)  # always un-mirror the wearable camera

            with self.lock:
                self.latest_frame = frame
                self.latest_timestamp = time.time()
                self.frame_count += 1

    # -------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------
    def get_frame(self):
        frame, _, _ = self.get_frame_with_metadata()
        return frame

    def get_frame_with_metadata(self):
        """Return the latest frame plus the stream sequence number and capture time."""
        with self.lock:
            if self.latest_frame is None:
                return None, 0, 0.0
            return self.latest_frame.copy(), self.frame_count, self.latest_timestamp

    def get_received_frame_count(self):
        with self.lock:
            return self.frame_count

    def is_alive(self, timeout: float = 3.0) -> bool:
        with self.lock:
            if self.latest_frame is None or self.latest_timestamp == 0:
                return False
            return (time.time() - self.latest_timestamp) < timeout

    def release(self):
        self.running = False
        if hasattr(self, "_transcribe_queue"):
            self._transcribe_queue.put(None)

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
