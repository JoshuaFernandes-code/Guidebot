import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from config import CONFIG

# Configurable Ollama request timeout (seconds). Set via LLM_TIMEOUT env var or config.py.
LLM_REQUEST_TIMEOUT = float(CONFIG.get("llm_timeout", 25.0))


class LLMAssistant:
    """
    Local LLM Assistant for GuideBot powered by Ollama.
    Handles unconfigured/conversational queries with contextual awareness:
    - Visible detections (objects, distances, directions)
    - Recent navigation/TTS history
    - Last OCR/reading result
    - Last recognized face
    
    Includes auto-start capability for Ollama:
    - Checks if Ollama is running at ollama_url
    - If not, attempts to launch `ollama serve` in a detached background process
    - If Ollama is not installed or fails to launch, logs clear warnings and activates
      a graceful fallback so GuideBot continues operating without crashing.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are GuideBot, an intelligent and empathetic wearable smart navigation assistant for a visually impaired user. "
        "Your answers are spoken aloud to the user via Text-to-Speech (TTS).\n"
        "CRITICAL RULES:\n"
        "1. Keep responses under 15 words. Be direct, crisp, and conversational.\n"
        "2. NEVER use markdown, formatting, asterisks, bullet points, emojis, or quotes. Output only plain speakable text.\n"
        "3. Base your answer strictly on the 'Current visible objects' provided. If an object is not in the visible list, state clearly that you do not see it.\n"
        "4. Never hallucinate or invent objects that are not listed in the current camera view."
    )

    def __init__(
        self,
        ollama_url: str = "http://127.0.0.1:11434",
        model: str = "llama3.2:1b",
        enabled: bool = True,
        auto_start: bool = True,
        tts_callback = None
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self.auto_start = auto_start
        self.tts_callback = tts_callback

        self._lock = threading.Lock()
        self._history_tts: List[str] = []
        self._last_ocr: Optional[str] = None
        self._last_face: Optional[str] = None
        self._last_detections: List[Dict[str, Any]] = []

        self._is_busy = False
        self._process: Optional[subprocess.Popen] = None
        self._service_ready = False

        if self.enabled:
            # Check availability or auto-start in background so startup isn't blocked
            threading.Thread(target=self._init_service, daemon=True).start()

    def _init_service(self):
        """Attempts to ensure Ollama service is reachable. If down and auto_start is true, spawns it."""
        if self._ping():
            self._service_ready = True
            print(f"[LLM] Connected to running Ollama service at {self.ollama_url} (model: {self.model})")
            return

        if not self.auto_start:
            print(f"[LLM] [WARNING] Ollama is not running at {self.ollama_url} and auto_start=False. Assistant fallback active.")
            return

        print(f"[LLM] Ollama not detected at {self.ollama_url}. Attempting auto-start in background...")
        ollama_bin = shutil.which("ollama") or shutil.which("ollama.exe")
        
        # Check standard Windows installation directory if not in PATH
        if not ollama_bin and os.name == "nt":
            standard_win_path = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
            if os.path.isfile(standard_win_path):
                ollama_bin = standard_win_path

        if not ollama_bin:
            print("[LLM] [FALLBACK ACTIVE] 'ollama' executable was not found on this system.")
            print("[LLM] To enable the local AI assistant, install Ollama from https://ollama.com and run: ollama pull llama3.2:3b")
            self._service_ready = False
            return

        try:
            # Spawn 'ollama serve' in background
            kwargs: Dict[str, Any] = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if os.name == "nt":
                # DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP prevents Ctrl+C propagation
                CREATE_NO_WINDOW = 0x08000000
                DETACHED_PROCESS = 0x00000008
                kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NO_WINDOW
            else:
                kwargs["start_new_session"] = True

            self._process = subprocess.Popen([ollama_bin, "serve"], **kwargs)
            print(f"[LLM] Started Ollama background process (PID: {self._process.pid}). Waiting for ready state...")

            # Ollama on Windows can take 8-15 seconds to discover GPUs and bind the port on cold boot.
            # Poll every 0.5s up to 30 seconds.
            ready = False
            for _ in range(60):
                time.sleep(0.5)
                if self._ping():
                    ready = True
                    break

            if ready:
                self._service_ready = True
                print(f"[LLM] Ollama successfully auto-started and ready at {self.ollama_url}!")
            else:
                print(f"[LLM] [FALLBACK ACTIVE] Timed out waiting for Ollama service to respond at {self.ollama_url}.")
                print("[LLM] GuideBot will continue with rule-based fallback responses.")
                self._service_ready = False

        except Exception as e:
            print(f"[LLM] [FALLBACK ACTIVE] Failed to auto-start Ollama: {e}")
            print("[LLM] GuideBot will continue with rule-based fallback responses.")
            self._service_ready = False

    def _ping(self) -> bool:
        """Fast ping to check if the Ollama endpoint is answering."""
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=1.2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if LLM service is available."""
        if not self.enabled:
            return False
        if self._service_ready and self._ping():
            return True
        # Re-check in case it came online later
        self._service_ready = self._ping()
        return self._service_ready

    def update_tts_history(self, message: str):
        """Track recent TTS outputs for conversational memory."""
        if not message:
            return
        with self._lock:
            self._history_tts.append(message)
            if len(self._history_tts) > 5:
                self._history_tts.pop(0)

    def update_ocr(self, text: str):
        """Update last read text result."""
        with self._lock:
            self._last_ocr = text

    def update_face(self, result: str):
        """Update last recognized face result."""
        with self._lock:
            self._last_face = result

    def update_detections(self, detections: List[Dict[str, Any]]):
        """Update latest snapshot of detected objects."""
        with self._lock:
            self._last_detections = list(detections)

    def _generate_rule_fallback(self, user_query: str) -> str:
        """
        Rule-based contextual fallback when Ollama is offline or uninstalled.
        Constructs a helpful answer using spatial detections and readings instead of failing silently.
        """
        q = user_query.lower()
        with self._lock:
            dets = list(self._last_detections)
            last_ocr = self._last_ocr
            last_face = self._last_face

        # Fallback 1: Text / reading check (specific)
        if any(w in q for w in ["text", "sign", "word", "written", "read"]):
            if last_ocr:
                return f"The last text I read was: '{last_ocr}'."
            return "I haven't read any text recently. Say 'read' to scan."

        # Fallback 2: Person / identity check (specific)
        if any(w in q for w in ["who", "person", "someone", "face"]):
            if last_face:
                return f"Recent face identification: {last_face}."
            return "No known faces recognized recently. Say 'who' to identify."

        # Fallback 3: Surrounding / nearby objects check
        if any(w in q for w in ["what", "around", "see", "look", "nearby", "drink", "cup", "chair", "object"]):
            if dets:
                det_strs = [f"{d.get('label', 'object')} {d.get('distance', '')} {d.get('direction', '')}".strip() for d in dets[:3]]
                return f"I can see: {', '.join(det_strs)}."
            return "I don't detect any objects directly in view right now."

        # Generic helpful guidebot response
        return "I am with you. You can ask what is ahead, left, right, say 'read' for text, or 'who' for faces."

    def _build_context_prompt(self, user_query: str, fresh_detections: Optional[List[Dict[str, Any]]] = None) -> str:
        with self._lock:
            # Prefer fresh_detections passed at call time (from the live frame) over stale cache
            dets = list(fresh_detections) if fresh_detections is not None else list(self._last_detections)
            tts_hist = list(self._history_tts)
            last_ocr = self._last_ocr
            last_face = self._last_face

        context_lines = []

        if dets:
            det_summaries = []
            for d in dets[:6]:
                label = d.get("label", "object")
                dist = d.get("distance", "unknown distance")
                direction = d.get("direction", "in view")
                det_summaries.append(f"{label} ({dist}, {direction})")
            context_lines.append(f"Current visible objects: {', '.join(det_summaries)}.")
        else:
            context_lines.append("Current visible objects: No objects detected in the current camera view.")

        if last_face:
            context_lines.append(f"Recent face check: {last_face}.")

        if last_ocr:
            context_lines.append(f"Recent text reading: '{last_ocr}'.")

        if tts_hist:
            recent_spoken = " | ".join(tts_hist[-3:])
            context_lines.append(f"Recent announcements you made: {recent_spoken}")

        context_block = "\n".join(context_lines)
        return (
            f"Context about what is currently seen:\n"
            f"{context_block}\n\n"
            f"User says: \"{user_query}\"\n"
            f"Respond concisely as GuideBot (do not mention data or context, speak naturally about what you see or don't see):"
        )

    def ask(self, user_query: str, on_response=None, fresh_detections: Optional[List[Dict[str, Any]]] = None):
        """
        Asynchronously ask the LLM. If on_response is not provided, defaults to self.tts_callback.
        Non-blocking: spawns background daemon thread so main loop remains smooth.
        If Ollama is offline or uninstalled, automatically provides an intelligent spatial fallback.
        fresh_detections: pass the live detection list from the current frame for maximum accuracy.
        """
        if not self.enabled:
            return

        target_cb = on_response or self.tts_callback

        def _worker():
            with self._lock:
                if self._is_busy:
                    print(f"[LLM] Skipping '{user_query}' - assistant is already thinking.")
                    return
                self._is_busy = True

            try:
                # Fresh availability check right before sending (not just cached startup flag).
                # If Ollama is unreachable now, use fallback immediately without waiting on a timeout.
                if not self._ping():
                    self._service_ready = False
                    fallback_reply = self._generate_rule_fallback(user_query)
                    print(f"[LLM / Fallback] Ollama unreachable. Responded with fallback: '{fallback_reply}'")
                    if target_cb:
                        target_cb(fallback_reply)
                    return

                prompt = self._build_context_prompt(user_query, fresh_detections=fresh_detections)
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "system": self.DEFAULT_SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,   # More deterministic / factual, less rambling
                        "num_predict": 40,    # ~40 tokens ≈ 1–2 short sentences, <0.8s on CPU
                    }
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.ollama_url}/api/generate",
                    data=data,
                    headers={"Content-Type": "application/json"}
                )

                t0 = time.time()
                with urllib.request.urlopen(req, timeout=LLM_REQUEST_TIMEOUT) as response:
                    res_body = json.loads(response.read().decode("utf-8"))
                    answer = res_body.get("response", "").strip()
                    # Clean up any potential markdown formatting
                    answer = answer.replace("*", "").replace("#", "").strip()

                dur = time.time() - t0
                print(f"[LLM] Answered in {dur:.2f}s: '{answer}'")

                if answer and target_cb:
                    target_cb(answer)

            except urllib.error.URLError as ue:
                print(f"[LLM] Connection to Ollama failed during query ({ue}). Using fallback.")
                fallback_reply = self._generate_rule_fallback(user_query)
                if target_cb:
                    target_cb(fallback_reply)
            except Exception as e:
                print(f"[LLM] Error answering query: {e}. Using fallback.")
                fallback_reply = self._generate_rule_fallback(user_query)
                if target_cb:
                    target_cb(fallback_reply)
            finally:
                with self._lock:
                    self._is_busy = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
