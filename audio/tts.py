import pyttsx3
import threading
import queue
import time
import re
from collections import deque

class TTS:
    def __init__(self):
        self.queue = queue.Queue()
        self._broadcast_fn = None  # set via set_broadcast() after Stream is created
        
        # Acoustic Echo Cancellation (AEC) State:
        # Used to mute mic intake while GuideBot speaks and prevent self-feedback loops
        self._is_speaking = False
        self._last_spoke_ts = 0.0
        self._lock = threading.Lock()
        self._recent_phrases = deque(maxlen=8)  # stores (clean_text, timestamp)

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def set_broadcast(self, fn):
        """Register a callback fn(text: str) that is called from the worker
        thread just before pyttsx3 speaks. Used to push TTS text to web clients."""
        self._broadcast_fn = fn

    def speak(self, text):
        if not text:
            return
        self.queue.put(text)

    def is_active(self, margin_s: float = 0.35) -> bool:
        """Returns True if TTS is actively speaking or finished less than margin_s ago
        (to account for room reverberation and sound propagation time)."""
        if self._is_speaking:
            return True
        return (time.time() - self._last_spoke_ts) < margin_s

    def is_echo(self, text: str, time_window_s: float = 6.0) -> bool:
        """Returns True if candidate text matches or is a substring of something
        GuideBot spoke within the last time_window_s seconds."""
        if not text:
            return False
        clean = re.sub(r"[^\w\s]", "", text.strip().lower())
        if not clean:
            return False
        
        now = time.time()
        with self._lock:
            for spoken, ts in self._recent_phrases:
                if (now - ts) <= time_window_s:
                    # Exact match, or candidate is inside spoken text, or spoken text is inside candidate
                    if clean == spoken or clean in spoken or (len(spoken) > 5 and spoken in clean):
                        return True
        return False

    def _worker_loop(self):
        while True:
            text = self.queue.get()
            if text is None:
                continue
            try:
                # Clean text for echo comparison
                clean_spoken = re.sub(r"[^\w\s]", "", text.strip().lower())
                with self._lock:
                    self._recent_phrases.append((clean_spoken, time.time()))

                # Broadcast to web clients BEFORE speaking locally
                if self._broadcast_fn:
                    try:
                        self._broadcast_fn(text)
                    except Exception as be:
                        print(f"[TTS] Broadcast error: {be}")

                self._is_speaking = True
                engine = pyttsx3.init()
                engine.setProperty("rate", 170)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                print(f"[TTS] Error: {e}")
            finally:
                self._is_speaking = False
                self._last_spoke_ts = time.time()