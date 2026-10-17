import pyttsx3
import threading
import queue

class TTS:
    def __init__(self):
        self.queue = queue.Queue()
        self._broadcast_fn = None  # set via set_broadcast() after Stream is created
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def set_broadcast(self, fn):
        """Register a callback fn(text: str) that is called from the worker
        thread just before pyttsx3 speaks. Used to push TTS text to web clients."""
        self._broadcast_fn = fn

    def speak(self, text):
        self.queue.put(text)

    def _worker_loop(self):
        while True:
            text = self.queue.get()
            if text is None:
                continue
            try:
                # Broadcast to web clients BEFORE speaking locally
                if self._broadcast_fn:
                    try:
                        self._broadcast_fn(text)
                    except Exception as be:
                        print(f"[TTS] Broadcast error: {be}")
                engine = pyttsx3.init()
                engine.setProperty("rate", 170)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                print(f"[TTS] Error: {e}")