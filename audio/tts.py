import pyttsx3
import threading
import queue

class TTS:
    def __init__(self):
        self.queue = queue.Queue()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def speak(self, text):
        self.queue.put(text)

    def _worker_loop(self):
        while True:
            text = self.queue.get()
            if text is None:
                continue
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 170)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                print(f"[TTS] Error: {e}")