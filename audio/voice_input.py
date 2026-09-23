import whisper
import sounddevice as sd
import numpy as np
import threading
import queue
from config import CONFIG

class VoiceInput:
    def __init__(self):
        print(f"[Voice] Loading Whisper ({CONFIG['whisper_model']})...")
        self.model = whisper.load_model(CONFIG["whisper_model"])
        print("[Voice] Whisper loaded")

        devices = sd.query_devices()
        default_input = sd.default.device[0]
        print(f"[Voice] Default input device: {devices[default_input]['name']}")

        self.command_queue = queue.Queue()
        self.running = True

        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def _record_audio(self):
        sample_rate = CONFIG["voice_sample_rate"]
        duration = CONFIG["voice_record_seconds"]

        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        return audio.flatten()

    def _listen_loop(self):
        while self.running:
            try:
                audio = self._record_audio()
                volume = np.abs(audio).mean()

                print(f"[Voice] Captured audio — volume level: {volume:.4f}")

                if volume < 0.003:
                    print("[Voice] Too quiet, skipping transcription")
                    continue

                result = self.model.transcribe(audio, fp16=False, language="en")
                text = result["text"].strip().lower()

                if text:
                    print(f"[Voice] Heard: '{text}'")
                    self.command_queue.put(text)
                else:
                    print("[Voice] Transcribed but got empty text")

            except Exception as e:
                print(f"[Voice] Error: {e}")

    def get_command(self):
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False