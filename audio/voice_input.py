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

        self.command_queue = queue.Queue()
        self.running = True
        self.use_web_mic = CONFIG.get("use_web_mic", False)
        self.use_esp32_mic = CONFIG.get("use_esp32_mic", False)

        if self.use_web_mic:
            print("[Voice] USE_WEB_MIC=true: Local microphone disabled. Commands accepted via Web UI at /audio.")
            self.thread = None
        elif self.use_esp32_mic:
            print("[Voice] USE_ESP32_MIC=true: Local microphone disabled. Commands stream from the ESP32-S3 I2S mic (via /ws/stream AUDIO_DATA).")
            self.thread = None
        else:
            try:
                devices = sd.query_devices()
                default_input = sd.default.device[0]
                print(f"[Voice] Default input device: {devices[default_input]['name']}")
            except Exception as e:
                print(f"[Voice] Warning: Could not query local audio devices: {e}")

            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()

    def submit_command(self, text: str):
        """Allows web interface or external callers to inject a command."""
        text = text.strip().lower()
        if text:
            print(f"[Voice / Web] Command received: '{text}'")
            self.command_queue.put(text)

    def transcribe_audio_file(self, file_path_or_bytes):
        """Transcribe an audio recording received from the web interface."""
        try:
            result = self.model.transcribe(file_path_or_bytes, fp16=False, language="en")
            text = result["text"].strip().lower()
            clean = text.rstrip(".?!,")
            NOISE = {"", ".", "..", "...", "you", "thank you", "subtitles", "the", "a"}
            if clean and clean not in NOISE and len(clean) >= 2:
                print(f"[Voice / Web] Transcribed audio: '{text}'")
                self.command_queue.put(clean)
                return clean
            return ""
        except Exception as e:
            print(f"[Voice / Web] Transcription error: {e}")
            return ""

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
        fail_count = 0
        NOISE = {"", ".", "..", "...", "you", "thank you", "subtitles", "the", "a"}
        while self.running:
            try:
                audio = self._record_audio()
                fail_count = 0
                volume = np.abs(audio).mean()

                if volume < 0.003:
                    continue

                result = self.model.transcribe(audio, fp16=False, language="en")
                text = result["text"].strip().lower()
                clean = text.rstrip(".?!,")

                if clean and clean not in NOISE and len(clean) >= 2:
                    print(f"[Voice] Heard: '{clean}'")
                    self.command_queue.put(clean)

            except Exception as e:
                fail_count += 1
                # Only print the first occurrence, then once every 10 failures
                if fail_count == 1 or fail_count % 10 == 0:
                    print(f"[Voice] Microphone error (attempt {fail_count}): {e}")
                # Backoff: cap at 5s so recovery is quick when mic becomes available
                import time
                time.sleep(min(fail_count * 0.5, 5.0))

    def get_command(self):
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False