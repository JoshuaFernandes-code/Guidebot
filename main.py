import cv2
import time
import threading
import numpy as np
from pipeline.stream import Stream
from pipeline.detection import Detector
from pipeline.depth import DepthEstimator
from pipeline.decision import DecisionEngine
from pipeline.ocr import OCRReader
from pipeline.face import FaceRecognizer
from audio.tts import TTS
from audio.voice_input import VoiceInput
from audio.llm import LLMAssistant
from config import CONFIG


class InferencePipeline:
    """
    Runs YOLO detection + MiDaS depth estimation in a background thread.
    The main display loop reads the latest results without blocking on inference.
    This decouples display FPS (= stream FPS) from inference FPS (= CPU speed).
    """

    def __init__(self, stream: Stream, detector: Detector,
                 depth_estimator: DepthEstimator, decision_engine: DecisionEngine):
        self.stream = stream
        self.detector = detector
        self.depth_estimator = depth_estimator
        self.decision_engine = decision_engine

        self.lock = threading.Lock()
        self.running = True

        # Latest output available for the display loop to read
        self._latest_results = None    # YOLO Results object
        self._latest_depth   = None    # depth_map numpy array
        self._latest_frame   = None    # frame that was actually inferred on
        self._latest_detections = []   # parsed detection dicts
        self._latest_frame_info = None # source stream frame number and capture time
        self._inference_fps  = 0.0

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        prev = time.time()
        while self.running:
            frame, frame_number, captured_at = self.stream.get_frame_with_metadata()
            if frame is None or frame.size == 0:
                time.sleep(0.02)
                continue

            frame_resized = cv2.resize(
                frame,
                (CONFIG["display_width"], CONFIG["display_height"])
            )
            frame_width = CONFIG["display_width"]

            try:
                results    = self.detector.detect(frame_resized)
                depth_map  = self.depth_estimator.estimate(frame_resized)
            except Exception as e:
                print(f"[Inference] Error: {e}")
                continue

            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label  = results.names[cls_id]
                dist   = self.depth_estimator.get_distance_at_box(
                    depth_map, x1, y1, x2, y2
                )
                direction = self.decision_engine.get_direction(x1, x2, frame_width)
                detections.append({
                    "label": label,
                    "conf":  float(box.conf[0]),
                    "box":   (x1, y1, x2, y2),
                    "distance": dist,
                    "direction": direction,
                })

            now = time.time()
            self._inference_fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            with self.lock:
                self._latest_results    = results
                self._latest_depth      = depth_map
                self._latest_frame      = frame_resized
                self._latest_detections = detections
                self._latest_frame_info = {
                    "number": frame_number,
                    "captured_at": captured_at,
                    "shape": tuple(frame_resized.shape),
                }

    def get_latest(self):
        """Returns (frame, detections, inference_fps) — all safe copies."""
        with self.lock:
            if self._latest_frame is None:
                return None, [], None, 0.0
            return (
                self._latest_frame.copy(),
                list(self._latest_detections),
                dict(self._latest_frame_info),
                self._inference_fps,
            )

    def stop(self):
        self.running = False


def is_window_closed(window_name: str = "GuideBot") -> bool:
    """Detects if the user clicked the window's [X] close button."""
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except Exception:
        return True


def draw_standby_screen(w: int, h: int, stream: Stream):
    """Renders a responsive, mathematically center-aligned standby canvas."""
    standby = np.full((h, w, 3), (25, 25, 28), dtype=np.uint8)
    dots = "." * (int(time.time() * 2) % 4)

    if stream.frame_count == 0:
        title = "Waiting for Camera Stream" + dots
        sub   = f"Receiver active on port {stream.port} (/ws/stream)"
        hint  = "Ensure ESP32 is powered on and connected to tunnel"
    else:
        sec_ago = max(int(time.time() - stream.latest_timestamp), 0)
        title = f"Stream Interrupted ({sec_ago}s ago)"
        sub   = "Reconnecting" + dots
        hint  = "Check ESP32 power, Wi-Fi, or tunnel connection"

    def put_centered_text(img, text, y, font, scale, color, thickness):
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        x = max((w - tw) // 2, 10)
        cv2.putText(img, text, (x, y), font, scale, color, thickness)

    put_centered_text(standby, "GuideBot Standby", h // 2 - 60,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    put_centered_text(standby, title, h // 2 - 15,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    put_centered_text(standby, sub, h // 2 + 25,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    put_centered_text(standby, hint, h // 2 + 55,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)
    put_centered_text(standby, "Press 'q' or click [X] to quit", h - 30,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1)

    return standby


def main():
    print("[GuideBot] Starting - Phase 6C (Face Recognition)")
    voice_input     = VoiceInput()
    detector        = Detector()
    depth_estimator = DepthEstimator()
    decision_engine = DecisionEngine()
    tts             = TTS()
    ocr_reader      = OCRReader()
    face_recognizer = FaceRecognizer()
    # Pass tts reference so Stream can mute mic while GuideBot is speaking (AEC gate)
    # Pass face_recognizer so /faces API can hot-reload embeddings after add/delete
    print("[Startup] Initializing stream receiver...")
    stream          = Stream(voice_input=voice_input, face_recognizer=face_recognizer, tts=tts)
    print("[Startup] Stream receiver ready.")

    # Wire TTS → Stream: broadcast spoken text to /ws/tts-feed web clients
    tts.set_broadcast(stream.broadcast_tts_text)

    # Local LLM Assistant (Ollama)
    llm = LLMAssistant(
        ollama_url=CONFIG.get("llm_url", "http://127.0.0.1:11434"),
        model=CONFIG.get("llm_model", "llama3.2:1b"),
        enabled=CONFIG.get("llm_enabled", False),
        auto_start=CONFIG.get("llm_auto_start", True),
        tts_callback=tts.speak,
        vision_enabled=CONFIG.get("vision_enabled", True),
        vision_model=CONFIG.get("vision_model", "qwen3-vl:4b-instruct"),
        vision_timeout=CONFIG.get("vision_timeout", 240.0),
        vision_frame_dir=CONFIG.get("vision_frame_dir", "logs/vision-frames"),
        vision_frame_limit=CONFIG.get("vision_frame_limit", 25),
        vision_image_max_width=CONFIG.get("vision_image_max_width", 512),
    )
    if llm.enabled:
        print(f"[LLM] Conversational assistant ENABLED (model: {llm.model})")
    else:
        print("[LLM] Conversational assistant DISABLED (set LLM_ENABLED=true in .env to activate)")
    if llm.vision_enabled:
        print(f"[Vision] Scene description ENABLED (model: {llm.vision_model})")
    else:
        print("[Vision] Scene description DISABLED (detector-based fallback remains active)")

    inference = InferencePipeline(stream, detector, depth_estimator, decision_engine)

    cv2.namedWindow("GuideBot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GuideBot", CONFIG["display_width"], CONFIG["display_height"])

    last_fps_update     = time.monotonic()
    last_received_count = stream.get_received_frame_count()
    receive_fps_text    = "0.0"
    inf_fps_text        = "0.0"
    frame_width         = CONFIG["display_width"]
    frame_height        = CONFIG["display_height"]

    while True:
        try:
            # ------------------------------------------------------------------
            # 1. Standby Check: Keep OpenCV window responsive when ESP is offline
            # ------------------------------------------------------------------
            if not stream.is_alive(timeout=3.0):
                standby = draw_standby_screen(frame_width, frame_height, stream)
                cv2.imshow("GuideBot", standby)
                
                # Still process voice commands even when video stream is in standby
                cmd_standby = voice_input.get_command()
                if cmd_standby:
                    print(f"[Voice / LLM (Standby)] User asked: '{cmd_standby}'")
                    llm.ask(cmd_standby)

                key = cv2.waitKey(30)
                if (key & 0xFF == ord('q')) or is_window_closed("GuideBot"):
                    break
                continue

            # ------------------------------------------------------------------
            # 2. Display: render latest inferred frame
            # ------------------------------------------------------------------
            annotated_frame, detections, frame_info, inf_fps = inference.get_latest()

            if annotated_frame is None or annotated_frame.size == 0:
                standby = draw_standby_screen(frame_width, frame_height, stream)
                cv2.imshow("GuideBot", standby)

                cmd_standby = voice_input.get_command()
                if cmd_standby:
                    print(f"[Voice / LLM (Standby)] User asked: '{cmd_standby}'")
                    llm.ask(cmd_standby)

                key = cv2.waitKey(30)
                if (key & 0xFF == ord('q')) or is_window_closed("GuideBot"):
                    break
                continue

            # Keep an unannotated copy for a vision request. Detection boxes are added below.
            vision_frame = annotated_frame.copy()

            # Draw detection boxes on the frame
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                label    = det["label"]
                conf     = det["conf"]
                distance = det["distance"]
                direction = det["direction"]

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f} | {distance} | {direction}"
                cv2.putText(annotated_frame, text, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # ------------------------------------------------------------------
            # Audio / Decision engine (non-blocking: uses last known detections)
            # ------------------------------------------------------------------
            if llm.enabled:
                llm.update_detections(detections)

            messages = decision_engine.decide(detections)
            for message in messages:
                print(f"[GuideBot Speaking] {message}")
                tts.speak(message)
                llm.update_tts_history(message)

            command = voice_input.get_command()
            if command:
                cmd_clean = command.strip().lower()
                # Remove common trailing punctuation
                cmd_clean = cmd_clean.rstrip(".?!,")
                words = cmd_clean.split()
                num_words = len(words)

                # Intent Classification:
                # 1. Fast Direct Navigation Queries (exact phrases or short commands <= 4 words)
                DIRECT_AHEAD = {
                    "ahead", "front", "forward", "what is ahead", "what's ahead",
                    "whats ahead", "what is in front", "whats in front",
                    "what is in front of me", "anything ahead", "anything in front"
                }
                DIRECT_LEFT = {
                    "left", "what is on my left", "what's on my left", "whats on my left",
                    "what is left", "whats left", "anything on my left"
                }
                DIRECT_RIGHT = {
                    "right", "what is on my right", "what's on my right", "whats on my right",
                    "what is right", "whats right", "anything on my right"
                }

                query = None
                if cmd_clean in DIRECT_AHEAD or (num_words <= 3 and any(w in words for w in ["ahead", "forward"])):
                    query = "ahead"
                elif cmd_clean in DIRECT_LEFT or (num_words <= 4 and "left" in words and any(w in words for w in ["what", "whats", "what's", "is", "on", "check"])):
                    query = "left"
                elif cmd_clean in DIRECT_RIGHT or (num_words <= 4 and "right" in words and any(w in words for w in ["what", "whats", "what's", "is", "on", "check"])):
                    query = "right"

                # 2. Fast Scene Description Queries (e.g. "what do you see", "describe scene")
                is_scene_query = (
                    cmd_clean in {
                        "what do you see", "what do you see now", "what can you see",
                        "what are you looking at", "what are you seeing", "describe",
                        "describe scene", "describe the scene", "describe what you see",
                        "what is there", "whats there", "what's there",
                        "what is around", "whats around", "what's around", "what is around me",
                        "look around", "what is in view", "tell me what you see"
                    } or
                    ("see" in words and any(w in words for w in ["what", "whats", "what's", "can", "do"])) or
                    (num_words <= 3 and "describe" in words) or
                    ("around" in words and any(w in words for w in ["what", "whats", "what's", "look"]))
                )

                # 3. Fast OCR Queries
                is_ocr_query = (
                    cmd_clean in {"read", "read text", "read this", "read sign", "read sign board", "read the text"} or
                    (num_words <= 3 and "read" in words)
                )

                # 4. Fast Face Identification Queries
                is_face_query = (
                    cmd_clean in {"who", "who is this", "who is that", "who is here", "who are you looking at", "identify face", "who is in front"} or
                    (num_words <= 4 and "who" in words and any(w in words for w in ["this", "that", "here", "is"]))
                )

                if query:
                    response = decision_engine.answer_query(query, detections, frame_width)
                    print(f"[GuideBot Speaking] {response}")
                    tts.speak(response)
                    llm.update_tts_history(response)
                elif is_scene_query:
                    # Vision sees the unannotated camera frame; detector output remains the fallback.
                    fallback_response = decision_engine.describe_scene(detections, frame_width)
                    print("[Vision] Scene request routed to local vision model.")
                    llm.ask_vision(
                        command,
                        vision_frame,
                        frame_info,
                        fallback_response,
                        fresh_detections=detections,
                    )
                elif is_ocr_query:
                    print("[OCR] Reading current frame...")
                    text_result = ocr_reader.read_frame(annotated_frame)
                    print(f"[GuideBot Speaking] {text_result}")
                    tts.speak(text_result)
                    llm.update_tts_history(text_result)
                    llm.update_ocr(text_result)
                elif is_face_query:
                    print("[Face] Identifying...")
                    face_result = face_recognizer.identify(annotated_frame)
                    print(f"[GuideBot Speaking] {face_result}")
                    tts.speak(face_result)
                    llm.update_tts_history(face_result)
                    llm.update_face(face_result)
                else:
                    # 5. Conversational / Contextual Question -> Route to Local LLM Assistant
                    # Discard single-word filler words or noise fragments
                    if num_words <= 1 and words[0] in {"you", "the", "a", "it", "so", "oh", "yeah", "yes", "no", "uh", "um", "bye"}:
                        continue
                    print(f"[Voice / LLM] Conversational query routed to assistant: '{command}'")
                    # Pass detections and the exact source-frame metadata to the LLM request.
                    llm.ask(command, fresh_detections=detections, frame_info=frame_info)

            # ------------------------------------------------------------------
            # FPS Overlay: compact receive and inference diagnostics
            # ------------------------------------------------------------------
            curr_time = time.monotonic()
            elapsed = curr_time - last_fps_update
            if elapsed >= 0.3:
                received_count = stream.get_received_frame_count()
                receive_fps_text = f"{(received_count - last_received_count) / elapsed:.1f}"
                inf_fps_text = f"{inf_fps:.1f}"
                last_received_count = received_count
                last_fps_update = curr_time

            cv2.rectangle(annotated_frame, (8, 6), (145, 46), (20, 20, 20), -1)
            cv2.rectangle(annotated_frame, (8, 6), (145, 46), (80, 80, 80), 1)

            cv2.putText(annotated_frame, f"Recv: {receive_fps_text} fps",
                        (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1)
            cv2.putText(annotated_frame, f"Infer: {inf_fps_text} fps",
                        (14, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

            if annotated_frame is not None and getattr(annotated_frame, "size", 0) > 0 and len(annotated_frame.shape) == 3 and annotated_frame.shape[0] > 0 and annotated_frame.shape[1] > 0:
                cv2.imshow("GuideBot", annotated_frame)

            key = cv2.waitKey(1)
            if (key & 0xFF == ord('q')) or is_window_closed("GuideBot"):
                break

        except KeyboardInterrupt:
            print("\n[GuideBot] Application terminated by user (Ctrl+C). Cleaning up...")
            break
        except cv2.error as e:
            print(f"[GuideBot] OpenCV display error (recovered): {e}")
            continue
        except Exception as e:
            print(f"[GuideBot] Frame error (recovered): {e}")
            continue

    inference.stop()
    stream.release()
    voice_input.stop()
    cv2.destroyAllWindows()
    print("[GuideBot] Shutdown")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[GuideBot] User killed application (Ctrl+C).")
