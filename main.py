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
        self._inference_fps  = 0.0

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        prev = time.time()
        while self.running:
            frame = self.stream.get_frame()
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

    def get_latest(self):
        """Returns (frame, detections, inference_fps) — all safe copies."""
        with self.lock:
            if self._latest_frame is None:
                return None, [], 0.0
            return (
                self._latest_frame.copy(),
                list(self._latest_detections),
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
    # Pass face_recognizer so /faces API can hot-reload embeddings after add/delete
    stream          = Stream(voice_input=voice_input, face_recognizer=face_recognizer)

    inference = InferencePipeline(stream, detector, depth_estimator, decision_engine)

    cv2.namedWindow("GuideBot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GuideBot", CONFIG["display_width"], CONFIG["display_height"])

    prev_time           = time.time()
    last_fps_update     = 0.0
    display_fps_text    = "0.0"
    inf_fps_text        = "0.0"
    smooth_display_fps  = 0.0
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
                key = cv2.waitKey(30)
                if (key & 0xFF == ord('q')) or is_window_closed("GuideBot"):
                    break
                continue

            # ------------------------------------------------------------------
            # 2. Display: render latest inferred frame
            # ------------------------------------------------------------------
            annotated_frame, detections, inf_fps = inference.get_latest()

            if annotated_frame is None or annotated_frame.size == 0:
                standby = draw_standby_screen(frame_width, frame_height, stream)
                cv2.imshow("GuideBot", standby)
                key = cv2.waitKey(30)
                if (key & 0xFF == ord('q')) or is_window_closed("GuideBot"):
                    break
                continue

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
            messages = decision_engine.decide(detections)
            for message in messages:
                print(f"[GuideBot Speaking] {message}")
                tts.speak(message)

            command = voice_input.get_command()
            if command:
                query = None
                if any(w in command for w in ["ahead", "front", "forward", "head"]):
                    query = "ahead"
                elif "left" in command:
                    query = "left"
                elif "right" in command:
                    query = "right"

                if query:
                    response = decision_engine.answer_query(query, detections, frame_width)
                    print(f"[GuideBot Speaking] {response}")
                    tts.speak(response)
                elif "read" in command:
                    print("[OCR] Reading current frame...")
                    text_result = ocr_reader.read_frame(annotated_frame)
                    print(f"[GuideBot Speaking] {text_result}")
                    tts.speak(text_result)
                elif "who" in command:
                    print("[Face] Identifying...")
                    face_result = face_recognizer.identify(annotated_frame)
                    print(f"[GuideBot Speaking] {face_result}")
                    tts.speak(face_result)
                else:
                    print(f"[Voice] Command not recognized: '{command}'")

            # ------------------------------------------------------------------
            # FPS Overlay: Smoothed & Fixed Positions (No Jitter)
            # ------------------------------------------------------------------
            curr_time = time.time()
            instant_display_fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # Exponential moving average to eliminate micro-fluctuations
            if smooth_display_fps == 0.0:
                smooth_display_fps = instant_display_fps
            else:
                smooth_display_fps = 0.85 * smooth_display_fps + 0.15 * instant_display_fps

            # Refresh the displayed text strings every 300ms so numbers don't flicker
            if curr_time - last_fps_update >= 0.3:
                display_fps_text = f"{smooth_display_fps:.1f}"
                inf_fps_text = f"{inf_fps:.1f}"
                last_fps_update = curr_time

            # Draw dark background badge for contrast
            cv2.rectangle(annotated_frame, (8, 6), (420, 38), (20, 20, 20), -1)
            cv2.rectangle(annotated_frame, (8, 6), (420, 38), (80, 80, 80), 1)

            # Draw Display at x=15 and Inference at fixed x=220
            # Because x=220 is independent, Inference will NEVER shift left or right!
            cv2.putText(annotated_frame, f"Display: {display_fps_text} fps",
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"| Inference: {inf_fps_text} fps",
                        (220, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("GuideBot", annotated_frame)

            key = cv2.waitKey(1)
            if (key & 0xFF == ord('q')) or is_window_closed("GuideBot"):
                break

        except Exception as e:
            print(f"[GuideBot] Frame error (recovered): {e}")
            continue

    inference.stop()
    stream.release()
    voice_input.stop()
    cv2.destroyAllWindows()
    print("[GuideBot] Shutdown")


if __name__ == "__main__":
    main()