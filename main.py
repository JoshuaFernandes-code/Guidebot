import cv2
import time
from pipeline.stream import Stream
from pipeline.detection import Detector
from pipeline.depth import DepthEstimator
from pipeline.decision import DecisionEngine
from pipeline.ocr import OCRReader
from audio.tts import TTS
from audio.voice_input import VoiceInput
from config import CONFIG

def main():
    print("[GuideBot] Starting - Phase 6 (OCR)")
    stream = Stream()
    detector = Detector()
    depth_estimator = DepthEstimator()
    decision_engine = DecisionEngine()
    tts = TTS()
    voice_input = VoiceInput()
    ocr_reader = OCRReader()

    cv2.namedWindow("GuideBot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GuideBot", CONFIG["display_width"], CONFIG["display_height"])

    prev_time = time.time()

    while True:
        try:
            frame = stream.get_frame()
            if frame is None or frame.size == 0:
                continue

            frame = cv2.resize(frame, (CONFIG["display_width"], CONFIG["display_height"]))
            frame_width = CONFIG["display_width"]

            results = detector.detect(frame)
            depth_map = depth_estimator.estimate(frame)

            annotated_frame = frame.copy()
            detections = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = results.names[cls_id]

                distance_value = depth_estimator.get_distance_at_box(depth_map, x1, y1, x2, y2)
                direction = decision_engine.get_direction(x1, x2, frame_width)

                detections.append({
                    "label": label,
                    "distance": distance_value,
                    "direction": direction
                })

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f} | {distance_value} | {direction}"
                cv2.putText(annotated_frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
                    text_result = ocr_reader.read_frame(frame)
                    print(f"[GuideBot Speaking] {text_result}")
                    tts.speak(text_result)
                else:
                    print(f"[Voice] Command not recognized: '{command}'")

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("GuideBot", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"[GuideBot] Frame error (recovered): {e}")
            continue

    stream.release()
    voice_input.stop()
    cv2.destroyAllWindows()
    print("[GuideBot] Shutdown")

if __name__ == "__main__":
    main()