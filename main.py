import cv2
from pipeline.stream import Stream
from pipeline.detection import Detector
from pipeline.depth import DepthEstimator
from pipeline.decision import DecisionEngine
from audio.tts import TTS
from config import CONFIG

def main():
    print("[GuideBot] Starting Phase 4 - Decision Engine + Audio")
    stream = Stream()
    detector = Detector()
    depth_estimator = DepthEstimator()
    decision_engine = DecisionEngine()
    tts = TTS()

    while True:
        frame = stream.get_frame()
        if frame is None or frame.size == 0:
            continue

        frame = cv2.resize(frame, (CONFIG["display_width"], CONFIG["display_height"]))

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

            detections.append({"label": label, "distance": distance_value})

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f} | {distance_value}"
            cv2.putText(annotated_frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        messages = decision_engine.decide(detections)
        for message in messages:
            print(f"[GuideBot Speaking] {message}")
            tts.speak(message)

        cv2.imshow("GuideBot - Phase 4", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()
    print("[GuideBot] Shutdown")

if __name__ == "__main__":
    main()