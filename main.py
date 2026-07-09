import cv2
from pipeline.stream import Stream
from pipeline.detection import Detector
from config import CONFIG

def main():
    print("[GuideBot] Starting Phase 2 - Object Detection")
    stream = Stream()
    detector = Detector()

    while True:
        frame = stream.get_frame()
        if frame is None or frame.size == 0:
            continue

        frame = cv2.resize(frame, (CONFIG["display_width"], CONFIG["display_height"]))

        results = detector.detect(frame)
        annotated_frame = results.plot()

        cv2.imshow("GuideBot - Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()
    print("[GuideBot] Shutdown")

if __name__ == "__main__":
    main()