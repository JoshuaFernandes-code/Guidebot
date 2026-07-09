import cv2
from pipeline.stream import Stream
from config import CONFIG

def main():
    print("[GuideBot] Starting Phase 1 - Live Stream")
    stream = Stream()

    while True:
        frame = stream.get_frame()
        if frame is None or frame.size ==0:
            continue

        # Resize for display
        frame = cv2.resize(frame, (CONFIG["display_width"], CONFIG["display_height"]))

        cv2.imshow("GuideBot - Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()
    print("[GuideBot] Shutdown")

if __name__ == "__main__":
    main()