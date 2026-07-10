CONFIG = {
    # ESP32 Stream Settings
    "stream_url": "http://10.26.199.220/stream",
    "stream_timeout": 10,
    "display_width": 800,
    "display_height": 600,

    # Object Detection Settings
    "yolo_model": "yolo11m.pt",
    "confidence_threshold": 0.5,
    "detection_classes": None,  # None = detect all classes

    # Depth Estimation Settings
    "midas_model": "MiDaS_small",  # Fast, good enough for real-time
    "depth_max_display": 10.0,     # Values beyond this treated as "far"
}