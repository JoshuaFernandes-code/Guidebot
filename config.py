CONFIG = {
    "stream_url": "http://10.233.192.220/stream",
    "stream_timeout": 10,
    "display_width": 800,
    "display_height": 600,
}

CONFIG = {
    # ESP32 Stream Settings
    "stream_url": "http://10.233.192.220/stream",
    "stream_timeout": 10,
    "display_width": 800,
    "display_height": 600,

    # Object Detection Settings
    "yolo_model": "yolo11m.pt",
    "confidence_threshold": 0.5,
    "detection_classes": None,  # None = detect all classes
}