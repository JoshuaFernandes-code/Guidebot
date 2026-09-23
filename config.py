CONFIG = {
    # ESP32 Stream Settings
    "stream_url": "http://10.26.199.220/stream",
    "stream_timeout": 10,
    "display_width": 800,
    "display_height": 600,

    # Object Detection Settings
    "yolo_model": "yolov8n.pt",
    "confidence_threshold": 0.65,
    "detection_classes": None,

    # Depth Estimation Settings
    "midas_model": "MiDaS_small",
    "depth_max_display": 10.0,

    # Decision Engine Settings
    "danger_classes": ["car", "truck", "bus", "motorcycle", "bicycle"],
    "warning_cooldown_seconds": 4,
    "max_repeats": 1,

    # Voice Command Settings
    "whisper_model": "base",
    "voice_sample_rate": 16000,
    "voice_record_seconds": 3,
    "wake_check_interval": 0.5,
}