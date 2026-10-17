import os
from dotenv import load_dotenv

# Load variables from .env file if present
load_dotenv()

CONFIG = {
    # ESP32 Stream Settings
    # stream_mode: "push_receiver" (ESP32 sends to Cloudflare tunnel -> local receiver)
    #              or "pull" (Python pulls from local stream_url via cv2.VideoCapture)
    "stream_mode": "push_receiver",
    "receiver_host": "0.0.0.0",
    "receiver_port": 8000,
    "stream_auth_token": os.getenv("STREAM_AUTH_TOKEN", "guidebot_secret_token"),
    # When True: server tells ESP32 to always use VGA regardless of network speed.
    # When False (default): ESP32 auto-downgrades to QVGA on slow networks.
    "force_high_res": os.getenv("FORCE_HIGH_RES", "false").strip().lower() in ("1", "true", "yes"),
    # When True (default): un-mirrors (horizontally flips) incoming camera frames for normal perspective.
    "flip_horizontal": os.getenv("FLIP_HORIZONTAL", "true").strip().lower() in ("1", "true", "yes"),
    "stream_url": "http://10.254.76.220/stream",
    "stream_timeout": 10,
    "display_width": 800,
    "display_height": 600,

    # Object Detection Settings
    "yolo_model": "yolov8n.pt",
    "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.45")),
    "detection_classes": None,

    # Depth Estimation Settings
    "midas_model": "MiDaS_small",
    "depth_max_display": 10.0,

    # Decision Engine Settings
    "danger_classes": ["car", "truck", "bus", "motorcycle", "bicycle"],
    "warning_cooldown_seconds": 4,
    "max_repeats": 1,

    # Voice Command Settings
    "use_web_mic": os.getenv("USE_WEB_MIC", "false").strip().lower() in ("1", "true", "yes"),
    "use_esp32_mic": os.getenv("USE_ESP32_MIC", "false").strip().lower() in ("1", "true", "yes"),
    "esp32_mic_sample_rate": int(os.getenv("ESP32_MIC_SAMPLE_RATE", "16000")),
    "esp32_mic_channels": 1,
    "esp32_mic_vad_threshold": float(os.getenv("ESP32_MIC_VAD_THRESHOLD", "0.005")),
    "whisper_model": "base",
    "voice_sample_rate": 16000,
    "voice_record_seconds": 3,
    "wake_check_interval": 0.5,

    # OCR Settings
    "ocr_languages": ["en"],
    "ocr_min_confidence": 0.4,

    # Face Recognition Settings
    "face_similarity_threshold": 0.5,

    # Local LLM Assistant Settings (Ollama)
    "llm_enabled": os.getenv("LLM_ENABLED", "true").strip().lower() in ("1", "true", "yes"),
    "llm_auto_start": os.getenv("LLM_AUTO_START", "true").strip().lower() in ("1", "true", "yes"),
    "llm_model": os.getenv("LLM_MODEL", "llama3.2:3b"),
    "llm_url": os.getenv("LLM_URL", "http://127.0.0.1:11434"),
    "llm_timeout": float(os.getenv("LLM_TIMEOUT", "25.0")),
}