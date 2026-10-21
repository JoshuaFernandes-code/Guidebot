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
    "stream_lan_host": os.getenv("STREAM_LAN_HOST", "").strip(),
    "stream_auth_token": os.getenv("STREAM_AUTH_TOKEN", "guidebot_secret_token"),
    # When False (default) or unset: server always forces VGA (640x480) — best detection quality.
    # When True: ESP32 is allowed to auto-downgrade to QVGA on slow/congested networks.
    "allow_low_res": os.getenv("ALLOW_LOW_RES", "false").strip().lower() in ("1", "true", "yes"),

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
    "esp32_mic_vad_threshold": float(os.getenv("ESP32_MIC_VAD_THRESHOLD", "0.015")),
    "whisper_model": os.getenv("WHISPER_MODEL", "base.en"),
    "voice_sample_rate": 16000,
    "voice_record_seconds": 3,
    "wake_check_interval": 0.5,

    # OCR Settings
    "ocr_languages": ["en"],
    "ocr_min_confidence": 0.4,

    # Face Recognition Settings
    "face_similarity_threshold": 0.5,

    # Local LLM Assistant Settings (Ollama)
    # LLM_ENABLED must be explicitly set to true/1/yes in .env to activate the assistant.
    # When false or not set, GuideBot operates exactly as it did before Ollama was added.
    "llm_enabled": os.getenv("LLM_ENABLED", "false").strip().lower() in ("1", "true", "yes"),
    "llm_auto_start": os.getenv("LLM_AUTO_START", "true").strip().lower() in ("1", "true", "yes"),
    "llm_model": os.getenv("LLM_MODEL", "llama3.2:1b"),
    "llm_url": os.getenv("LLM_URL", "http://127.0.0.1:11434"),
    "llm_timeout": float(os.getenv("LLM_TIMEOUT", "25.0")),

    # Local Vision Scene Description Settings (Ollama vision model)
    "vision_enabled": os.getenv("VISION_ENABLED", "true").strip().lower() in ("1", "true", "yes"),
    "vision_model": os.getenv("VISION_MODEL", "qwen3-vl:4b-instruct"),
    "vision_timeout": float(os.getenv("VISION_TIMEOUT", "240")),
    "vision_frame_dir": os.getenv("VISION_FRAME_DIR", "logs/vision-frames"),
    "vision_frame_limit": int(os.getenv("VISION_FRAME_LIMIT", "25")),
    "vision_image_max_width": int(os.getenv("VISION_IMAGE_MAX_WIDTH", "512")),
}
