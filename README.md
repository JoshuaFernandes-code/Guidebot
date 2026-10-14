# GuideBot 🦯

An AI-powered wearable navigation assistant for people who are blind or losing their vision. GuideBot watches the environment through a camera and speaks out loud when something is close or potentially hazardous.

**Status:** Active development — this reflects what currently works.

---

## What Works Right Now

- Live camera streaming from a Seeed XIAO ESP32-S3 Sense to a laptop over WiFi
- Real-time object detection (YOLOv11m)
- Distance estimation — objects classified as Close / Medium / Far (MiDaS)
- Spoken audio alerts when something is nearby

No manual input needed — it detects, judges distance, and speaks automatically.

---

## How It Works

The ESP32 only captures and streams video. All AI processing runs on the laptop.

---

## Tech Stack

Python, OpenCV, YOLOv11 (Ultralytics), MiDaS, pyttsx3, Arduino (ESP32-S3)

---

## Hardware

- Seeed XIAO ESP32-S3 Sense
- Laptop with NVIDIA GPU (developed on RTX 3070 Ti)
- Planned: Jetson Orin Nano, bone conduction headphones, custom enclosure

---

## Roadmap

- [x] Live video streaming
- [x] Object detection
- [x] Depth estimation
- [x] Decision engine + audio feedback
- [x] Voice commands
- [x] Text reading (OCR)
- [ ] Currency recognition
- [x] Face recognition
- [ ] Scene description
- [ ] Final polish + wearable build

---

## Author

Joshua Fernandes — B.E. Electronics and Computer Engineering, Agnel Institute of Engineering and Management, Goa University
