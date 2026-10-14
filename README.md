# GuideBot 🦯

An AI-powered wearable navigation assistant for people who are blind or losing their vision. GuideBot watches the environment through a camera and speaks out loud when something is close or potentially hazardous.

**Status:** Active development — this reflects what currently works.

---

## What Works Right Now

- **Remote & Local Video Streaming:** Streams real-time video from a Seeed XIAO ESP32-S3 Sense to a host laptop across **any network** (cellular hotspot, remote Wi-Fi, or local LAN) via WebSocket / Cloudflare Tunnel / VS Code Ports tunnel.
- **Adaptive Network Quality:** Automatically downscales to QVGA (320x240) if network speed is constrained, or stays in full VGA (640x480) when network allows (configurable via `.env`).
- **Decoupled Real-Time Inference:** Video display runs at full camera frame rate while AI runs asynchronously in the background.
- **Object Detection:** Real-time object recognition (YOLOv8/v11).
- **Depth Estimation:** Distance estimation classified into Close / Medium / Far (MiDaS).
- **Audio & Spoken Alerts:** Spoken hazard warnings with directional cues (pyttsx3).
- **Voice Commands:** Spoken queries for navigation guidance (OpenAI Whisper).
- **Text Reading (OCR):** Reads signs and text on demand (EasyOCR).
- **Face Recognition:** Identifies known individuals (InsightFace).

---

## Hardware

- **Camera / Microcontroller:** Seeed Studio XIAO ESP32-S3 Sense (with camera module)
- **Host Laptop:** Windows / Linux laptop with an NVIDIA GPU (recommended, developed on RTX 3070 Ti; CPU fallback supported)
- **Audio Output:** Headphones / Bone-conduction earphones
- **Audio Input:** USB microphone / Headset mic

---

## Tech Stack

Python 3.11, PyTorch (CUDA), OpenCV, Ultralytics YOLO, MiDaS, OpenAI Whisper, EasyOCR, InsightFace, FastAPI, Uvicorn, WebSockets, Arduino (ESP32-S3).

---

## Setup & Installation Guide

### 1. Clone & Navigate to Folder
Open a terminal (Command Prompt or PowerShell) and navigate into the project directory:
```cmd
cd C:\Users\nathan\DevLobo\Guidebot
```

### 2. Create Virtual Environment
GuideBot requires **Python 3.11.x**. Check your Python version first:
```cmd
python --version
```
Create a clean virtual environment:
```cmd
python -m venv venv
```

### 3. Activate the Virtual Environment
**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```
**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```
*(You should see `(venv)` appear at the beginning of your terminal prompt).*

### 4. Install Dependencies
Run the following installation commands in order inside the activated virtual environment:

```cmd
:: 1. PyTorch with CUDA 12.4 support (recommended for NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

:: 2. Core Computer Vision & AI libraries
pip install numpy ultralytics timm
pip install opencv-python
pip install pyttsx3
pip install openai-whisper sounddevice
pip install easyocr
pip install insightface onnxruntime-gpu

:: 3. Remote Streaming Server & Utilities
pip install fastapi uvicorn websockets python-dotenv

:: 4. Resolve OpenCV GUI conflict (easyocr pulls in headless opencv, which breaks cv2.imshow)
pip uninstall opencv-python-headless -y
pip install --force-reinstall opencv-python
```

---

### 5. Configure Environment Variables (`.env`)
Create your local `.env` file by copying the provided template:
```cmd
copy .env.example .env
```
Open `.env` and configure your settings:
```env
# Shared secret token for the camera stream (must match ESP32 secrets.h)
STREAM_AUTH_TOKEN=guidebot_secret_token

# Set to true to always force VGA (640x480) even if network speed drops.
# Set to false (default) to let the ESP32 dynamically adapt to QVGA (320x240) on slow networks.
FORCE_HIGH_RES=false
```

---

### 6. Network & Tunnel Setup (Port 8000)
GuideBot runs an internal video receiver on port `8000`. To receive video over mobile hotspots or across the internet, choose **one** of the tunneling options below:

#### Option A: Cloudflare Tunnel (Recommended for permanent domains)
1. In Cloudflare Zero Trust / Cloudflare Tunnels dashboard, add a Public Hostname to your tunnel:
   - **Service Type:** Select `HTTP` *(Cloudflare automatically upgrades WebSockets over HTTP)*
   - **URL:** `localhost:8000`
   - **Subdomain:** e.g. `stream.yourdomain.com`
2. Use this subdomain in your ESP32 configuration.

#### Option B: VS Code Built-in Port Tunneling ("VS Code Ports")
If you don't have a Cloudflare domain, you can use VS Code's free built-in tunneling:
1. Open the project in **VS Code**.
2. Open the **Ports** panel in the bottom dock (`Ctrl + ~`, then click the **Ports** tab).
3. Click **Forward a Port** and enter `8000`.
4. Right-click the forwarded port row -> **Port Visibility** -> change from *Private* to **Public**.
5. Copy the generated **Forwarded Address** (e.g. `xxxx-8000.inc1.devtunnels.ms`).
6. Use that host name as the server host in your ESP32 configuration.

#### Option C: Local LAN Mode (No internet required)
If you are testing on the same local Wi-Fi without any tunnel:
1. In `config.py`, set `"stream_mode": "pull"`.
2. Set `"stream_url": "http://<ESP32_IP>/stream"`.

---

### 7. ESP32-S3 Firmware Setup
1. In `esp32/guidebot_stream/`, copy `secrets.h.example` to `secrets.h`:
   ```cmd
   copy esp32\guidebot_stream\secrets.h.example esp32\guidebot_stream\secrets.h
   ```
2. Open `esp32/guidebot_stream/secrets.h` and update:
   - `SECRET_SSID` and `SECRET_PASSWORD` (your Wi-Fi or phone hotspot)
   - `SECRET_SERVER_HOST` (your Cloudflare subdomain or devtunnel host, e.g. `"stream.yourdomain.com"`)
   - `SECRET_AUTH_TOKEN` (must match `STREAM_AUTH_TOKEN` in your `.env`)
3. Open `esp32/guidebot_stream/guidebot_stream.ino` in **Arduino IDE**.
4. In Arduino IDE, install the required library:
   - Go to **Sketch** -> **Include Library** -> **Manage Libraries...**
   - Search for **WebSockets** by **Markus Sattler** and click **Install**.
5. Select Board: **Seeed XIAO ESP32S3** (and the matching COM port).
6. Click **Upload**.

---

### 8. Add Known Faces
Place at least one clear portrait photo (`.jpg` or `.png`) into the `data/known_faces/` folder:
```text
data/known_faces/
    ├── joshua.jpg
    └── friend.jpg
```
The filename (without extension) will be spoken when GuideBot recognizes that person.

---

### 9. Run GuideBot
Start GuideBot on your workstation:
```cmd
python main.py
```
Power on the ESP32 (from any Wi-Fi or phone hotspot). Once connected, the live stream will appear in the "GuideBot" window with real-time detection, depth estimation, and voice responses!

Press `q` in the video window to quit cleanly.

---

## Roadmap

- [x] Live video streaming (LAN + Cloudflare Tunnel + VS Code devtunnels)
- [x] Adaptive resolution for low-bandwidth connections
- [x] Asynchronous real-time inference pipeline
- [x] Object detection
- [x] Depth estimation
- [x] Decision engine + audio feedback
- [x] Voice commands
- [x] Text reading (OCR)
- [x] Face recognition
- [ ] Currency recognition
- [ ] Scene description
- [ ] Final polish + wearable enclosure build

---

## Author

Joshua Fernandes — B.E. Electronics and Computer Engineering, Agnel Institute of Engineering and Management, Goa University
