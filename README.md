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
- **Voice Commands & Hardware Mic Streaming:** Spoken queries for navigation guidance (OpenAI Whisper) supported via three flexible input modes:
  - **ESP32-S3 Sense Onboard Mic:** Live audio streaming directly from the built-in PDM MEMS microphone on the Sense camera expansion board over WebSocket.
  - **Browser Web Audio:** Dedicated voice command recording page at `/audio`.
  - **Host PC Mic:** Local sounddevice recording.
- **Local LLM Conversational Assistant (Ollama):** Optional intelligent assistance powered by a local Ollama instance (e.g. `llama3.2:3b`) for conversational QA and scene inquiries.
- **Remote ESP Management Dashboard (`/esp`):** Scan & switch Wi-Fi networks, reboot, toggle the microphone, and wipe credentials into hotspot mode remotely over the tunnel without physical access. Smart password handling: open networks hide the password field; previously saved networks allow blank password to reuse stored credentials; new secured networks require a password before connecting.
- **Captive Portal Wi-Fi Provisioning:** Setup hotspot (`GuideBot-Setup` at `192.168.4.1` / `http://guidebot.local`) for credential configuration with hidden tokens and eye visibility toggles. Same smart password logic as the remote dashboard applies here too.
- **Text Reading (OCR):** Reads signs and text on demand (EasyOCR).
- **Face Recognition:** Identifies known individuals (InsightFace) with multi-image person support and real-time hot-reloading.
- **Known Faces Dashboard (`/faces`):** Add & manage known faces from any browser. Supports live webcam photo snapshots and file uploads with drag-and-drop, SQLite-backed name mapping, automatic duplicate file naming (`name_1.png`, `name_2.png`), thumbnail viewing, and instant hot-reloading without restarting the inference server.
- **Unified Navigation Hub (`/`):** Quick-launch cards linking to ESP Remote Control (`/esp`), Known Faces Management (`/faces`), and Voice Input (`/audio`).

---

## Hardware

- **Camera & Microphone Board:** Seeed Studio XIAO ESP32-S3 Sense (equipped with camera daughterboard, microSD slot, and onboard digital PDM MEMS microphone `MSM261D3526H1CPM`)
- **Host Laptop:** Windows / Linux laptop with an NVIDIA GPU (recommended, developed on RTX 3070 Ti; CPU fallback supported)
- **Audio Output:** Headphones / Bone-conduction earphones
- **Audio Input:** Seeed XIAO ESP32-S3 Sense onboard mic / Browser Web Audio (`/audio`) / USB microphone / Headset mic

---

## Tech Stack

Python 3.11, PyTorch (CUDA), OpenCV, Ultralytics YOLO, MiDaS, OpenAI Whisper, EasyOCR, InsightFace, FastAPI, Uvicorn, WebSockets, Arduino (ESP32-S3).

---

## Setup & Installation Guide

### 1. Clone & Navigate to Folder
Open a terminal (Command Prompt or PowerShell) and navigate into the project directory:
```cmd
cd \path\to\Guidebot
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

# Audio Input Options:
# Set USE_ESP32_MIC=true to stream live audio from the Seeed XIAO Sense onboard PDM mic
USE_ESP32_MIC=true
ESP32_MIC_SAMPLE_RATE=16000
ESP32_MIC_VAD_THRESHOLD=0.005

# Set to true to enable the browser-based Web Voice & Command interface at /audio
# and disable the local PC microphone thread.
USE_WEB_MIC=false

# Local LLM Conversational Assistant (Ollama)
LLM_ENABLED=true
LLM_MODEL=llama3.2:3b or llama3.2:1b
LLM_URL=http://127.0.0.1:11434
LLM_TIMEOUT=25.0

# Local vision scene description (requires: ollama pull qwen3-vl:4b-instruct)
VISION_ENABLED=true
VISION_MODEL=qwen3-vl:4b-instruct
VISION_TIMEOUT=240
VISION_FRAME_DIR=logs/vision-frames
VISION_FRAME_LIMIT=25
VISION_IMAGE_MAX_WIDTH=512
```

---

### 6. Network & Tunnel Setup (Port 8000)
GuideBot runs an internal receiver on port `8000`. The ESP32 connects to the configured tunnel URL for commands and audio. The server advertises its LAN IPv4 address over that connection; when the ESP32 can reach that address, camera frames go directly to the server over the LAN. If it cannot, frames fall back to the tunnel. Thus, direct streaming requires the ESP32 and server to be reachable on the same LAN; it does not bypass routing between different networks.

If the server has multiple network adapters, set `STREAM_LAN_HOST` in `.env` to the IPv4 address reachable by the ESP32. Otherwise it is detected from the server's default route. Ensure port `8000` is allowed through the host firewall.

The ESP32 remembers up to five Wi-Fi profiles entered through its setup portal or the `/esp` dashboard. It checks the Python server with heartbeats; after 35 seconds without a response, it tries the other saved networks (up to 15 seconds to associate with each) and enables the `GuideBot-Setup` hotspot if none restore communication. Wi-Fi station and setup hotspot can run together, so the device can remain connected to a network while exposing the setup page at `http://192.168.4.1`.

#### Option A: Cloudflare Tunnel (Recommended for permanent domains)
1. In Cloudflare Zero Trust / Cloudflare Tunnels dashboard, add a Public Hostname to your tunnel:
   - **Service Type:** Select `HTTP` *(Cloudflare automatically upgrades WebSockets over HTTP)*
   - **URL:** `localhost:8000`
   - **Subdomain:** e.g. `stream.yourdomain.com`
2. Use this subdomain in your ESP32 configuration. This connection remains active for device commands and audio, and carries video only when direct LAN streaming is unavailable.

#### Option B: VS Code Built-in Port Tunneling ("VS Code Ports")
If you don't have a Cloudflare domain, you can use VS Code's free built-in tunneling:
1. Open the project in **VS Code**.
2. Open the **Ports** panel in the bottom dock (`Ctrl + ~`, then click the **Ports** tab).
3. Click **Forward a Port** and enter `8000`.
4. Right-click the forwarded port row -> **Port Visibility** -> change from *Private* to **Public**.
5. Copy the generated **Forwarded Address** (e.g. `xxxx-8000.inc1.devtunnels.ms`).
6. Use that host name as the server host in your ESP32 configuration. As with Cloudflare, the tunnel is used for discovery, commands, audio, and video fallback.

#### Option C: LAN-only setup (No tunnel)
If both devices are on the same local Wi-Fi and you do not need remote commands/audio, you can set the ESP32 server host to the computer's LAN address directly. The tunnel-based discovery and fallback described above will not be used.

---

### 7. ESP32-S3 Firmware Setup
1. In `esp32/guidebot_stream/`, copy `secrets.h.example` to `secrets.h`:
   ```cmd
   copy esp32\guidebot_stream\secrets.h.example esp32\guidebot_stream\secrets.h
   ```
2. Open `esp32/guidebot_stream/secrets.h` and configure default fallbacks:
   - `SECRET_SERVER_HOST` (your Cloudflare subdomain or devtunnel host, e.g. `"stream.yourdomain.com"`)
   - `SECRET_AUTH_TOKEN` (must match `STREAM_AUTH_TOKEN` in your `.env`)
   *(Note: Wi-Fi credentials are no longer hardcoded! They are provisioned dynamically via the GuideBot setup portal).*
3. Open `esp32/guidebot_stream/guidebot_stream.ino` in **Arduino IDE**.
4. In Arduino IDE, install the required library:
   - Go to **Sketch** -> **Include Library** -> **Manage Libraries...**
   - Search for **WebSockets** by **Markus Sattler** and click **Install**.
5. Select Board: **XIAO_ESP32S3** (or `ESP32S3 Dev Module`):
   - **Board Package:** Ensure `esp32 by Espressif Systems` (v3.0.x or newer) is installed in the Boards Manager.
   - **PSRAM:** `OPI PSRAM` *(Critical: Must be enabled for camera and audio DMA buffers)*
   - **Flash Mode:** `QIO 80MHz`
   - **Flash Size:** `8MB (64Mb)`
   - **USB CDC On Boot:** `Enabled`
   *(Hardware Note: Firmly press the camera/mic daughterboard onto the XIAO ESP32-S3 core board until the board-to-board connector clicks flush).*
6. Click **Upload**.
7. **Wi-Fi Setup via Captive Portal:**
   - On first boot (or after an erase), the ESP32 launches an open hotspot: **`GuideBot-Setup`**.
   - Connect your phone or laptop to `GuideBot-Setup` and open `http://192.168.4.1` (or `http://guidebot.local`).
   - Pick your Wi-Fi network, enter the password, and save. The ESP32 saves credentials to NVS flash and automatically begins streaming video and microphone audio to your server.
   - You can later switch networks, reboot, toggle the mic, or wipe credentials remotely anytime via `http://<server-host>/esp`.

---

### 8. Add Known Faces
Place at least one clear portrait photo (`.jpg` or `.png`) into the `data/known_faces/` folder (or upload directly via the web dashboard at `/faces`):
```text
data/known_faces/
    ├── joshua.jpg
    └── friend.jpg
```
The filename (without extension) will be spoken when GuideBot recognizes that person.

---

### 9. Local LLM Setup (Ollama Assistant)

GuideBot includes an on-device conversational AI assistant powered by [Ollama](https://ollama.com). It handles open-ended spoken questions (e.g., *"Is there a chair in front of me?"*, *"Do you see my drink?"*, *"What's around me?"*) by combining real-time visible detections, distance estimates, OCR text, and recognized faces into a direct, speakable response.

#### 1. Install Ollama
- Download and run the installer for Windows, macOS, or Linux from **[ollama.com/download](https://ollama.com/download)**.
- Verify the installation by opening a new terminal and running:
  ```cmd
  ollama --version
  ```

#### 2. Pull the Recommended Model
GuideBot defaults to `llama3.2:3b`, which provides high conversational accuracy while running fast enough on consumer GPUs (or CPU) alongside YOLO and MiDaS:
```cmd
ollama pull llama3.2:3b
```
*(For machines with limited VRAM or CPU-only setups, lighter models like `llama3.2:1b` or `qwen2.5:1.5b` can also be used by setting `LLM_MODEL` in `.env`).*

#### 3. Pull the Vision Model for Scene Descriptions
Scene-description commands send the current camera frame to a local vision model. Pull it once:

```cmd
ollama pull qwen3-vl:4b-instruct
```

The exact JPEG sent to Ollama is saved in `logs/vision-frames/`. To keep CPU-only descriptions responsive, frames wider than 512 pixels are downscaled before encoding; GuideBot logs the source and transmitted dimensions. It retains the newest 25 snapshots and logs the frame ID, file path, size, and hash in `guidebot.log`.

#### 4. Service Lifecycle
- **Automatic:** If `LLM_AUTO_START=true` (the default), GuideBot automatically checks if Ollama is listening at `http://127.0.0.1:11434`. If not, it launches `ollama serve` in the background.
- **Manual (Optional):** You can also start the Ollama desktop application or run `ollama serve` in a dedicated terminal before starting GuideBot.
- **Graceful Fallback:** If Ollama is not installed or unreachable, GuideBot logs a clear warning and falls back to rule-based keyword commands without crashing.

#### 5. Configure `.env`
Verify your settings in `.env`:
```env
LLM_ENABLED=true
LLM_AUTO_START=true
LLM_MODEL=llama3.2:3b
LLM_URL=http://127.0.0.1:11434
LLM_TIMEOUT=25.0
VISION_ENABLED=true
VISION_MODEL=qwen3-vl:4b-instruct
VISION_TIMEOUT=240
VISION_FRAME_DIR=logs/vision-frames
VISION_FRAME_LIMIT=25
VISION_IMAGE_MAX_WIDTH=512
```

---

### 10. Run GuideBot
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
- [x] Object detection (YOLOv8)
- [x] Depth estimation (MiDaS)
- [x] Decision engine + audio hazard alerts
- [x] Voice commands (OpenAI Whisper)
- [x] ESP32-S3 Sense onboard PDM microphone streaming
- [x] Text reading (OCR with EasyOCR)
- [x] Face recognition (InsightFace) + web management dashboard (`/faces`)
- [x] Local LLM assistant integration (Ollama / Llama 3.2)
- [ ] Scene description
- [ ] Final polish + wearable enclosure build

---

## Author

* [**Joshua Fernandes**](https://github.com/JoshuaFernandes-code) — B.E. Electronics and Computer Engineering, Agnel Institute of Engineering and Management, Goa University
* [**Nathan Lobo**](https://github.com/nathanlobo) — B.E. Electronics and Computer Engineering, Agnel Institute of Engineering and Management, Goa University
