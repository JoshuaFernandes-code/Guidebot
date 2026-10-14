// ============================================================================
// PROTOCOL MODE SELECTION (Must be at the very top for Arduino IDE)
// ============================================================================
// - Set to true:  Uses WebSocket (Recommended: lowest latency, continuous stream).
//                 Requires the "WebSockets" library by Markus Sattler in Arduino IDE.
// - Set to false: Uses HTTP POST with Keep-Alive.
//                 Zero external dependencies (uses built-in ESP32 HTTPClient).
#define USE_WEBSOCKET true

#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiClientSecure.h>

#if USE_WEBSOCKET
  #include <WebSocketsClient.h>
  void webSocketEvent(WStype_t type, uint8_t * payload, size_t length);
#else
  #include <HTTPClient.h>
#endif

// ============================================================================
// CONFIGURATION & SECRETS
// ============================================================================

// Load private credentials from secrets.h (which is ignored by Git).
// Copy secrets.h.example -> secrets.h to customize your credentials.
#if __has_include("secrets.h")
  #include "secrets.h"
  const char* ssid        = SECRET_SSID;
  const char* password    = SECRET_PASSWORD;
  const char* server_host = SECRET_SERVER_HOST;
  const int   server_port = SECRET_SERVER_PORT;
  const char* auth_token  = SECRET_AUTH_TOKEN;
#else
  #warning "secrets.h not found! Create esp32/guidebot_stream/secrets.h based on secrets.h.example"
  const char* ssid        = "YOUR_WIFI_SSID";
  const char* password    = "YOUR_WIFI_PASSWORD";
  const char* server_host = "stream.yourdomain.com";
  const int   server_port = 443;
  const char* auth_token  = "guidebot_secret_token";
#endif

// Target frame rate interval in ms.
// 66ms = ~15 FPS target. The actual FPS will be limited by network upload speed.
// Lower this if network is fast; raise it if frames are backing up.
#define FRAME_INTERVAL_MS 66

// ============================================================================
// PIN DEFINITIONS (Seeed Studio XIAO ESP32-S3 Sense)
// ============================================================================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// ============================================================================
// ADAPTIVE RESOLUTION STATE
// ============================================================================
// Tracks actual FPS over a rolling window and switches resolution accordingly.
// The server can override this by sending "FORCE_HIGH_RES:1" on connect.

bool          force_high_res      = false;  // set by server command on WS connect
bool          currently_high_res  = true;   // tracks current sensor state (starts at VGA)
unsigned long adaptive_window_ms  = 0;      // start of current measurement window
int           adaptive_frame_count = 0;     // frames sent in current window
const float   DOWNGRADE_THRESHOLD  = 0.65f; // drop to QVGA if FPS < 65% of target
const float   UPGRADE_THRESHOLD    = 0.85f; // restore VGA  if FPS >= 85% of target
const unsigned long ADAPTIVE_WINDOW = 3000; // measure every 3 seconds

void setResolution(framesize_t size) {
    sensor_t* s = esp_camera_sensor_get();
    if (!s) return;
    s->set_framesize(s, size);
    currently_high_res = (size == FRAMESIZE_VGA);
    Serial.printf("[Adaptive] Resolution set to %s\n",
                  currently_high_res ? "VGA (640x480)" : "QVGA (320x240)");
}

void checkAdaptiveResolution() {
    if (force_high_res) return;  // server said keep high res — never downgrade

    unsigned long now = millis();
    if (now - adaptive_window_ms < ADAPTIVE_WINDOW) return;

    float actual_fps = adaptive_frame_count * 1000.0f / (float)(now - adaptive_window_ms);
    float target_fps = 1000.0f / FRAME_INTERVAL_MS;

    Serial.printf("[Adaptive] Window FPS: %.1f / %.1f target (res: %s)\n",
                  actual_fps, target_fps,
                  currently_high_res ? "VGA" : "QVGA");

    if (currently_high_res && actual_fps < target_fps * DOWNGRADE_THRESHOLD) {
        Serial.println("[Adaptive] Slow network detected — switching to QVGA");
        setResolution(FRAMESIZE_QVGA);
    } else if (!currently_high_res && actual_fps >= target_fps * UPGRADE_THRESHOLD) {
        Serial.println("[Adaptive] Network recovered — switching back to VGA");
        setResolution(FRAMESIZE_VGA);
    }

    // Reset window
    adaptive_window_ms  = now;
    adaptive_frame_count = 0;
}

// ============================================================================
// PROTOCOL IMPLEMENTATION
// ============================================================================
#if USE_WEBSOCKET
  WebSocketsClient webSocket;
  bool is_connected = false;

  void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
      switch(type) {
          case WStype_DISCONNECTED:
              is_connected = false;
              Serial.println("[WS] Disconnected from GuideBot server");
              break;
          case WStype_CONNECTED:
              is_connected = true;
              Serial.printf("[WS] Connected to: %s\n", payload);
              break;
          case WStype_TEXT: {
              String msg = String((char*)payload);
              Serial.printf("[WS] Server command: %s\n", msg.c_str());
              if (msg.startsWith("FORCE_HIGH_RES:")) {
                  force_high_res = msg.substring(15).toInt() == 1;
                  Serial.printf("[Adaptive] Force high res: %s\n",
                                force_high_res ? "YES (always VGA)" : "NO (adaptive)");
                  if (force_high_res) {
                      setResolution(FRAMESIZE_VGA);
                  }
              }
              break;
          }
          case WStype_ERROR:
              Serial.println("[WS] Connection Error");
              break;
          default:
              break;
      }
  }

  void setupStreaming() {
      String wsPath = String("/ws/stream?token=") + auth_token;
      // Connect using TLS on port 443 through Cloudflare Tunnel
      webSocket.beginSSL(server_host, server_port, wsPath.c_str());
      webSocket.onEvent(webSocketEvent);
      webSocket.setReconnectInterval(2000);
      Serial.printf("[Stream] WebSocket configured for wss://%s:%d%s\n", server_host, server_port, wsPath.c_str());
  }

  void sendFrame() {
      webSocket.loop();
      if (!is_connected) {
          return;
      }

      // Check if we should adapt resolution before capturing
      checkAdaptiveResolution();

      camera_fb_t *fb = esp_camera_fb_get();
      if (!fb) {
          Serial.println("[Camera] Capture failed");
          return;
      }

      bool ok = webSocket.sendBIN(fb->buf, fb->len);
      if (ok) {
          adaptive_frame_count++;
      } else {
          Serial.println("[WS] Frame send FAILED (buffer full or disconnected)");
      }
      esp_camera_fb_return(fb);
  }

#else
  WiFiClientSecure client;
  HTTPClient http;

  void setupStreaming() {
      client.setInsecure(); // Cloudflare manages SSL certificates, skip root cert check to conserve RAM
      Serial.printf("[Stream] HTTP POST configured for https://%s/frame\n", server_host);
  }

  void sendFrame() {
      camera_fb_t *fb = esp_camera_fb_get();
      if (!fb) {
          Serial.println("[Camera] Capture failed");
          return;
      }

      Serial.printf("[HTTP] Sending frame: %zu bytes to https://%s/frame\n", fb->len, server_host);

      String url = String("https://") + server_host + "/frame";
      http.begin(client, url);
      http.addHeader("Content-Type", "image/jpeg");
      http.addHeader("X-Auth-Token", auth_token);
      http.setReuse(true);

      int httpResponseCode = http.POST(fb->buf, fb->len);
      if (httpResponseCode == 200) {
          Serial.println("[HTTP] Frame sent OK (200)");
      } else if (httpResponseCode > 0) {
          Serial.printf("[HTTP] Server returned error: %d — %s\n",
                        httpResponseCode, http.getString().c_str());
      } else {
          Serial.printf("[HTTP] Connection failed: %s\n",
                        HTTPClient::errorToString(httpResponseCode).c_str());
      }
      http.end();
      esp_camera_fb_return(fb);
  }
#endif

// ============================================================================
// ARDUINO SETUP & LOOP
// ============================================================================
unsigned long last_frame_time = 0;

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n--- GuideBot ESP32-S3 Remote Stream Client ---");

    // Camera Configuration
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size   = FRAMESIZE_VGA;   // Start at VGA (640x480). Adaptive logic will
                                            // downgrade to QVGA if network is too slow.
    config.jpeg_quality = 15;              // 0=best/largest, 63=worst/smallest.
                                            // 15 is good starting quality; adaptive may
                                            // benefit from smaller frames at QVGA anyway.
    config.fb_count     = 2;
    config.grab_mode    = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[Camera] Init failed with error: 0x%x\n", err);
        return;
    }
    Serial.println("[Camera] Initialized successfully");

    // Connect to WiFi
    Serial.printf("[WiFi] Connecting to %s", ssid);
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\n[WiFi] Connected successfully");
    Serial.printf("[WiFi] IP Address: %s\n", WiFi.localIP().toString().c_str());

    // Initialize Streaming Client
    setupStreaming();

    // Start adaptive FPS measurement window
    adaptive_window_ms  = millis();
    adaptive_frame_count = 0;
}

void loop() {
    // Monitor Wi-Fi connection
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("[WiFi] Lost connection. Reconnecting...");
        WiFi.reconnect();
        delay(2000);
        return;
    }

    // Rate-limited frame capture and upload
    unsigned long now = millis();
    if (now - last_frame_time >= FRAME_INTERVAL_MS) {
        last_frame_time = now;
        sendFrame();
    }

    #if USE_WEBSOCKET
      webSocket.loop();
    #endif

    delay(2);
}