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
#include <Preferences.h>
#include <DNSServer.h>
#include <WebServer.h>
#include <ESPmDNS.h>

#if USE_WEBSOCKET
  #include <WebSocketsClient.h>
  void webSocketEvent(WStype_t type, uint8_t * payload, size_t length);
#else
  #include <HTTPClient.h>
#endif

// ============================================================================
// FALLBACK DEFAULTS (Used if no server host or auth token are saved in Flash yet)
// Note: Wi-Fi credentials are NEVER hardcoded; configured via GuideBot-Setup portal.
// ============================================================================
#if __has_include("secrets.h")
  #include "secrets.h"
  const char* DEFAULT_SERVER_HOST = SECRET_SERVER_HOST;
  const int   DEFAULT_SERVER_PORT = SECRET_SERVER_PORT;
  const char* DEFAULT_AUTH_TOKEN  = SECRET_AUTH_TOKEN;
#else
  const char* DEFAULT_SERVER_HOST = "guidebot.codinx.app";
  const int   DEFAULT_SERVER_PORT = 443;
  const char* DEFAULT_AUTH_TOKEN  = "guidebot_secret_token";
#endif

// Target frame rate interval in ms (66ms = ~15 FPS target)
#define FRAME_INTERVAL_MS 66

// Hotspot SSID for initial setup (open, no password required)
#define AP_SSID "GuideBot-Setup"

// ============================================================================
// RUNTIME CONFIGURATION (Stored in ESP32 Flash Memory via Preferences)
// ============================================================================
String config_ssid;
String config_password;
String config_server_host;
int    config_server_port = 443;
String config_auth_token;

bool   config_is_configured = false;

Preferences prefs;
DNSServer dnsServer;
WebServer portalServer(80);

// Remote Command Flags (received over WebSocket, handled safely in loop())
volatile bool flag_erase_wifi      = false;
volatile bool flag_reboot          = false;
volatile bool flag_scan_wifi       = false;
volatile bool flag_set_wifi        = false;
String        pending_wifi_payload = "";

void sanitizeConfig() {
    config_ssid.trim();
    config_password.trim();
    config_auth_token.trim();

    // Strip http://, https://, wss://, ws:// prefixes
    config_server_host.trim();
    if (config_server_host.startsWith("https://")) config_server_host.remove(0, 8);
    else if (config_server_host.startsWith("http://")) config_server_host.remove(0, 7);
    else if (config_server_host.startsWith("wss://")) config_server_host.remove(0, 6);
    else if (config_server_host.startsWith("ws://")) config_server_host.remove(0, 5);

    // Strip trailing slashes or paths
    int slash_idx = config_server_host.indexOf('/');
    if (slash_idx != -1) {
        config_server_host = config_server_host.substring(0, slash_idx);
    }

    // Extract port if formatted as host:port
    int colon_idx = config_server_host.indexOf(':');
    if (colon_idx != -1) {
        config_server_port = config_server_host.substring(colon_idx + 1).toInt();
        config_server_host = config_server_host.substring(0, colon_idx);
    }
    if (config_server_port <= 0) {
        config_server_port = 443;
    }
}

void loadConfiguration() {
    prefs.begin("guidebot", false);
    // Explicitly check if Wi-Fi has ever been configured by user/portal
    config_is_configured = prefs.getBool("configured", false);

    if (config_is_configured) {
        config_ssid        = prefs.getString("ssid", "");
        config_password    = prefs.getString("password", "");
        config_server_host = prefs.getString("host", DEFAULT_SERVER_HOST);
        config_server_port = prefs.getInt("port", DEFAULT_SERVER_PORT);
        config_auth_token  = prefs.getString("token", DEFAULT_AUTH_TOKEN);
    } else {
        // Unconfigured or reset: clear runtime Wi-Fi so board does not attempt auto-join
        config_ssid        = "";
        config_password    = "";
        config_server_host = prefs.getString("host", DEFAULT_SERVER_HOST);
        config_server_port = prefs.getInt("port", DEFAULT_SERVER_PORT);
        config_auth_token  = prefs.getString("token", DEFAULT_AUTH_TOKEN);
    }
    prefs.end();

    sanitizeConfig();

    Serial.println("[Config] Loaded configuration from flash memory:");
    Serial.printf("         Configured:  %s\n", config_is_configured ? "YES" : "NO (Needs Setup)");
    Serial.printf("         SSID:        %s\n", config_ssid.length() > 0 ? config_ssid.c_str() : "(none)");
    Serial.printf("         Server Host: %s:%d\n", config_server_host.c_str(), config_server_port);
    Serial.printf("         Auth Token:  %s\n", config_auth_token.length() > 0 ? "***" : "none");
}

void saveConfiguration(const String& new_ssid, const String& new_pass, const String& new_host, const String& new_token) {
    config_ssid          = new_ssid;
    config_password      = new_pass;
    config_server_host   = new_host;
    config_server_port   = 443;
    config_auth_token    = new_token;
    config_is_configured = (new_ssid.length() > 0);
    sanitizeConfig();

    prefs.begin("guidebot", false);
    prefs.putBool("configured", config_is_configured);
    prefs.putString("ssid", config_ssid);
    prefs.putString("password", config_password);
    prefs.putString("host", config_server_host);
    prefs.putInt("port", config_server_port);
    prefs.putString("token", config_auth_token);
    prefs.end();
    Serial.println("[Config] Successfully saved new credentials to flash.");
}

void performClearCredentials() {
    Serial.println("\n[Config] Clearing all saved Wi-Fi credentials...");
    // Clear application preferences
    prefs.begin("guidebot", false);
    prefs.clear();
    prefs.putBool("configured", false);
    prefs.end();

    // Wipe ESP32 hardware station cache and disconnect from router
    WiFi.disconnect(true, true);
    WiFi.mode(WIFI_OFF);

    config_is_configured = false;
    config_ssid          = "";
    config_password      = "";
    Serial.println("[Config] Stored Wi-Fi erased & hardware cache wiped successfully.");
}

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
bool          force_high_res       = false;
bool          currently_high_res   = true;
unsigned long adaptive_window_ms   = 0;
int           adaptive_frame_count = 0;
const float   DOWNGRADE_THRESHOLD  = 0.65f;
const float   UPGRADE_THRESHOLD    = 0.85f;
const unsigned long ADAPTIVE_WINDOW = 3000;

void setResolution(framesize_t size) {
    bool want_high = (size == FRAMESIZE_VGA);
    if (want_high == currently_high_res) {
        return; // Already at this resolution! Avoid camera sensor reset & FB-OVF
    }
    sensor_t* s = esp_camera_sensor_get();
    if (!s) return;
    s->set_framesize(s, size);
    currently_high_res = want_high;
    Serial.printf("[Adaptive] Resolution changed to %s\n",
                  currently_high_res ? "VGA (640x480)" : "QVGA (320x240)");
}

void checkAdaptiveResolution() {
    if (force_high_res) return;

    unsigned long now = millis();
    if (now - adaptive_window_ms < ADAPTIVE_WINDOW) return;

    float actual_fps = adaptive_frame_count * 1000.0f / (float)(now - adaptive_window_ms);
    float target_fps = 1000.0f / FRAME_INTERVAL_MS;

    if (currently_high_res && actual_fps < target_fps * DOWNGRADE_THRESHOLD) {
        Serial.printf("[Adaptive] FPS dropped to %.1f (target %.1f) -> Switching to QVGA\n", actual_fps, target_fps);
        setResolution(FRAMESIZE_QVGA);
    } else if (!currently_high_res && actual_fps >= target_fps * UPGRADE_THRESHOLD) {
        Serial.printf("[Adaptive] FPS recovered to %.1f (target %.1f) -> Switching to VGA\n", actual_fps, target_fps);
        setResolution(FRAMESIZE_VGA);
    }

    adaptive_window_ms   = now;
    adaptive_frame_count = 0;
}

// ============================================================================
// PROTOCOL IMPLEMENTATION (WebSocket / HTTP)
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
              webSocket.sendTXT("DEVICE_INFO:{\"ssid\":\"" + config_ssid + "\"}");
              break;
          case WStype_TEXT: {
              // Convert payload safely using exact length to avoid reading beyond buffer
              String msg = "";
              for (size_t i = 0; i < length; i++) {
                  msg += (char)payload[i];
              }
              msg.trim();
              Serial.printf("[WS] Server command: '%s'\n", msg.c_str());

              if (msg.startsWith("FORCE_HIGH_RES:")) {
                  force_high_res = msg.substring(15).toInt() == 1;
                  Serial.printf("[Adaptive] Force high res: %s\n",
                                force_high_res ? "YES (always VGA)" : "NO (adaptive)");
                  if (force_high_res) {
                      setResolution(FRAMESIZE_VGA);
                  }
              } else if (msg == "SCAN_WIFI" || msg.startsWith("SCAN_WIFI")) {
                  flag_scan_wifi = true;
              } else if (msg.startsWith("SET_WIFI:")) {
                  pending_wifi_payload = msg.substring(9);
                  flag_set_wifi = true;
              } else if (msg == "REBOOT" || msg.startsWith("REBOOT")) {
                  flag_reboot = true;
              } else if (msg == "ERASE_WIFI" || msg.startsWith("ERASE_WIFI")) {
                  flag_erase_wifi = true;
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
      String wsPath = String("/ws/stream?token=") + config_auth_token;
      if (config_server_port == 443) {
          webSocket.beginSSL(config_server_host.c_str(), config_server_port, wsPath.c_str());
      } else {
          webSocket.begin(config_server_host.c_str(), config_server_port, wsPath.c_str());
      }
      webSocket.onEvent(webSocketEvent);
      webSocket.setReconnectInterval(2000);
      Serial.printf("[Stream] WebSocket configured for %s://%s:%d%s\n",
                    (config_server_port == 443 ? "wss" : "ws"),
                    config_server_host.c_str(), config_server_port, wsPath.c_str());
  }

  void sendFrame(camera_fb_t *fb) {
      bool ok = webSocket.sendBIN(fb->buf, fb->len);
      if (ok) {
          adaptive_frame_count++;
      } else {
          Serial.println("[WS] Frame send FAILED (buffer full or network lag)");
      }
  }

#else
  WiFiClientSecure client;
  HTTPClient http;

  void setupStreaming() {
      client.setInsecure();
      Serial.printf("[Stream] HTTP POST configured for https://%s/frame\n", config_server_host.c_str());
  }

  void sendFrame(camera_fb_t *fb) {
      String url = String("https://") + config_server_host + "/frame";
      http.begin(client, url);
      http.addHeader("Content-Type", "image/jpeg");
      http.addHeader("X-Auth-Token", config_auth_token);
      http.setReuse(true);

      int httpResponseCode = http.POST(fb->buf, fb->len);
      if (httpResponseCode == 200) {
          // OK
      } else if (httpResponseCode > 0) {
          Serial.printf("[HTTP] Server returned error: %d\n", httpResponseCode);
      }
      http.end();
  }
#endif

// ============================================================================
// CAPTIVE PORTAL & WEB DASHBOARD
// ============================================================================
void handlePortalRoot() {
    Serial.println("[Portal] Scanning nearby Wi-Fi networks...");
    int n = WiFi.scanNetworks();
    Serial.printf("[Portal] Found %d networks\n", n);

    bool is_wifi_live = (WiFi.status() == WL_CONNECTED);

    String html = "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1'>";
    html += "<title>GuideBot Settings</title>";
    html += "<style>";
    html += ":root { --bg:#090a0f; --surface:#13151c; --border:#222634; --text:#f1f5f9; --text-muted:#94a3b8; --primary:#3b82f6; --primary-hover:#2563eb; --danger:#ef4444; --danger-hover:#dc2626; --success-bg:#064e3b; --success-border:#059669; --success-text:#a7f3d0; --warn-bg:#451a03; --warn-border:#b45309; --warn-text:#fde68a; }";
    html += "* { box-sizing:border-box; margin:0; padding:0; font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',Roboto,sans-serif; }";
    html += "body { background:var(--bg); color:var(--text); padding:28px 16px; display:flex; justify-content:center; -webkit-font-smoothing:antialiased; }";
    html += ".container { background:var(--surface); border:1px solid var(--border); border-radius:18px; padding:28px; width:100%; max-width:440px; box-shadow:0 12px 36px rgba(0,0,0,0.45); }";
    html += ".header { text-align:center; margin-bottom:24px; }";
    html += ".header h1 { font-size:20px; font-weight:700; letter-spacing:-0.4px; color:#fff; display:flex; align-items:center; justify-content:center; gap:8px; }";
    html += ".header p { font-size:13px; color:var(--text-muted); margin-top:4px; }";
    html += ".status-card { padding:14px 16px; border-radius:12px; margin-bottom:22px; font-size:13px; line-height:1.6; }";
    html += ".status-online { background:var(--success-bg); border:1px solid var(--success-border); color:var(--success-text); }";
    html += ".status-hotspot { background:var(--warn-bg); border:1px solid var(--warn-border); color:var(--warn-text); }";
    html += ".field-group { margin-bottom:16px; }";
    html += "label { font-size:12px; font-weight:600; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.6px; display:block; margin-bottom:7px; }";
    html += "select, input { width:100%; padding:11px 14px; background:#1a1d27; border:1px solid var(--border); border-radius:10px; color:#fff; font-size:14px; outline:none; transition:border-color 0.15s; }";
    html += "select:focus, input:focus { border-color:var(--primary); }";
    html += ".helper { font-size:11px; color:#64748b; margin-top:5px; }";
    html += ".btn { width:100%; padding:12px 16px; border-radius:10px; font-size:14px; font-weight:600; cursor:pointer; border:none; transition:background-color 0.15s, opacity 0.15s; display:flex; align-items:center; justify-content:center; gap:8px; }";
    html += ".btn-primary { background:var(--primary); color:#fff; margin-top:22px; }";
    html += ".btn-primary:hover { background:var(--primary-hover); }";
    html += ".btn-danger { background:transparent; border:1px solid #7f1d1d; color:#fca5a5; margin-top:12px; }";
    html += ".btn-danger:hover { background:rgba(239,68,68,0.12); border-color:var(--danger); color:#fff; }";
    html += ".btn-ghost { background:transparent; border:1px solid #2d3345; color:var(--text-muted); font-size:12px; padding:9px 12px; margin-top:10px; }";
    html += ".btn-ghost:hover { border-color:#475569; color:#cbd5e1; }";
    html += ".modal-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.75); backdrop-filter:blur(4px); display:none; align-items:center; justify-content:center; padding:16px; z-index:99; }";
    html += ".modal { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:24px; max-width:380px; width:100%; text-align:center; box-shadow:0 20px 40px rgba(0,0,0,0.6); }";
    html += ".modal h3 { font-size:17px; font-weight:700; color:#fff; margin-bottom:8px; }";
    html += ".modal p { font-size:13px; color:var(--text-muted); line-height:1.5; margin-bottom:20px; }";
    html += ".modal-actions { display:flex; gap:10px; }";
    html += ".modal-actions button { flex:1; padding:11px; font-size:13px; font-weight:600; border-radius:9px; border:none; cursor:pointer; }";
    html += ".btn-modal-cancel { background:#222634; color:var(--text); }";
    html += ".btn-modal-confirm { background:var(--danger); color:#fff; }";
    html += ".pw-wrap { position:relative; display:flex; }";
    html += ".pw-wrap input { flex:1; padding-right:42px; }";
    html += ".eye-btn { position:absolute; right:0; top:0; height:100%; width:40px; background:none; border:none; cursor:pointer; font-size:16px; color:var(--text-muted); opacity:0.45; display:flex; align-items:center; justify-content:center; transition:opacity 0.15s; }";
    html += ".eye-btn:hover { opacity:0.8; }";
    html += "</style>";
    html += "<script>";
    html += "function onSsidChange(sel){";
    html += "  var c=document.getElementById('custom_ssid_box');";
    html += "  var pwGroup=document.getElementById('pw_field_group');";
    html += "  var pw=document.getElementById('wifiPw');";
    html += "  var pwHint=document.getElementById('pw_hint');";
    html += "  var isCustom=(sel.value==='__custom__');";
    html += "  c.style.display=isCustom?'block':'none';";
    html += "  if(isCustom){";
    html += "    pwGroup.style.display='block';";
    html += "    pw.placeholder='Enter Wi-Fi password (leave empty if open)';";
    html += "    pwHint.style.display='none';";
    html += "    return;";
    html += "  }";
    html += "  var opt=sel.options[sel.selectedIndex];";
    html += "  var isOpen=(opt&&opt.getAttribute('data-open')==='1');";
    html += "  var isSaved=(opt&&opt.getAttribute('data-saved')==='1');";
    html += "  if(isOpen){";
    html += "    pwGroup.style.display='none';";
    html += "    pw.value='';";
    html += "    pwHint.style.display='none';";
    html += "  }else{";
    html += "    pwGroup.style.display='block';";
    html += "    if(isSaved){";
    html += "      pw.value='';";
    html += "      pw.placeholder='Leave blank to use saved password';";
    html += "      pwHint.innerText='Previously saved network. Leave blank to keep existing password.';";
    html += "      pwHint.style.display='block';";
    html += "    }else{";
    html += "      pw.value='';";
    html += "      pw.placeholder='Enter Wi-Fi password';";
    html += "      pwHint.style.display='none';";
    html += "    }";
    html += "  }";
    html += "}";
    html += "function validateSaveForm(){";
    html += "  var sel=document.querySelector(\"select[name='ssid']\");";
    html += "  var isCustom=(sel.value==='__custom__');";
    html += "  var pw=document.getElementById('wifiPw');";
    html += "  if(isCustom){";
    html += "    var cSsid=document.getElementById('customSsid').value.trim();";
    html += "    if(!cSsid){";
    html += "      showAlert('Network Required','Please enter a network SSID.');";
    html += "      return false;";
    html += "    }";
    html += "  }else{";
    html += "    var opt=sel.options[sel.selectedIndex];";
    html += "    var isOpen=(opt&&opt.getAttribute('data-open')==='1');";
    html += "    var isSaved=(opt&&opt.getAttribute('data-saved')==='1');";
    html += "    if(!isOpen&&!isSaved){";
    html += "      if(!pw.value||pw.value.trim()===''){";
    html += "        showAlert('Password Required','Password is required for this secured network.');";
    html += "        return false;";
    html += "      }";
    html += "    }";
    html += "  }";
    html += "  return true;";
    html += "}";
    html += "var targetForm=null;";
    html += "function showAlert(title,desc){";
    html += "  document.getElementById('mTitle').innerText=title;";
    html += "  document.getElementById('mDesc').innerText=desc;";
    html += "  document.getElementById('mActions').innerHTML=\"<button type='button' class='btn btn-primary' style='width:100%' onclick='closeModal()'>OK</button>\";";
    html += "  document.getElementById('modal').style.display='flex';";
    html += "}";
    html += "function openModal(formId,title,desc){";
    html += "  targetForm=document.getElementById(formId);";
    html += "  document.getElementById('mTitle').innerText=title;";
    html += "  document.getElementById('mDesc').innerText=desc;";
    html += "  document.getElementById('mActions').innerHTML=\"<button type='button' class='btn-modal-cancel' onclick='closeModal()'>Cancel</button><button type='button' class='btn-modal-confirm' onclick='confirmAction()'>Confirm</button>\";";
    html += "  document.getElementById('modal').style.display='flex';";
    html += "}";
    html += "function closeModal(){document.getElementById('modal').style.display='none';targetForm=null;}";
    html += "function confirmAction(){if(targetForm) targetForm.submit(); closeModal();}";
    html += "function togglePw(id,btn){var el=document.getElementById(id);el.type=(el.type==='password')?'text':'password';btn.style.opacity=(el.type==='text')?'1':'0.45';}";
    html += "window.addEventListener('DOMContentLoaded',function(){var s=document.querySelector(\"select[name='ssid']\");if(s)onSsidChange(s);});";
    html += "</script>";
    html += "</head><body>";
    html += "<div class='container'>";
    html += "<div class='header'><h1>GuideBot Device Setup</h1><p>Network & Streaming Endpoint Configuration</p></div>";

    // Status Banner
    if (is_wifi_live) {
        html += "<div class='status-card status-online'>";
        html += "<strong>Network:</strong> " + WiFi.SSID() + "<br>";
        html += "<strong>Device IP:</strong> " + WiFi.localIP().toString() + " (" + String(WiFi.RSSI()) + " dBm)<br>";
        html += "<strong>Hostname:</strong> <a href='http://guidebot.local' style='color:#a7f3d0;text-decoration:underline;'>http://guidebot.local</a><br>";
        html += "<strong>Stream Endpoint:</strong> " + config_server_host + "</div>";
    } else {
        html += "<div class='status-card status-hotspot'>";
        html += "<strong>Mode: Setup Hotspot Active</strong><br>";
        html += "Connected via <code>GuideBot-Setup</code> (192.168.4.1). Enter network credentials below.</div>";
    }

    // Save Form
    html += "<form id='saveForm' action='/save' method='POST' onsubmit='return validateSaveForm()'>";

    // Wi-Fi Dropdown
    html += "<div class='field-group'><label>Wi-Fi Network</label>";
    html += "<select name='ssid' onchange='onSsidChange(this)'>";
    if (n == 0) {
        html += "<option value='__custom__'>No networks found (Enter manually)</option>";
    } else {
        for (int i = 0; i < n; ++i) {
            String netName = WiFi.SSID(i);
            int rssi = WiFi.RSSI(i);
            bool isOpen = (WiFi.encryptionType(i) == WIFI_AUTH_OPEN);
            bool isSaved = (netName == config_ssid);
            String isCurrent = (netName == config_ssid) ? " selected" : "";
            html += "<option value='" + netName + "' data-open='" + (isOpen ? "1" : "0") + "' data-saved='" + (isSaved ? "1" : "0") + "'" + isCurrent + ">";
            html += (isOpen ? "&#128275; " : "&#128274; ") + netName + " (" + (isOpen ? "Open" : String(rssi) + " dBm") + (isSaved ? " - Saved" : "") + ")</option>";
        }
        html += "<option value='__custom__'>Other network (Enter manually)...</option>";
    }
    html += "</select></div>";

    // Custom SSID Box
    html += "<div id='custom_ssid_box' class='field-group' style='display:" + String(n == 0 ? "block" : "none") + ";'>";
    html += "<label>Network Name (SSID)</label>";
    html += "<input type='text' id='customSsid' name='custom_ssid' placeholder='Enter network SSID'>";
    html += "</div>";

    // Password with eye toggle (hidden if open, shown if locked)
    html += "<div id='pw_field_group' class='field-group'><label>Password</label>";
    html += "<div class='pw-wrap'>";
    html += "<input type='password' id='wifiPw' name='password' placeholder='Enter password'>";
    html += "<button type='button' class='eye-btn' onclick=\"togglePw('wifiPw',this)\" aria-label='Show/hide password'>&#128065;</button>";
    html += "</div>";
    html += "<div id='pw_hint' class='helper' style='display:none;'></div>";
    html += "</div>";

    // Server Domain
    html += "<div class='field-group'><label>Server Domain / Tunnel Host</label>";
    html += "<input type='text' name='host' value='" + config_server_host + "' required>";
    html += "<div class='helper'>e.g. guidebot.codinx.app or cloudflare tunnel domain</div></div>";

    // Auth Token — stored value is never exposed; user submits new token or leaves blank to keep existing
    html += "<div class='field-group'><label>Authentication Token</label>";
    html += "<div class='pw-wrap'>";
    html += "<input type='password' id='authTok' name='token' placeholder='Leave blank to keep current token'>";
    html += "<button type='button' class='eye-btn' onclick=\"togglePw('authTok',this)\" aria-label='Show/hide token'>&#128065;</button>";
    html += "</div>";
    html += "<div class='helper'>Must match STREAM_AUTH_TOKEN in your GuideBot environment</div></div>";

    html += "<button type='submit' class='btn btn-primary'>Save & Connect</button>";
    html += "</form>";

    // Reset Form (Hidden, triggered via custom modal)
    html += "<form id='eraseForm' action='/erase' method='POST'></form>";

    // Trigger Button
    html += "<button type='button' class='btn btn-danger' onclick=\"openModal('eraseForm','Clear Saved Wi-Fi?','This will wipe all saved networks and switch GuideBot into Setup Hotspot mode immediately.')\">Clear Wi-Fi Credentials</button>";

    html += "</div>";

    // Custom Modal Container
    html += "<div id='modal' class='modal-overlay'><div class='modal'>";
    html += "<h3 id='mTitle'>Confirm Action</h3>";
    html += "<p id='mDesc'>Are you sure you want to proceed?</p>";
    html += "<div class='modal-actions' id='mActions'>";
    html += "<button type='button' class='btn-modal-cancel' onclick='closeModal()'>Cancel</button>";
    html += "<button type='button' class='btn-modal-confirm' onclick='confirmAction()'>Confirm</button>";
    html += "</div></div></div>";

    html += "</body></html>";

    portalServer.send(200, "text/html", html);
}

void handlePortalSave() {
    String selected_ssid = portalServer.arg("ssid");
    if (selected_ssid == "__custom__") {
        selected_ssid = portalServer.arg("custom_ssid");
    }
    selected_ssid.trim();
    String new_pass  = portalServer.arg("password");
    String new_host  = portalServer.arg("host");
    String new_token = portalServer.arg("token");
    new_host.trim();
    new_token.trim();

    // If user left the token field blank, keep the currently stored token unchanged
    if (new_token.length() == 0) {
        new_token = config_auth_token;
    }

    // If reconnecting to previously saved network with empty password, use previous password
    if (selected_ssid == config_ssid && new_pass.length() == 0) {
        new_pass = config_password;
    }

    saveConfiguration(selected_ssid, new_pass, new_host, new_token);

    String html = "<!DOCTYPE html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>";
    html += "<style>body{background:#121214;color:#ffeb3b;font-family:sans-serif;text-align:center;padding:50px 20px;}h2{color:#fff;}</style>";
    html += "</head><body>";
    html += "<h2>Settings Saved!</h2>";
    html += "<p>GuideBot is restarting and connecting to <b>" + selected_ssid + "</b>...</p>";
    html += "<p style='color:#999;font-size:13px;'>You can now access the dashboard at the new IP assigned by your router.</p>";
    html += "</body></html>";

    portalServer.send(200, "text/html", html);
    delay(2000);
    ESP.restart();
}

void handlePortalErase() {
    performClearCredentials();

    String html = "<!DOCTYPE html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>";
    html += "<style>body{background:#121214;color:#ff5252;font-family:sans-serif;text-align:center;padding:50px 20px;}h2{color:#fff;}</style>";
    html += "</head><body>";
    html += "<h2>Credentials Erased!</h2>";
    html += "<p style='color:#ccc;'>All stored Wi-Fi networks have been wiped from memory.</p>";
    html += "<p>GuideBot is restarting in <b>Setup Hotspot Mode</b>...</p>";
    html += "<p style='color:#888;font-size:13px;'>In a few seconds, look for Wi-Fi <b>'GuideBot-Setup'</b> on your phone/laptop and connect to configure.</p>";
    html += "</body></html>";

    portalServer.send(200, "text/html", html);
    delay(1500);
    ESP.restart();
}

void handleCaptiveRedirect() {
    portalServer.sendHeader("Location", "http://192.168.4.1/", true);
    portalServer.send(302, "text/plain", "");
}

void setupPortalRoutes() {
    portalServer.on("/", handlePortalRoot);
    portalServer.on("/save", HTTP_POST, handlePortalSave);
    portalServer.on("/erase", HTTP_POST, handlePortalErase);

    portalServer.on("/hotspot-detect.html", handlePortalRoot);
    portalServer.on("/generate_204", handlePortalRoot);
    portalServer.on("/gen_204", handlePortalRoot);
    portalServer.on("/ncsi.txt", handlePortalRoot);
    portalServer.onNotFound(handleCaptiveRedirect);
}

void startConfigPortal() {
    Serial.println("\n========================================================");
    Serial.println("   LAUNCHING GUIDEBOT WI-FI SETUP HOTSPOT");
    Serial.println("========================================================");
    Serial.println("1. Connect your phone or laptop to Wi-Fi: GuideBot-Setup");
    Serial.println("2. Open browser and visit: http://192.168.4.1");
    Serial.println("========================================================\n");

    WiFi.disconnect(true);
    delay(100);
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID);

    IPAddress apIP = WiFi.softAPIP();
    Serial.printf("[Portal] Hotspot active at IP: %s\n", apIP.toString().c_str());

    // Start DNS server redirecting all domain queries to 192.168.4.1 (Captive Portal)
    dnsServer.start(53, "*", apIP);

    setupPortalRoutes();
    portalServer.begin();

    // Stay in setup loop indefinitely until user saves credentials and device reboots
    while (true) {
        dnsServer.processNextRequest();
        portalServer.handleClient();
        delay(5);
    }
}

// ============================================================================
// CAMERA INITIALIZATION
// ============================================================================
bool initCamera() {
    Serial.println("\n[Camera] Checking PSRAM...");
    if (psramFound()) {
        Serial.printf("[Camera] PSRAM is enabled (%d KB available).\n", ESP.getPsramSize() / 1024);
    } else {
        Serial.println("****************************************************************");
        Serial.println("[CRITICAL WARNING] PSRAM NOT DETECTED!");
        Serial.println("On Seeed Studio XIAO ESP32-S3 Sense, external PSRAM is REQUIRED.");
        Serial.println("In Arduino IDE: Go to Tools -> PSRAM -> Select 'OPI PSRAM'");
        Serial.println("Without OPI PSRAM, camera frame buffer overflow (FB-OVF) occurs!");
        Serial.println("****************************************************************");
    }

    camera_config_t config = {};
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
    // 10 MHz is a valid ESP32-S3 XCLK divisor and keeps the capture rate manageable.
    config.xclk_freq_hz = 10000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size   = FRAMESIZE_VGA;

    if (psramFound()) {
        // Lower numbers create larger JPEGs. 12 can exceed the driver's VGA
        // JPEG receive buffer on noisy/dim scenes, producing cam_hal: FB-OVF.
        config.jpeg_quality = 20;
        config.fb_count     = 2;
        config.fb_location  = CAMERA_FB_IN_PSRAM;
        config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
        // The automatic VGA JPEG buffer is only 640 * 480 / 5 (61,440 bytes).
        // Allocate enough room for noisy/dim frames, which compress poorly.
        config.jpeg_buffer_size = 128 * 1024;
    } else {
        config.frame_size   = FRAMESIZE_QVGA;
        config.jpeg_quality = 16;
        config.fb_count     = 1;
        config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
    }

    Serial.printf("[Camera] JPEG receive buffer: %u KB\n",
                  (unsigned)(config.jpeg_buffer_size / 1024));

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[Camera] Init failed with error: 0x%x\n", err);
        return false;
    }
    Serial.println("[Camera] Initialized successfully");
    return true;
}

// ============================================================================
// ARDUINO SETUP & LOOP
// ============================================================================
unsigned long last_frame_time = 0;
unsigned long wifi_disconnect_start = 0;

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n--- GuideBot ESP32-S3 Smart Client ---");

    // Load credentials from flash memory (or fallback to secrets.h)
    loadConfiguration();

    // Check if user sends command during initial 2 seconds to wipe credentials
    Serial.println("[Boot] Type 'erase' in Serial Monitor within 2s to wipe stored credentials...");
    unsigned long boot_check = millis();
    while (millis() - boot_check < 2000) {
        if (Serial.available()) {
            String c = Serial.readStringUntil('\n');
            c.trim();
            if (c.equalsIgnoreCase("erase") || c.equalsIgnoreCase("reset")) {
                performClearCredentials();
                Serial.println("[Boot] Stored credentials wiped! Starting hotspot...");
                break;
            }
        }
        delay(50);
    }

    // Attempt to connect to saved Wi-Fi if available and configured
    bool connected = false;
    if (config_is_configured && config_ssid.length() > 0) {
        Serial.printf("[WiFi] Attempting to connect to '%s'", config_ssid.c_str());
        WiFi.mode(WIFI_STA);
        WiFi.begin(config_ssid.c_str(), config_password.c_str());

        unsigned long start_attempt = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - start_attempt < 10000) {
            delay(500);
            Serial.print(".");
        }

        if (WiFi.status() == WL_CONNECTED) {
            connected = true;
            Serial.println("\n[WiFi] Connected successfully!");
            Serial.printf("[WiFi] IP Address: %s\n", WiFi.localIP().toString().c_str());
        } else {
            Serial.println("\n[WiFi] Could not connect to Wi-Fi within 10 seconds.");
        }
    } else {
        Serial.println("[WiFi] No saved Wi-Fi credentials found.");
    }

    // If Wi-Fi failed or not configured, launch captive portal setup hotspot
    if (!connected) {
        startConfigPortal();
    } else {
        // Run live dashboard on local Wi-Fi IP so user can manage/erase settings anytime
        setupPortalRoutes();
        portalServer.begin();

        if (MDNS.begin("guidebot")) {
            MDNS.addService("http", "tcp", 80);
            Serial.println("[Dashboard] MDNS responder started at http://guidebot.local/");
        }
        Serial.printf("[Dashboard] Live web control panel active at: http://%s/ (or http://guidebot.local/)\n", WiFi.localIP().toString().c_str());
    }

    // Once connected, initialize camera and start video stream
    if (!initCamera()) {
        Serial.println("[Error] Halting due to camera failure.");
        while(true) delay(1000);
    }

    setupStreaming();

    adaptive_window_ms   = millis();
    adaptive_frame_count = 0;
}

void loop() {
    // -------------------------------------------------------------
    // REMOTE COMMAND HANDLERS (Dispatched safely from main loop)
    // -------------------------------------------------------------
    if (flag_erase_wifi) {
        flag_erase_wifi = false;
        Serial.println("\n[Remote] Executing ERASE_WIFI command...");
        #if USE_WEBSOCKET
          webSocket.disconnect();
        #endif
        delay(100);
        esp_camera_deinit();
        performClearCredentials();
        Serial.println("[Remote] Launching Setup Hotspot immediately...");
        startConfigPortal();
        return;
    }

    if (flag_reboot) {
        flag_reboot = false;
        Serial.println("\n[Remote] Executing REBOOT command...");
        #if USE_WEBSOCKET
          webSocket.disconnect();
        #endif
        delay(100);
        esp_camera_deinit();
        delay(100);
        ESP.restart();
        return;
    }

    if (flag_set_wifi) {
        flag_set_wifi = false;
        Serial.println("\n[Remote] Executing SET_WIFI command...");
        String new_ssid = "";
        String new_pass = "";
        int ssidKeyIdx = pending_wifi_payload.indexOf("\"ssid\":");
        if (ssidKeyIdx != -1) {
            int startQuote = pending_wifi_payload.indexOf('\"', ssidKeyIdx + 7);
            int endQuote = pending_wifi_payload.indexOf('\"', startQuote + 1);
            if (startQuote != -1 && endQuote != -1) {
                new_ssid = pending_wifi_payload.substring(startQuote + 1, endQuote);
            }
        }
        int passKeyIdx = pending_wifi_payload.indexOf("\"password\":");
        if (passKeyIdx != -1) {
            int startQuote = pending_wifi_payload.indexOf('\"', passKeyIdx + 11);
            int endQuote = pending_wifi_payload.indexOf('\"', startQuote + 1);
            if (startQuote != -1 && endQuote != -1) {
                new_pass = pending_wifi_payload.substring(startQuote + 1, endQuote);
            }
        }
        if (new_ssid.length() > 0) {
            // If reconnecting to previously saved network with empty password, use previous password
            if (new_ssid == config_ssid && new_pass.length() == 0) {
                new_pass = config_password;
            }
            Serial.printf("[Remote] Saving new Wi-Fi '%s' and restarting...\n", new_ssid.c_str());
            saveConfiguration(new_ssid, new_pass, config_server_host, config_auth_token);
            #if USE_WEBSOCKET
              webSocket.disconnect();
            #endif
            delay(100);
            esp_camera_deinit();
            delay(100);
            ESP.restart();
            return;
        } else {
            Serial.println("[Remote] ERROR: Empty SSID received in SET_WIFI");
        }
    }

    if (flag_scan_wifi) {
        flag_scan_wifi = false;
        Serial.println("\n[Remote] Executing SCAN_WIFI request...");
        int n = WiFi.scanNetworks();
        Serial.printf("[Remote] Scan found %d networks\n", n);
        String jsonResp = "SCAN_RESULTS:[";
        for (int i = 0; i < n; i++) {
            if (i > 0) jsonResp += ",";
            String ssid = WiFi.SSID(i);
            ssid.replace("\"", "\\\"");
            bool isOpen = (WiFi.encryptionType(i) == WIFI_AUTH_OPEN);
            bool isSaved = (WiFi.SSID(i) == config_ssid);
            jsonResp += "{\"ssid\":\"" + ssid + "\",\"rssi\":" + String(WiFi.RSSI(i)) + ",\"open\":" + (isOpen ? "true" : "false") + ",\"saved\":" + (isSaved ? "true" : "false") + "}";
        }
        jsonResp += "]";
        #if USE_WEBSOCKET
          webSocket.sendTXT(jsonResp);
        #endif
        WiFi.scanDelete();
    }

    // Keep web dashboard responsive at all times
    portalServer.handleClient();

    // Check for Serial commands ("erase", "reset", "portal")
    if (Serial.available()) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd.equalsIgnoreCase("portal") || cmd.equalsIgnoreCase("setup") || cmd.equalsIgnoreCase("config")) {
            Serial.println("\n[Command] Opening Setup Hotspot upon user request...");
            startConfigPortal();
        } else if (cmd.equalsIgnoreCase("reset") || cmd.equalsIgnoreCase("erase")) {
            Serial.println("\n[Command] Clearing saved credentials and rebooting into Hotspot...");
            performClearCredentials();
            delay(500);
            ESP.restart();
        }
    }

    // Check if Wi-Fi dropped
    if (WiFi.status() != WL_CONNECTED) {
        if (wifi_disconnect_start == 0) {
            wifi_disconnect_start = millis();
            Serial.println("[WiFi] Lost connection. Attempting to reconnect...");
            WiFi.reconnect();
        }

        // Still drain camera while waiting for reconnect (prevents FB-OVF)
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);

        // If disconnected for more than 25 seconds, launch setup portal
        if (millis() - wifi_disconnect_start > 25000) {
            Serial.println("[WiFi] Reconnection timed out. Opening Setup Hotspot...");
            startConfigPortal();
        }
        delay(2);
        return;
    } else {
        wifi_disconnect_start = 0;
    }

    unsigned long now = millis();

    #if USE_WEBSOCKET
      webSocket.loop();

      static unsigned long last_ws_log = 0;
      if (!is_connected && (now - last_ws_log > 5000)) {
          last_ws_log = now;
          Serial.printf("[WS] Connecting to %s://%s:%d/ws/stream...\n",
                        (config_server_port == 443 ? "wss" : "ws"),
                        config_server_host.c_str(), config_server_port);
      }
    #endif

    // Always return a captured frame promptly; frame buffers are owned by the
    // camera driver until esp_camera_fb_return() is called.
    bool should_send = (now - last_frame_time >= FRAME_INTERVAL_MS);

    #if USE_WEBSOCKET
      if (should_send && is_connected) {
          // Do not reset sensor timing while a frame buffer is checked out.
          checkAdaptiveResolution();
      }
    #endif

    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) {
        if (should_send) {
            last_frame_time = now;
            #if USE_WEBSOCKET
              if (is_connected) {
                  sendFrame(fb);
              }
            #else
              sendFrame(fb);
            #endif
        }
        esp_camera_fb_return(fb);
    }

    delay(2);
}
