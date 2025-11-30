

#include "Arduino.h"
#include "esp_timer.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "NeuralNetwork.h"
#include <WiFi.h>
#include <WebServer.h>
#include "esp_timer.h"
#include "esp_camera.h"
#include <PubSubClient.h>

#define CAMERA_W 96
#define CAMERA_H 96
#define LED_BUILT_IN 21

#define DEBUG_TFLITE 1
#define USE_CAMERA 1
#define CAMERA_MODEL_XIAO_ESP32S3 1

#define RESIZE_INPUT 1

#if RESIZE_INPUT 
  #define INPUT_W 48
  #define INPUT_H 48
#else
  #define INPUT_W 96
  #define INPUT_H 96
#endif

#include "esp_camera.h"
#include "img_converters.h"
#include "fb_gfx.h"
#include "camera_pins.h"

const char* ssid = "";
const char* password = "";

#ifdef MQTT
WebServer server(80);
#endif
const char* mqtt_topic = "esp32/state";

const char* mqtt_server = "homeassistant.local";
const int mqtt_port = 1883;
const char* mqtt_user = "";
const char* mqtt_password = "";

#define FMODE

#ifndef FMODE
const char* gesture_labels[] = {"Thumbs Down", "Fist", "Thumbs Up", "None", "Palm"};
const int NUM_GESTURES = 5;
#endif

#ifdef FMODE
const char* gesture_labels[] = {"Thumbs Down", "Fist", "Thumbs Up", "Palm"};
const int NUM_GESTURES = 4;
#endif

NeuralNetwork *g_nn;
camera_fb_t *g_last_frame = NULL;
WiFiClient streamClient;
unsigned long fist_pause_until = 0;  

#ifndef MQTT
PubSubClient client(streamClient);
#endif

uint32_t rgb565torgb888(uint16_t color)
{
  uint8_t hb, lb;
  uint32_t r, g, b;

  lb = (color >> 8) & 0xFF;
  hb = color & 0xFF;

  r = (lb & 0x1F) << 3;
  g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
  b = (hb & 0xF8);

  return (r << 16) | (g << 8) | b;
}


#if RESIZE_INPUT
// Resize image using nearest neighbor downsampling
int GetImage(camera_fb_t * fb, TfLiteTensor* input)
{
  assert(fb->format == PIXFORMAT_RGB565);

  float *image_data = input->data.f;
  int post = 0;

  // Calculate scale factors
  float scale_x = (float)CAMERA_W / INPUT_W;
  float scale_y = (float)CAMERA_H / INPUT_H;

  // Calculate starting position for center crop
  int startx = (fb->width - CAMERA_W) / 2;
  int starty = (fb->height - CAMERA_H);

  for (int out_y = 0; out_y < INPUT_H; out_y++) {
    for (int out_x = 0; out_x < INPUT_W; out_x++) {
      // Nearest neighbor sampling
      int src_x = (int)(out_x * scale_x);
      int src_y = (int)(out_y * scale_y);

      int getPos = (starty + src_y) * fb->width + startx + src_x;
      uint16_t color = ((uint16_t *)fb->buf)[getPos];
      uint32_t rgb = rgb565torgb888(color);

      image_data[post * 3 + 0] = ((rgb >> 16) & 0xFF);
      image_data[post * 3 + 1] = ((rgb >> 8) & 0xFF);
      image_data[post * 3 + 2] = (rgb & 0xFF);
      post++;
    }
  }
  return 0;
}
#else
// No resize - direct crop from center
int GetImage(camera_fb_t * fb, TfLiteTensor* input)
{
  assert(fb->format == PIXFORMAT_RGB565);

  float *image_data = input->data.f;
  int post = 0;
  int startx = (fb->width - INPUT_W) / 2;
  int starty = (fb->height - INPUT_H);

  for (int y = 0; y < INPUT_H; y++) {
    for (int x = 0; x < INPUT_W; x++) {
      int getPos = (starty + y) * fb->width + startx + x;
      uint16_t color = ((uint16_t *)fb->buf)[getPos];
      uint32_t rgb = rgb565torgb888(color);

      image_data[post * 3 + 0] = ((rgb >> 16) & 0xFF);
      image_data[post * 3 + 1] = ((rgb >> 8) & 0xFF);
      image_data[post * 3 + 2] = (rgb & 0xFF);
      post++;
    }
  }
  return 0;
}
#endif

#ifdef MQTT
void handle_snapshot() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  uint8_t * jpg_buf = NULL;
  size_t jpg_len = 0;
  bool converted = frame2jpg(fb, 80, &jpg_buf, &jpg_len);

  if (!converted) {
    esp_camera_fb_return(fb);
    server.send(500, "text/plain", "JPEG conversion failed");
    return;
  }

  WiFiClient client = server.client();


  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: image/jpeg");
  client.print("Content-Length: ");
  client.println(jpg_len);
  client.println("Connection: close");
  client.println();


  client.write(jpg_buf, jpg_len);


  free(jpg_buf);
  esp_camera_fb_return(fb);
}


void handle_stream() {

  streamClient = server.client();

  streamClient.println("HTTP/1.1 200 OK");
  streamClient.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
  streamClient.println();

  Serial.println("Stream client connected");
}

void sendStreamFrame() {

  if (!streamClient || !streamClient.connected()) {
    if (streamClient) {
      streamClient.stop();
    }
    return;
  }

  if (g_last_frame == NULL) {
    return;
  }

  uint8_t * jpg_buf = NULL;
  size_t jpg_len = 0;
  bool converted = frame2jpg(g_last_frame, 80, &jpg_buf, &jpg_len);

  if (!converted) {
    return;
  }

  streamClient.println("--frame");
  streamClient.println("Content-Type: image/jpeg");
  streamClient.print("Content-Length: ");
  streamClient.println(jpg_len);
  streamClient.println();
  streamClient.write(jpg_buf, jpg_len);
  streamClient.println();


  free(jpg_buf);
}

#endif

#ifndef MQTT
void reconnectMQTT() {
  int mqttAttempts = 0;
  while (!client.connected()) {
    Serial.println();
    Serial.println("========================================");
    Serial.print("Attempting MQTT connection to: ");
    Serial.print(mqtt_server);
    Serial.print(":");
    Serial.println(mqtt_port);

    String clientId = "ESP32Client-";
    clientId += String(random(0xffff), HEX);
    Serial.print("Client ID: ");
    Serial.println(clientId);

    bool connected;
    if (strlen(mqtt_user) > 0) {
      Serial.print("Using authentication - Username: ");
      Serial.println(mqtt_user);
      connected = client.connect(clientId.c_str(), mqtt_user, mqtt_password);
    } else {
      Serial.println("No authentication");
      connected = client.connect(clientId.c_str());
    }

    if (connected) {
      Serial.println("MQTT CONNECTED!");
      Serial.println("========================================");
      Serial.println("Ready to publish messages.");
      Serial.print("Publish topic: ");
      Serial.println(mqtt_topic);
      Serial.println("Type a message and press Enter to publish");
      Serial.println("========================================");
    } else {
      Serial.print("MQTT Connection FAILED! Error code: ");
      Serial.println(client.state());
      Serial.println("Error meanings:");
      Serial.println("  -4: Connection timeout");
      Serial.println("  -3: Connection lost");
      Serial.println("  -2: Connection failed");
      Serial.println("  -1: Disconnected");
      Serial.println("   1: Bad protocol");
      Serial.println("   2: Bad client ID");
      Serial.println("   3: Server unavailable");
      Serial.println("   4: Bad credentials");
      Serial.println("   5: Unauthorized");

      mqttAttempts++;
      if (mqttAttempts > 10) {
        Serial.println("========================================");
        Serial.println("MQTT connection failed after 10 attempts!");
        Serial.println("Possible issues:");
        Serial.println("  - MQTT broker hostname/IP incorrect");
        Serial.println("  - MQTT broker not running");
        Serial.println("  - Wrong port number");
        Serial.println("  - Invalid credentials");
        Serial.println("  - Firewall blocking connection");
        Serial.println("Restarting in 10 seconds...");
        Serial.println("========================================");
        delay(10000);
        ESP.restart();
      }

      Serial.println("Retrying in 5 seconds...");
      Serial.println("========================================");
      delay(5000);
    }
  }
}
#endif
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);

  while(!Serial) {
    static int retries = 0;
    delay(2000);
    if (retries++ > 5) {
      break;
    }
  }
  delay(5000);
  Serial.setDebugOutput(false);

  // Initialize neural network BEFORE camera to ensure clean memory allocation
  Serial.println("Initializing neural network...");
  Serial.println("About to call NeuralNetwork constructor...");
  g_nn = new NeuralNetwork();
  Serial.println("NeuralNetwork constructor returned");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_XGA;  
  config.pixel_format = PIXFORMAT_RGB565;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  pinMode(LED_BUILT_IN, OUTPUT);

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  Serial.printf("Camera init success!\n");
  Serial.printf("frame_size=%d\n", config.frame_size);
  Serial.printf("pixel_format=%d\n", config.pixel_format);

  if (g_nn == nullptr) {
    Serial.println("ERROR: Failed to create NeuralNetwork!");
    while(1) delay(1000);
  }

  Serial.println("Checking if neural network is ready...");
  if (g_nn->getInput() == nullptr) {
    Serial.println("ERROR: Input tensor is null!");
    while(1) delay(1000);
  }
  if (g_nn->getOutput() == nullptr) {
    Serial.println("ERROR: Output tensor is null!");
    while(1) delay(1000);
  }

  Serial.println("Neural network initialized successfully!");

  // Initialize WiFi after NN to ensure NN gets clean memory allocation
  Serial.println("Connecting to Wi-Fi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int wifi_retries = 0;
  while (WiFi.status() != WL_CONNECTED && wifi_retries < 20) {
    delay(500);
    Serial.print(".");
    wifi_retries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWi-Fi connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWi-Fi connection failed, continuing without WiFi");
  }

  #ifdef MQTT
  server.on("/", HTTP_GET, []() {
    server.send(200, "text/html",
      "<!DOCTYPE html>"
      "<html><head>"
      "<title>ESP32 Gesture Recognition</title>"
      "<style>"
      "body { font-family: Arial, sans-serif; text-align: center; margin: 20px; }"
      "img { max-width: 800px; border: 2px solid #333; }"
      "h1 { color: #333; }"
      "</style>"
      "</head><body>"
      "<h1>ESP32 Camera Stream</h1>"
      "<img src=\"/stream\" />"
      "<p>Live video stream from ESP32 camera</p>"
      "<p><a href=\"/snapshot\">Click for single snapshot</a></p>"
      "</body></html>");
  });

  server.on("/snapshot", HTTP_GET, handle_snapshot);
  server.on("/stream", HTTP_GET, handle_stream);

   server.begin();
  Serial.println("HTTP server started");
  #endif

  #ifndef MQTT
  client.setServer(mqtt_server, mqtt_port);

  reconnectMQTT();
  #endif
 
}


void loop() {
  #ifndef MQTT
  if (!client.connected()) {
    Serial.println("\n[WARNING] MQTT disconnected! Reconnecting...");
    reconnectMQTT();
  }
  client.loop();
  #endif


  unsigned long current_time = millis();
  if (fist_pause_until > 0 && current_time < fist_pause_until) {

    #ifndef MQTT
    Serial.println("Fist pause active - sending OFF");
    client.publish(mqtt_topic, "OFF");
    #endif

    #ifdef MQTT
    server.handleClient();
    sendStreamFrame();
    #endif

    delay(100); 
    return;
  } else if (fist_pause_until > 0 && current_time >= fist_pause_until) {

    Serial.println("Fist pause ended - resuming normal operation");
    fist_pause_until = 0;
  }

  uint64_t start, dur_prep = 0, dur_infer;

  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;

  if (g_last_frame != NULL) {
    esp_camera_fb_return(g_last_frame);
    g_last_frame = NULL;
  }

  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    res = ESP_FAIL;
    return;
  }

  g_last_frame = fb;

  if(fb->format != PIXFORMAT_JPEG){
    start = esp_timer_get_time();
    GetImage(fb, g_nn->getInput());
    dur_prep = esp_timer_get_time() - start;
  }

  #ifdef MQTT
  server.handleClient();

  #endif
  start = esp_timer_get_time();
  TfLiteStatus invoke_status = g_nn->predict();
  dur_infer = esp_timer_get_time() - start;

  Serial.printf("Preprocessing: %llu ms, Inference: %llu ms\n", dur_prep/1000, dur_infer/1000);

  TfLiteTensor* output_tensor = g_nn->getOutput();
  Serial.println("Raw output scores:");
  for (int i = 0; i < NUM_GESTURES; i++) {
    Serial.printf("  %s: %.3f\n", gesture_labels[i], output_tensor->data.f[i]);
  }

  #ifndef MQTT
  if (output_tensor->data.f[0] > 0.65) {
    Serial.println("Gesture Detected: Thumbs Down");
    client.publish(mqtt_topic, "DIM");
  } else if (output_tensor->data.f[1] > 0.65) {
    Serial.println("Gesture Detected: Fist");
    client.publish(mqtt_topic, "OFF");
    fist_pause_until = millis() + 20000;
    Serial.println("Starting 2 second fist pause");
  } else if (output_tensor->data.f[2] >0.65) {
    Serial.println("Gesture Detected: Thumbs Up");
    client.publish(mqtt_topic, "BRIGHTEN");

  }
  #ifdef FMODE
  else if (output_tensor->data.f[3] > 0.65) {
    Serial.println("Gesture Detected: Palm");
    client.publish(mqtt_topic, "ON");
  } else {
    Serial.println("No significant gesture detected.");
  }
  #else
  else if (output_tensor->data.f[4] > 0.7) {
    Serial.println("Gesture Detected: Palm");
    client.publish(mqtt_topic, "ON");
  } else {
    Serial.println("No significant gesture detected.");
  }
  #endif
  #endif

  #ifdef MQTT
  sendStreamFrame();

  #endif
}