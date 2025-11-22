

#include "Arduino.h"
#include "esp_timer.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "NeuralNetwork.h"

#define INPUT_W 160
#define INPUT_H 160
#define LED_BUILT_IN 21

#define DEBUG_TFLITE 1
#define USE_CAMERA 0

#if USE_CAMERA==1
#include "esp_camera.h"
#include "img_converters.h"
#include "fb_gfx.h"
#endif

#if DEBUG_TFLITE==1
#include "img.h"
#endif

const char* gesture_labels[] = {"Thumbs Down", "Fist", "Open Hand", "Thumbs Up"};
const int NUM_GESTURES = 4;

NeuralNetwork *g_nn;

#if USE_CAMERA==1
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

int GetImage(camera_fb_t * fb, TfLiteTensor* input)
{
  assert(fb->format == PIXFORMAT_RGB565);

  int post = 0;
  int startx = (fb->width - INPUT_W) / 2;
  int starty = (fb->height - INPUT_H);
  for (int y = 0; y < INPUT_H; y++) {
    for (int x = 0; x < INPUT_W; x++) {
      int getPos = (starty + y) * fb->width + startx + x;
      uint16_t color = ((uint16_t *)fb->buf)[getPos];
      uint32_t rgb = rgb565torgb888(color);

      float *image_data = input->data.f;

      image_data[post * 3 + 0] = ((rgb >> 16) & 0xFF);
      image_data[post * 3 + 1] = ((rgb >> 8) & 0xFF);
      image_data[post * 3 + 2] = (rgb & 0xFF);
      post++;
    }
  }
  return 0;
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

#if USE_CAMERA==1
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
  config.frame_size = FRAMESIZE_QVGA;
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
#else
  pinMode(LED_BUILT_IN, OUTPUT);
  Serial.println("Running in DEBUG mode - Camera disabled");
#endif

  Serial.println("Initializing neural network...");
  Serial.println("About to call NeuralNetwork constructor...");
  g_nn = new NeuralNetwork();
  Serial.println("NeuralNetwork constructor returned");

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

}


void loop() {
  uint64_t start, dur_prep = 0, dur_infer;

#if USE_CAMERA==1
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;

  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    res = ESP_FAIL;
    return;
  }

  if(fb->format != PIXFORMAT_JPEG){
    start = esp_timer_get_time();
    GetImage(fb, g_nn->getInput());
    dur_prep = esp_timer_get_time() - start;
  }
#else
  float *input_data = g_nn->getInput()->data.f;
  for (int i = 0; i < sizeof(img_data); i++) {
    input_data[i] = (float)img_data[i];
  }
  Serial.printf("Loaded test image. First 3 pixels: %.0f %.0f %.0f\n",
    input_data[0], input_data[1], input_data[2]);
#endif

  start = esp_timer_get_time();
  g_nn->predict();
  dur_infer = esp_timer_get_time() - start;

#if USE_CAMERA==1
  Serial.printf("Preprocessing: %llu ms, Inference: %llu ms\n", dur_prep/1000, dur_infer/1000);
#else
  Serial.printf("Inference: %llu ms\n", dur_infer/1000);
  float fps_estimate = (dur_infer > 0) ? (1000000.0 / dur_infer) : 0.0;
  Serial.printf("Estimated max FPS: %.2f\n", fps_estimate);
#endif


  TfLiteTensor* output_tensor = g_nn->getOutput();

  Serial.printf("Output tensor type: %d, dims: %d, elements: %d\n",
                output_tensor->type,
                output_tensor->dims->size,
                output_tensor->dims->data[0]);

  float* output = output_tensor->data.f;


  Serial.print("Raw output values: [");
  for (int i = 0; i < NUM_GESTURES; i++) {
    Serial.printf("%.6f", output[i]);
    if (i < NUM_GESTURES - 1) Serial.print(", ");
  }
  Serial.println("]");


  int max_idx = 0;
  float max_prob = output[0];
  for (int i = 1; i < NUM_GESTURES; i++) {
    if (output[i] > max_prob) {
      max_prob = output[i];
      max_idx = i;
    }
  }


  Serial.print("Probabilities: ");
  for (int i = 0; i < NUM_GESTURES; i++) {
    Serial.printf("%s: %.3f  ", gesture_labels[i], output[i]);
  }
  Serial.println();


  Serial.printf("Detected: %s (%.3f)\n", gesture_labels[max_idx], max_prob);


  if (max_prob > 0.6) {
    digitalWrite(LED_BUILT_IN, LOW);
  } else {
    digitalWrite(LED_BUILT_IN, HIGH);
  }

#if USE_CAMERA==1
  esp_camera_fb_return(fb);
  fb = NULL;
#else
  delay(3000);
#endif
}