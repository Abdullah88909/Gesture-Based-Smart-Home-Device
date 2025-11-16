# Gesture-Based Smart Home Device

This project uses **hand gestures** and **embedded machine learning** to control a smart home light. Built using an **ESP32-S3 microcontroller**, **TensorFlow Lite**, and a custom trained model.

---

## How It Works

1. **Gesture Recognition**  
   A simple CNN model is trained to recognize 4 hand gestures:
   - 🖐 `open` → Turn light **ON**  
   - ✊ `fist` → Turn light **OFF**  
   - 👍 `up` → Increase brightness  
   - 👎 `down` → Decrease brightness  

2. **ESP32 Inference**  
   The trained `.tflite` model runs directly on the ESP32. Based on the predicted gesture, Wi-Fi commands are sent to a smart bulb (e.g., Amazon Smart Bulb).

3. **Smart Bulb Control**  
   ESP32 triggers REST/MQTT commands over Wi-Fi to toggle power or brightness.

---



