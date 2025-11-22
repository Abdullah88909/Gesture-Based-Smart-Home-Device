import tensorflow as tf
import numpy as np


interpreter = tf.lite.Interpreter(model_path="gesture_4class_mnet.tflite")
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== Model Input Details ===")
for i, detail in enumerate(input_details):
    print(f"Input {i}:")
    print(f"  Name: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Type: {detail['dtype']}")
    print(f"  Quantization: scale={detail['quantization'][0]}, zero_point={detail['quantization'][1]}")

print("\n=== Model Output Details ===")
for i, detail in enumerate(output_details):
    print(f"Output {i}:")
    print(f"  Name: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Type: {detail['dtype']}")
    print(f"  Quantization: scale={detail['quantization'][0]}, zero_point={detail['quantization'][1]}")
