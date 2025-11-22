#include "NeuralNetwork.h"
#include "model_data.h"

#include <esp_attr.h>
#include <Arduino.h>

int kTensorArenaSize = 3 * 1024 * 1024;

NeuralNetwork::NeuralNetwork()
{
    printf("\n=== NeuralNetwork Constructor Started ===\n");
    printf("Free heap: %d\n", ESP.getFreeHeap());
    printf("largest size (8bit): %d\n", heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    printf("largest size (default): %d\n", heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT));
    printf("largest size (spiram): %d\n", heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
    printf("largest size (internal): %d\n", heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

    printf("Model address: %p\n", gmodel);
    printf("Model size: %u bytes\n", gmodel_len);

    model = tflite::GetModel(gmodel);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        MicroPrintf("Model provided is schema version %d not equal to supported "
                    "version %d.",
                    model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    printf("Model schema version: %d (expected %d)\n", model->version(), TFLITE_SCHEMA_VERSION);
#if USE_ALLOPS==1
    static tflite::AllOpsResolver micro_op_resolver;
#else
    static tflite::MicroMutableOpResolver<16> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddLogistic();
    micro_op_resolver.AddTanh();
    micro_op_resolver.AddRelu6();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSub();
    micro_op_resolver.AddSoftmax();
#endif
    tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (tensor_arena == NULL)
    {
        printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
        return;
    }
    printf("Free heap: %d\n", ESP.getFreeHeap());

    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    MicroPrintf("interpreter initialization");
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        MicroPrintf("AllocateTensors() failed");
        return;
    }

    size_t used_bytes = interpreter->arena_used_bytes();
    MicroPrintf("Used bytes %d\n", used_bytes);
    printf("Tensor arena utilization: %d / %d bytes (%.1f%%)\n",
           used_bytes, kTensorArenaSize, (used_bytes * 100.0) / kTensorArenaSize);

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (input == nullptr) {
        printf("ERROR: Failed to get input tensor!\n");
        return;
    }
    if (output == nullptr) {
        printf("ERROR: Failed to get output tensor!\n");
        return;
    }

    printf("tensor_arena: %p, input: %p\n", tensor_arena, input->data.uint8);
    printf("input->dims->size: %d\n", input->dims->size);
    printf("input->dims->data[0]: %d\n", input->dims->data[0]);
    printf("input->dims->data[1]: %d\n", input->dims->data[1]);
    printf("input->dims->data[2]: %d\n", input->dims->data[2]);
    printf("input->dims->data[3]: %d\n", input->dims->data[3]);
    printf("input->type: %d\n", input->type);
    printf("input->params.scale: %.3f\n", input->params.scale);
    printf("input->params.zero_point: %d\n", input->params.zero_point);

    printf("output tensor: %p\n", output->data.f);
    printf("output->dims->size: %d\n", output->dims->size);
    for (int i = 0; i < output->dims->size; i++) {
        printf("output->dims->data[%d]: %d\n", i, output->dims->data[i]);
    }
    printf("output->type: %d\n", output->type);
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;
    printf("output scale=%.3f, zero_point=%d\n", scale, zero_point);

    printf("=== NeuralNetwork Constructor Completed Successfully ===\n\n");
} 

TfLiteTensor* NeuralNetwork::getInput()
{
    return input;
}

TfLiteStatus NeuralNetwork::predict()
{
    return interpreter->Invoke();
}

TfLiteTensor* NeuralNetwork::getOutput()
{
    return output;
}