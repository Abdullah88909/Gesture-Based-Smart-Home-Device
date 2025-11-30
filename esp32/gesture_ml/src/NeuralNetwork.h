#ifndef __NeuralNetwork__
#define __NeuralNetwork__



#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite
{
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class Model;
    class MicroInterpreter;
}


class NeuralNetwork
{
private:
#if USE_ALLOPS==1
    tflite::AllOpsResolver *resolver;
#else
    tflite::MicroMutableOpResolver<16> *resolver;
#endif
    const tflite::Model *model;
    tflite::MicroInterpreter *interpreter;
    TfLiteTensor *input;
    TfLiteTensor *output;
    uint8_t *tensor_arena;

public:
    NeuralNetwork();
    TfLiteTensor* getInput();
    TfLiteStatus predict();
    TfLiteTensor* getOutput();
};


#endif 