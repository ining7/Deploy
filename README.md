1. `convert`
   - 从`pytorch`中转出`.onnx`模型
   - 通过`.pt`格式进行精度对比
2. `runInference`
   - 在`C++`中通过`OnnxRuntime`调用`.onnx`模型进行推理
   - 通过`.bin`格式进行精度对比