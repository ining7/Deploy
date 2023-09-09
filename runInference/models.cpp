#include <cassert>
#include <iostream>
#include <fstream>

#include "onnxruntime_c_api.h"
#include "provider_options.h"

#include "models.hpp"
#include "yanic.hpp"

const OrtApi *ONNXContext::gctx = OrtGetApiBase()->GetApi(ORT_API_VERSION);

#define ORT_ABORT_ON_ERROR(expr)                                               \
  do {                                                                         \
    OrtStatus *onnx_status = (expr);                                           \
    if (onnx_status != NULL) {                                                 \
      const char *msg = ONNXContext::gctx->GetErrorMessage(onnx_status);       \
      std::cout << msg << std::endl;                                           \
      ONNXContext::gctx->ReleaseStatus(onnx_status);                           \
      abort();                                                                 \
    }                                                                          \
  } while (0);

void ConstONNXTensor(ONNXContext &octx, const vector<Tensor *> &tensor,
                     vector<OrtValue *> &onnx_tensor,
                     vector<const char *> &onnx_name) {
  for (int i = 0; i < tensor.size(); i++) {
    vector<int64_t> shape;
    int64_t len = sizeof(float);
    for (auto sh : tensor[i]->shape) {
      shape.push_back(sh);
      len *= sh;
    }
    ORT_ABORT_ON_ERROR(octx.gctx->CreateTensorWithDataAsOrtValue(
        octx.minf, tensor[i]->data, len, shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &onnx_tensor[i]));
    onnx_name[i] = tensor[i]->name.c_str();
  }
}

void DeConstONNXTensor(ONNXContext &octx, vector<OrtValue *> &onnx_tensor) {
  for (int i = 0; i < onnx_tensor.size(); i++) {
    octx.gctx->ReleaseValue(onnx_tensor[i]);
  }
}

ONNXModel::ONNXModel(const NNModelConfig *cfg) : NNModel(cfg) {
  assert(_cfg->etype == ONNX_CPU);
  ORT_ABORT_ON_ERROR(_octx.gctx->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                           "yanic_onnx_env", &_octx.env));
  ORT_ABORT_ON_ERROR(_octx.gctx->CreateSessionOptions(&_octx.sopt));
  ORT_ABORT_ON_ERROR(_octx.gctx->CreateSession(_octx.env, _cfg->mpath.c_str(),
                                               _octx.sopt, &_octx.sess));
  ORT_ABORT_ON_ERROR(_octx.gctx->CreateCpuMemoryInfo(
      OrtArenaAllocator, OrtMemTypeDefault, &_octx.minf));
}

ONNXModel::~ONNXModel() {
  _octx.gctx->ReleaseMemoryInfo(_octx.minf);
  _octx.gctx->ReleaseSession(_octx.sess);
  _octx.gctx->ReleaseSessionOptions(_octx.sopt);
  _octx.gctx->ReleaseEnv(_octx.env);
}

void ONNXModel::RunInfer(const vector<Tensor *> &in,
                         const vector<Tensor *> &out) {
  vector<OrtValue *> in_tensor(in.size()), out_tensor(out.size());
  vector<const char *> in_name(in.size()), out_name(out.size());
  ConstONNXTensor(_octx, in, in_tensor, in_name);
  ConstONNXTensor(_octx, out, out_tensor, out_name);
  ORT_ABORT_ON_ERROR(_octx.gctx->Run(
      _octx.sess, NULL, in_name.data(), in_tensor.data(), in_tensor.size(),
      out_name.data(), out_tensor.size(), out_tensor.data()));
  DeConstONNXTensor(_octx, in_tensor);
  DeConstONNXTensor(_octx, out_tensor);
}
