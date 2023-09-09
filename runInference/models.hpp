#pragma once

#include <yanic.hpp>

typedef struct OrtApi OrtApi;
typedef struct OrtEnv OrtEnv;
typedef struct OrtSessionOptions OrtSessionOptions;
typedef struct OrtSession OrtSession;
typedef struct OrtMemoryInfo OrtMemoryInfo;
typedef struct OrtValue OrtValue;

struct ONNXContext {
  static const OrtApi *gctx;
  OrtEnv *env;
  OrtSession *sess;
  OrtSessionOptions *sopt;
  OrtMemoryInfo *minf;
  vector<OrtValue *> in;
  vector<OrtValue *> out;
};

class ONNXModel final : public NNModel {
public:
  ONNXModel(const NNModelConfig *);
  ~ONNXModel() override;
  void RunInfer(const vector<Tensor *> &, const vector<Tensor *> &) override;

private:
  ONNXContext _octx;
};
