#include <cassert>
#include <iostream>

#include "models.hpp"
#include "yanic.hpp"

NNModel *NNModel::MakeNNModel(const NNModelConfig *cfg) {
  NNModel *mod;
  if (cfg->etype == ONNX_CPU) {
    mod = new ONNXModel(cfg);
  } else {
    assert(0);
  }
  return mod;
}

void NNModel::DelNNModel(NNModel *mod) { delete mod; }

NNModel::NNModel(const NNModelConfig *cfg) { _cfg = cfg; }

NNModel::~NNModel() {}
