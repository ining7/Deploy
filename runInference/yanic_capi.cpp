#include "yanic.hpp"

Tensor *MakeTensor(TensorDtype dtype, TensorDevice device,
                   const vector<int> &shape, const string &name, void *data) {
  Tensor *tensor = new Tensor();
  tensor->dtype = dtype;
  tensor->device = device;
  tensor->shape = shape;
  tensor->name = name;
  tensor->data = data;
  return tensor;
}
void DelTensor(Tensor *tensor) { delete tensor; }

NNModelConfig *MakeNNModelConfig(EngineType etype, const string &mpath) {
  NNModelConfig *cfg = new NNModelConfig();
  cfg->etype = etype;
  cfg->mpath = mpath;
  return cfg;
}
void DelNNModelConfig(NNModelConfig *cfg) { delete cfg; }
