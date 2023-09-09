#pragma once

#include <string>
#include <vector>

using namespace std;

typedef enum {
  INT8,
  INT32,
  FLOAT16,
  FLOAT32,
} TensorDtype;

typedef enum {
  CPU,
} TensorDevice;

typedef enum {
  ONNX_CPU,
} EngineType;

typedef enum {
  BALLE2018,
} CodecType;

struct Tensor {
  TensorDtype dtype;
  TensorDevice device;
  vector<int> shape;
  string name;
  void *data;
  // vector<int> stride; // not sure whether we need this field ...
};

struct BitStream {
  int bits;
  void *data;
};

struct NNModelConfig {
  EngineType etype;
  string mpath;
};

struct EncoderConfig {
  CodecType ctype;
};

struct DecoderConfig {
  CodecType ctype;
};

class Node {
public:
  Node() = default;
  virtual ~Node() = default;
  Node(const Node &) = delete;
  Node(Node &&) = delete;
  Node &operator=(const Node &) = delete;
  Node &operator=(Node &&) = delete;
};

class NNModel : public Node {
public:
  static NNModel *MakeNNModel(const NNModelConfig *);
  static void DelNNModel(NNModel *);
  NNModel() = delete;
  NNModel(const NNModelConfig *);
  virtual ~NNModel();
  virtual void RunInfer(const vector<Tensor *> &, const vector<Tensor *> &) = 0;

protected:
  const NNModelConfig *_cfg;
};

class Encoder : public Node {
public:
  static Encoder *MakeEncoder(const EncoderConfig *);
  static void DelEncoder(Encoder *);
  Encoder() = delete;
  Encoder(const EncoderConfig *);
  virtual ~Encoder();
  virtual void Compress(const Tensor *, BitStream *);

protected:
  const EncoderConfig *_cfg;
};

class Decoder : public Node {
public:
  static Decoder *MakeDecoder(const DecoderConfig *);
  static void DelDecoder(Decoder *);
  Decoder() = delete;
  Decoder(const DecoderConfig *);
  virtual ~Decoder();
  virtual void DeCompress(const BitStream *, Tensor *);

protected:
  const DecoderConfig *_cfg;
};
