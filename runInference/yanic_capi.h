typedef struct Tensor Tensor;
typedef struct BitStream BitStream;
typedef struct NNModelConfig NNModelConfig;
typedef struct EncoderConfig EncoderConfig;
typedef struct DecoderConfig DecoderConfig;

Tensor *MakeTensor(int, int, const vector<int> &, const string &, void *);
void DelTensor(Tensor *);

NNModelConfig *MakeNNModelConfig(int, const string &);
void DelNNModelConfig(NNModelConfig *);
