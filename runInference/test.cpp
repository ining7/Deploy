#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <png.h>

#include "yanic.hpp"

using namespace std;

template <typename Iter>
void write_output_to_bin(const std::string& file_path, Iter begin, Iter end) {
  std::ofstream output_file(file_path, std::ios::out | std::ios::binary);
  if (!output_file.is_open()) {
      std::cerr << "Error: Cannot open the output file." << std::endl;
      return;
  }
  for (Iter it = begin; it != end; ++it) {
      float value = static_cast<float>(*it);
      output_file.write(reinterpret_cast<char*>(&value), sizeof(float));
  }
  output_file.close();
}

bool read_bin_to_float_vector(const std::string& file_path, std::vector<float>& input_data) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << file_path << std::endl;
        return false;
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    input_data.resize(size / sizeof(float));
    file.read(reinterpret_cast<char*>(input_data.data()), size);

    file.close();
    return true;
}

template <std::size_t N>
bool read_bin_to_float_array(const std::string& file_path, std::array<float, N>& arr) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << file_path << std::endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(arr.data()), N * sizeof(float));
    if (!file) {
        std::cerr << "Warning: Not enough data to fill the array." << std::endl;
        return false;
    }
    file.close();
    return true;
}

int main(int argc, char *argv[]) {
  const std::string model_dir = argv[1];
  const char *input_file = argv[2];
  const std::string output_txt_path = argv[3];

  const std::string g_a_model_path = model_dir + "/g_a.onnx";
  const std::string h_a_model_path = model_dir + "/h_a.onnx";
  const std::string h_s_model_path = model_dir + "/h_s.onnx";
  const std::string g_s_model_path = model_dir + "/g_s.onnx";

  // input data 
  std::vector<float> input_data;
  read_bin_to_float_vector(input_file, input_data);

//   g_a
  NNModelConfig mcfg_g_a = {ONNX_CPU, g_a_model_path};
  NNModel* runner_g_a = NNModel::MakeNNModel(&mcfg_g_a);
  Tensor input_g_a = {FLOAT32, CPU, {1, 3, 512, 768}, "x_input", input_data.data()};
  std::array<float, 1 * 192 * 32 * 48> output_data_g_a;
  read_bin_to_float_array(input_file, output_data_g_a);
  Tensor output_g_a = {FLOAT32, CPU, {1, 192, 32, 48}, "y_output", output_data_g_a.data()};
  runner_g_a->RunInfer({&input_g_a}, {&output_g_a});
  NNModel::DelNNModel(runner_g_a);
write_output_to_bin(output_txt_path, output_data_g_a.begin(), output_data_g_a.end());
return 0;

  // h_a
  NNModelConfig mcfg_h_a = {ONNX_CPU, h_a_model_path};
  NNModel* runner_h_a = NNModel::MakeNNModel(&mcfg_h_a);
  Tensor input_h_a = {FLOAT32, CPU, {1, 192, 32, 48}, "y_input", output_data_g_a.data()};
  std::array<float, 1 * 128 * 8 * 12> output_data_h_a;
  read_bin_to_float_array(input_file, output_data_h_a);
  Tensor output_h_a = {FLOAT32, CPU, {1, 128, 8, 12}, "z_hat_output", output_data_h_a.data()};
  runner_h_a->RunInfer({&input_h_a}, {&output_h_a});
  NNModel::DelNNModel(runner_h_a);
write_output_to_bin(output_txt_path, output_data_h_a.begin(), output_data_h_a.end());
return 0;

  // h_s
  NNModelConfig mcfg_h_s = {ONNX_CPU, h_s_model_path};
  NNModel* runner_h_s = NNModel::MakeNNModel(&mcfg_h_s);
  Tensor input_h_s = {FLOAT32, CPU, {1, 128, 8, 12}, "z_hat_input", output_data_h_a.data()};
  std::array<float, 1 * 192 * 32 * 48> output_data_h_s;
  read_bin_to_float_array(input_file, output_data_h_s);
  Tensor output_h_s = {FLOAT32, CPU, {1, 192, 32, 48}, "scales_hat_output", output_data_h_s.data()};
  runner_h_s->RunInfer({&input_h_s}, {&output_h_s});
  NNModel::DelNNModel(runner_h_s);
write_output_to_bin(output_txt_path, output_data_h_s.begin(), output_data_h_s.end());
return 0;

  // g_s
  NNModelConfig mcfg_g_s = {ONNX_CPU, g_s_model_path};
  NNModel* runner_g_s = NNModel::MakeNNModel(&mcfg_g_s);
  Tensor input_g_s = {FLOAT32, CPU, {1, 192, 32, 48}, "y_hat_input", output_data_h_s.data()};
  std::array<float, 1 * 3 * 512 * 768> output_data_g_s;
  Tensor output_g_s = {FLOAT32, CPU, {1, 3, 512, 768}, "x_hat_output", output_data_g_s.data()};
  runner_g_s->RunInfer({&input_g_s}, {&output_g_s});
  NNModel::DelNNModel(runner_g_s);

  write_output_to_bin(output_txt_path, output_data_g_s.begin(), output_data_g_s.end());

  return 0;
}