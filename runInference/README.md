​	以在`C++`中调用`.onnx`模型为例，完成一次模型的推理，并核对精度问题

​	使用图片为`kodim02.png`；对于`C++`而言输入输出保持为`.bin`格式

1. Code
   - `test.cpp`
2. 文档

   1. `run_inference.md`
   2. `check_accuracy.md`
3. 运行

   ```shell
   # run inference
   mkdir build
   cd build
   cmake ..
   make
   ./test ${onnx_model_path} ${input_path} ${output_file}
   
   # check accuracy
   python pt2bin.py --input ${pt_path} --output ${onnx_path}
   python check_acc_bin.py --pt_file ${pt_path} --bin_file ${bin_path}
   ```

   

