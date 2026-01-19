// Minimal SDXL VAE decoder run using CLML.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "clml_decoder.h"
#include "nn_framework/clml_utils_nn.h"
#include "utils/clml_fp_conv_utils.h"
#include "utils/clml_utils.h"

static std::vector<cl_half> ConvertFP32VectorToFP16(const std::vector<cl_float>& src) {
  std::vector<cl_half> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = fp32ToFP16(src[i]);
  }
  return dst;
}

static std::vector<cl_float> MakeRandomFp32(size_t count, float scale, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, scale);
  std::vector<cl_float> fp32(count);
  for (size_t i = 0; i < count; ++i) {
    fp32[i] = dist(rng);
  }
  return fp32;
}

static void WriteFp16(const std::string& path, const std::vector<cl_half>& data) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(cl_half)));
}

static void WriteFp32(const std::string& path, const std::vector<cl_float>& data) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(cl_float)));
}

static std::vector<cl_float> ReadFp32(const std::string& path, size_t expected_count) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "[Error] Failed to open input file: " << path << "\n";
    return {};
  }
  in.seekg(0, std::ios::end);
  const std::streamsize bytes = in.tellg();
  in.seekg(0, std::ios::beg);
  const size_t expected_bytes = expected_count * sizeof(cl_float);
  if (static_cast<size_t>(bytes) != expected_bytes) {
    std::cerr << "[Error] File size mismatch: " << path << " bytes=" << bytes
              << " expected=" << expected_bytes << "\n";
    return {};
  }
  std::vector<cl_float> data(expected_count);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expected_bytes));
  return data;
}

static void PrintStats(const std::vector<cl_float>& data) {
  if (data.empty()) {
    std::cout << "[Warn] Empty output\n";
    return;
  }
  size_t nan_count = 0;
  size_t inf_count = 0;
  cl_float min_v = data[0];
  cl_float max_v = data[0];
  long double sum = 0.0;
  for (cl_float v : data) {
    if (std::isnan(v)) {
      nan_count++;
      continue;
    }
    if (std::isinf(v)) {
      inf_count++;
      continue;
    }
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
    sum += v;
  }
  const long double denom = static_cast<long double>(data.size() - nan_count - inf_count);
  const long double mean = denom > 0 ? (sum / denom) : 0.0;
  std::cout << "[Out] min=" << min_v << " max=" << max_v << " mean=" << mean
            << " nan=" << nan_count << " inf=" << inf_count << "\n";
}

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  std::string base_dir = "sdxl_clml/";
  int iters = 1;
  int warmup = 0;
  bool dump_io = false;
  float input_scale = 0.1f;
  int latent_h = 64;
  int latent_w = 64;
  bool optimize_mem = true;
  bool latent_scaled = false;
  std::string latent_path;
  if (argc > 1) {
    base_dir = argv[1];
    if (!base_dir.empty() && base_dir.back() != '/') {
      base_dir += "/";
    }
  }
  if (argc > 2) {
    iters = std::max(1, std::atoi(argv[2]));
  }
  if (argc > 3) {
    warmup = std::max(0, std::atoi(argv[3]));
  }
  if (argc > 4) {
    dump_io = std::atoi(argv[4]) != 0;
  }
  if (argc > 5) {
    input_scale = std::max(0.0f, static_cast<float>(std::atof(argv[5])));
  }
  if (argc > 6) {
    latent_h = std::max(1, std::atoi(argv[6]));
  }
  if (argc > 7) {
    latent_w = std::max(1, std::atoi(argv[7]));
  }
  if (argc > 8) {
    optimize_mem = std::atoi(argv[8]) != 0;
  }
  if (argc > 9) {
    latent_path = argv[9];
  }

  const char* latent_env = std::getenv("SDXL_VAE_LATENT");
  if (latent_path.empty() && latent_env != nullptr && std::strlen(latent_env) > 0) {
    latent_path = latent_env;
  }
  const char* latent_scaled_env = std::getenv("SDXL_VAE_LATENT_SCALED");
  if (latent_scaled_env != nullptr && std::atoi(latent_scaled_env) != 0) {
    latent_scaled = true;
  }

  NNModelDesc model_desc = {
      NNModelMode::INFERENCE,
      CL_HALF_FLOAT,
      CL_TENSOR_LAYOUT_NCHW_QCOM,
      true,
      base_dir,
      false,   // enable_model_tuning
      "",      // tuning_cache_file_name
      false,   // enable_layer_dumping
      optimize_mem,    // optimize_device_mem
      false,   // enable_model_profiling
      false,   // apply_zero_copy
      false,   // use_recordable_queue
      true,    // enable_winograd_conv
      false    // enable_gmem_buffers
  };

  std::cout << "[Info] create CL env\n";
  CLEnvironment cl_env = createCLEnvironment(false);
  std::cout << "[Info] CL env ready\n";
  const cl_int minor_version = 0;
  CLMLInterfaceV4QCOM* clml_intf = clGetMLInterfaceV4QCOM(minor_version);
  if (clml_intf == nullptr) {
    std::cerr << "[Error] clGetMLInterfaceV4QCOM failed\n";
    return 1;
  }
  std::cout << "[Info] CLML interface ok\n";

  std::cout << "[Config] latent_h=" << latent_h << " latent_w=" << latent_w
            << " optimize_device_mem=" << (optimize_mem ? 1 : 0)
            << " latent_scaled=" << (latent_scaled ? 1 : 0)
            << " latent_path=" << (latent_path.empty() ? "none" : latent_path)
            << "\n";

  tensor_dims_t latent_dims = {1, 4,
                               static_cast<cl_uint>(latent_h),
                               static_cast<cl_uint>(latent_w)};
  std::cout << "[Info] create latent tensor\n";
  MLTensor latent = createMLTensor(
      cl_env, clml_intf, latent_dims, model_desc.model_dtype, CL_TENSOR_LAYOUT_NCHW_QCOM,
      CL_TENSOR_USAGE_CNN_QCOM);
  std::cout << "[Info] create decoder\n";

  Decoder decoder(model_desc, cl_env);
  std::vector<MLTensor> outputs = decoder.create({latent});
  if (outputs.size() != 1) {
    std::cerr << "[Error] Decoder output size invalid: " << outputs.size() << "\n";
    return 1;
  }

  std::cout << "[Info] decoder initParams\n";
  decoder.initParams();
  std::cout << "[Info] decoder initParams done\n";

  const size_t latent_count = static_cast<size_t>(latent_dims.n) * latent_dims.c * latent_dims.h * latent_dims.w;
  std::vector<cl_float> latent_fp32;
  if (!latent_path.empty()) {
    latent_fp32 = ReadFp32(latent_path, latent_count);
    if (latent_fp32.empty()) {
      return 1;
    }
  } else {
    latent_fp32 = MakeRandomFp32(latent_count, input_scale, 0);
    latent_scaled = false;
  }
  if (!latent_scaled) {
    const float scale = 1.0f / 0.13025f;
    for (auto& v : latent_fp32) {
      v *= scale;
    }
  }
  std::vector<cl_half> latent_fp16 = ConvertFP32VectorToFP16(latent_fp32);

  if (dump_io) {
    createDirectory("output");
    WriteFp16("output/sdxl_vae_latent.qfp16", latent_fp16);
  }

  std::cout << "[Info] upload latent\n";
  latent.uploadDataIntoGPUMem(latent_fp16.data(), model_desc.tensor_data_src_layout);

  std::cout << "[Info] decoder warmup\n";
  for (int i = 0; i < warmup; ++i) {
    decoder.forward();
    clFinish(cl_env.queue);
  }

  std::cout << "[Info] decoder forward\n";
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    decoder.forward();
    clFinish(cl_env.queue);
  }
  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = t1 - t0;
  double sec_per_iter = elapsed.count() / static_cast<double>(iters);
  std::cout << "[Perf] iters=" << iters << " warmup=" << warmup << " s/it=" << sec_per_iter << "\n";

  std::vector<cl_float> output = outputs[0].downloadDataFromGPUMem();
  if (dump_io) {
    createDirectory("output");
    WriteFp32("output/sdxl_vae_out.qfp32", output);
  }
  PrintStats(output);

  releaseCLEnvironment(cl_env);
  return 0;
}
