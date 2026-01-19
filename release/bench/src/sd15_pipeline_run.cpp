// SD1.5 end-to-end pipeline run using CLML (text encoder + UNet + VAE decoder).
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "clml_stable_diffusion_pl.h"
#include "utils/clml_utils.h"

int main(int argc, char** argv) {
  std::string base_dir = "sd15_clml/sd15_clml_weights/";
  int num_steps = 20;
  if (argc > 1) {
    base_dir = argv[1];
    if (!base_dir.empty() && base_dir.back() != '/') {
      base_dir += "/";
    }
  }
  if (argc > 2) {
    num_steps = std::max(1, std::atoi(argv[2]));
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
      true,    // optimize_device_mem
      false,   // enable_model_profiling
      false,   // apply_zero_copy
      false,   // use_recordable_queue
      true,    // enable_winograd_conv
      false    // enable_gmem_buffers
  };

  CLEnvironment cl_env = createCLEnvironment(false);
  StableDiffusionPL sd_pipeline(model_desc, cl_env);
  sd_pipeline.create();

  std::string cond_tokens_path = base_dir + "cond_text_tokens.qint32";
  std::vector<cl_int> cond_tokens = readINT32FromFile(cond_tokens_path);
  if (cond_tokens.empty()) {
    std::cerr << "[Error] cond_text_tokens.qint32 not found or empty: " << cond_tokens_path << "\n";
    return 1;
  }

  auto t0 = std::chrono::steady_clock::now();
  sd_pipeline.inference(cond_tokens.data(), static_cast<cl_uint>(num_steps));
  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = t1 - t0;
  double total_s = elapsed.count();
  double s_per_step = total_s / static_cast<double>(num_steps);
  std::cout << "[Perf] steps=" << num_steps << " total_s=" << total_s
            << " s/step=" << s_per_step << "\n";

  releaseCLEnvironment(cl_env);
  return 0;
}
