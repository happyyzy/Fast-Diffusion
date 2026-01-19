// SDXL pipeline run: MNN CLIP (CPU) + CLML UNet/VAE.
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "clml_decoder.h"
#include "clml_scheduler.h"
#include "clml_unet.h"
#include "nn_framework/clml_utils_nn.h"
#include "utils/clml_fp_conv_utils.h"
#include "utils/clml_utils.h"

namespace {

constexpr int kBatch = 2;
constexpr int kClipTokens = 77;
constexpr int kClipEmbedDim = 768;
constexpr int kClip2EmbedDim = 1280;
constexpr int kTextEmbedDim = kClipEmbedDim + kClip2EmbedDim;
constexpr int kPooledDim = 1280;
constexpr int kTimeEmbedDim = 320;
constexpr int kAddTimeEmbedDim = 256;
constexpr int kAddEmbedsDim = 2816;  // 1280 pooled + 6 * 256

enum class ClipInputType { kIds, kEmbedding, kUnknown };

const char* ClipInputTypeName(ClipInputType type) {
  switch (type) {
    case ClipInputType::kIds:
      return "input_ids";
    case ClipInputType::kEmbedding:
      return "input_embedding";
    default:
      return "unknown";
  }
}

ClipInputType DetectClipInputType(MNN::Interpreter* interp,
                                  MNN::Session* session,
                                  bool prefer_ids_if_both) {
  if (!interp || !session) return ClipInputType::kUnknown;
  const auto& inputs = interp->getSessionInputAll(session);
  if (!inputs.empty()) {
    std::string names;
    for (const auto& kv : inputs) {
      if (!names.empty()) names += ",";
      names += kv.first;
    }
    std::cout << "[CLIP] inputs: " << names << "\n";
  }
  bool has_ids = inputs.find("input_ids") != inputs.end();
  bool has_embedding = inputs.find("input_embedding") != inputs.end();
  if (has_embedding && has_ids) {
    return prefer_ids_if_both ? ClipInputType::kIds : ClipInputType::kEmbedding;
  }
  if (has_embedding) return ClipInputType::kEmbedding;
  if (has_ids) return ClipInputType::kIds;
  return ClipInputType::kUnknown;
}

struct ClipOutputs {
  std::vector<float> last_hidden;
  std::vector<float> pooled;
};

std::vector<int32_t> LoadIds(const std::string& path) {
  std::vector<int32_t> ids = readINT32FromFile(path);
  if (ids.size() != static_cast<size_t>(kBatch * kClipTokens)) {
    std::cerr << "[Error] Invalid token count: " << path << " size=" << ids.size()
              << " expect=" << (kBatch * kClipTokens) << "\n";
    return {};
  }
  return ids;
}

ClipOutputs RunMnnClipIds(const std::string& model_path,
                          const std::vector<int32_t>& ids,
                          int embed_dim,
                          int pooled_dim) {
  ClipOutputs out;
  if (ids.size() != static_cast<size_t>(kBatch * kClipTokens)) {
    std::cerr << "[Error] CLIP ids size mismatch\n";
    return out;
  }

  auto net = std::shared_ptr<MNN::Interpreter>(
      MNN::Interpreter::createFromFile(model_path.c_str()),
      MNN::Interpreter::destroy);
  if (!net) {
    std::cerr << "[Error] Failed to load MNN model: " << model_path << "\n";
    return out;
  }

  std::string weight_path = model_path + ".weight";
  if (std::filesystem::exists(weight_path)) {
    net->setExternalFile(weight_path.c_str());
    std::cout << "[CLIP] external weights: " << weight_path << "\n";
  }

  MNN::ScheduleConfig cfg;
  cfg.type = MNN_FORWARD_CPU;
  cfg.numThread = 4;
  MNN::BackendConfig backend;
  backend.memory = MNN::BackendConfig::Memory_Low;
  backend.power = MNN::BackendConfig::Power_High;
  cfg.backendConfig = &backend;

  auto session = net->createSession(cfg);
  if (!session) {
    std::cerr << "[Error] Failed to create MNN session: " << model_path << "\n";
    return out;
  }

  ClipInputType input_type = DetectClipInputType(net.get(), session, true);
  if (input_type != ClipInputType::kIds) {
    std::cerr << "[Error] CLIP expects " << ClipInputTypeName(input_type)
              << ", input_ids required\n";
    return out;
  }

  auto input = net->getSessionInput(session, "input_ids");
  if (!input) {
    std::cerr << "[Error] Missing input_ids tensor in CLIP\n";
    return out;
  }
  net->resizeTensor(input, {1, kClipTokens});
  net->resizeSession(session);
  net->releaseModel();

  out.last_hidden.resize(static_cast<size_t>(kBatch) * kClipTokens * embed_dim);
  if (pooled_dim > 0) {
    out.pooled.resize(static_cast<size_t>(kBatch) * pooled_dim);
  }

  for (int b = 0; b < kBatch; ++b) {
    MNN::Tensor input_host(input, MNN::Tensor::CAFFE);
    std::memcpy(input_host.host<int>(),
                ids.data() + b * kClipTokens,
                kClipTokens * sizeof(int32_t));
    input->copyFromHostTensor(&input_host);

    net->runSession(session);

    auto out_tensor = net->getSessionOutput(session, "last_hidden_state");
    if (!out_tensor) {
      std::cerr << "[Error] Missing last_hidden_state output\n";
      return ClipOutputs();
    }
    MNN::Tensor out_host(out_tensor, MNN::Tensor::CAFFE);
    out_tensor->copyToHostTensor(&out_host);
    std::memcpy(out.last_hidden.data() + b * kClipTokens * embed_dim,
                out_host.host<float>(),
                kClipTokens * embed_dim * sizeof(float));

    if (pooled_dim > 0) {
      auto pooled_tensor = net->getSessionOutput(session, "pooled_output");
      if (!pooled_tensor) {
        std::cerr << "[Error] Missing pooled_output\n";
        return ClipOutputs();
      }
      MNN::Tensor pooled_host(pooled_tensor, MNN::Tensor::CAFFE);
      pooled_tensor->copyToHostTensor(&pooled_host);
      std::memcpy(out.pooled.data() + b * pooled_dim,
                  pooled_host.host<float>(),
                  pooled_dim * sizeof(float));
    }
  }

  return out;
}

MNN::Tensor* PickMnnInput(MNN::Interpreter* net,
                          MNN::Session* session,
                          const std::string& preferred) {
  if (!net || !session) return nullptr;
  const auto& inputs = net->getSessionInputAll(session);
  if (!preferred.empty()) {
    auto it = inputs.find(preferred);
    if (it != inputs.end()) return it->second;
  }
  if (!inputs.empty()) return inputs.begin()->second;
  return nullptr;
}

MNN::Tensor* PickMnnOutput(MNN::Interpreter* net,
                           MNN::Session* session,
                           const std::string& preferred) {
  if (!net || !session) return nullptr;
  const auto& outputs = net->getSessionOutputAll(session);
  if (!preferred.empty()) {
    auto it = outputs.find(preferred);
    if (it != outputs.end()) return it->second;
  }
  if (!outputs.empty()) return outputs.begin()->second;
  return nullptr;
}

std::vector<float> RunMnnVae(const std::string& model_path,
                             const std::vector<float>& latent,
                             int latent_h,
                             int latent_w) {
  std::vector<float> out;
  auto net = std::shared_ptr<MNN::Interpreter>(
      MNN::Interpreter::createFromFile(model_path.c_str()),
      MNN::Interpreter::destroy);
  if (!net) {
    std::cerr << "[Error] Failed to load MNN VAE: " << model_path << "\n";
    return out;
  }

  std::string weight_path = model_path + ".weight";
  if (std::filesystem::exists(weight_path)) {
    net->setExternalFile(weight_path.c_str());
    std::cout << "[VAE] external weights: " << weight_path << "\n";
  }

  MNN::ScheduleConfig cfg;
  cfg.type = MNN_FORWARD_CPU;
  cfg.numThread = 4;
  MNN::BackendConfig backend;
  backend.memory = MNN::BackendConfig::Memory_Low;
  backend.power = MNN::BackendConfig::Power_High;
  cfg.backendConfig = &backend;

  auto session = net->createSession(cfg);
  if (!session) {
    std::cerr << "[Error] Failed to create MNN VAE session\n";
    return out;
  }

  MNN::Tensor* input = PickMnnInput(net.get(), session, "latent_sample");
  if (!input) {
    std::cerr << "[Error] Missing VAE input\n";
    return out;
  }
  net->resizeTensor(input, {1, 4, latent_h, latent_w});
  net->resizeSession(session);
  net->releaseModel();

  const size_t expect = static_cast<size_t>(4) * latent_h * latent_w;
  if (latent.size() != expect) {
    std::cerr << "[Error] VAE latent size mismatch: " << latent.size()
              << " expect=" << expect << "\n";
    return out;
  }

  MNN::Tensor input_host(input, MNN::Tensor::CAFFE);
  std::memcpy(input_host.host<float>(),
              latent.data(),
              latent.size() * sizeof(float));
  input->copyFromHostTensor(&input_host);

  net->runSession(session);
  MNN::Tensor* output = PickMnnOutput(net.get(), session, "sample");
  if (!output) {
    std::cerr << "[Error] Missing VAE output\n";
    return out;
  }
  MNN::Tensor out_host(output, MNN::Tensor::CAFFE);
  output->copyToHostTensor(&out_host);
  const int count = out_host.elementSize();
  out.assign(out_host.host<float>(), out_host.host<float>() + count);
  return out;
}

std::vector<float> MakeSinusoidalEmbedding(float value, int embed_dim) {
  const int half = embed_dim / 2;
  const float max_period = 10000.0f;
  std::vector<float> data(embed_dim);
  for (int i = 0; i < embed_dim; ++i) {
    const float exp_term = std::exp((-std::log(max_period) * (i < half ? i : i - half)) / half);
    const float scaled = exp_term * value;
    data[i] = (i < half) ? std::cos(scaled) : std::sin(scaled);
  }
  return data;
}

std::vector<float> BuildAddTimeEmbeds(const std::vector<float>& time_ids,
                                      int embed_dim) {
  std::vector<float> out;
  out.reserve(time_ids.size() * embed_dim);
  for (float v : time_ids) {
    std::vector<float> emb = MakeSinusoidalEmbedding(v, embed_dim);
    out.insert(out.end(), emb.begin(), emb.end());
  }
  return out;
}

std::vector<cl_half> ConvertFP32ToFP16(const float* src, size_t count) {
  std::vector<cl_half> dst(count);
  for (size_t i = 0; i < count; ++i) {
    dst[i] = fp32ToFP16(src[i]);
  }
  return dst;
}

std::vector<float> MakeRandomNormal(size_t count, float scale, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, scale);
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) {
    out[i] = dist(rng);
  }
  return out;
}

void WriteFp32(const std::string& path, const std::vector<float>& data) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  const char* dump_args_env = std::getenv("SDXL_DUMP_ARGS");
  if (dump_args_env != nullptr && std::atoi(dump_args_env) != 0) {
    std::cout << "[Args] argc=" << argc << "\n";
    for (int i = 0; i < argc; ++i) {
      std::cout << "[Args] argv[" << i << "]=" << argv[i] << "\n";
    }
  }
  std::string base_dir = "sdxl_clml/sdxl_clml_weights/";
  std::string mnn_dir =
      "/media/happyyzy/Data/pyproject/npu_sdxl/sdxl_qnn_out/mnn_sdxl_clip/";
  std::string clip_l_tokens_path =
      "/media/happyyzy/Data/pyproject/npu_sdxl/output/"
      "clip_int8_debug_dump_20260107_191913/debug_dump/clip_l_ids.i32";
  std::string clip_g_tokens_path =
      "/media/happyyzy/Data/pyproject/npu_sdxl/output/"
      "clip_int8_debug_dump_20260107_191913/debug_dump/clip_g_ids.i32";
  int num_steps = 20;
  float cfg_scale = 7.5f;
  uint32_t seed = 0;
  int height = 1024;
  int width = 1024;
  std::string out_path = "sdxl_clml_output.qfp32";
  bool optimize_mem = true;
  bool decoder_optimize_mem = true;
  bool use_mnn_vae = false;
  bool mnn_vae_expects_scaled = false;
  bool unet_only = false;
  int early_decode_k = 0;
  bool early_decode_x0 = false;
  bool export_x0 = false;
  std::string latent_out_path;
  std::string mnn_vae_path;

  if (argc > 1) base_dir = argv[1];
  if (argc > 2) mnn_dir = argv[2];
  if (argc > 3) clip_l_tokens_path = argv[3];
  if (argc > 4) clip_g_tokens_path = argv[4];
  if (argc > 5) num_steps = std::max(1, std::atoi(argv[5]));
  if (argc > 6) cfg_scale = std::max(0.0f, static_cast<float>(std::atof(argv[6])));
  if (argc > 7) seed = static_cast<uint32_t>(std::atoi(argv[7]));
  if (argc > 8) height = std::max(8, std::atoi(argv[8]));
  if (argc > 9) width = std::max(8, std::atoi(argv[9]));
  if (argc > 10) out_path = argv[10];
  if (argc > 11) optimize_mem = std::atoi(argv[11]) != 0;
  if (argc > 12) {
    mnn_vae_path = argv[12];
    use_mnn_vae = true;
  }

  if (!base_dir.empty() && base_dir.back() != '/') base_dir += "/";
  if (!mnn_dir.empty() && mnn_dir.back() != '/') mnn_dir += "/";

  if (height % 8 != 0 || width % 8 != 0) {
    std::cerr << "[Error] height/width must be multiples of 8\n";
    return 1;
  }

  std::vector<float> text_embedding;
  std::vector<float> pooled_embedding;
  bool use_embed_files = false;
  const char* text_embed_env = std::getenv("SDXL_TEXT_EMBEDS");
  const char* pooled_embed_env = std::getenv("SDXL_POOLED_EMBEDS");
  if ((text_embed_env && *text_embed_env) || (pooled_embed_env && *pooled_embed_env)) {
    if (!text_embed_env || !*text_embed_env || !pooled_embed_env ||
        !*pooled_embed_env) {
      std::cerr << "[Error] SDXL_TEXT_EMBEDS and SDXL_POOLED_EMBEDS must both be set\n";
      return 1;
    }
    text_embedding = readFP32FromFile(text_embed_env);
    pooled_embedding = readFP32FromFile(pooled_embed_env);
    const size_t expect_text =
        static_cast<size_t>(kBatch) * kClipTokens * kTextEmbedDim;
    const size_t expect_pooled = static_cast<size_t>(kBatch) * kPooledDim;
    if (text_embedding.size() != expect_text) {
      std::cerr << "[Error] text embedding size mismatch: " << text_embedding.size()
                << " expect=" << expect_text << "\n";
      return 1;
    }
    if (pooled_embedding.size() != expect_pooled) {
      std::cerr << "[Error] pooled embedding size mismatch: " << pooled_embedding.size()
                << " expect=" << expect_pooled << "\n";
      return 1;
    }
    use_embed_files = true;
    std::cout << "[CLIP] use precomputed embeddings\n";
  }

  if (!std::getenv("CLML_NO_REUSE_TNN")) {
    setenv("CLML_NO_REUSE_TNN", "1", 1);
  }

  const char* unet_only_env = std::getenv("SDXL_UNET_ONLY");
  if (unet_only_env != nullptr && std::atoi(unet_only_env) != 0) {
    unet_only = true;
  }
  const char* early_decode_env = std::getenv("SDXL_EARLY_DECODE_K");
  if (early_decode_env != nullptr && std::strlen(early_decode_env) > 0) {
    early_decode_k = std::max(0, std::atoi(early_decode_env));
  }
  const char* early_decode_x0_env = std::getenv("SDXL_EARLY_DECODE_X0");
  if (early_decode_x0_env != nullptr && std::atoi(early_decode_x0_env) != 0) {
    early_decode_x0 = true;
  }
  const char* export_x0_env = std::getenv("SDXL_EXPORT_X0");
  if (export_x0_env != nullptr && std::atoi(export_x0_env) != 0) {
    export_x0 = true;
  }
  const char* latent_out_env = std::getenv("SDXL_LATENT_OUT");
  if (latent_out_env != nullptr && std::strlen(latent_out_env) > 0) {
    latent_out_path = latent_out_env;
  }
  std::string latent_init_out_path;
  const char* latent_init_env = std::getenv("SDXL_LATENT_INIT_OUT");
  if (latent_init_env != nullptr && std::strlen(latent_init_env) > 0) {
    latent_init_out_path = latent_init_env;
  }
  std::string latent_init_path;
  bool latent_init_unscaled = false;
  const char* latent_init_in_env = std::getenv("SDXL_LATENT_INIT");
  if (latent_init_in_env != nullptr && std::strlen(latent_init_in_env) > 0) {
    latent_init_path = latent_init_in_env;
  }
  const char* latent_init_unscaled_env = std::getenv("SDXL_LATENT_INIT_UNSCALED");
  if (latent_init_unscaled_env != nullptr && std::atoi(latent_init_unscaled_env) != 0) {
    latent_init_unscaled = true;
  }
  std::string latent_step_dir;
  int latent_step_every = 1;
  const char* latent_step_dir_env = std::getenv("SDXL_LATENT_STEP_DIR");
  if (latent_step_dir_env != nullptr && std::strlen(latent_step_dir_env) > 0) {
    latent_step_dir = latent_step_dir_env;
  }
  const char* latent_step_every_env = std::getenv("SDXL_LATENT_STEP_EVERY");
  if (latent_step_every_env != nullptr && std::strlen(latent_step_every_env) > 0) {
    latent_step_every = std::max(1, std::atoi(latent_step_every_env));
  }
  if (!latent_step_dir.empty() && latent_step_dir.back() != '/') {
    latent_step_dir += "/";
  }
  std::string debug_dump_dir;
  int debug_dump_every = 1;
  const char* debug_dump_dir_env = std::getenv("SDXL_DEBUG_DUMP_DIR");
  if (debug_dump_dir_env != nullptr && std::strlen(debug_dump_dir_env) > 0) {
    debug_dump_dir = debug_dump_dir_env;
  }
  const char* debug_dump_every_env = std::getenv("SDXL_DEBUG_DUMP_EVERY");
  if (debug_dump_every_env != nullptr && std::strlen(debug_dump_every_env) > 0) {
    debug_dump_every = std::max(1, std::atoi(debug_dump_every_env));
  }
  if (!debug_dump_dir.empty() && debug_dump_dir.back() != '/') {
    debug_dump_dir += "/";
  }

  std::string scheduler_betas =
      base_dir + "weights/scheduler/scheduler_betas.qfp32";
  if (!std::filesystem::exists(scheduler_betas)) {
    std::cerr << "[Error] Missing scheduler betas: " << scheduler_betas << "\n";
    return 1;
  }

  std::vector<int32_t> clip_l_ids;
  std::vector<int32_t> clip_g_ids;
  if (!use_embed_files) {
    clip_l_ids = LoadIds(clip_l_tokens_path);
    clip_g_ids = LoadIds(clip_g_tokens_path);
    if (clip_l_ids.empty() || clip_g_ids.empty()) {
      return 1;
    }
  }

  std::string clip_l_path = mnn_dir + "clip_int8.mnn";
  std::string clip_g_path = mnn_dir + "clip_2_int8.mnn";
  if (!use_embed_files) {
    if (!std::filesystem::exists(clip_l_path) ||
        !std::filesystem::exists(clip_g_path)) {
      std::cerr << "[Error] Missing MNN CLIP models in: " << mnn_dir << "\n";
      return 1;
    }

    auto clip_start = std::chrono::steady_clock::now();
    ClipOutputs clip_l = RunMnnClipIds(clip_l_path, clip_l_ids, kClipEmbedDim, 0);
    ClipOutputs clip_g =
        RunMnnClipIds(clip_g_path, clip_g_ids, kClip2EmbedDim, kPooledDim);
    auto clip_end = std::chrono::steady_clock::now();
    if (clip_l.last_hidden.empty() || clip_g.last_hidden.empty() ||
        clip_g.pooled.empty()) {
      std::cerr << "[Error] CLIP outputs are empty\n";
      return 1;
    }

    text_embedding.resize(static_cast<size_t>(kBatch) * kClipTokens *
                          kTextEmbedDim);
    for (int b = 0; b < kBatch; ++b) {
      for (int t = 0; t < kClipTokens; ++t) {
        size_t dst_off =
            (static_cast<size_t>(b) * kClipTokens + t) * kTextEmbedDim;
        size_t l_off =
            (static_cast<size_t>(b) * kClipTokens + t) * kClipEmbedDim;
        size_t g_off =
            (static_cast<size_t>(b) * kClipTokens + t) * kClip2EmbedDim;
        std::memcpy(text_embedding.data() + dst_off,
                    clip_l.last_hidden.data() + l_off,
                    kClipEmbedDim * sizeof(float));
        std::memcpy(text_embedding.data() + dst_off + kClipEmbedDim,
                    clip_g.last_hidden.data() + g_off,
                    kClip2EmbedDim * sizeof(float));
      }
    }
    pooled_embedding = clip_g.pooled;

    auto clip_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        clip_end - clip_start).count();
    std::cout << "[Perf] CLIP ms=" << clip_ms << "\n";
  }

  std::vector<float> time_ids = {
      static_cast<float>(height),
      static_cast<float>(width),
      0.0f,
      0.0f,
      static_cast<float>(height),
      static_cast<float>(width),
  };
  std::vector<float> add_time_embeds =
      BuildAddTimeEmbeds(time_ids, kAddTimeEmbedDim);

  std::vector<float> add_embeds_uncond(kAddEmbedsDim);
  std::vector<float> add_embeds_cond(kAddEmbedsDim);
  std::memcpy(add_embeds_uncond.data(),
              pooled_embedding.data(),
              kPooledDim * sizeof(float));
  std::memcpy(add_embeds_uncond.data() + kPooledDim,
              add_time_embeds.data(),
              add_time_embeds.size() * sizeof(float));
  std::memcpy(add_embeds_cond.data(),
              pooled_embedding.data() + kPooledDim,
              kPooledDim * sizeof(float));
  std::memcpy(add_embeds_cond.data() + kPooledDim,
              add_time_embeds.data(),
              add_time_embeds.size() * sizeof(float));
  const char* swap_env = std::getenv("SDXL_EMBEDS_SWAP");
  if (swap_env != nullptr && std::atoi(swap_env) != 0) {
    const size_t text_stride = static_cast<size_t>(kClipTokens) * kTextEmbedDim;
    const size_t pooled_stride = static_cast<size_t>(kPooledDim);
    for (size_t i = 0; i < text_stride; ++i) {
      std::swap(text_embedding[i], text_embedding[text_stride + i]);
    }
    for (size_t i = 0; i < pooled_stride; ++i) {
      std::swap(add_embeds_uncond[i], add_embeds_cond[i]);
    }
    std::cout << "[CLIP] swap uncond/cond embeddings\n";
  }

  const char* use_mnn_vae_env = std::getenv("SDXL_USE_MNN_VAE");
  if (use_mnn_vae_env != nullptr && std::atoi(use_mnn_vae_env) != 0) {
    use_mnn_vae = true;
  }
  const char* mnn_scaled_env = std::getenv("MNN_VAE_EXPECTS_SCALED");
  if (mnn_scaled_env != nullptr && std::atoi(mnn_scaled_env) != 0) {
    mnn_vae_expects_scaled = true;
  }
  if (mnn_vae_path.empty()) {
    mnn_vae_path = mnn_dir + "vae_decoder.mnn";
  }

  decoder_optimize_mem = optimize_mem;
  if (height >= 1024 || width >= 1024) {
    decoder_optimize_mem = false;
  }
  const char* decoder_no_opt_env = std::getenv("CLML_DECODER_NO_OPT");
  if (decoder_no_opt_env != nullptr && std::atoi(decoder_no_opt_env) != 0) {
    decoder_optimize_mem = false;
  }

  std::cout << "[Config] optimize_device_mem=" << (optimize_mem ? 1 : 0)
            << " decoder_optimize_mem=" << (decoder_optimize_mem ? 1 : 0)
            << " use_mnn_vae=" << (use_mnn_vae ? 1 : 0)
            << " mnn_vae_expects_scaled=" << (mnn_vae_expects_scaled ? 1 : 0)
            << " unet_only=" << (unet_only ? 1 : 0)
            << " early_decode_k=" << early_decode_k
            << " early_decode_x0=" << (early_decode_x0 ? 1 : 0)
            << " export_x0=" << (export_x0 ? 1 : 0)
            << "\n";

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

  const cl_int minor_version = 0;
  CLEnvironment unet_env = createCLEnvironment(false);
  CLMLInterfaceV4QCOM* unet_clml_intf = clGetMLInterfaceV4QCOM(minor_version);
  if (!unet_clml_intf) {
    std::cerr << "[Error] clGetMLInterfaceV4QCOM failed\n";
    return 1;
  }

  const int latent_h = height / 8;
  const int latent_w = width / 8;
  const size_t sample_count = static_cast<size_t>(4) * latent_h * latent_w;
  std::cout << "[Config] height=" << height << " width=" << width
            << " latent_h=" << latent_h << " latent_w=" << latent_w
            << " sample_count=" << sample_count << "\n";
  std::vector<float> latent_sample;
  if (early_decode_k >= num_steps) {
    std::cerr << "[Error] SDXL_EARLY_DECODE_K must be < num_steps\n";
    return 1;
  }
  const int unet_steps = num_steps - early_decode_k;
  Scheduler scheduler(model_desc);
  scheduler.setTimeSteps(static_cast<cl_uint>(num_steps));
  if (!latent_init_path.empty()) {
    latent_sample = readFP32FromFile(latent_init_path);
    if (latent_sample.size() != sample_count) {
      std::cerr << "[Error] latent init size mismatch: " << latent_sample.size()
                << " expect=" << sample_count << "\n";
      return 1;
    }
    if (latent_init_unscaled) {
      latent_sample = scheduler.applyInitialNoiseSigma(latent_sample);
      std::cout << "[Init] latent init (unscaled)\n";
    } else {
      std::cout << "[Init] latent init (scaled)\n";
    }
  } else {
    latent_sample = MakeRandomNormal(sample_count, 1.0f, seed);
    latent_sample = scheduler.applyInitialNoiseSigma(latent_sample);
  }
  if (!latent_init_out_path.empty()) {
    std::filesystem::path init_path(latent_init_out_path);
    if (!init_path.parent_path().empty()) {
      std::error_code ec;
      std::filesystem::create_directories(init_path.parent_path(), ec);
    }
    WriteFp32(latent_init_out_path, latent_sample);
    std::cout << "[Out] latent_init=" << latent_init_out_path << "\n";
  }
  if (!latent_step_dir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(latent_step_dir, ec);
    std::cout << "[Config] latent_step_dir=" << latent_step_dir
              << " every=" << latent_step_every << "\n";
  }
  if (!debug_dump_dir.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(debug_dump_dir, ec);
    std::cout << "[Config] debug_dump_dir=" << debug_dump_dir
              << " every=" << debug_dump_every << "\n";
  }

  auto unet_init_start = std::chrono::steady_clock::now();
  std::vector<float> early_decode_latent;
  {
    tensor_dims_t sample_dims = {1, 4,
                                 static_cast<cl_uint>(latent_h),
                                 static_cast<cl_uint>(latent_w)};
    tensor_dims_t text_dims = {1, 1, kClipTokens, kTextEmbedDim};
    tensor_dims_t tstep_dims = {1, 1, 1, kTimeEmbedDim};
    tensor_dims_t add_dims = {1, 1, 1, kAddEmbedsDim};

    MLTensor sample = createMLTensor(
        unet_env, unet_clml_intf, sample_dims, model_desc.model_dtype,
        CL_TENSOR_LAYOUT_NCHW_QCOM, CL_TENSOR_USAGE_CNN_QCOM);
    MLTensor text_embedding_tensor = createMLTensor(
        unet_env, unet_clml_intf, text_dims, model_desc.model_dtype,
        CL_TENSOR_LAYOUT_NCHW_QCOM, CL_TENSOR_USAGE_TNN_QCOM);
    MLTensor tstep_embedding_tensor = createMLTensor(
        unet_env, unet_clml_intf, tstep_dims, model_desc.model_dtype,
        CL_TENSOR_LAYOUT_NCHW_QCOM, CL_TENSOR_USAGE_TNN_QCOM);
    MLTensor add_embeds_tensor = createMLTensor(
        unet_env, unet_clml_intf, add_dims, model_desc.model_dtype,
        CL_TENSOR_LAYOUT_NCHW_QCOM, CL_TENSOR_USAGE_TNN_QCOM);

    UNet unet(model_desc, unet_env);
    std::vector<MLTensor> unet_outputs =
        unet.create({sample, text_embedding_tensor, tstep_embedding_tensor,
                     add_embeds_tensor});
    if (unet_outputs.size() != 1) {
      std::cerr << "[Error] UNet output size invalid: " << unet_outputs.size()
                << "\n";
      return 1;
    }
    unet.initParams();
    auto unet_init_end = std::chrono::steady_clock::now();
    auto unet_init_s = std::chrono::duration_cast<std::chrono::duration<double>>(
        unet_init_end - unet_init_start)
                           .count();
    std::cout << "[Perf] UNet init_s=" << unet_init_s << "\n";

    const size_t text_batch_stride = static_cast<size_t>(kClipTokens) * kTextEmbedDim;
    const float* text_uncond = text_embedding.data();
    const float* text_cond = text_embedding.data() + text_batch_stride;
    std::vector<cl_half> text_uncond_fp16 =
        ConvertFP32ToFP16(text_uncond, text_batch_stride);
    std::vector<cl_half> text_cond_fp16 =
        ConvertFP32ToFP16(text_cond, text_batch_stride);
    std::vector<cl_half> add_uncond_fp16 =
        ConvertFP32ToFP16(add_embeds_uncond.data(), kAddEmbedsDim);
    std::vector<cl_half> add_cond_fp16 =
        ConvertFP32ToFP16(add_embeds_cond.data(), kAddEmbedsDim);

    std::vector<float> pred_uncond;
    std::vector<float> pred_cond;

    auto unet_loop_start = std::chrono::steady_clock::now();
    for (int step = 0; step < unet_steps; ++step) {
      float time_step = scheduler.getCurrentTimeStep(static_cast<cl_uint>(step));
      float time_step_embed = time_step;
      const char* round_t_env = std::getenv("SDXL_TSTEP_ROUND");
      if (round_t_env != nullptr && std::atoi(round_t_env) != 0) {
        time_step_embed = std::round(time_step);
      }
      std::vector<float> sample_input =
          scheduler.scaleLatentSample(latent_sample, time_step);

      std::vector<cl_half> sample_fp16 =
          ConvertFP32ToFP16(sample_input.data(), sample_count);
      std::vector<float> tstep_fp32 =
          MakeSinusoidalEmbedding(time_step_embed, kTimeEmbedDim);
      std::vector<cl_half> tstep_fp16 =
          ConvertFP32ToFP16(tstep_fp32.data(), tstep_fp32.size());

      pred_uncond.clear();
      pred_cond.clear();

      sample.uploadDataIntoGPUMem(sample_fp16.data(),
                                  model_desc.tensor_data_src_layout);
      tstep_embedding_tensor.uploadDataIntoGPUMem(
          tstep_fp16.data(), model_desc.tensor_data_src_layout);

      text_embedding_tensor.uploadDataIntoGPUMem(
          text_uncond_fp16.data(), model_desc.tensor_data_src_layout);
      add_embeds_tensor.uploadDataIntoGPUMem(
          add_uncond_fp16.data(), model_desc.tensor_data_src_layout);
      unet.forward();
      pred_uncond = unet_outputs[0].downloadDataFromGPUMem();

      text_embedding_tensor.uploadDataIntoGPUMem(
          text_cond_fp16.data(), model_desc.tensor_data_src_layout);
      add_embeds_tensor.uploadDataIntoGPUMem(
          add_cond_fp16.data(), model_desc.tensor_data_src_layout);
      unet.forward();
      pred_cond = unet_outputs[0].downloadDataFromGPUMem();

      if (pred_uncond.size() != pred_cond.size()) {
        std::cerr << "[Error] UNet output size mismatch\n";
        return 1;
      }

      std::vector<float> pred_noise(pred_cond.size());
      for (size_t i = 0; i < pred_noise.size(); ++i) {
        pred_noise[i] =
            pred_uncond[i] + cfg_scale * (pred_cond[i] - pred_uncond[i]);
      }

      if (!debug_dump_dir.empty() && ((step + 1) % debug_dump_every == 0)) {
        const std::string prefix =
            debug_dump_dir + "step_" + std::to_string(step) + "_";
        WriteFp32(prefix + "latent_in.qfp32", latent_sample);
        WriteFp32(prefix + "sample_input.qfp32", sample_input);
        WriteFp32(prefix + "pred_uncond.qfp32", pred_uncond);
        WriteFp32(prefix + "pred_cond.qfp32", pred_cond);
        WriteFp32(prefix + "pred_noise.qfp32", pred_noise);
        std::vector<float> tstep_vals = {time_step, time_step_embed};
        WriteFp32(prefix + "timestep.qfp32", tstep_vals);
        WriteFp32(prefix + "tstep_embed.qfp32", tstep_fp32);
      }

      if ((export_x0 || (early_decode_k > 0 && early_decode_x0)) &&
          step == (unet_steps - 1)) {
        float sigma = scheduler.getSigma(time_step);
        early_decode_latent.resize(latent_sample.size());
        for (size_t i = 0; i < latent_sample.size(); ++i) {
          early_decode_latent[i] = latent_sample[i] - sigma * pred_noise[i];
        }
      }

      latent_sample = scheduler.step(latent_sample, pred_noise, time_step);
      std::cout << "[UNet] step " << (step + 1) << "/" << unet_steps;
      if (early_decode_k > 0) {
        std::cout << " (early decode k=" << early_decode_k << ")";
      }
      std::cout << "\n";
      if (!latent_step_dir.empty() && ((step + 1) % latent_step_every == 0)) {
        const std::string step_path = latent_step_dir + "latent_step_" +
                                      std::to_string(step + 1) + ".qfp32";
        WriteFp32(step_path, latent_sample);
      }
    }
    auto unet_loop_end = std::chrono::steady_clock::now();
    auto unet_loop_s = std::chrono::duration_cast<std::chrono::duration<double>>(
        unet_loop_end - unet_loop_start)
                           .count();
    std::cout << "[Perf] UNet loop_s=" << unet_loop_s
              << " s/step=" << (unet_loop_s / static_cast<double>(num_steps))
              << "\n";
  }

  if (early_decode_k > 0) {
    std::cout << "[Info] UNet early decode at step " << unet_steps << "/"
              << num_steps << "\n";
  }
  std::cout << "[Info] UNet done, releasing CL env\n";
  releaseCLEnvironment(unet_env);
  std::cout << "[Info] UNet CL env released\n";

  const std::vector<float>* latent_for_decode = &latent_sample;
  if ((export_x0 || (early_decode_k > 0 && early_decode_x0)) &&
      !early_decode_latent.empty()) {
    latent_for_decode = &early_decode_latent;
  }

  if (unet_only) {
    if (latent_out_path.empty()) {
      latent_out_path = out_path;
    }
    if (latent_out_path.empty()) {
      std::cerr << "[Error] latent output path is empty\n";
      return 1;
    }
    std::filesystem::path latent_path(latent_out_path);
    if (!latent_path.parent_path().empty()) {
      std::error_code ec;
      std::filesystem::create_directories(latent_path.parent_path(), ec);
    }
    WriteFp32(latent_out_path, *latent_for_decode);
    if (export_x0 || (early_decode_k > 0 && early_decode_x0)) {
      std::cout << "[Out] x0=" << latent_out_path << " (unscaled)\n";
    } else {
      std::cout << "[Out] latent=" << latent_out_path << " (unscaled)\n";
    }
    return 0;
  }

  std::cout << "[Info] VAE init\n";
  auto vae_start = std::chrono::steady_clock::now();
  if (use_mnn_vae) {
    if (!std::filesystem::exists(mnn_vae_path)) {
      std::cerr << "[Error] Missing MNN VAE model: " << mnn_vae_path << "\n";
      return 1;
    }
    std::vector<float> vae_input = *latent_for_decode;
    if (mnn_vae_expects_scaled) {
      const float vae_scaling = 0.13025f;
      for (float& v : vae_input) {
        v *= (1.0f / vae_scaling);
      }
    }
    std::vector<float> image = RunMnnVae(mnn_vae_path, vae_input, latent_h, latent_w);
    if (image.empty()) {
      std::cerr << "[Error] MNN VAE output is empty\n";
      return 1;
    }
    WriteFp32(out_path, image);
  } else {
    std::cout << "[Info] VAE create CL env\n";
    CLEnvironment vae_env = createCLEnvironment(false);
    std::cout << "[Info] VAE CL env ready\n";
    CLMLInterfaceV4QCOM* vae_clml_intf = clGetMLInterfaceV4QCOM(minor_version);
    if (!vae_clml_intf) {
      std::cerr << "[Error] clGetMLInterfaceV4QCOM failed\n";
      return 1;
    }
    std::cout << "[Info] VAE create decoder\n";
    const float vae_scaling = 0.13025f;
    std::vector<float> scaled_latent(latent_for_decode->size());
    for (size_t i = 0; i < latent_for_decode->size(); ++i) {
      scaled_latent[i] = (*latent_for_decode)[i] * (1.0f / vae_scaling);
    }
    tensor_dims_t latent_dims = {1, 4,
                                 static_cast<cl_uint>(latent_h),
                                 static_cast<cl_uint>(latent_w)};
    MLTensor decoder_input = createMLTensor(
        vae_env, vae_clml_intf, latent_dims, model_desc.model_dtype,
        CL_TENSOR_LAYOUT_NCHW_QCOM, CL_TENSOR_USAGE_CNN_QCOM);
    NNModelDesc decoder_desc = model_desc;
    decoder_desc.optimize_device_mem = decoder_optimize_mem;
    Decoder decoder(decoder_desc, vae_env);
    std::vector<MLTensor> decoder_outputs = decoder.create({decoder_input});
    if (decoder_outputs.size() != 1) {
      std::cerr << "[Error] Decoder output size invalid: " << decoder_outputs.size()
                << "\n";
      return 1;
    }
    decoder.initParams();

    std::vector<cl_half> latent_fp16 =
        ConvertFP32ToFP16(scaled_latent.data(), scaled_latent.size());
    decoder_input.uploadDataIntoGPUMem(latent_fp16.data(),
                                       model_desc.tensor_data_src_layout);
    decoder.forward();
    std::vector<float> image = decoder_outputs[0].downloadDataFromGPUMem();
    WriteFp32(out_path, image);
    releaseCLEnvironment(vae_env);
  }
  auto vae_end = std::chrono::steady_clock::now();

  auto vae_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      vae_end - vae_start).count();
  std::cout << "[Perf] VAE ms=" << vae_ms << "\n";
  std::cout << "[Out] " << out_path << "\n";

  return 0;
}
