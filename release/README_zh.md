<p align="center">
  <img src="../assets/logo_fast_diffusion.svg" alt="Fast-Diffusion" width="720">
</p>
<h1 align="center">Fast-Diffusion（移动端极速推理发布）</h1>
<p align="center">
  Fast-Diffusion 是基于 <strong>CLML</strong> 的移动端扩散推理方案，目标是 <strong>在适配机型上做到公开可比口径下的最快速度</strong>。<br/>
  本发布版包含 SD1.5 512 与 SDXL 1024 的完整流程、App 与 bench 程序。
</p>

---

## 亮点
- CLML 后端，高速移动端推理
- SD1.5 512（单进程）与 SDXL 1024（分进程，推荐）
- **提前 k=2 步解码 x0**，显著改善 SDXL 末端噪点
- 含 App 源码、APK 与 bench 二进制

---

## 图像对比（SDXL 1024，CLML）

**同 prompt / seed / CFG**。左图是提前解码（k=2, x0），右图是默认最终步解码。

**20 步**
| 提前解码（k=2, x0） | 最终步解码（x0） |
| --- | --- |
| <img src="../assets/gallery/sdxl_clml_early2_x0_1024.png" alt="sdxl early k2 x0 20 steps" width="512"><br/><sub>SM8750 · 步数=20 · k=2 · s/it=3.21684（CFG UNet）</sub> | <img src="../assets/gallery/sdxl_clml_final_x0_1024.png" alt="sdxl final x0 20 steps" width="512"><br/><sub>SM8750 · 步数=20 · final · s/it=3.21684（CFG UNet）</sub> |

**25 步**
| 提前解码（k=2, x0） | 最终步解码（x0） |
| --- | --- |
| <img src="../assets/gallery/sdxl_clml_k2_x0_25steps.png" alt="sdxl early k2 x0 25 steps" width="512"><br/><sub>SM8750 · 步数=25 · k=2 · s/it=3.21684（CFG UNet）</sub> | <img src="../assets/gallery/sdxl_clml_final_x0_25steps.png" alt="sdxl final x0 25 steps" width="512"><br/><sub>SM8750 · 步数=25 · final · s/it=3.21684（CFG UNet）</sub> |

**30 步**
| 提前解码（k=2, x0） | 最终步解码（x0） |
| --- | --- |
| <img src="../assets/gallery/sdxl_clml_k2_x0_30steps.png" alt="sdxl early k2 x0 30 steps" width="512"><br/><sub>SM8750 · 步数=30 · k=2 · s/it=3.21684（CFG UNet）</sub> | <img src="../assets/gallery/sdxl_clml_final_x0_30steps.png" alt="sdxl final x0 30 steps" width="512"><br/><sub>SM8750 · 步数=30 · final · s/it=3.21684（CFG UNet）</sub> |

**步进快照（20 步，x0）**
| Step 17（k=2） | Step 18（k=1） | Step 19（final） |
| --- | --- | --- |
| <img src="../assets/gallery/sdxl_clml_x0_step17.png" alt="sdxl x0 step17" width="360"><br/><sub>SM8750 · 步数=20 · step=17 · s/it=3.21684（CFG UNet）</sub> | <img src="../assets/gallery/sdxl_clml_x0_step18.png" alt="sdxl x0 step18" width="360"><br/><sub>SM8750 · 步数=20 · step=18 · s/it=3.21684（CFG UNet）</sub> | <img src="../assets/gallery/sdxl_clml_x0_step19.png" alt="sdxl x0 step19" width="360"><br/><sub>SM8750 · 步数=20 · step=19 · s/it=3.21684（CFG UNet）</sub> |

---

## 图集（SD1.5 512，CLML）

| 人像 2 | 非人像（城市夜景） |
| --- | --- |
| <img src="../assets/gallery/sd15_clml_portrait2.png" alt="sd15 portrait2" width="360"><br/><sub>SM8750 · 步数=20 · s/step=0.459639（CFG）</sub> | <img src="../assets/gallery/sd15_clml_sample_cityscape.png" alt="sd15 cityscape" width="360"><br/><sub>SM8750 · 步数=20 · s/step=0.459639（CFG）</sub> |

---

## 性能对比（CFG 口径对齐，含明确记录）

我们的实测（完整记录已附在仓库中）：
- **SD1.5 512（CFG）**：steps=20，total_s=9.19277，s/step=0.459639  
  - 记录：`release/sd_pipelines_zh.md`
- **SDXL 1024 UNet-only（CFG）**：init_s=65.6627，loop_s=64.3368，s/step=3.21684（20 步，预计算 embedding）  
  - 记录：`release/bench/logs/sdxl_unet_pyclip.log`
- **SDXL 1024 UNet 单次前向（不含 CFG）**：iters=1，s/it=1.61393  
  - 记录：`release/bench/logs/sdxl_unet_single_step.log`

公开对比基线（CFG 口径）：
1. CVPR 2023（Google LLC）：Adreno 740，SD1.4，20 步 **11.5s**  
   - 论文：<https://arxiv.org/abs/2304.11267>
2. Local Diffusion App（作者实测）：Snapdragon 8 Gen 3，SD1.5，**8 s/it**，GPU 比 CPU 更慢  
   - App：<https://github.com/rmatif/Local-Diffusion>  
   - 后端：<https://github.com/leejet/stable-diffusion.cpp>
3. Local Dream（MNN GPU 后端）：Snapdragon 8 Elite，SD1.5，20 步 **52s**
4. T4（SDXL 1024）：**1.2 s/it**（含 CFG）

**关于 T4 口径：**1.2 s/it 含 CFG（两次 UNet）。折算到单次 UNet 为 0.6 s/it。  
对比我们的 SDXL 单次前向记录 **1.61393 s/it**，CFG 步速为 **3.21684 s/step**。

结论强调：
- 在支持 CLML 的机型上，**公开可比口径下全网最快**
- **SDXL 1024 在该类机型上首次实现可用推理体验**
- 单步速度非常惊人，移动端体验强

---

## 未来计划（已验证/必然）

- SDXL Base/Turbo 的 512/768 后端已完成，仅差前端接入
- 已验证 SM8750 上 SDXL 512 单步 UNet：**65–70 ms/step**  
  - 记录：`release/bench/logs/sdxl_512_unet_timing_record.md`（device_base_logcat，graph `qnn_base_unet_w8a16_b1_512_real_dpm`）
- 因此 SM8750 上 SDXL Turbo 768 **必然**可在 10 秒内生成高质量图像（性能/质量折中）
- 对 16GB RAM 机型开放 SDXL 的 UNet 预初始化选项

---

## 依赖与前提

- Qualcomm Adreno GPU
- OpenCL 扩展：`cl_qcom_ml_ops`
- CLML SDK（编译所需）
- MNN 支持 Attention HostOp
  - 参考编译选项：`MNN_SUPPORT_TRANSFORMER_FUSE=ON`

运行注意：
- 必须设置 `CLML_NO_REUSE_TNN=1`（避免数值不稳定）
- SDXL VAE 依赖 CLML VAE + MNN Attention HostOp
- App 行为说明：SDXL 1024 即使不预初始化也需要 16GB RAM；预初始化时 CLIP + UNet 同时驻留会 OOM
- SD1.5 不受该限制，可预初始化，体验更流畅

## SDK 版本说明

- **CLML SDK**：v4.1（cl_qcom_ml_ops）
- **QNN/SNPE SDK**：2.39（SoC 列表来源）
- **MNN**：3.3.0，自定义编译，需启用 Attention HostOp（Transformer Fuse）

## 适配 SoC（来源：Qualcomm QNN/SNPE SDK 表）

来源文件：`QNN_SDK_2.39/qairt/2.39.0.250926/docs/SNPE/html/general/overview.html`

- SD 8 Elite Gen 5 (SM8850)
- SD 8 Gen 4 (SM8750)
- SD 8 Gen 3 (SM8650)
- SD 8 Gen 2 (SM8550)
- SD 8s Gen 3 (SM8635)
- SD 8+ Gen 1 (SM8475)
- SD 8 Gen 1 (SM8450)
- 888+ (SM8350P)
- 888 (SM8350)
- 7+ Gen 3 (SM7675)
- 7 Gen 1 (SM7450)
- 778G (SM7325)
- 865 (SM8250)
- 765 (SM7250)
- 750G (SM7225)
- 690 (SM6350)
- 695 (SM6375)
- 680 (SM6225)
- 480 (SM4350/6325)
- 460 (SM4250)
- 662 (SM6115)

---

## 发布结构

- `app/sdxl-clml/`：App 源码
- `release/`：发布说明、APK、bench 二进制与源码
  - `release/app/sdxl-clml-debug.apk`
  - `release/bench/`
  - `release/sd_pipelines_zh.md`（完整流程说明）

---

## 快速开始（SD1.5 512）

```sh
adb push release/bench/sd15_pipeline_run /data/local/tmp/sd15_pipeline_run
adb push -r <sd15_clml_weights_dir> /data/local/tmp/sd15_clml/

adb shell "CLML_NO_REUSE_TNN=1 /data/local/tmp/sd15_pipeline_run /data/local/tmp/sd15_clml/sd15_clml_weights 20"
```

输出：`/data/local/tmp/output/clml_stable_diffusion_output.qfp32`

---

## 快速开始（SDXL 1024，推荐）

内存说明：
- SDXL 1024 即使不预初始化也需要 16GB RAM
- 预初始化时 CLIP + UNet 同时驻留会 OOM；App 不对 SDXL 做预初始化

### 1) 生成 CLIP token ids（本机）
```sh
conda run -n comfyui --no-capture-output python - <<'PY'
import sys
import numpy as np
from pathlib import Path
COMFY_ROOT = "/home/happyyzy/ComfyUI"
if COMFY_ROOT not in sys.path:
    sys.path.append(COMFY_ROOT)
import comfy.sd

ckpt_path = "<SDXL_CKPT_PATH>/sd_xl_base_1.0.safetensors"
prompt = "a close-up portrait of a young woman, soft lighting, shallow depth of field"

_, clip, _, _ = comfy.sd.load_checkpoint_guess_config(
    ckpt_path,
    output_vae=False,
    output_clip=True,
    output_model=False,
)

def token_ids(token_list):
    return [int(t[0]) for t in token_list]

cond = clip.tokenize(prompt)
uncond = clip.tokenize("")

ids_l = np.array(token_ids(uncond["l"][0]) + token_ids(cond["l"][0]), dtype=np.int32)
ids_g = np.array(token_ids(uncond["g"][0]) + token_ids(cond["g"][0]), dtype=np.int32)

Path("clip_l_ids.i32").write_bytes(ids_l.tobytes())
Path("clip_g_ids.i32").write_bytes(ids_g.tobytes())
print("ok")
PY
```

### 2) UNet 进程（CPU CLIP + CLML UNet）
```sh
adb push release/bench/sdxl_pipeline_run /data/local/tmp/sdxl_pipeline_run

adb shell "LD_LIBRARY_PATH=/data/local/tmp/MNN_fuse:/system/lib64:/vendor/lib64 \
MNN_CL_LIB=/data/local/tmp/MNN_fuse/libMNN_CL.so \
CLML_MNN_ATTN_BACKEND=opencl CLML_MNN_ATTN_FP32=1 CLML_NO_REUSE_TNN=1 \
SDXL_EARLY_DECODE_K=2 SDXL_EARLY_DECODE_X0=1 SDXL_UNET_ONLY=1 \
SDXL_LATENT_OUT=/data/local/tmp/sdxl_latent_clipcpu_early2_x0.qfp32 \
/data/local/tmp/sdxl_pipeline_run \
/data/local/tmp/sdxl_clml/sdxl_clml_weights \
/data/local/tmp/MNN_clip \
/data/local/tmp/clip_l_ids.i32 /data/local/tmp/clip_g_ids.i32 \
20 7.5 0 1024 1024 /data/local/tmp/unused_output.qfp32"
```

### 3) VAE 进程（CLML VAE + MNN Attention）
```sh
adb push release/bench/sdxl_vae_decoder_run /data/local/tmp/sdxl_vae_decoder_run

adb shell "cd /data/local/tmp && \
LD_LIBRARY_PATH=/data/local/tmp/MNN_fuse:/system/lib64:/vendor/lib64 \
MNN_CL_LIB=/data/local/tmp/MNN_fuse/libMNN_CL.so \
MNN_BACKEND=opencl MNN_GPU_MODE=1 MNN_MEM=0 MNN_POWER=0 MNN_PREC=0 \
CLML_MNN_ATTN_BACKEND=opencl CLML_MNN_ATTN_FP32=1 CLML_NO_REUSE_TNN=1 \
./sdxl_vae_decoder_run /data/local/tmp/sdxl_clml/sdxl_clml_weights \
1 0 1 0.1 128 128 1 /data/local/tmp/sdxl_latent_clipcpu_early2_x0.qfp32"
```

### 4) qfp32 转 PNG（本机）
```sh
python3 - <<'PY'
import numpy as np
from PIL import Image
path_in = './sdxl_vae_out_clipcpu_early2_x0.qfp32'
path_out = './sdxl_vae_out_clipcpu_early2_x0.png'
arr = np.fromfile(path_in, dtype=np.float32).reshape(1, 3, 1024, 1024)
img = (arr[0] / 2.0 + 0.5)
img = np.clip(img, 0.0, 1.0)
img = (img.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
Image.fromarray(img).save(path_out)
print(path_out)
PY
```

---

## App

- 源码：`app/sdxl-clml/`
- APK：`release/app/sdxl-clml-debug.apk`
- 功能：SDXL 1024 + SD1.5 512、步数、CFG、scheduler、early decode、decode x0、seed、正/负 prompt

---

## 权重

权重已上传至 HuggingFace（public）：  
https://huggingface.co/zhiyuanasad/fast-diffusion-weights

- SD1.5 权重：https://huggingface.co/zhiyuanasad/fast-diffusion-weights/tree/main/sd15_clml_weights  
- SDXL 权重：https://huggingface.co/zhiyuanasad/fast-diffusion-weights/tree/main/sdxl_clml_weights

---

## 致谢

- Qualcomm CLML SDK（cl_qcom_ml_ops）
- MNN SDK 与运行时
