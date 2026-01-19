# SD1.5 512 / SDXL 1024 成功模型与端到端流水线汇总

本文档汇总当前已验证可跑通的两套方案：
- SD1.5 512 分辨率（单进程端到端）
- SDXL 1024 分辨率（分进程端到端，推荐）

## 通用前提
- 运行时请设置 `CLML_NO_REUSE_TNN=1`（复用 TNN 会导致数值偏差或不稳定）。
- SDXL VAE 使用 **CLML VAE + MNN Attention HostOp**，必须加载带 Attention 的 MNN 库。
  - 推荐库路径：`/data/local/tmp/MNN_fuse/`
  - 运行时环境变量：
    - `LD_LIBRARY_PATH=/data/local/tmp/MNN_fuse:/system/lib64:/vendor/lib64`
    - `MNN_CL_LIB=/data/local/tmp/MNN_fuse/libMNN_CL.so`
  - 避免把 `/data/local/tmp` 放在前面，否则可能加载到不支持 Attention 的旧库。
  - 内存说明：SDXL 1024 即使不预初始化也需要 16GB RAM；预初始化时 CLIP + UNet 同时驻留会 OOM

## SD1.5 512 端到端（单进程）
### 资源与二进制
- 权重与资产：`test_archive/sd15_clml_weights`
- 生成资产脚本：`test_archive/sd15_clml_prepare_pipeline_assets.py`
- 二进制：`test_archive/clml_tests/sd15_pipeline_run`

### 设备端运行示例
```sh
adb push test_archive/clml_tests/sd15_pipeline_run /data/local/tmp/sd15_pipeline_run
adb push -r test_archive/sd15_clml_weights /data/local/tmp/sd15_clml/

adb shell "CLML_NO_REUSE_TNN=1 /data/local/tmp/sd15_pipeline_run /data/local/tmp/sd15_clml/sd15_clml_weights 20"
```

### 输出与测速
- 输出文件：`/data/local/tmp/output/clml_stable_diffusion_output.qfp32`
- 速度：程序会打印
  - `[Perf] steps=20 total_s=... s/step=...`

## SDXL 1024 端到端（分进程，推荐）
单进程 CLIP+UNet+VAE 容易 OOM（UNet 卸载不彻底），因此推荐：
1) UNet 单独进程生成 `step-k` 的 `x0` latent
2) VAE 单独进程解码

内存说明：
- SDXL 1024 即使不预初始化也需要 16GB RAM
- 预初始化时 CLIP + UNet 同时驻留会 OOM

### 必备资源
- SDXL 权重：`/data/local/tmp/sdxl_clml/sdxl_clml_weights`
- CLIP 模型（CPU，MNN int8）：
  - `/data/local/tmp/MNN_clip/clip_int8.mnn`
  - `/data/local/tmp/MNN_clip/clip_2_int8.mnn`
  - `/data/local/tmp/MNN_clip/clip_2_int8.mnn.weight`
- CLIP token ids（2x77，顺序为 [uncond, cond]）：
  - `/data/local/tmp/clip_l_ids.i32`
  - `/data/local/tmp/clip_g_ids.i32`
- MNN_fuse 库：`/data/local/tmp/MNN_fuse/libMNN.so`、`libMNN_Express.so`、`libMNN_CL.so`

### CLIP token ids 生成示例（本机）
```sh
conda run -n comfyui --no-capture-output python - <<'PY'
import sys
import numpy as np
from pathlib import Path
COMFY_ROOT = "/home/happyyzy/ComfyUI"
if COMFY_ROOT not in sys.path:
    sys.path.append(COMFY_ROOT)
import comfy.sd

ckpt_path = "/media/happyyzy/Data/ComfyUI_Zluda_New/models/checkpoints/sd_xl_base_1.0.safetensors"
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

Path("/media/happyyzy/Data/pyproject/mlcl_tvm_sd1.5/test_archive/output/clip_l_ids.i32").write_bytes(ids_l.tobytes())
Path("/media/happyyzy/Data/pyproject/mlcl_tvm_sd1.5/test_archive/output/clip_g_ids.i32").write_bytes(ids_g.tobytes())
print("ok")
PY
```

### 步骤 A：UNet 单独进程（CPU CLIP + CLML UNet）
- 推荐：提前 `k=2` 步并输出 `x0`（减少末端色块）
- 环境变量：
  - `SDXL_EARLY_DECODE_K=2`
  - `SDXL_EARLY_DECODE_X0=1`
  - `SDXL_UNET_ONLY=1`
  - `SDXL_LATENT_OUT=/data/local/tmp/sdxl_latent_clipcpu_early2_x0.qfp32`

```sh
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

测速输出：
- CLIP：`[Perf] CLIP ms=...`
- UNet：`[Perf] UNet loop_s=... s/step=...`

### 步骤 B：VAE 单独进程（CLML VAE + MNN Attention）
- 使用上一步输出的 latent
- 必须在 `/data/local/tmp` 下运行以写入 `output/`

```sh
adb shell "cd /data/local/tmp && \
LD_LIBRARY_PATH=/data/local/tmp/MNN_fuse:/system/lib64:/vendor/lib64 \
MNN_CL_LIB=/data/local/tmp/MNN_fuse/libMNN_CL.so \
MNN_BACKEND=opencl MNN_GPU_MODE=1 MNN_MEM=0 MNN_POWER=0 MNN_PREC=0 \
CLML_MNN_ATTN_BACKEND=opencl CLML_MNN_ATTN_FP32=1 CLML_NO_REUSE_TNN=1 \
./sdxl_vae_decoder_run /data/local/tmp/sdxl_clml/sdxl_clml_weights \
1 0 1 0.1 128 128 1 /data/local/tmp/sdxl_latent_clipcpu_early2_x0.qfp32"
```

输出：
- `/data/local/tmp/output/sdxl_vae_out.qfp32`
- VAE 性能：`[Perf] iters=1 warmup=0 s/it=...`

### 步骤 C：把 qfp32 图像转 PNG
- VAE 输出是 `[-1,1]`，需做 `(x/2 + 0.5)` 再保存。

```sh
python3 - <<'PY'
import numpy as np
from PIL import Image
path_in = '/media/happyyzy/Data/pyproject/mlcl_tvm_sd1.5/test_archive/output/sdxl_vae_out_clipcpu_early2_x0.qfp32'
path_out = '/media/happyyzy/Data/pyproject/mlcl_tvm_sd1.5/test_archive/output/sdxl_vae_out_clipcpu_early2_x0.png'
arr = np.fromfile(path_in, dtype=np.float32).reshape(1, 3, 1024, 1024)
img = (arr[0] / 2.0 + 0.5)
img = np.clip(img, 0.0, 1.0)
img = (img.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
Image.fromarray(img).save(path_out)
print(path_out)
PY
```

## SDXL 关键开关说明
- `SDXL_EARLY_DECODE_K=<k>`：UNet 只跑 `num_steps - k`，避免最后几步色块。
- `SDXL_EARLY_DECODE_X0=1`：输出 `x0 = x_t - sigma * eps` 作为解码输入。
- `SDXL_EXPORT_X0=1`：即使 `k=0` 也导出最终步的 `x0`。
- `SDXL_TEXT_EMBEDS` / `SDXL_POOLED_EMBEDS`：设置后跳过 CLIP，直接用预计算 embedding。

## 当前已验证的速度（记录文件在 release/bench/logs）
- SD1.5 512（设备端 20 步）：`total_s=9.19277`，`s/step=0.459639`
- SDXL 1024 UNet-only（20 步，预计算 embedding）：`init_s=65.6627`，`loop_s=64.3368`，`s/step=3.21684`
  - 记录：`release/bench/logs/sdxl_unet_pyclip.log`
- SDXL 1024 UNet 单步：`iters=1`，`s/it=1.61393`
  - 记录：`release/bench/logs/sdxl_unet_single_step.log`
