<p align="center">
  <img src="assets/logo_fast_diffusion.svg" alt="Fast-Diffusion" width="720">
</p>
<h1 align="center">Fast-Diffusion</h1>
<p align="center">
  Mobile-first CLML diffusion. <strong>Fastest on supported devices</strong> in publicly comparable settings.<br/>
  SDXL 1024 on phones with a practical end-to-end pipeline.
</p>

## Highlights
- CLML backend optimized for **fast on-device inference**
- SD1.5 512 (single-process) and SDXL 1024 (multi-process, recommended)
- Early-decoded `x0` flow for cleaner SDXL results
- Includes app source, APK, and bench binaries

---

## Gallery (SDXL 1024, CLML)

**Same prompt/seed/CFG**. Left is early decode (`k=2`, `x0`), right is the default final-step decode.

**20 steps**
| Early decode (k=2, x0) | Final-step decode (x0) |
| --- | --- |
| <img src="assets/gallery/sdxl_clml_early2_x0_1024.png" alt="sdxl early k2 x0 20 steps" width="512"><br/><sub>SM8750 • steps=20 • k=2 • s/it=3.21684 (CFG UNet)</sub> | <img src="assets/gallery/sdxl_clml_final_x0_1024.png" alt="sdxl final x0 20 steps" width="512"><br/><sub>SM8750 • steps=20 • final • s/it=3.21684 (CFG UNet)</sub> |

**25 steps**
| Early decode (k=2, x0) | Final-step decode (x0) |
| --- | --- |
| <img src="assets/gallery/sdxl_clml_k2_x0_25steps.png" alt="sdxl early k2 x0 25 steps" width="512"><br/><sub>SM8750 • steps=25 • k=2 • s/it=3.21684 (CFG UNet)</sub> | <img src="assets/gallery/sdxl_clml_final_x0_25steps.png" alt="sdxl final x0 25 steps" width="512"><br/><sub>SM8750 • steps=25 • final • s/it=3.21684 (CFG UNet)</sub> |

**30 steps**
| Early decode (k=2, x0) | Final-step decode (x0) |
| --- | --- |
| <img src="assets/gallery/sdxl_clml_k2_x0_30steps.png" alt="sdxl early k2 x0 30 steps" width="512"><br/><sub>SM8750 • steps=30 • k=2 • s/it=3.21684 (CFG UNet)</sub> | <img src="assets/gallery/sdxl_clml_final_x0_30steps.png" alt="sdxl final x0 30 steps" width="512"><br/><sub>SM8750 • steps=30 • final • s/it=3.21684 (CFG UNet)</sub> |

**Step snapshots (20 steps, x0)**
| Step 17 (k=2) | Step 18 (k=1) | Step 19 (final) |
| --- | --- | --- |
| <img src="assets/gallery/sdxl_clml_x0_step17.png" alt="sdxl x0 step17" width="360"><br/><sub>SM8750 • steps=20 • step=17 • s/it=3.21684 (CFG UNet)</sub> | <img src="assets/gallery/sdxl_clml_x0_step18.png" alt="sdxl x0 step18" width="360"><br/><sub>SM8750 • steps=20 • step=18 • s/it=3.21684 (CFG UNet)</sub> | <img src="assets/gallery/sdxl_clml_x0_step19.png" alt="sdxl x0 step19" width="360"><br/><sub>SM8750 • steps=20 • step=19 • s/it=3.21684 (CFG UNet)</sub> |

---

## Gallery (SD1.5 512, CLML)

| Portrait 2 | Non-portrait (cityscape) |
| --- | --- |
| <img src="assets/gallery/sd15_clml_portrait2.png" alt="sd15 portrait2" width="360"><br/><sub>SM8750 • steps=20 • s/step=0.459639 (CFG)</sub> | <img src="assets/gallery/sd15_clml_sample_cityscape.png" alt="sd15 cityscape" width="360"><br/><sub>SM8750 • steps=20 • s/step=0.459639 (CFG)</sub> |

---

## Performance (CFG-aligned, with explicit records)

Our measured results (full records are included in this repo):
- **SD1.5 512 (CFG)**: steps=20, total_s=9.19277, s/step=0.459639  
  - Record: `release/sd_pipelines_zh.md`
- **SDXL 1024 UNet-only (CFG)**: init_s=65.6627, loop_s=64.3368, s/step=3.21684 (20 steps, precomputed embeddings)  
  - Record: `release/bench/logs/sdxl_unet_pyclip.log`
- **SDXL 1024 UNet single pass (no CFG)**: iters=1, s/it=1.61393  
  - Record: `release/bench/logs/sdxl_unet_single_step.log`

Public comparison baselines (CFG enabled or equivalent):
1. CVPR 2023 (Google LLC), Adreno 740, SD1.4, 20 steps: **11.5s**  
   - Paper: https://arxiv.org/abs/2304.11267
2. Local Diffusion app (author report), Snapdragon 8 Gen 3, SD1.5: **8 s/it**, GPU slower than CPU  
   - App: https://github.com/rmatif/Local-Diffusion  
   - Backend: https://github.com/leejet/stable-diffusion.cpp
3. Local Dream (MNN GPU backend), Snapdragon 8 Elite, SD1.5, 20 steps: **52s**
4. T4 baseline for SDXL 1024: **1.2 s/it** (CFG enabled)

**Note on T4:** 1.2 s/it includes CFG (two UNet passes). Normalized to a single pass: 0.6 s/it.  
For comparison, our SDXL UNet single-pass record is **1.61393 s/it**, and CFG step is **3.21684 s/step**.

Conclusion highlights:
- **Fastest in publicly comparable settings** on CLML-supported phones
- **SDXL 1024 on mobile is practical here for the first time**
- Per-step speed is remarkable for a phone-class device

---

## Roadmap

- Add SDXL Base/Turbo support at 512 and 768 resolutions
- On 16GB RAM devices, enable optional UNet pre-init for SDXL
- Target: SDXL Turbo 768 on SM8750 generates a high-quality image in under 10 seconds (quality/perf balance)

---

## Requirements

- Qualcomm Adreno GPU
- OpenCL device extension: `cl_qcom_ml_ops`
- CLML SDK (for building)
- MNN with Attention HostOp enabled
  - Example build flags: `MNN_SUPPORT_TRANSFORMER_FUSE=ON`

Runtime notes:
- Always set `CLML_NO_REUSE_TNN=1` (TNN reuse causes numerical instability)
- SDXL VAE requires CLML VAE + MNN Attention HostOp
- SDXL 1024 needs 16GB RAM even without pre-init; pre-init with CLIP + UNet co-resident OOMs
- SD1.5 does not have this issue and supports pre-init for smoother UX

## SDK Versions

- **CLML SDK**: v4.1 (cl_qcom_ml_ops)
- **QNN/SNPE SDK**: 2.39 (used for SoC table source)
- **MNN**: 3.3.0 custom build with Attention HostOp enabled (Transformer Fuse)

## Supported SoCs (from Qualcomm QNN/SNPE SDK table)

Source: `QNN_SDK_2.39/qairt/2.39.0.250926/docs/SNPE/html/general/overview.html`

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

## Repository Layout (Release)

- `app/sdxl-clml/` - Android app source
- `release/` - release artifacts
  - `release/app/sdxl-clml-debug.apk`
  - `release/bench/` (binaries + source)
  - `release/sd_pipelines_zh.md` (full pipeline notes, Chinese)

---

## Quick Start (SD1.5 512)

```sh
adb push release/bench/sd15_pipeline_run /data/local/tmp/sd15_pipeline_run
adb push -r <sd15_clml_weights_dir> /data/local/tmp/sd15_clml/

adb shell "CLML_NO_REUSE_TNN=1 /data/local/tmp/sd15_pipeline_run /data/local/tmp/sd15_clml/sd15_clml_weights 20"
```

Output: `/data/local/tmp/output/clml_stable_diffusion_output.qfp32`

---

## Quick Start (SDXL 1024, recommended)

Memory note:
- SDXL 1024 needs 16GB RAM even without pre-init
- Pre-init with CLIP + UNet co-resident OOMs; app does not pre-init SDXL

### 1) Generate CLIP token ids (host)
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

### 2) UNet process (CPU CLIP + CLML UNet)
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

### 3) VAE process (CLML VAE + MNN Attention)
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

### 4) Convert qfp32 to PNG (host)
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

- Source: `app/sdxl-clml/`
- APK: `release/app/sdxl-clml-debug.apk`
- Features: SDXL 1024 + SD1.5 512, steps, CFG, scheduler, early decode, decode x0, seed, prompt/negative prompt

---

## Weights

Weights are hosted on HuggingFace (public):  
https://huggingface.co/zhiyuanasad/fast-diffusion-weights

- SD1.5 weights: https://huggingface.co/zhiyuanasad/fast-diffusion-weights/tree/main/sd15_clml_weights  
- SDXL weights: https://huggingface.co/zhiyuanasad/fast-diffusion-weights/tree/main/sdxl_clml_weights

---

## Acknowledgements

- Qualcomm CLML SDK (cl_qcom_ml_ops)
- MNN SDK and runtime

---

## Notes

- Full pipeline records are in `release/sd_pipelines_zh.md` (Chinese)
- Chinese release doc: `release/README_zh.md`
- For reproducibility, ensure CLML and MNN runtime libraries match the expected build options
