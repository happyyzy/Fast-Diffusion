# SDXL Experiments Timeline (Workspace Evidence Only)

This file summarizes what can be proven from artifacts in `output/`,
`sdxl_qnn_out/`, and `tmp_qnn_outputs/`. If a step has no matching log/dump,
it is marked as unknown.

## Evidence Sources
- `output/*logcat*.txt`, `output/*_debug_dump*`, `output/*_context_info.json`
- `tmp_qnn_outputs/*/execution_metadata.yaml`
- `sdxl_qnn_out/*` model binaries
- code diff: `local-dream-v231/` vs `local-dream/`

## Model Binaries Generated (UNet)
- 2025-12-31: `sdxl_qnn_out/qnn_base_unet_w8a16_b1_512.bin` (FP->QNN base)
- 2026-01-02 21:11: `sdxl_qnn_out/qnn_base_unet_w8a16_b1_512_realcalib.bin` (2,568,560,640 bytes)
- 2026-01-04 11:33: `sdxl_qnn_out/qnn_base_unet_w8a16_b1_512_realcalib2.bin` (2,568,560,640 bytes)
- 2026-01-04 13:51: `sdxl_qnn_out/qnn_base_unet_w8a16_b1_512_real_dpm.bin` (2,568,560,640 bytes)
- 2026-01-03 17:27: `tmp_backup_sdxl/models/sdxl/unet.bin` (2,606,981,120 bytes)
- 2026-01-05 22:20: `output/unet_device.bin` (copy of device context, same size)
  - Context metadata: `output/unet_device_context_info.json` shows timestep
    scale 1.5259e-05.

## Runs / Outcomes (Ordered by Artifact Time)
1) 2026-01-04 10:58-11:12: `output/sdxl_debug_dump2/`
   - `step0_unet_out.f32` non-constant (min -8.69, max 5.83, mean -0.0530,
     std 1.1151).
   - Input shape `2,4,64,64`, timestep `940`, time_ids `[512,512,0,0,512,512]`
     repeated.
   - No logcat in `output/` for this run; UNet binary unknown.

2) 2026-01-04 20:03-20:04: `output/sdxl_debug_dump/`
   - `step0_unet_out.f32` non-constant (min -8.39, max 5.70, mean -0.0506,
     std 1.1095).
   - Same input shape/timestep as above.
   - No logcat in `output/`; UNet binary unknown.

3) 2026-01-05 18:56: `output/device_base_logcat.txt`
   - SDXL backend command uses `--unet .../sdxl/unet.so`.
   - Graph created: `qnn_base_unet_w8a16_b1_512_real_dpm`.
   - UNet timing ~65-70 ms per step.

4) 2026-01-05 19:19: `output/v231_sdxl_logcat.txt` + `output/v231_debug_dump/`
   - SDXL backend command uses `--unet .../sdxl/unet.bin`.
   - Graph: `qnn_base_unet_w8a16_b1_512_realcalib2`.
   - `step0_unet_out.f32` constant (-3.952423; std 0.0).

5) 2026-01-05 20:25: `output/sdxl_fix1_logcat_20260105_202403.txt`
   - Graph: `qnn_base_unet_w8a16_b1_512_realcalib2`.
   - `output/sdxl_fix1_debug_dump_20260105_202403/step0_unet_out.f32` constant.

6) 2026-01-05 20:38: `output/sdxl_fix2_logcat_20260105_203855.txt`
   - Graph: `qnn_base_unet_w8a16_b1_512_realcalib2`.
   - `output/sdxl_fix2_debug_dump_20260105_203855/step0_unet_out.f32` constant.

7) 2026-01-05 21:52: `output/sd15_debug_dump_20260105_215211/`
   - SD1.5 sanity run after timestep handling fix (not SDXL).

8) 2026-01-06 00:50: device rerun (qnn-net-run, real_dpm)
   - Device outputs pulled to `tmp_qnn_outputs/real_dpm_all_model_new_device/`
     and flattened in `tmp_qnn_outputs/real_dpm_all_model_new_flat/`.
   - VAE output pulled to `tmp_qnn_outputs/vae_new_device_flat/images.raw`.
   - PC comparison report: `output/compare_full_pipeline_real_dpm_device_new.txt`
     (UNet MAE ~1.41e-2, cosine ~0.9997; VAE MAE ~0.526, cosine ~0.825).

9) 2026-01-06 10:24: device VAE variants (qnn-net-run, model .so)
   - Models tested on device:
     - `/data/local/tmp/sdxl_ctx/libqnn_base_vae_decoder_fp16_b1_512.so`
     - `/data/local/tmp/sdxl_ctx/libqnn_base_vae_decoder_w16a16_b1_512_real.so`
     - `/data/local/tmp/sdxl_ctx/libqnn_base_vae_decoder_w8a16_b1_512_real.so`
   - Outputs pulled to:
     - `tmp_qnn_outputs/vae_fp16_device_flat/images.raw`
     - `tmp_qnn_outputs/vae_w16a16_real_device_flat/images.raw`
     - `tmp_qnn_outputs/vae_w8a16_real_device_flat/images.raw`
   - PC comparison report: `output/compare_vae_variants_device.txt`
     - fp16 output contains NaNs.
     - w16a16_real MAE ~0.496, cosine ~0.851.
     - w8a16_real MAE ~0.502, cosine ~0.848.

## qnn-net-run Validation
- `tmp_qnn_outputs/cond/execution_metadata.yaml` and
  `tmp_qnn_outputs/uncond/execution_metadata.yaml` show qnn-net-run with
  `/data/local/tmp/sdxl_ctx/out/sdxl_unet_realcalib2.serialized.bin` and graph
  `qnn_base_unet_w8a16_b1_512_realcalib2`.

## Code Changes Observed (v231 -> current, no timestamps)
- Removed SDXL NCHW<->NHWC conversions in UNet/VAE paths
  (no `nchw_to_nhwc`/`nhwc_to_nchw` usage in current `main.cpp`).
- CLIP input handling changed: detect input type (ids vs embedding),
  clip_v2 embedding support, and prompt processing changes.
- VAE scaling unified to `vae_scaling` with SDXL value 0.13025; tiling overlap
  split into `overlap_x`/`overlap_y`; tiled decode logic updated.
- Extra debug dumps added (e.g., `final_latents` and `final_latents_scaled`).

## Gaps / Missing Links
- No logcat for the two non-constant UNet runs (`sdxl_debug_dump*`), so the
  exact UNet binary for those runs is unknown.
- No archived APK or code snapshot tied to the non-constant UNet output.
- `output/sdxl_fix1_mem_20260105_202403.txt` and
  `output/sdxl_fix2_mem_20260105_203855.txt` contain "No process found", so
  memory data is missing for those runs.
