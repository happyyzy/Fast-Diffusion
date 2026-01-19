# Bench 程序（Android 可执行）

本目录提供已编译的 Android 可执行程序与对应源码，便于快速复现测速。

## 程序列表（可执行）
- `sd15_pipeline_run`：SD1.5 512 单进程端到端
- `sdxl_pipeline_run`：SDXL 1024 UNet/CLIP 管线（支持 early decode）
- `sdxl_vae_decoder_run`：SDXL VAE 单进程解码

## 源码位置
`release/bench/src/`

- `sd15_pipeline_run.cpp`
- `sdxl_pipeline_run.cpp`
- `sdxl_vae_decoder_run.cpp`

## 运行说明
完整运行命令与环境变量说明请直接查看：
`test_archive/output/sd_pipelines_zh.md`
