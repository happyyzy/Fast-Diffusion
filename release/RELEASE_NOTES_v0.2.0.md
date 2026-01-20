Fast-Diffusion v0.2.0

Changes
- Added runtime checks for OpenCL/CLML and device RAM; SDXL 1024 is auto-disabled when requirements are not met.
- Added missing-only model downloads from HuggingFace with SHA256 verification and a Stop Download button.
- Improved UI layout for environment status, downloads, and action buttons.
- Adjusted SDXL RAM gate to allow devices reporting ~14.5GB.
