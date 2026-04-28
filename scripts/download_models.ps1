$ErrorActionPreference = "Stop"

Write-Host "Downloading Qwen2.5-VL checkpoint..."
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir checkpoints/Qwen2.5-VL-7B-Instruct

Write-Host "Downloading DINOv3 checkpoint..."
hf download facebook/dinov3-vits16-pretrain-lvd1689m --local-dir checkpoints/dinov3-vitb16

Write-Host "Done. Install SAM3 separately from external/sam3 after cloning the official repository."

