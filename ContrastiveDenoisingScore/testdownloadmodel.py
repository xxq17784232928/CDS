from diffusers import StableDiffusionPipeline
from pipeline_cds import CDSPipeline
import torch

weight_dtype = torch.float32  # 或 torch.float16 取决于项目需求


# 下载模型
stable = CDSPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=weight_dtype, revision="main", use_safetensors=False)

