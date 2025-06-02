import torch
import torch.nn as nn
import time
import psutil
import os
from pathlib import Path
from diffusers import StableDiffusionPipeline, AutoencoderKL
from optimum.exporters.onnx import export
from optimum.exporters.onnx.model_configs import (
    CLIPTextOnnxConfig,
    UNetOnnxConfig,
    VaeDecoderOnnxConfig,
)
from transformers import CLIPProcessor, CLIPModel
from torch.profiler import profile, ProfilerActivity, record_function
from PIL import Image

# Set model and output directory
model_id = "CompVis/stable-diffusion-v1-4"
onnx_output_dir = Path.cwd() / "stable_diffusion_onnx_optimum"
onnx_output_dir.mkdir(parents=True, exist_ok=True)

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")

prompt = "A scenic view of a beach at sunset with a palm tree"

# 🧪 Profile Inference
print("🧪 Running inference with profiling...")
start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

# Profiled Inference Timing (use wall-clock for total latency)
start_time = time.time()
start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    with_stack=True,
    profile_memory=True
) as prof:
    with record_function("stable_diffusion_inference"):
        image = pipe(prompt).images[0]

end_time = time.time()
end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

image_path = onnx_output_dir / "sample_output3.png"
image.save(image_path)

# ✅ Fixed: use wall time for latency
inference_latency = (end_time - start_time) * 1000  # in milliseconds
mem_used = end_mem - start_mem

# ✅ Op-level breakdown
top_ops = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)

# 🔎 CLIP Score
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    clip_score = clip_model(**clip_inputs).logits_per_image[0].item()

# 📤 Export ONNX models (only if they don't exist)

# 1. Text Encoder
text_encoder_file = onnx_output_dir / "text_encoder" / "text_encoder.onnx"
if not text_encoder_file.exists():
    print("📤 Exporting Text Encoder...")
    text_encoder_file.parent.mkdir(parents=True, exist_ok=True)
    text_encoder_config = CLIPTextOnnxConfig(pipe.text_encoder.config)
    export(model=pipe.text_encoder, config=text_encoder_config, output=text_encoder_file, opset=14)
else:
    print("✅ Text Encoder already exists. Skipping export.")

# 2. UNet
unet_file = onnx_output_dir / "unet" / "unet.onnx"
if not unet_file.exists():
    print("📤 Exporting UNet...")
    unet_file.parent.mkdir(parents=True, exist_ok=True)
    unet_config = UNetOnnxConfig(pipe.unet.config)
    export(model=pipe.unet, config=unet_config, output=unet_file, opset=14)
else:
    print("✅ UNet already exists. Skipping export.")

# 3. VAE Decoder
vae_file = onnx_output_dir / "vae_decoder" / "vae.onnx"
if not vae_file.exists():
    print("📤 Exporting VAE Decoder...")
    vae_file.parent.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to("cpu")

    class VaeDecoder(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.post_quant_conv = vae.post_quant_conv
            self.decoder = vae.decoder
            self.config = vae.config

        def forward(self, latent_sample):
            return self.decoder(self.post_quant_conv(latent_sample))

    vae_decoder = VaeDecoder(vae)
    vae_decoder_config = VaeDecoderOnnxConfig(vae.config)
    export(model=vae_decoder, config=vae_decoder_config, output=vae_file, opset=14)
else:
    print("✅ VAE Decoder already exists. Skipping export.")

# 📊 Final Summary
print("\n📊 Performance Summary:")
print(f"Prompt: {prompt}")
print(f"Inference Time (profiled): {inference_latency:.3f} ms")
print(f"Memory Used: {mem_used:.2f} MB")
print(f"CLIP Score: {clip_score:.4f}")
print(f"Sample Output: {image_path}")
print(f"ONNX Models Saved In: {onnx_output_dir}")

print("\n🔥 Top Time-Consuming Ops:")
print(top_ops)