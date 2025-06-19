import torch
import os
import time
import psutil
from tqdm import tqdm
import gc
import torch.nn as nn
import onnx
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import onnxruntime as ort
from transformers import CLIPTokenizer, CLIPTextModel,logging
from diffusers import StableDiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler,UNet2DConditionModel,DPMSolverSinglestepScheduler
from optimum.exporters.onnx import export,main_export
from optimum.exporters.onnx.model_configs import (
    CLIPTextOnnxConfig,
    UNetOnnxConfig,
    VaeDecoderOnnxConfig,
)
from optimum.onnxruntime import ORTQuantizer, ORTModel
from optimum.onnxruntime.configuration import AutoQuantizationConfig,QuantizationConfig
from onnx import load_model,save_model
from onnx.external_data_helper import convert_model_from_external_data
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
    QuantizationMode,
    StaticQuantConfig,
    quantize_dynamic
)
from onnxruntime.quantization import quant_pre_process, write_calibration_table

# Set model and output directory
model_id = "CompVis/stable-diffusion-v1-4"
onnx_output_dir = Path.cwd() / "stable_diffusion_onnx_optimum"
onx_quant_output_dir = onnx_output_dir / "quantized"
onx_quant_output_dir.mkdir(parents=True, exist_ok=True)

print("⚙️ Loading Stable Diffusion pipeline in fp32 and reducing UNet resolution to 64x64...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")
pipe.unet.config.sample_size = 64

# Tokenizer and Scheduler
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
num_inference_steps = 30
scheduler.set_timesteps(num_inference_steps)

# Export ONNX Models
opset_version = 15
text_encoder_file = onnx_output_dir / "text_encoder" / "text_encoder.onnx"
unet_file = onnx_output_dir / "unet" /"unet.onnx"
vae_file = onnx_output_dir / "vae" /"vae.onnx"

def export_unet():
    if unet_file.exists():
        print("UNet ONNX already exists, skipping export.")
        return
    print("Exporting UNet ONNX...")
    unet_file.parent.mkdir(parents=True, exist_ok=True)
    model = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    model.eval()
    sample = torch.randn(1, 4, 64, 64)
    timestep = torch.tensor([1.0])
    encoder_hidden_states = torch.randn(1, 77, 768)
    torch.onnx.export(
        model,
        (sample, timestep, encoder_hidden_states),
        str(unet_file),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["out_sample"],
        dynamic_axes={
            "sample": {0: "batch"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "out_sample": {0: "batch"},
        },
        opset_version=16,
        external_data=True,
    )
    print("Exported UNet ONNX.")


class PatchedUNetOnnxConfig(UNetOnnxConfig):
    @property
    def inputs(self):
        # Copy from original class if needed, or override safely
        return super().inputs

    @property
    def text_encoder_projection_dim(self):
        # Skip access to the missing attribute
        return None  # Or raise NotImplementedError if necessary

if not text_encoder_file.exists():
    print("📤 Exporting Text Encoder...")
    export(pipe.text_encoder.eval(), CLIPTextOnnxConfig(pipe.text_encoder.config), text_encoder_file, opset=opset_version)
    gc.collect()
if not unet_file.exists():
    print("📤 Exporting UNet...")
    # export(pipe.unet.to(torch.float32), UNetOnnxConfig(pipe.unet.config), unet_file, opset=17)
    # export(pipe.unet.to(torch.float32), PatchedUNetOnnxConfig(pipe.unet.config), unet_file, opset=opset_version)
    
   
    export_unet()
    gc.collect()

if not vae_file.exists():
    print("📤 Exporting VAE Decoder...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to("cpu")
    class VaeDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.post_quant_conv = vae.post_quant_conv
            self.decoder = vae.decoder
            self.config = vae.config
        def forward(self, latent_sample):
            return self.decoder(self.post_quant_conv(latent_sample))
    vae_decoder = VaeDecoder(vae)
    export(vae_decoder.to(torch.float32), VaeDecoderOnnxConfig(vae.config), vae_file, opset=opset_version)
    del vae, vae_decoder
    gc.collect()



text_encoder_test = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder_test.eval()
def get_encoder_hidden_states(prompt: str):
    inputs = tokenizer([prompt], return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    with torch.no_grad():
        
        outputs = text_encoder_test(**inputs)
    return outputs.last_hidden_state.numpy().astype(np.float32)  # Shape: [1, 77, 768]
# === DUMMY DATA READER ===
def generate_synthetic_unet_dataset(batch_size=1, latent_shape=(4, 64, 64), steps=1):
    import torch
    for _ in range(steps):
        latents = torch.randn(batch_size, *latent_shape)
        timestep = torch.randint(1, 1000, (batch_size,)).long()
        encoder_hidden_states = torch.randn(batch_size, 77, 768)
        print(f"[Dataset] Step {_}: latents {latents.shape}, timestep {timestep.shape}, encoder_hidden_states {encoder_hidden_states.shape}")
        yield {"sample": latents, "timestep": timestep, "encoder_hidden_states": encoder_hidden_states}

def generate_synthetic_vae_dataset(batch_size=1, shape=(1, 4, 64, 64), steps=2):
    import torch
    for _ in range(steps):
        yield {"latent_sample": torch.randn(*shape)}

def generate_synthetic_text_encoder_dataset(batch_size=1, seq_len=77, steps=2):
    import torch
    for _ in range(steps):
        yield {"input_ids": torch.randint(0, 10000, (batch_size, seq_len), dtype=torch.int32)}

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, CalibrationMethod

class UNetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader):
        self.enum_data = iter(dataloader)
        self._cache = None

    def get_next(self):
        if self._cache is None:
            try:
                self._cache = next(self.enum_data)
            except StopIteration:
                return None
        result = self._cache
        self._cache = None

        # Convert all tensors to float32 regardless of original dtype
        converted = {
        k: v.detach().cpu().numpy().astype(np.float32)
        for k, v in result.items()
        }
        for k, v in converted.items():
            print(f"[DataReader] Input {k} shape: {v.shape}")
        return converted
    

class GenericCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader):
        self.enum_data = iter(dataloader)
        self._cache = None

    def get_next(self):
        if self._cache is None:
            try:
                self._cache = next(self.enum_data)
            except StopIteration:
                return None
        result = self._cache
        self._cache = None

        # Cast each input to appropriate dtype
        input_dict = {}
        for k, v in result.items():
            if v.dtype == torch.int64:  # force int32 if needed
                v = v.to(dtype=torch.int32)
            input_dict[k] = v.numpy()
        print(f"[DataReader] Input {list(input_dict.keys())} shape: {[v.shape for v in input_dict.values()]}, dtype: {[v.dtype for v in input_dict.values()]}")
        return input_dict

   

def quantize_model_statically(model_path, output_path, dataloader_fn):
    if output_path.exists():
        print(f"Quantized model already exists at {output_path}. Skipping.")
        return

    print(f"Quantizing model at {model_path.name} using static QDQ format...")
    if 'text_encoder' in model_path.name:
        data_reader=GenericCalibrationDataReader(dataloader_fn())
    else:
        data_reader = UNetCalibrationDataReader(dataloader_fn())

    quantize_static(
        model_input=str(model_path),
        model_output=str(output_path),
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=True
    )

    print(f"✅ Quantized model saved to: {output_path}")

def quantize_model_dynamic():

    quantize_dynamic(
    model_input=str(unet_file),
    model_output=str(onx_quant_output_dir / "unet" / "unet_quantized.onnx"),
    weight_type=QuantType.QInt8,
    use_external_data_format=True  # if original model used it
    )


gc.collect()
# model = load_model(onnx_output_dir / "unet" / "unet.onnx", load_external_data=True)
# convert_model_from_external_data(model)
# onnx.save_model(model, "unet_merged.onnx")


if not (onx_quant_output_dir / "unet" / "unet_quantized.onnx").exists():
    print("🔧 Quantizing UNet...")
    (onx_quant_output_dir / "unet").mkdir(parents=True, exist_ok=True)
    # quantize_model_statically(
    #     onnx_output_dir / "unet" /"unet.onnx",
    #     onx_quant_output_dir / "unet" / "unet_quantized.onnx",
    #     generate_synthetic_unet_dataset
    # )
    quantize_model_dynamic()
    gc.collect()

if not (onx_quant_output_dir / "vae_decoder" / "vae_quantized.onnx").exists():
    print("🔧 Quantizing VAE Decoder...")
    (onx_quant_output_dir / "vae_decoder").mkdir(parents=True, exist_ok=True)
    quantize_model_statically(
        vae_file,
        onx_quant_output_dir / "vae_decoder" / "vae_quantized.onnx",
        generate_synthetic_vae_dataset
    )
    gc.collect()

if not (onx_quant_output_dir / "text_encoder" / "text_encoder_quantized.onnx").exists():
    print("🔧 Quantizing Text Encoder...")
    (onx_quant_output_dir / "text_encoder").mkdir(parents=True, exist_ok=True)
    quantize_model_statically(
        text_encoder_file,
        onx_quant_output_dir / "text_encoder" / "text_encoder_quantized.onnx",
        generate_synthetic_text_encoder_dataset
    )
    gc.collect()


# unet_quant_path = onx_quant_output_dir / "unet" / "unet_quantized.onnx"
# model = load_model(str(unet_quant_path))
# op_types = set(node.op_type for node in model.graph.node)
# print(f"Quantized UNet ops: {op_types}")

# Load ONNX Runtime sessions
def load_sessions(folder):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu_cores = os.cpu_count() or 1
    opts.intra_op_num_threads = cpu_cores
    opts.inter_op_num_threads = cpu_cores
    providers = ["CPUExecutionProvider"]
    return [
        ort.InferenceSession(str(folder / "text_encoder" /"text_encoder.onnx"), opts, providers),
        ort.InferenceSession(str(folder / "unet" / "unet.onnx"), opts, providers),
        ort.InferenceSession(str(folder / "vae" / "vae.onnx"), opts, providers),
    ]

# Load ONNX Runtime sessions
def load_quantized_sessions(folder):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu_cores = os.cpu_count() or 1
    opts.intra_op_num_threads = cpu_cores
    opts.inter_op_num_threads = cpu_cores
    providers = ["CPUExecutionProvider"]
    return [
        ort.InferenceSession(str(folder / "text_encoder" / "text_encoder_quantized.onnx"), opts, providers),
        ort.InferenceSession(str(onnx_output_dir / "unet" / "unet.onnx"), opts, providers),
        ort.InferenceSession(str(folder / "vae_decoder" / "vae_quantized.onnx"), opts, providers),
    ]


prompt = "A scenic view of a beach at sunset with a palm tree"
latent_shape = (1, 4, 64, 64)

# Inference function
def run_pipeline(sessions, label):
    print(f"🚀 Running inference: {label}")
    text_encoder_sess, unet_sess, vae_decoder_sess = sessions

    # Recreate scheduler for each run to avoid state carryover
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
    )
    scheduler.set_timesteps(30)

    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=77, truncation=True)
    uncond_inputs = tokenizer([""], return_tensors="np", padding="max_length", max_length=77, truncation=True)

    text_emb = text_encoder_sess.run(None, {text_encoder_sess.get_inputs()[0].name: inputs["input_ids"].astype(np.int32)})[0].astype(np.float32)
    uncond_emb = text_encoder_sess.run(None, {text_encoder_sess.get_inputs()[0].name: uncond_inputs["input_ids"].astype(np.int32)})[0].astype(np.float32)
    text_embeddings = np.concatenate([uncond_emb, text_emb], axis=0)

    latents = np.random.randn(*latent_shape).astype(np.float32) * scheduler.init_noise_sigma
    latents = np.concatenate([latents, latents], axis=0)

    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    start_time = time.time()

    # Denoising loop (correct loop over scheduler.timesteps)
    for i, t in enumerate(scheduler.timesteps):
        # t_array = np.array(t * latents.shape[0], dtype=np.float32)
        t_array = np.full((latents.shape[0],), t, dtype=np.float32)

        unet_inputs = {
            unet_sess.get_inputs()[0].name: latents,
            unet_sess.get_inputs()[1].name: t_array,
            unet_sess.get_inputs()[2].name: text_embeddings,
        }

        noise_pred = unet_sess.run(None, unet_inputs)[0].astype(np.float32)
        gc.collect()

        uncond_pred, text_pred = np.split(noise_pred, 2, axis=0)
        guided_pred = uncond_pred + 10.0 * (text_pred - uncond_pred)
        del noise_pred, uncond_pred, text_pred
        gc.collect()

        latents_torch = torch.from_numpy(latents[:1])
        guided_pred_torch = torch.from_numpy(guided_pred)
        t_tensor = torch.tensor(int(t))

        latents_torch = scheduler.step(guided_pred_torch, t_tensor, latents_torch)["prev_sample"]
        latents = np.concatenate([latents_torch.numpy(), latents_torch.numpy()], axis=0)

        del latents_torch, guided_pred_torch, t_tensor, guided_pred
        gc.collect()

    denoising_time = time.time() - start_time
    end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    memory_used = end_mem - start_mem

    latents = latents[:1]
    print("Memory (MB):", psutil.Process().memory_info().rss / 1024 ** 2)
    decoded = vae_decoder_sess.run(None, {vae_decoder_sess.get_inputs()[0].name: latents})[0]
    decoded = np.clip((decoded + 1) / 2, 0, 1)
    image = (decoded[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    pil_image = Image.fromarray(image).filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))

    path = onnx_output_dir / f"output_{label}.png"
    pil_image.save(path)
    print(f"✅ {label} image saved at: {path}")
    gc.collect()
    return denoising_time, memory_used, path

def get_model_size(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)  # MB
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob("*.onnx")) / (1024 * 1024)
    else:
        return 0.0

# Run and benchmark
base_sess = load_sessions(onnx_output_dir)
gc.collect()
base_time, base_mem, base_path = run_pipeline(base_sess, "base")
del base_sess
gc.collect()
quant_sess = load_quantized_sessions(onx_quant_output_dir)
gc.collect()
quant_time, quant_mem, quant_path = run_pipeline(quant_sess, "quantized")
del quant_sess
gc.collect()


print("\n📊 Benchmark Results:")
print(f"Base UNet 64x64 Denoising Time:   {base_time:.2f}s")
print(f"Quantized UNet 64x64 Time:        {quant_time:.2f}s")
print(f"Speedup:                          {base_time / quant_time:.2f}x")
print(f"Base Peak Memory Used:           {base_mem:.2f} MB")
print(f"Quantized Peak Memory Used:      {quant_mem:.2f} MB")
print(f"Image Outputs:                   \n  Base: {base_path}\n  Quant: {quant_path}")

print("\n📦 Model Size Comparison (in MB):")
base_size = (
    get_model_size(onnx_output_dir / "text_encoder" /"text_encoder.onnx")
    + get_model_size(onnx_output_dir / "unet" / "unet.onnx")
    + get_model_size(onnx_output_dir / "vae" / "vae.onnx")
)
quant_size = (
    get_model_size(onx_quant_output_dir / "text_encoder" / "text_encoder_quantized.onnx")
    + get_model_size(onx_quant_output_dir / "unet" / "unet_quantized.onnx")
    + get_model_size(onx_quant_output_dir / "vae_decoder" / "vae_quantized.onnx")
)

print(f"Base ONNX Models Total Size:      {base_size:.2f} MB")
print(f"Quantized ONNX Models Size:       {quant_size:.2f} MB")
print(f"Compression Ratio:                {base_size / quant_size:.2f}x smaller")
