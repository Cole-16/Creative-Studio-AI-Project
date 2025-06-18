import os
import numpy as np
import torch
import gc
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageFilter
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPTokenizer,SamModel,SamProcessor
from transformers.onnx import FeaturesManager, export as export_onnx
from diffusers import StableDiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from optimum.exporters.onnx import export
from optimum.exporters.onnx.model_configs import CLIPTextOnnxConfig, UNetOnnxConfig, VaeDecoderOnnxConfig
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# --- Configuration ---
gpt2_model_id = "openai-community/gpt2-medium"
sd_model_id = "CompVis/stable-diffusion-v1-4"
onnx_dir = Path("onnx")
onnx_dir.mkdir(parents=True, exist_ok=True)
gpt2_onnx_path = onnx_dir / "gpt2" / "model.onnx"
text_encoder_path = onnx_dir / "text_encoder" / "text_encoder.onnx"
unet_path = onnx_dir / "unet" / "unet.onnx"
vae_path = onnx_dir / "vae_decoder" / "vae.onnx"

onnx_dir.mkdir(parents=True, exist_ok=True)
opset = 14

# # --- Step 1: Prompt for user keyword ---
# keyword = input("Enter a word or concept to generate an image of: ").strip()

# # --- Step 2: Load/export GPT-2 ONNX model ---
# tokenizer = AutoTokenizer.from_pretrained(gpt2_model_id)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# if not gpt2_onnx_path.exists():
#     print("📦 Exporting GPT-2 to ONNX...")
#     gpt2_onnx_path.parent.mkdir(parents=True, exist_ok=True)
#     model = AutoModelForCausalLM.from_pretrained(gpt2_model_id)
#     feature = "causal-lm"
#     model_kind, onnx_config_class = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
#     onnx_config = onnx_config_class(model.config)
#     export_onnx(preprocessor=tokenizer, model=model, config=onnx_config, opset=opset, output=gpt2_onnx_path)

# gpt2_sess = ort.InferenceSession(str(gpt2_onnx_path), providers=["CPUExecutionProvider"])

# # --- Step 3: Run GPT-2 inference for descriptions ---
# template = """Keyword: forest  
# Description: A mystical forest with golden sunlight and ancient trees.

# Keyword: ocean  
# Description: A vast ocean with a pastel sunset and gentle waves.

# Keyword: desert  
# Description: A golden desert with rolling sand dunes and a clear blue sky.

# Keyword: {keyword}  
# Description:""".format(keyword=keyword)

# batch_prompts = [template] * 3
# inputs = tokenizer(batch_prompts, return_tensors="np", padding=True)
# input_ids = inputs["input_ids"].astype(np.int64)
# attention_mask = inputs["attention_mask"].astype(np.int64)
# batch_size = input_ids.shape[0]

# def nucleus_sample(logits, p=0.9, temperature=0.8):
#     logits = logits / temperature
#     sorted_indices = np.argsort(logits)[::-1]
#     sorted_logits = logits[sorted_indices]
#     probs = np.exp(sorted_logits - np.max(sorted_logits))
#     probs /= np.sum(probs)
#     cum_probs = np.cumsum(probs)
#     cutoff = np.searchsorted(cum_probs, p) + 1
#     selected_indices = sorted_indices[:cutoff]
#     selected_probs = probs[:cutoff]
#     selected_probs /= selected_probs.sum()
#     return np.random.choice(selected_indices, p=selected_probs)

# def extract_gpt2_description(text: str, keyword: str) -> str:
#     keyword_tag = f"Keyword: {keyword}"
#     start_idx = text.rfind(keyword_tag)
#     if start_idx == -1:
#         return "[Keyword not found]"
    
#     segment = text[start_idx:]
#     desc_start = segment.find("Description:")
#     if desc_start == -1:
#         return "[Description not found]"
    
#     desc_segment = segment[desc_start + len("Description:"):]

#     # Optional: stop at next "Keyword:" or just take first sentence
#     keyword_next = desc_segment.find("Keyword:")
#     if keyword_next != -1:
#         desc_segment = desc_segment[:keyword_next]

#     first_sentence = desc_segment.strip().split(".")[0].strip()
#     return first_sentence + "."

# max_new_tokens = 40
# generated = input_ids.copy()
# for step in range(max_new_tokens):
#     ort_inputs = {"input_ids": generated.astype(np.int64), "attention_mask": np.ones_like(generated,dtype=np.int64)}
#     logits = gpt2_sess.run(None, ort_inputs)[0]
#     next_tokens = [nucleus_sample(logits[i, -1]) for i in range(batch_size)]
#     generated = np.concatenate([generated, np.array(next_tokens).reshape(batch_size, 1)], axis=1)
#     if all(t == tokenizer.eos_token_id for t in next_tokens):
#         break

# print("\n📝 GPT-2 Outputs:")
# descriptions = []
# for i, seq in enumerate(generated):
#     decoded = tokenizer.decode(seq, skip_special_tokens=True)
#     desc = extract_gpt2_description(decoded, keyword)
#     # desc = decoded.split("Keyword:")[-1].strip().split("Description:")[-1].split(".")[0].strip() + "."
#     descriptions.append(desc)
#     print(f"{i + 1}. {desc}")



# # --- Step 4: Let user select a description ---
# choice = int(input("\nChoose one of the outputs (1-3): ").strip()) - 1
# selected_prompt = descriptions[choice]
# print(f"\n🖼️ Using: \"{selected_prompt}\" as the image prompt.")

# # After GPT-2 inference is done
# del gpt2_sess, inputs, input_ids, attention_mask, generated, batch_prompts
# gc.collect()

# # --- Step 5: Load and export SD ONNX components if needed ---
# pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float32).to("cpu")
# pipe.unet.config.sample_size = 64  # Reduce UNet latent resolution

# if not text_encoder_path.exists():
#     text_encoder_path.parent.mkdir(parents=True, exist_ok=True)
#     export(pipe.text_encoder.eval(), CLIPTextOnnxConfig(pipe.text_encoder.config), text_encoder_path, opset=15)
# if not unet_path.exists():
#     unet_path.parent.mkdir(parents=True, exist_ok=True)
#     export(pipe.unet.to(torch.float32), UNetOnnxConfig(pipe.unet.config), unet_path, opset=15)
# if not vae_path.exists():
#     vae_path.parent.mkdir(parents=True, exist_ok=True)
#     vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", torch_dtype=torch.float32).to("cpu")
#     class VaeDecoder(torch.nn.Module):
#         def __init__(self, vae):
#             super().__init__()
#             self.post_quant_conv = vae.post_quant_conv
#             self.decoder = vae.decoder
#             self.config = vae.config
#         def forward(self, latent_sample):
#             return self.decoder(self.post_quant_conv(latent_sample))
#     export(VaeDecoder(vae).to(torch.float32), VaeDecoderOnnxConfig(vae.config), vae_path, opset=15)

# # After all SD ONNX exports
# del pipe
# if 'vae' in locals():
#     del vae
# gc.collect()

# # --- Image Generation ---
# text_encoder_sess = ort.InferenceSession(str(text_encoder_path), providers=["CPUExecutionProvider"])
# unet_sess = ort.InferenceSession(str(unet_path), providers=["CPUExecutionProvider"])
# vae_decoder_sess = ort.InferenceSession(str(vae_path), providers=["CPUExecutionProvider"])
# scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_model_id, subfolder="scheduler")
# scheduler.set_timesteps(50)

# tokenizer_clip = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# input_ids = tokenizer_clip([selected_prompt], return_tensors="np", padding="max_length", max_length=77)["input_ids"]
# uncond_ids = tokenizer_clip([""], return_tensors="np", padding="max_length", max_length=77)["input_ids"]
# text_emb = text_encoder_sess.run(None, {text_encoder_sess.get_inputs()[0].name: input_ids.astype(np.int64)})[0].astype(np.float32)
# uncond_emb = text_encoder_sess.run(None, {text_encoder_sess.get_inputs()[0].name: uncond_ids.astype(np.int64)})[0].astype(np.float32)
# text_embeddings = np.concatenate([uncond_emb, text_emb], axis=0)

# latents = np.random.randn(1, 4, 64, 64).astype(np.float32) * scheduler.init_noise_sigma
# latents = np.concatenate([latents, latents], axis=0)

# for t in tqdm(scheduler.timesteps, desc="Denoising"):
#     inputs = {
#         unet_sess.get_inputs()[0].name: latents,
#         unet_sess.get_inputs()[1].name: np.array(t, dtype=np.float32),
#         unet_sess.get_inputs()[2].name: text_embeddings,
#     }
#     noise_pred = unet_sess.run(None, inputs)[0]
#     uncond, cond = np.split(noise_pred, 2, axis=0)
#     guided = uncond + 10.0 * (cond - uncond)
#     latents_torch = torch.from_numpy(latents[:1])
#     guided_torch = torch.from_numpy(guided)
#     latents_torch = scheduler.step(guided_torch, torch.tensor(int(t)), latents_torch)["prev_sample"]
#     latents = np.concatenate([latents_torch.numpy(), latents_torch.numpy()], axis=0)

# latents = latents[:1] / np.std(latents)
# decoded = vae_decoder_sess.run(None, {vae_decoder_sess.get_inputs()[0].name: latents})[0]
# image = (np.clip((decoded + 1) / 2, 0, 1)[0].transpose(1, 2, 0) * 255).astype(np.uint8)

# pil_image = Image.fromarray(image).resize((768, 768), Image.BICUBIC)
# pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
# output_path = onnx_dir / "output.png"
# pil_image.save(output_path)


# print(f"\n✅ Image saved at: {output_path}")
# # After saving the image
# del text_encoder_sess, unet_sess, vae_decoder_sess, latents, decoded
# gc.collect()

## load yolov8 for image detection and masking
# --- Step 1: Select box manually ---
coords = []

def on_select(eclick, erelease):
    global coords
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    coords = [x1, y1, x2, y2]
    plt.close()

def select_box(image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img)
    toggle_selector = RectangleSelector(ax, on_select, useblit=True,
                                    button=[1],  # Left click
                                    minspanx=5, minspany=5,
                                    spancoords='pixels', interactive=True)
    plt.title("Draw a box around the object and close the window.")
    plt.show()
    return coords, img
# --- Step 1: Load Models ---
sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# --- Step 2: Load Input Image ---
image_path = "onnx/CaptureTree.png"
image_pil = Image.open(image_path).convert("RGB")
image_np = np.array(image_pil)


################################## POSSIBLY CHECK OUT clipseg-rd64-refined FOR THE MASKING AS IT DOES IT BASED ON A PROMPT ###################################################################


# --- Step 3: Run SAM prediction ---
def run_sam(image_pil, box):
    inputs = sam_processor(image_pil, input_boxes=[[box]], return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )
    return masks[0][0][0].numpy()  # First mask, height x width

# --- Step 4: Main execution ---
image_path = "onnx/CaptureTree.png"  # Replace with your image
box_coords, image_rgb = select_box(image_path)

if not box_coords:
    print("❌ No box selected.")
else:
    print(f"✅ Box selected: {box_coords}")
    image_pil = Image.fromarray(image_rgb)
    mask = run_sam(image_pil, box_coords)

    # Save mask
    mask_path = "onnx/output_mask.png"
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
    print(f"✅ Mask saved to {mask_path}")

    # Optional: Show mask overlaid
    overlay = image_rgb.copy()
    overlay[mask > 0.5] = [255, 0, 0]  # Red mask overlay
    plt.imshow(overlay)
    plt.title("Mask Overlay")
    plt.axis("off")
    plt.show()



