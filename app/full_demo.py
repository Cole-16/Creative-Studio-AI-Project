import os
import numpy as np
import torch
import gc
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageFilter
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPTokenizer,SamModel,SamProcessor,CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers.onnx import FeaturesManager, export as export_onnx
from diffusers import StableDiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from optimum.exporters.onnx import export
from optimum.exporters.onnx.model_configs import CLIPTextOnnxConfig, UNetOnnxConfig, VaeDecoderOnnxConfig
# from optimum.exporters.onnx.model_configs.diffusers import UNet2DConditionOnnxConfig
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# --- Configuration ---
gpt2_model_id = "openai-community/gpt2-medium"
sd_model_id = "CompVis/stable-diffusion-v1-4"
parent_path=Path.cwd().parent
onnx_dir = parent_path / "models" / "onnx"
onnx_dir.mkdir(parents=True, exist_ok=True)
gpt2_onnx_path = onnx_dir / "gpt2" / "model.onnx"
text_encoder_path = onnx_dir / "text_encoder" / "text_encoder.onnx"
unet_path = onnx_dir / "unet" / "unet.onnx"
vae_path = onnx_dir / "vae_decoder" / "vae.onnx"

onnx_dir.mkdir(parents=True, exist_ok=True)
opset = 14

# --- Step 1: Prompt for user keyword ---
keyword = input("Enter a word or concept to generate an image of: ").strip()

# --- Step 2: Load/export GPT-2 ONNX model ---
tokenizer = AutoTokenizer.from_pretrained(gpt2_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if not gpt2_onnx_path.exists():
    print("📦 Exporting GPT-2 to ONNX...")
    gpt2_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(gpt2_model_id)
    feature = "causal-lm"
    model_kind, onnx_config_class = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
    onnx_config = onnx_config_class(model.config)
    export_onnx(preprocessor=tokenizer, model=model, config=onnx_config, opset=opset, output=gpt2_onnx_path)

gpt2_sess = ort.InferenceSession(str(gpt2_onnx_path), providers=["CPUExecutionProvider"])

# --- Step 3: Run GPT-2 inference for descriptions ---
template = """Keyword: forest  
Description: A mystical forest with golden sunlight and ancient trees.

Keyword: ocean  
Description: A vast ocean with a pastel sunset and gentle waves.

Keyword: desert  
Description: A golden desert with rolling sand dunes and a clear blue sky.

Keyword: {keyword}  
Description:""".format(keyword=keyword)

batch_prompts = [template] * 3
inputs = tokenizer(batch_prompts, return_tensors="np", padding=True)
input_ids = inputs["input_ids"].astype(np.int64)
attention_mask = inputs["attention_mask"].astype(np.int64)
batch_size = input_ids.shape[0]

def nucleus_sample(logits, p=0.9, temperature=0.8):
    logits = logits / temperature
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    probs = np.exp(sorted_logits - np.max(sorted_logits))
    probs /= np.sum(probs)
    cum_probs = np.cumsum(probs)
    cutoff = np.searchsorted(cum_probs, p) + 1
    selected_indices = sorted_indices[:cutoff]
    selected_probs = probs[:cutoff]
    selected_probs /= selected_probs.sum()
    return np.random.choice(selected_indices, p=selected_probs)

def extract_gpt2_description(text: str, keyword: str) -> str:
    keyword_tag = f"Keyword: {keyword}"
    start_idx = text.rfind(keyword_tag)
    if start_idx == -1:
        return "[Keyword not found]"
    
    segment = text[start_idx:]
    desc_start = segment.find("Description:")
    if desc_start == -1:
        return "[Description not found]"
    
    desc_segment = segment[desc_start + len("Description:"):]

    # Optional: stop at next "Keyword:" or just take first sentence
    keyword_next = desc_segment.find("Keyword:")
    if keyword_next != -1:
        desc_segment = desc_segment[:keyword_next]

    first_sentence = desc_segment.strip().split(".")[0].strip()
    return first_sentence + "."

max_new_tokens = 40
generated = input_ids.copy()
for step in range(max_new_tokens):
    ort_inputs = {"input_ids": generated.astype(np.int64), "attention_mask": np.ones_like(generated,dtype=np.int64)}
    logits = gpt2_sess.run(None, ort_inputs)[0]
    next_tokens = [nucleus_sample(logits[i, -1]) for i in range(batch_size)]
    generated = np.concatenate([generated, np.array(next_tokens).reshape(batch_size, 1)], axis=1)
    if all(t == tokenizer.eos_token_id for t in next_tokens):
        break

print("\n📝 GPT-2 Outputs:")
descriptions = []
for i, seq in enumerate(generated):
    decoded = tokenizer.decode(seq, skip_special_tokens=True)
    desc = extract_gpt2_description(decoded, keyword)
    # desc = decoded.split("Keyword:")[-1].strip().split("Description:")[-1].split(".")[0].strip() + "."
    descriptions.append(desc)
    print(f"{i + 1}. {desc}")



# --- Step 4: Let user select a description ---
choice = int(input("\nChoose one of the outputs (1-3): ").strip()) - 1
selected_prompt = descriptions[choice]
print(f"\n🖼️ Using: \"{selected_prompt}\" as the image prompt.")

# After GPT-2 inference is done
del gpt2_sess, inputs, input_ids, attention_mask, generated, batch_prompts
gc.collect()

# --- Step 5: Load and export SD ONNX components if needed ---
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float32).to("cpu")
pipe.unet.config.sample_size = 64  # Reduce UNet latent resolution

if not text_encoder_path.exists():
    text_encoder_path.parent.mkdir(parents=True, exist_ok=True)
    export(pipe.text_encoder.eval(), CLIPTextOnnxConfig(pipe.text_encoder.config), text_encoder_path, opset=15)
if not unet_path.exists():
    unet_path.parent.mkdir(parents=True, exist_ok=True)
    export(pipe.unet.to(torch.float32), UNetOnnxConfig(pipe.unet.config), unet_path, opset=15)
if not vae_path.exists():
    vae_path.parent.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", torch_dtype=torch.float32).to("cpu")
    class VaeDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.post_quant_conv = vae.post_quant_conv
            self.decoder = vae.decoder
            self.config = vae.config
        def forward(self, latent_sample):
            return self.decoder(self.post_quant_conv(latent_sample))
    export(VaeDecoder(vae).to(torch.float32), VaeDecoderOnnxConfig(vae.config), vae_path, opset=15)

# After all SD ONNX exports
del pipe
if 'vae' in locals():
    del vae
gc.collect()

# --- Image Generation ---
text_encoder_sess = ort.InferenceSession(str(text_encoder_path), providers=["CPUExecutionProvider"])
unet_sess = ort.InferenceSession(str(unet_path), providers=["CPUExecutionProvider"])
vae_decoder_sess = ort.InferenceSession(str(vae_path), providers=["CPUExecutionProvider"])
scheduler = DPMSolverMultistepScheduler.from_pretrained(sd_model_id, subfolder="scheduler")
scheduler.set_timesteps(50)

tokenizer_clip = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
input_ids = tokenizer_clip([selected_prompt], return_tensors="np", padding="max_length", max_length=77)["input_ids"]
uncond_ids = tokenizer_clip([""], return_tensors="np", padding="max_length", max_length=77)["input_ids"]
text_emb = text_encoder_sess.run(None, {text_encoder_sess.get_inputs()[0].name: input_ids.astype(np.int64)})[0].astype(np.float32)
uncond_emb = text_encoder_sess.run(None, {text_encoder_sess.get_inputs()[0].name: uncond_ids.astype(np.int64)})[0].astype(np.float32)
text_embeddings = np.concatenate([uncond_emb, text_emb], axis=0)

latents = np.random.randn(1, 4, 64, 64).astype(np.float32) * scheduler.init_noise_sigma
latents = np.concatenate([latents, latents], axis=0)

for t in tqdm(scheduler.timesteps, desc="Denoising"):
    inputs = {
        unet_sess.get_inputs()[0].name: latents,
        unet_sess.get_inputs()[1].name: np.array(t, dtype=np.float32),
        unet_sess.get_inputs()[2].name: text_embeddings,
    }
    noise_pred = unet_sess.run(None, inputs)[0]
    uncond, cond = np.split(noise_pred, 2, axis=0)
    guided = uncond + 10.0 * (cond - uncond)
    latents_torch = torch.from_numpy(latents[:1])
    guided_torch = torch.from_numpy(guided)
    latents_torch = scheduler.step(guided_torch, torch.tensor(int(t)), latents_torch)["prev_sample"]
    latents = np.concatenate([latents_torch.numpy(), latents_torch.numpy()], axis=0)

latents = latents[:1] / np.std(latents)
gc.collect()
decoded = vae_decoder_sess.run(None, {vae_decoder_sess.get_inputs()[0].name: latents})[0]
image = (np.clip((decoded + 1) / 2, 0, 1)[0].transpose(1, 2, 0) * 255).astype(np.uint8)

pil_image = Image.fromarray(image).resize((768, 768), Image.BICUBIC)
pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
output_path = onnx_dir / "test_output.png"
pil_image.save(output_path)


print(f"\n✅ Image saved at: {output_path}")
# After saving the image
del text_encoder_sess, unet_sess, vae_decoder_sess, latents, decoded
gc.collect()



# ## manual masking commented out to save if needed later
# #--- Step 1: Select box manually ---
# coords = []

# def on_select(eclick, erelease):
#     global coords
#     x1, y1 = int(eclick.xdata), int(eclick.ydata)
#     x2, y2 = int(erelease.xdata), int(erelease.ydata)
#     coords = [x1, y1, x2, y2]
#     plt.close()

# def select_box(image_path):
#     img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#     fig, ax = plt.subplots()
#     ax.imshow(img)
#     toggle_selector = RectangleSelector(ax, on_select, useblit=True,
#                                     button=[1],  # Left click
#                                     minspanx=5, minspany=5,
#                                     spancoords='pixels', interactive=True)
#     plt.title("Draw a box around the object and close the window.")
#     plt.show()
#     return coords, img
# # --- Step 1: Load Models ---
# sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
# sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# # --- Step 2: Load Input Image ---
# image_path = onnx_dir / "test_output.png"
# image_pil = Image.open(image_path).convert("RGB")
# image_np = np.array(image_pil)


# ################################## POSSIBLY CHECK OUT clipseg-rd64-refined FOR THE MASKING AS IT DOES IT BASED ON A PROMPT ###################################################################


# # --- Step 3: Run SAM prediction ---
# def run_sam(image_pil, box):
#     inputs = sam_processor(image_pil, input_boxes=[[box]], return_tensors="pt").to("cpu")
#     with torch.no_grad():
#         outputs = sam_model(**inputs)
#     masks = sam_processor.image_processor.post_process_masks(
#         outputs.pred_masks.cpu(),
#         inputs["original_sizes"].cpu(),
#         inputs["reshaped_input_sizes"].cpu()
#     )
#     return masks[0][0][0].numpy()  # First mask, height x width

# # --- Step 4: Main execution ---
# image_path = onnx_dir / "test_output.png"  # Replace with your image
# box_coords, image_rgb = select_box(str(image_path))

# if not box_coords:
#     print("❌ No box selected.")
# else:
#     print(f"✅ Box selected: {box_coords}")
#     image_pil = Image.fromarray(image_rgb)
#     mask = run_sam(image_pil, box_coords)

#     # Save mask
#     mask_path = onnx_dir / "test_mask_output.png"
#     cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
#     print(f"✅ Mask saved to {mask_path}")

#     # Optional: Show mask overlaid
#     overlay = image_rgb.copy()
#     overlay[mask > 0.5] = [255, 0, 0]  # Red mask overlay
#     plt.imshow(overlay)
#     plt.title("Mask Overlay")
#     plt.axis("off")
#     plt.show()
 ## clipseg masking 

# --- Setup ---
clipseg_model_id = "CIDAS/clipseg-rd64-refined"
clipseg_model = CLIPSegForImageSegmentation.from_pretrained(clipseg_model_id)
clipseg_processor = CLIPSegProcessor.from_pretrained(clipseg_model_id)

image_filenames = ["test_output.png", "quality_image.png"]
onnx_dir = onnx_dir  # update this if needed

# --- Get masking and recoloring prompts ---
clipseg_prompt = input("🖋️ What do you want to mask in the image? (e.g., 'sky'): ").strip()
color_prompt = input("🎨 Enter a color change prompt (e.g., 'change to purple'): ").strip()

# --- Color mapping ---
color_map = {
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "purple": [128, 0, 128],
    "yellow": [255, 255, 0],
    "orange": [255, 165, 0],
    "pink": [255, 192, 203],
    "brown": [139, 69, 19],
    "white": [255, 255, 255],
    "black": [0, 0, 0],
}

def extract_color_from_prompt(prompt):
    prompt = prompt.lower()
    for name in color_map:
        if name in prompt:
            return np.array(color_map[name], dtype=np.uint8)
    return None

target_rgb = extract_color_from_prompt(color_prompt)
if target_rgb is None:
    print("❌ No recognized color in the prompt.")
    exit()

def apply_clipseg_and_blend(image_pil, text_prompt, color_rgb, alpha=0.6):
    # Prepare image and model input
    inputs = clipseg_processor(
        text=[text_prompt],
        images=[image_pil],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = clipseg_model(**inputs).logits  # shape: [1, 352, 352]

    # Convert logits to soft mask
    mask = torch.sigmoid(logits[0]).cpu().numpy()  # (352, 352)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_img.resize(image_pil.size, resample=Image.BICUBIC)
    mask_np = np.array(mask_resized).astype(np.float32) / 255.0  # [0,1]

    # Blend with color
    image_rgb = np.array(image_pil).astype(np.float32)
    color_rgb_f = color_rgb.astype(np.float32).reshape(1, 1, 3)
    blended = (
        (1 - alpha * mask_np[..., None]) * image_rgb +
        (alpha * mask_np[..., None]) * color_rgb_f
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return np.array(image_pil), blended  # original, edited

# --- Run on both images ---
image_results = []
for filename in image_filenames:
    # if the quality image exists for comparison grab it from docs location
    if filename == 'quality_image.png':
        image_path = parent_path / "docs" / "quality_image.png"
    else:
        image_path = onnx_dir / filename
    image_pil = Image.open(image_path).convert("RGB")
    original, edited = apply_clipseg_and_blend(image_pil, clipseg_prompt, target_rgb)
    image_results.append((original, edited))

    # Save edited image
    edited_filename = filename.replace(".png", "_edited.png")
    Image.fromarray(edited).save(onnx_dir / edited_filename)
    print(f"✅ Saved: {edited_filename}")

# --- Extended plotting with inferno mask ---
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Grid:
# Row 0 = test_output.png
# Row 1 = quality_output.png

# Add masks to image_results so we can plot them
image_results_with_masks = []
for filename in image_filenames:
    image_path = onnx_dir / filename
    image_pil = Image.open(image_path).convert("RGB")
    inputs = clipseg_processor(text=[clipseg_prompt], images=[image_pil], return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = clipseg_model(**inputs).logits  # [1, 352, 352]

    # Convert to sigmoid mask
    mask = torch.sigmoid(logits[0]).cpu().numpy()
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_img.resize(image_pil.size, resample=Image.BICUBIC)
    mask_np = np.array(mask_resized).astype(np.float32) / 255.0

    # Inferno mask for display only
    image_results_with_masks.append((np.array(image_pil), mask_np, None))  # We'll set edited next

# Fill in the edited images
for i in range(2):
    image_results_with_masks[i] = (
        image_results[i][0],  # original
        image_results_with_masks[i][1],  # mask
        image_results[i][1],  # edited
    )

# Unpack and display all
titles = [
    "Original (test)", "Mask (test)", "Edited (test)",
    "Original (quality)", "Mask (quality)", "Edited (quality)"
]

for idx, ax in enumerate(axs.flatten()):
    row = idx // 3
    col = idx % 3
    original, mask, edited = image_results_with_masks[row]

    if col == 0:
        ax.imshow(original)
    elif col == 1:
        ax.imshow(original)  # show base image
        ax.imshow(mask, alpha=0.5, cmap="inferno")  # inferno overlay
    elif col == 2:
        ax.imshow(edited)

    ax.set_title(titles[idx])
    ax.axis("off")

plt.suptitle(f"Mask prompt: '{clipseg_prompt}' | Color edit: '{color_prompt}'", fontsize=15)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()







