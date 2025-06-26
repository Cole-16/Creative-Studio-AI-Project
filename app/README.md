# 🧠 Creative Studio AI Project – ONNX Optimization & Benchmarking

## 📘 Overview

This project contains all Python scripts used to convert models, run performance tests, and enable various features like prompt generation, image creation, and editable AI-generated content. As development progresses, more functionality will be added.

---

## 🎯 Milestones

This section outlines each development phase, the goals achieved, and which files implement those goals.

---

## 🛠️ Phase 1: Model Selection, Resource Constraints & ONNX Export

### ✅ Selected Models

- `CompVis/stable-diffusion-v1-4` – for image generation (Stable Diffusion)
- `openai-community/gpt2-medium` – for creative text prompt generation

### ⚠️ Resource Constraints

- Limited RAM and CPU availability
- Large model storage requirements

### 🧪 Optimization Strategies

To reduce resource usage and file sizes:
- 🔄 **Quantization** – minimize model size and memory usage
- ✂️ **Pruning / KV Caching** – GPT-2 optimization for faster inference
- ⏱️ **UNet timestep/sample size reduction** – improves performance at the cost of image quality

### 🧩 General Full Demo Design

1. Export models to ONNX  
2. Prompt user for a concept (word/phrase)  
3. Generate creative text prompt (via GPT-2)  
4. Generate image from the prompt (via Stable Diffusion)  
5. Ask user what object they’d like to edit  
6. Use object detection/masking model to create a mask  
7. Ask user what edit to apply (e.g. "change color")  
8. Apply and display final edited image  

### 📄 Relevant Files

| Filename | Description |
|----------|-------------|
| `stable_diffusion_to_onnx_64px_quantized.py` | ONNX export & quantization of Stable Diffusion |
| `gpt2_pruned_vs_base.py` | GPT-2 pruning, KV caching, benchmarking |
| `convert_gpt2_to_onnx.py` | Basic ONNX export & benchmarking with UI output |

> 📝 Note: Both of the first two files for diffusion and gpt2 compares the benchmarks between base and optimized. They started out as original onnx exports and benchmarking but changed and updated throughout the project as I went along to contain more milestones as you will see in the next phases as well.
> 📝 Note: The `convert_gpt2_to_onnx.py` script is an early utility and includes performance testing, but lacks some later features.

---

## ⚙️ Phases 2–3: Optimizing Text & Image Generation

These phases include benchmarking and performance tuning for the GPT-2 and Stable Diffusion models.

### 📄 Optimization Files

- `stable_diffusion_to_onnx_64px_quantized.py`
- `gpt2_pruned_vs_base.py`

---

### ✍️ GPT-2 (Text Generation) Optimization

#### Techniques:
- **Pruning**
- **KV Caching**

#### Results:
- ✅ Faster inference, reduced memory  
- ❌ Poorer text quality (off-topic, repetitive, incoherent output)

Despite better performance, the optimizations harmed creativity. The full demo uses the **base GPT-2 model** for best prompt quality.

> Notes/Takeaways: Both of these optimizations did help in terms of speed and resources but it badly affected the quality of the generated text from what I saw. Using no optimization techniques my text generation stayed on track and produced rather good creative outputs but with the optimization it struggled to stay on topic and usually drifted into typing nonsense. I even tried some repetition handling and warnings where it would basically tell the model it was wrong for having repition in its outputs and that still did not help. That is why these files are seperate from the full demo as the tradeoff for the functionality was not worth it from what I saw in my examples. So, I decided to stick with the base model with no optimization for the full demo but did keep them in these other files for benchmarking purposes.

#### 📊 Benchmark Example:

| Pruned & KV Cached | Base |
|--------------------|------|
| ![image](https://github.com/user-attachments/assets/98d9d4a3-24fa-423e-9b57-b9e6c8092783) | ![image](https://github.com/user-attachments/assets/418397f8-04d4-4265-ae09-1be34bdd3cc3) |

---

### 🖼️ Stable Diffusion Optimization

#### Techniques:
- **Quantization** – Reduced model size, but poor visual quality
- **UNet Downgrade** – 50 timesteps and 64×64 samples to reduce memory

> ⚠️ Tradeoff: Smearing/artifacts in output image when using quantization. Best results: **64x64 sample size without quantization**

#### 📊 Benchmark Example:

| Quantized Calls | Sample Static Quantization Code | Calibration Data Setup |
|-------------------|-------------|-------------------|
| ![image](https://github.com/user-attachments/assets/ad6c2fd6-88c9-4d8a-b737-52e98d3f73a3) | ![image](https://github.com/user-attachments/assets/a2c068ba-426a-4215-a172-9a2fb9c256c4) | ![image](https://github.com/user-attachments/assets/6e2a32d2-d411-422f-971f-5f6c64c114c3) |

> Notes/Takeaways: Once again using these techniques I was able to reduce resources needed and also reduce the size of the model files but the tradeoff was that the quality of the image that was generated was greatly affected. With quantization the image looked like smeared paint and then without quantization and just the 64x64 sample size would create a fairly decent image but again at a drastically reduced quality. In the stable_diffusion_64px_quantized.py file I have kept the quantization for benchmarking to compare the regular to quantized models but in the full demo I made the trade off of no quantization but kept the 64x64 sample size as the image is viewable and does not take 20 minutes plus to create.

#### 🖼️ Sample Outputs:

![image](https://github.com/user-attachments/assets/762618c1-604b-4c54-9b56-2243d42d3c64)  
![image](https://github.com/user-attachments/assets/e8207ba5-a2ec-49d7-90e6-b10b929b1f3b)  
![image](https://github.com/user-attachments/assets/7a0219a9-11b2-4959-bbf3-bed62d73ca56)

📂 Saved to: `/benchmark/diffusion`

---

## 🧠 Phase 4+: Image Detection & Final Editing Demo

### 🔍 Masking & Detection Model

- **Model**: `CIDAS/clipseg-rd64-refined`  
- Used after image generation to create a mask based on a user prompt (e.g. “sky”)

| Input Prompt | Generated Mask |
|--------------|----------------|
| “sky” | ![image](https://github.com/user-attachments/assets/01b29ae6-47ab-49cb-b080-ce4e179d8ae5) |

### 📄 File:

- `full_demo.py`

### 🧪 Final Demo Workflow

1. Export models to ONNX  
2. Prompt user for a keyword/phrase  
3. GPT-2 generates 3 creative prompt candidates using batching
4. User selects best prompt  
5. Stable Diffusion generates image based on selected prompt 
6. User chooses object to edit (e.g. "tree")  
7. User specifies transformation (e.g. "change color to red")  
8. ClipSeg creates mask of object  
9. Color blending applied to selected region  
10. Final edited image is shown, along with a high-res “quality” image that I created using the same diffusion model just with better resolution

#### Demo Output:

![image](https://github.com/user-attachments/assets/0cc33dc2-5984-49e8-aca8-3736286c5b08)

---

## 📁 Current Files

### `convert_gpt2_to_onnx.py`

Converts GPT-2 to ONNX and runs 100 inference tests to generate a performance report.

📊 Example Output:

![image](https://github.com/user-attachments/assets/dc48c495-2554-49f7-a1de-810498503b82)

---

### `gpt2_pruned_vs_base.py`

Benchmarks pruned vs base GPT-2 models with KV caching.

📊 Example Output:

![image](https://github.com/user-attachments/assets/6fccfe6e-57b9-4786-b123-4ff7a5a7d8e6)  
![image](https://github.com/user-attachments/assets/fe1022e7-da59-46ec-a7f2-e21e72fad896)

---

### `stable_diffusion_to_onnx_64px_quantized.py`

Converts Stable Diffusion to ONNX, applies quantization, and benchmarks results.

📊 Example Output:

![image](https://github.com/user-attachments/assets/2b3a691a-e270-4ae9-93bf-26e170508acd)  
![image](https://github.com/user-attachments/assets/5d60bee3-77e6-461c-b2f1-049d3992c1d6)  
![image](https://github.com/user-attachments/assets/6192a934-1977-4057-ae12-93a8275be00f)

📂 Output saved to: `/benchmark/diffusion`

![image](https://github.com/user-attachments/assets/2c3fa3c2-c581-4655-af41-407d3dd00b98)

---
