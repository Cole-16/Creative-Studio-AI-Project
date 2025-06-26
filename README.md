# 🧠 Creative Studio AI – ONNX-Optimized Generative Pipeline

## 📘 Project Overview

**Creative Studio AI** is an end-to-end generative AI pipeline designed to run on resource-constrained environments using ONNX-optimized models. It combines:

- 🧠 **GPT-2 ONNX** for creative prompt generation  
- 🎨 **Stable Diffusion ONNX** for AI image generation  
- 🔍 **ClipSeg ONNX** for object detection and masking  
- 🧰 Custom logic for region-based editing and visualization  

The goal is to provide a lightweight, fast, and extensible pipeline for generating and editing creative AI outputs **entirely offline** using CPU-only inference, pruning, quantization, and model simplification.

This repository contains materials for the Content Creation AI project choice  for the "Deploying Edge AI" master's level course.

![image](https://github.com/user-attachments/assets/4ec4c32c-ad25-407e-a9f4-fffabe35cf00)



---

## 📊 Architectural Diagram

Below is a high-level overview of the full demo process from user input to final output:



**Workflow Steps:**

1. **User Input** → Keyword or phrase
2. **GPT-2 ONNX** → Generates 3 creative text prompts
3. **User Selection** → Chooses one prompt
4. **Stable Diffusion ONNX** → Generates a 64x64 image
5. **User Input** → Specifies object to edit (e.g., "tree")
6. **ClipSeg ONNX** → Creates inferno-style mask
7. **User Input** → Specifies desired edit (e.g., "change color to red")
8. **Editing Logic** → Applies blended color edit using the mask
9. **Output** → Saves edited image, original, and a higher-quality version
10. **UI Display** → Shows side-by-side image comparison

---

## 🚀 Quick Start Guide

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/your-username/creative-studio-ai.git
cd creative-studio-ai
```

### 🐍 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On macOS/Linux
venv\Scripts\activate           # On Windows
```

### 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Some dependencies may require system packages (e.g., PyTorch/ONNXRuntime); refer to the `requirements.txt` for exact versions.

### ▶️ 4. Run the Full Demo

```bash
cd app
python full_demo.py
```

This will:

- Load/export ONNX models (if not already cached)
- Prompt the user for a keyword
- Generate an image and let the user interactively apply an edit
- Save and display a side-by-side comparison of the result

---

## 📂 More Information

For **detailed milestone breakdowns**, **benchmark results**, and **per-script explanations**, be sure to read:

📄 [`app/README.md`](app/README.md)

This includes:

- 📊 Benchmark graphs for quantization, pruning, and KV caching
- 📄 Script-specific optimization summaries
- 🔍 Trade-offs between performance and quality
- 🎯 Phase-based development progress

---
