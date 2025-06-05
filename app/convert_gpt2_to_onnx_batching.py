from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.onnx import FeaturesManager, export
import torch
import os
import onnxruntime as ort
import psutil
import time
import numpy as np
import json
import random

model_id = "openai-community/gpt2"
output_path = Path("onnx/gpt2")
output_path.mkdir(parents=True, exist_ok=True)

# Step 1: Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Step 2: ONNX Export
onnx_model_path = output_path / "model.onnx"
feature = "causal-lm"
model_kind, onnx_config_class = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = onnx_config_class(model.config)

if not os.path.exists(onnx_model_path):
    dummy_inputs = onnx_config.generate_dummy_inputs(tokenizer, framework="pt")
    export(preprocessor=tokenizer, model=model, config=onnx_config, opset=14, output=onnx_model_path)
else:
    print("ONNX model already exists.")

print("✅ Exported openai-community/gpt2 to ONNX:", onnx_model_path)

print("Running inference with ONNX model...")

# Enable profiling in ONNX Runtime
session_options = ort.SessionOptions()
session_options.enable_profiling = True
session_options.log_severity_level = 3  # Reduce logging noise
session_options.profile_file_prefix = 'onnxruntime_profile'
ort_session = ort.InferenceSession(str(onnx_model_path), session_options)

# === Sampling helpers ===
def top_k_sample(logits, k=20, temperature=0.8):
    logits = logits / temperature
    top_k_indices = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_indices]
    exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
    probs = exp_logits / exp_logits.sum()
    next_token = np.random.choice(top_k_indices, p=probs)
    return next_token

def nucleus_sample(logits, top_p=0.9, temperature=0.8):
    logits = logits / temperature
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
    probs = exp_logits / exp_logits.sum()
    cumulative_probs = np.cumsum(probs)

    filtered_indices = sorted_indices[cumulative_probs <= top_p]
    if len(filtered_indices) == 0:
        filtered_indices = sorted_indices[:1]

    filtered_logits = logits[filtered_indices]
    exp_filtered = np.exp(filtered_logits - np.max(filtered_logits))
    filtered_probs = exp_filtered / exp_filtered.sum()
    next_token = np.random.choice(filtered_indices, p=filtered_probs)
    return next_token

# === Prompt Generation Setup ===
keyword = "beach"

example_pool = [
    f"A surreal {keyword} covered in glowing mist.",
    f"A tropical {keyword} lit by bioluminescent waves.",
    f"A futuristic {keyword} with hovering surfboards.",
    "A sci-fi planet with colorful foliage and a bright pink sky.",
    "A city in a bottle floating in the sky.",
    "An enchanted forest with moonlit mushrooms and glowing owls.",
    "A ghost town overrun by vines and floating lanterns.",
    "A dreamy landscape with floating islands and cascading waterfalls.",
    "A neon-lit street at night bustling with futuristic vehicles.",
    "A misty mountain valley glowing with ethereal light.",
]

keyword_examples = [ex for ex in example_pool if keyword.lower() in ex.lower()]
nonkeyword_examples = [ex for ex in example_pool if keyword.lower() not in ex.lower()]
prompt_starters = ["A", f"{keyword} with", f"Scenic view of a {keyword} with", "The"]

# === Batch size selection ===
batch_size = random.choice([1, 2, 3, 4])
print(f"Using batch size: {batch_size}")

def generate_batch_start_prompts(batch_size):
    batch_prompts = []
    for _ in range(batch_size):
        if keyword_examples and nonkeyword_examples:
            example_lines = [random.choice(keyword_examples), random.choice(nonkeyword_examples)]
        else:
            example_lines = random.sample(example_pool, 2)
        example_block = "\n".join(example_lines)
        start_prompt = f"{example_block}\n{random.choice(prompt_starters)}"
        batch_prompts.append((example_block, start_prompt))
    return batch_prompts

batch_prompt_data = generate_batch_start_prompts(batch_size)

# Tokenize batch prompts with padding
batch_inputs = tokenizer(
    [bp[1] for bp in batch_prompt_data],
    return_tensors="np",
    padding=True,
)

input_ids = batch_inputs["input_ids"]
attention_mask = batch_inputs["attention_mask"]

# Inference settings
max_new_tokens = 20
temperature = random.choice([0.9, 1.0])
top_k = random.choice([50, 70, 80])
top_p = 0.9
sampling_method = random.choice(["top_k", "nucleus"])
print(f"Sampling method: {sampling_method}, temperature: {temperature}, top_k: {top_k}, top_p: {top_p}")

max_attempts_per_prompt = 5
generated_prompts = set()
generated = input_ids.copy()

for attempt in range(max_attempts_per_prompt):
    for _ in range(max_new_tokens):
        onnx_inputs = {
            "input_ids": generated.astype(np.int64),
            "attention_mask": np.ones_like(generated).astype(np.int64)
        }
        outputs = ort_session.run(None, onnx_inputs)
        logits = outputs[0]  # shape (batch_size, seq_len, vocab_size)

        next_token_ids = []
        for i in range(batch_size):
            logit = logits[i, -1, :]
            if sampling_method == "top_k":
                next_token_id = top_k_sample(logit, k=top_k, temperature=temperature)
            else:
                next_token_id = nucleus_sample(logit, top_p=top_p, temperature=temperature)
            next_token_ids.append(next_token_id)

        next_tokens_arr = np.array(next_token_ids).reshape(batch_size, 1)
        generated = np.concatenate([generated, next_tokens_arr], axis=1)

    for i in range(batch_size):
        example_block, _ = batch_prompt_data[i]
        output_text = tokenizer.decode(generated[i], skip_special_tokens=True)
        output_tail = output_text.split(example_block.split("\n")[-1])[-1].strip()
        words = output_tail.split()
        trimmed = " ".join(words[:15])
        for j, word in enumerate(words[:15]):
            if word.endswith((".", "!", "?")):
                trimmed = " ".join(words[: j + 1])
                break

        if keyword.lower() in trimmed.lower() and trimmed not in generated_prompts:
            generated_prompts.add(trimmed)
            print(f"\n🖼️ Prompt {len(generated_prompts)}: {trimmed}")
        else:
            print(f"🔁 Attempt {attempt + 1}/{max_attempts_per_prompt} for batch item {i + 1} — keyword missing or duplicate.")

    if len(generated_prompts) >= batch_size:
        break
else:
    print("⚠️ Max attempts reached without generating enough valid prompts.")

print(f"\n✅ Generated {len(generated_prompts)} unique prompts with keyword '{keyword}'.")
print(f"Prompts: {generated_prompts}")

# === Profiling and stats ===

model_size = os.path.getsize(onnx_model_path) / (1024 ** 2)
latencies = []
process = psutil.Process(os.getpid())
mem_usage = []

# Warmup run
onnx_inputs = {
    "input_ids": generated.astype(np.int64),
    "attention_mask": np.ones_like(generated).astype(np.int64),
}
_ = ort_session.run(None, onnx_inputs)

for i in range(100):
    mem_usage.append(process.memory_info().rss / 1024 / 1024)
    start = time.time()
    _ = ort_session.run(None, onnx_inputs)
    latencies.append((time.time() - start) * 1000)

profile_file = ort_session.end_profiling()
print("Original profiling file:", repr(profile_file))

print(f'Model size MB : {model_size:.2f}')
print(f'Average latency ms: {np.mean(latencies):.2f}')
print(f'p95 latency ms: {np.percentile(latencies, 95):.2f}')
print(f'Average mem usage mb: {np.mean(mem_usage):.2f}')
print("ONNX output keys:", ort_session.get_outputs())
# print("First output shape:", _.shape)
print(f"📁 Profiling data saved to: {repr(profile_file)}")

# Analyze top ops from profile JSON
with open(profile_file, "r") as f:
    profile_data = json.load(f)

print("\n🔍 Top Time-Consuming Operations:")
events = [e for e in profile_data if e.get("cat") == "Node"]
sorted_ops = sorted(events, key=lambda x: x.get("dur", 0), reverse=True)

for op in sorted_ops[:5]:
    print(f"- {op['name']:30} | Duration: {op['dur'] / 1000:.3f} ms | OpType: {op['args'].get('op_name')}")