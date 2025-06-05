from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.onnx import FeaturesManager, export
from optimum.exporters.onnx import main_export
import torch
import os
import onnxruntime as ort
import psutil
import time
import numpy
import json
import random
import re

# === MODEL SETUP ===
model_id = "openai-community/gpt2"  # Optimized smaller GPT-2 model
output_path = Path("optimized_onnx/gpt2")
output_path.mkdir(parents=True, exist_ok=True)
onnx_model_path = output_path / "model.onnx"
quantized_model_path = output_path / "model_quant.onnx"

# === Load model & tokenizer ===
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# === Optional: Apply pruning before export ===
# from torch.nn.utils import prune
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Linear):
#         prune.l1_unstructured(module, name='weight', amount=0.2)
#         prune.remove(module, 'weight')  # Clean-up for ONNX export

## === EXPORT MODEL WITH KV CACHE ===
if not onnx_model_path.exists():
    main_export(
        model_name_or_path=model_id,
        output=output_path,
        task="text-generation",
        opset=14,
        use_auth_token=False,
        use_past=True
    )
    print("✅ ONNX model exported with KV cache support.")
else:
    print("📦 Using existing ONNX model.")

def benchmark_model(model_path, ort_inputs, description=""):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_profiling = False
    ort_session = ort.InferenceSession(str(model_path), session_options)

    latencies = []
    mem_usage = []
    process = psutil.Process(os.getpid())

    for _ in range(50):  # Reasonable sample size
        mem_usage.append(process.memory_info().rss / 1024 / 1024)
        start = time.time()
        ort_session.run(None, ort_inputs)
        latencies.append((time.time() - start) * 1000)

    model_size = os.path.getsize(model_path) / (1024 ** 2)

    stats = {
        "model": description,
        "size_mb": model_size,
        "latency_avg": numpy.mean(latencies),
        "latency_p95": numpy.percentile(latencies, 95),
        "ram_avg": numpy.mean(mem_usage)
    }
    return stats

# === ONNX Dynamic Quantization ===
from onnxruntime.quantization import quantize_dynamic, QuantType

if not quantized_model_path.exists():
    quantize_dynamic(
        model_input=str(onnx_model_path),
        model_output=str(quantized_model_path),
        weight_type=QuantType.QInt8
    )
    print(f"✅ Quantized model saved: {quantized_model_path}")
else:
    print("📦 Using existing quantized ONNX model.")

def compare_stats(base, quant):
    print("\n📊 Benchmark Comparison:")
    print(f"{'Metric':<20} | {'Baseline':>10} | {'Quantized':>10} | {'Improvement':>12}")
    print("-" * 60)
    for key in ["size_mb", "latency_avg", "latency_p95", "ram_avg"]:
        baseline_val = base[key]
        quant_val = quant[key]
        improvement = baseline_val - quant_val
        direction = "↓" if improvement > 0 else "↑"
        print(f"{key:<20} | {baseline_val:10.2f} | {quant_val:10.2f} | {direction} {abs(improvement):10.2f}")

# === Inference Setup with Optimizations ===
session_options = ort.SessionOptions()
session_options.enable_profiling = True
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.log_severity_level = 3  # Reduce log spam
ort_session = ort.InferenceSession(str(quantized_model_path), session_options)

# === Prompt Setup ===
keyword = "beach"
example_pool = [
    f"A surreal {keyword} covered in glowing mist.",
    f"A tropical {keyword} lit by bioluminescent waves.",
    f"A futuristic {keyword} with hovering surfboards.",
]

keyword_examples = [ex for ex in example_pool if keyword.lower() in ex.lower()]
nonkeyword_examples = [ex for ex in example_pool if keyword.lower() not in ex.lower()]
example_lines = [random.choice(keyword_examples), random.choice(nonkeyword_examples)] if keyword_examples and nonkeyword_examples else random.sample(example_pool, 2)
example_block = "\n".join(example_lines)
prompt_starters = [
    f"A {keyword} with",
    f"A {keyword} surrounded by",
    f"A {keyword} where the",
    f"A {keyword} during the",
    f"A {keyword} that contains"
]

start_prompt = f"{example_block}\n{random.choice(prompt_starters)} "

print(f"\n🪄 Prompt:\n{start_prompt}")

# === Tokenize input ===
inputs = tokenizer(start_prompt, return_tensors="np")
input_ids = inputs["input_ids"]
prompt_len = input_ids.shape[1]

# === Sampling parameters ===
temperature = 0.5
top_p = 0.9
top_k = 50
num_prompts = 5
max_new_tokens = 30
max_attempts_per_prompt = 5
no_repeat_ngram_size = 4  # Reduced from 6 for better repetition control
repetition_penalty = 1.5

def extract_first_sentence(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    for sent in sentences:
        if sent.strip():
            return sent.strip()
    return text.strip()  # fallback 

def sample_next_token(logits, generated_tokens, temperature=1.0, top_p=0.9, no_repeat_ngram_size=3, repetition_penalty=1.3, debug=False):
    logits = logits.copy()

    # Apply repetition penalty
    token_set = set(generated_tokens.tolist() if hasattr(generated_tokens, 'tolist') else generated_tokens)
    for token_id in token_set:
        logits[token_id] /= repetition_penalty

    # Temperature scaling
    logits = logits / temperature

    # Stable softmax
    max_logit = numpy.max(logits)
    exp_logits = numpy.exp(logits - max_logit)
    probs = exp_logits / exp_logits.sum()

    # Top-p filtering
    sorted_indices = numpy.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = numpy.cumsum(sorted_probs)
    cutoff_index = numpy.searchsorted(cumulative_probs, top_p) + 1
    filtered_indices = sorted_indices[:cutoff_index]
    filtered_probs = sorted_probs[:cutoff_index]
    filtered_probs /= filtered_probs.sum()

    # No-repeat ngram filtering
    if len(generated_tokens) >= no_repeat_ngram_size - 1:
        ngrams = {}
        tokens_list = generated_tokens.tolist() if hasattr(generated_tokens, 'tolist') else generated_tokens
        for i in range(len(tokens_list) - no_repeat_ngram_size + 1):
            prefix = tuple(tokens_list[i:i + no_repeat_ngram_size - 1])
            next_token = tokens_list[i + no_repeat_ngram_size - 1]
            ngrams.setdefault(prefix, set()).add(next_token)
        prefix = tuple(tokens_list[-(no_repeat_ngram_size - 1):])
        banned_tokens = ngrams.get(prefix, set())

        filtered_final = [(idx, prob) for idx, prob in zip(filtered_indices, filtered_probs) if idx not in banned_tokens]

        if filtered_final:
            indices, probs_final = zip(*filtered_final)
            probs_final = numpy.array(probs_final)
            probs_final /= probs_final.sum()
            if debug:
                print(f"Banned tokens: {banned_tokens}")
                print(f"Sampling from filtered tokens: {indices}")
            return numpy.random.choice(indices, p=probs_final)
        else:
            # All tokens banned fallback to most probable
            if debug:
                print("All tokens banned by n-gram filter; falling back to top token")
            return filtered_indices[0]

    # Sample from filtered distribution
    if debug:
        print(f"Sampling from tokens: {filtered_indices}")
    return numpy.random.choice(filtered_indices, p=filtered_probs)

# === Generation Loop ===
generated_prompts = set()

while len(generated_prompts) < num_prompts:
    for attempt in range(max_attempts_per_prompt):
        generated = input_ids.copy()
        response_tokens = []

        seq_len = generated.shape[1]
        position_ids = numpy.arange(seq_len).reshape(1, seq_len).astype(numpy.int64)

        # First forward pass
        onnx_inputs = {
            "input_ids": generated.astype(numpy.int64),
            "attention_mask": numpy.ones_like(generated).astype(numpy.int64),
            "position_ids": position_ids,
        }
        outputs = ort_session.run(None, onnx_inputs)
        logits = outputs[0]
        past_key_values = outputs[1:]

        # Sample first token
        next_token = sample_next_token(
            logits[0, -1, :],
            generated_tokens=generated[0],
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            debug=False
        )
        if next_token == tokenizer.eos_token_id:
            break
        response_tokens.append(next_token)

        # Generate remaining tokens
        for _ in range(max_new_tokens - 1):
            current_pos = generated.shape[1] + len(response_tokens) - 1
            position_ids = numpy.array([[current_pos]]).astype(numpy.int64)
            new_input = numpy.array([[response_tokens[-1]]]).astype(numpy.int64)

            onnx_inputs = {
                "input_ids": new_input,
                "attention_mask": numpy.ones_like(new_input).astype(numpy.int64),
                "position_ids": position_ids,
            }
            for i, (k, v) in enumerate(zip(past_key_values[::2], past_key_values[1::2])):
                onnx_inputs[f"past_key_values.{i}.key"] = k
                onnx_inputs[f"past_key_values.{i}.value"] = v

            outputs = ort_session.run(None, onnx_inputs)
            logits = outputs[0]
            past_key_values = outputs[1:]

            combined_tokens = numpy.concatenate([generated[0], numpy.array(response_tokens)])
            next_token = sample_next_token(
                logits[0, -1, :],
                generated_tokens=combined_tokens,
                temperature=temperature,
                top_p=top_p,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                debug=False
            )
            if next_token == tokenizer.eos_token_id:
                break
            response_tokens.append(next_token)

        # Decode full output
        full_sequence = numpy.concatenate([generated, numpy.array([response_tokens])], axis=1)
        output_text = tokenizer.decode(full_sequence[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print("FULL OUTPUT TEXT:", repr(output_text))

        output_tail = output_text.split(example_lines[-1])[-1].strip()
        trimmed = extract_first_sentence(output_tail)
        print(f"TRIMMED TEXT: {trimmed}")

        if keyword.lower() in trimmed.lower() and trimmed not in generated_prompts:
            generated_prompts.add(trimmed)
            print(f"\n🖼️ Prompt {len(generated_prompts)}: {trimmed}")
            break
        else:
            print(f"\n🔁 Retry {attempt + 1}/{max_attempts_per_prompt} — keyword missing or duplicate.")
    else:
        print("⚠️ Max attempts reached — skipping.")

# === Final output ===
print(f"\n✅ Finished generating {len(generated_prompts)} prompts containing keyword '{keyword}':")
for i, prompt in enumerate(generated_prompts, 1):
    print(f"{i}. {prompt}")

# === Performance Metrics ===
model_size = os.path.getsize(quantized_model_path) / (1024 ** 2)
latencies = []
mem_usage = []
process = psutil.Process(os.getpid())

for _ in range(100):
    mem_usage.append(process.memory_info().rss / 1024 / 1024)
    start = time.time()
    ort_session.run(None, onnx_inputs)
    latencies.append((time.time() - start) * 1000)

# === Profiling ===
profile_file = ort_session.end_profiling()
print("\n📁 Profiling file:", profile_file)

# === Print Stats ===
print(f"\n📊 Model size: {model_size:.2f} MB")
print(f"⚡ Average latency: {numpy.mean(latencies):.2f} ms")
print(f"📈 p95 latency: {numpy.percentile(latencies, 95):.2f} ms")
print(f"🧠 Avg RAM usage: {numpy.mean(mem_usage):.2f} MB")
print("ONNX output keys:", ort_session.get_outputs())
print("First output shape:", outputs[0].shape)

# === Analyze Profiling ===
with open(profile_file, "r") as f:
    profile_data = json.load(f)

print("\n🔍 Top Time-Consuming Operations:")
events = [e for e in profile_data if e.get("cat") == "Node"]
top_ops = sorted(events, key=lambda x: x.get("dur", 0), reverse=True)[:5]
for op in top_ops:
    print(f"- {op['name'][:30]:30} | {op['dur']/1000:.3f} ms | OpType: {op['args'].get('op_name')}")

# === Run Benchmarks and Compare ===
benchmark_inputs = {
    "input_ids": dummy_inputs["input_ids"].numpy().astype(numpy.int64),
    "attention_mask": dummy_inputs["attention_mask"].numpy().astype(numpy.int64)
}

baseline_stats = benchmark_model(onnx_model_path, benchmark_inputs, "Baseline")
quant_stats = benchmark_model(quantized_model_path, benchmark_inputs, "Quantized")
compare_stats(baseline_stats, quant_stats)