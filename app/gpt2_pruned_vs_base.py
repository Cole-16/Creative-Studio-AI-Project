# this should cover the onnx conversion for week 1, the pruning for week 2, and also the kv caching for week 3 and compare all the benchmarking.
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.onnx import FeaturesManager, export
import torch
import torch.nn.utils.prune as prune
import os
import onnxruntime as ort
import psutil
import time
import numpy
import json
import random
import gc
import pandas as pd
import matplotlib.pyplot as plt

model_id = "openai-community/gpt2-medium"
output_dir = Path.cwd().parent / "models" / "onnx" / "gpt2"
output_dir.mkdir(parents=True, exist_ok=True)

# Load base model & tokenizer
base_model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Helper to apply structured pruning
def apply_pruning(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, 'weight')
    return model

# Export helper
def export_to_onnx(model, onnx_path):
    feature = "causal-lm"
    model_kind, onnx_config_class = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
    onnx_config = onnx_config_class(model.config)
    export(preprocessor=tokenizer, model=model, config=onnx_config, opset=14, output=onnx_path)

# Benchmark helper
def benchmark_onnx_model(onnx_path, use_kv_cache=False):
    ort_sess = ort.InferenceSession(str(onnx_path))
    n_head = base_model.config.n_head
    d_model = base_model.config.n_embd
    head_dim = d_model // n_head

    prompt = "A beach scene with"
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    onnx_inputs = {
        "input_ids": input_ids.astype(numpy.int64),
        "attention_mask": attention_mask.astype(numpy.int64),
    }

    if use_kv_cache:
        past_input_names = [inp.name for inp in ort_sess.get_inputs() if inp.name.startswith("past_key_values.")]
        past_kv = [numpy.zeros((1, n_head, 0, head_dim), dtype=numpy.float32) for _ in past_input_names]
        for i, pkv in enumerate(past_kv):
            onnx_inputs[f"past_key_values.{i}"] = pkv

    latencies, mem_usage = [], []
    process = psutil.Process(os.getpid())

    for _ in range(50):
        mem_usage.append(process.memory_info().rss / 1024 / 1024)
        start = time.time()
        outputs = ort_sess.run(None, onnx_inputs)
        logits = outputs[0]
        latencies.append((time.time() - start) * 1000)
        del outputs, logits
        gc.collect()

    avg_latency = numpy.mean(latencies)
    tokens_per_sec = 1000.0 / avg_latency
    return {
        "model_size": os.path.getsize(onnx_path) / (1024 ** 2),
        "avg_latency": avg_latency,
        "p95_latency": numpy.percentile(latencies, 95),
        "memory": numpy.mean(mem_usage),
        "tokens_per_sec": tokens_per_sec,
    }

# Export models
base_onnx_path = output_dir / "model_base.onnx"
if not base_onnx_path.exists():
    export_to_onnx(base_model, base_onnx_path)
    print("✅ Exported base model")
else:
    print("🟢 Base model already exported")

pruned_model = apply_pruning(AutoModelForCausalLM.from_pretrained(model_id))
pruned_onnx_path = output_dir / "model_pruned.onnx"
export_to_onnx(pruned_model, pruned_onnx_path)
print("✅ Exported pruned model")

# Benchmark all combinations
print("📈 Benchmarking all variants...")
base_no_kv = benchmark_onnx_model(base_onnx_path, use_kv_cache=False)
base_kv = benchmark_onnx_model(base_onnx_path, use_kv_cache=True)
pruned_no_kv = benchmark_onnx_model(pruned_onnx_path, use_kv_cache=False)
pruned_kv = benchmark_onnx_model(pruned_onnx_path, use_kv_cache=True)

benchmark_dir = Path.cwd().parent / "benchmark" / "gpt2"
benchmark_dir.mkdir(parents=True, exist_ok=True)


# Table output
def print_comparison_table():
    headers = ["Metric", "Base", "Base + KV", "Pruned", "Pruned + KV"]
    rows = [
        ["Model Size (MB)",
         f"{base_no_kv['model_size']:.2f}", f"{base_kv['model_size']:.2f}",
         f"{pruned_no_kv['model_size']:.2f}", f"{pruned_kv['model_size']:.2f}"],
        ["Avg Latency (ms)",
         f"{base_no_kv['avg_latency']:.2f}", f"{base_kv['avg_latency']:.2f}",
         f"{pruned_no_kv['avg_latency']:.2f}", f"{pruned_kv['avg_latency']:.2f}"],
        ["P95 Latency (ms)",
         f"{base_no_kv['p95_latency']:.2f}", f"{base_kv['p95_latency']:.2f}",
         f"{pruned_no_kv['p95_latency']:.2f}", f"{pruned_kv['p95_latency']:.2f}"],
        ["Avg Memory (MB)",
         f"{base_no_kv['memory']:.2f}", f"{base_kv['memory']:.2f}",
         f"{pruned_no_kv['memory']:.2f}", f"{pruned_kv['memory']:.2f}"],
        ["Tokens/sec",
         f"{base_no_kv['tokens_per_sec']:.2f}", f"{base_kv['tokens_per_sec']:.2f}",
         f"{pruned_no_kv['tokens_per_sec']:.2f}", f"{pruned_kv['tokens_per_sec']:.2f}"],
    ]
    print("\n📊 Full Benchmark Comparison:")
    print(" | ".join(headers))
    print("-" * 70)
    for row in rows:
        print(" | ".join(row))
    return rows,headers

rows,headers=print_comparison_table()

# Convert to DataFrame
df = pd.DataFrame(rows, columns=headers)

# Save as CSV and Markdown
csv_path = benchmark_dir / "gpt2_benchmark_results.csv"
md_path = benchmark_dir / "gpt2_benchmark_results.md"
df.to_csv(csv_path, index=False)
df.to_markdown(md_path, index=False)

print(f"\n✅ Benchmark saved to:\nCSV: {csv_path}\nMarkdown: {md_path}")

# Plot grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(df))
bar_width = 0.2

# Offsets for each group
for i, col in enumerate(headers[1:]):
    ax.bar(
        [xi + i * bar_width for xi in x],
        df[col].astype(float),
        width=bar_width,
        label=col
    )

ax.set_xticks([xi + bar_width * 1.5 for xi in x])
ax.set_xticklabels(df["Metric"], rotation=45)
ax.set_ylabel("Value")
ax.set_title("GPT-2 Benchmark Comparison")
ax.legend()
plt.tight_layout()

chart_path = benchmark_dir / "gpt2_benchmark_chart.png"
plt.savefig(chart_path)
plt.close()

print(f"📈 Chart saved to: {chart_path}")