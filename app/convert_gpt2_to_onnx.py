from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.onnx import FeaturesManager, export
from transformers.onnx.features import FeaturesManager
import torch
import os
import onnxruntime as ort
import psutil
import time
import numpy
import json

model_id = "openai-community/gpt2"
output_path = Path("onnx/gpt2")
output_path.mkdir(parents=True, exist_ok=True)

# Step 1: Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Step 2: Choose ONNX feature
  # For language modeling

output_path = Path("onnx/gpt2")
onnx_model_path = Path('onnx/gpt2/model.onnx')


feature = "causal-lm"
model_kind, onnx_config_class = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = onnx_config_class(model.config)
dummy_inputs = onnx_config.generate_dummy_inputs(tokenizer, framework="pt")
if not os.path.exists(output_path):
    export(preprocessor=tokenizer, model=model, config=onnx_config, opset=14,
        output=output_path / "model.onnx")
    
else:
    print("ONNX model already exists.")

print("✅ Exported openai-community/gpt2 to ONNX:", output_path / "model.onnx")


print("Running inference with ONNX model...")
# Enable profiling in ONNX Runtime
session_options = ort.SessionOptions()
session_options.enable_profiling = True
session_options.log_severity_level = 3  # Reduce logging noise
session_options.profile_file_prefix = 'onnxruntime_profile'
ort_session = ort.InferenceSession(onnx_model_path,session_options)

# Convert inputs to numpy
onnx_inputs = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
# measuring stats
model_size=os.path.getsize(onnx_model_path) / (1024 **2)
latencies=[]
process = psutil.Process(os.getpid())
mem_usage=[]
for i in range(100):
    mem_usage.append(process.memory_info().rss / 1024 / 1024)
    start = time.time()

    outputs = ort_session.run(None, onnx_inputs)
    latencies.append((time.time() - start ) * 1000)

# Retrieve and save ONNX Runtime profile
profile_file = ort_session.end_profiling()
print("Original profiling file:", repr(profile_file))



# Show results
print(f'Model size MB : {model_size}')
print(f'Average latency ms: {numpy.mean(latencies)}')
print(f'p95 latency ms: {numpy.percentile(latencies, 95)}')
print(f'Average mem usage mb: {numpy.mean(mem_usage)}')
print("ONNX output keys:", ort_session.get_outputs())
print("First output shape:", outputs[0].shape)
print(f"📁 Profiling data saved to: {repr(profile_file)}")

# Analyze top ops from profile JSON
with open(profile_file, "r") as f:
    profile_data = json.load(f)


# Extract top time-consuming operators
print("\n🔍 Top Time-Consuming Operations:")
events = [e for e in profile_data if e.get("cat") == "Node"]
sorted_ops = sorted(events, key=lambda x: x.get("dur", 0), reverse=True)

for op in sorted_ops[:5]:
    print(f"- {op['name']:30} | Duration: {op['dur'] / 1000:.3f} ms | OpType: {op['args'].get('op_name')}")