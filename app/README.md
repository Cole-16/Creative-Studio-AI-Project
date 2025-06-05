## Overview 

This section holds all of the projects python scripts that will be needed to convert the models, run performance testing, and all other functionality. This will grow as the project expands.

## Current files

### convert_gpt2_to_onnx.py
This script converts the openai-community/gpt2 huggingface model to onnx and then runs through 100 tests to create a performance report so that we  can see exactly how the model is performing on the system.

Example output:
![image](https://github.com/user-attachments/assets/dc48c495-2554-49f7-a1de-810498503b82)



### convert_diffusion_to_onnx_with_stats.py
This scirpt converts the CompVis/stable-diffusion-v1-4 huggingface model to onnx and then runs through creating an image based on the prompt given and gives a performance report so that we can see how the model is performing on the system. It also gives the CLIP score to determine how accurate the image is based on the prompt given.

Example output: 
![image](https://github.com/user-attachments/assets/614c095f-d2f3-4c4c-9b9d-73c69a0a4065)



### Quantization Stats 

Base models: 

![image](https://github.com/user-attachments/assets/616f8b4e-464e-489c-a89c-6dd4d858d066)

Quantized Models: 

![image](https://github.com/user-attachments/assets/6c56d152-8546-4af5-a7be-8191306e3b23)

