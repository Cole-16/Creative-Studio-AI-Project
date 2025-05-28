# Content-Creation-AI-Project

## Overview
This repository contains materials for the Content Creation AI project choice  for the "Deploying Edge AI" master's level course. The current models in use for this project are openai-community/gpt2 and CompVis/stable-diffusion-v1-4. These will update as the project expands. The main goal of this project is to complete this project scope. 



## Architecture Diagram

## Quick Start Guide

### If venv is already created.
1. Once the virtual environment is created the first step is to clone the repo down so that you have the code in place in your venv. Once that is done make sure all packages    are installed that are needed with this project. Run "pip install -r requirements.txt" to install the required packages.

2. When that is completed you are ready to test out the models. All python scripts are located under the /app/ folder. Currently there is one file and a readme in that location. The readme will explain in further detail waht each script contains and what it does. It will expand as the project grows. 

3. As the project stands right now to start the model export and performance testing all you need to do is run the "convert_gpt2_to_onnx.py" script: 

    - Make sure you are in the /app/ directory.
    - run "python .\convert_gpt2_to_onnx.py" 
    - You should recieve and output similar to this: 
        ![image](https://github.com/user-attachments/assets/5f205cad-27a7-48dd-9d7e-f60d40a1eb5e)

    
