#!/bin/bash

### Supported Models
# Qwen/Qwen2. 5-VL-7B-Instruct
# Qwen/Qwen2.5-VL-32B-Instruct
# Qwen/Qwen2. 5-VL-72B-Instruct
# google/gemma-3-4b-pt
# google/gemma-3-12b-pt
# google/gemma-3-27b-pt

### Supported Datasets
# HuggingFaceM4/ChartQA
# xai-org/RealworldQA
# Lmms-lab/RealWorldQA


## Start training
accelerate launch train.py \
    --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_name "HuggingFaceM4/ChartQA" \
    --num_train_epochs 3 \
    --output_dir "results"