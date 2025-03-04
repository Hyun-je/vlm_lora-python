#!/bin/bash

# Basic settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify GPUs to use

# Model and data paths
MODEL_NAME="lmms-lab/llama3-llava-next-8b"
MODEL_DIR="./models/llama3-llava-next-8b"  # Local model directory
DATASET_NAME="lmms-lab/LLaVA-NeXT-Data"
DATASET_DIR="./data"  # Local dataset directory
OUTPUT_DIR="./outputs/llava-next-lora"

# Training hyperparameters
BATCH_SIZE=1
GRAD_ACCUMULATION=16
NUM_EPOCHS=3
LEARNING_RATE=2e-4
WARMUP_RATIO=0.03

# LoRA parameters
LORA_RANK=64
LORA_ALPHA=16
LORA_DROPOUT=0.1

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if model exists locally
if [ -d "$MODEL_DIR" ]; then
    MODEL_PATH=$MODEL_DIR
else
    MODEL_PATH=$MODEL_NAME
    echo "Warning: Local model not found at $MODEL_DIR, will download from HuggingFace Hub"
fi

# Start training
torchrun --nproc_per_node=4 --master_port=29501 train.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATASET_NAME \
    --dataset_dir $DATASET_DIR \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --torch_dtype float16 \
    --fp16 True \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --evaluation_strategy "no" \
    --report_to "none" \
    --ddp_find_unused_parameters False \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --overwrite_output_dir 