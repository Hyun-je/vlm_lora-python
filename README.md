# Vision-Language Model LoRA Fine-tuning

This repository contains code for LoRA (Low-Rank Adaptation) fine-tuning of vision-language models, with specific support for LLaVA-NeXT and similar models.

## Features

- LoRA fine-tuning for vision-language models
- Multi-GPU training support for NVIDIA H100
- Compatible with Hugging Face's transformers and PEFT libraries
- Optimized for the LLaVA-NeXT architecture
- Support for custom datasets and model configurations

## Requirements

- Python 3.8+
- NVIDIA H100 GPUs
- CUDA 12.0+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vlm_lora-python.git
cd vlm_lora-python
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To start training with default parameters:

```bash
python train.py \
    --model_name_or_path "lmms-lab/llama3-llava-next-8b" \
    --dataset_name "lmms-lab/LLaVA-NeXT-Data" \
    --output_dir "./output" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --lora_rank 64 \
    --lora_alpha 16
```

### Configuration

Key parameters:
- `--model_name_or_path`: Base model to fine-tune
- `--dataset_name`: Training dataset
- `--lora_rank`: Rank of LoRA matrices
- `--lora_alpha`: Alpha parameter for LoRA scaling
- `--learning_rate`: Learning rate for training
- `--num_train_epochs`: Number of training epochs
- `--gradient_accumulation_steps`: Number of steps for gradient accumulation
- `--per_device_train_batch_size`: Batch size per GPU

## Model Architecture

The code is primarily designed for the LLaVA-NeXT model but is compatible with other vision-language models that follow a similar architecture. The LoRA adaptation is applied to key attention layers for efficient fine-tuning.

## Dataset

The default dataset is LLaVA-NeXT-Data, which includes:
- High-quality image-text pairs
- Multi-turn conversations
- Rich visual instruction data

## License

[Your chosen license]
