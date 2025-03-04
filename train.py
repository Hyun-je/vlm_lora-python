import os
import torch
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from pathlib import Path

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="lmms-lab/llama3-llava-next-8b",
        metadata={"help": "Path to pretrained model or identifier from huggingface.co/models"}
    )
    use_4bit: bool = field(default=True, metadata={"help": "Use 4-bit quantization"})
    use_nested_quant: bool = field(default=False, metadata={"help": "Use nested quantization"})
    bnb_4bit_compute_dtype: str = field(default="float16", metadata={"help": "Compute dtype for 4-bit base model"})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "Quantization type (fp4 or nf4)"})

@dataclass
class DataArguments:
    dataset_name: str = field(
        default="lmms-lab/LLaVA-NeXT-Data",
        metadata={"help": "The name of the dataset to use (via the datasets library) or local path"}
    )
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Local directory containing the dataset files"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of training examples."}
    )

@dataclass
class LoraArguments:
    lora_rank: int = field(default=64, metadata={"help": "Rank of LoRA matrices"})
    lora_alpha: int = field(default=16, metadata={"help": "Alpha parameter for LoRA scaling"})
    lora_dropout: float = field(default=0.1, metadata={"help": "Dropout probability for LoRA layers"})
    lora_target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
        metadata={"help": "Comma-separated list of target modules to apply LoRA"}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Load dataset
    if data_args.dataset_dir and os.path.exists(os.path.join(data_args.dataset_dir, "train")):
        logger.info(f"Loading dataset from local directory: {data_args.dataset_dir}")
        dataset = load_from_disk(os.path.join(data_args.dataset_dir, "train"))
        dataset = {"train": dataset}
    else:
        logger.info(f"Downloading dataset: {data_args.dataset_name}")
        dataset = load_dataset(data_args.dataset_name)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_target_modules = lora_args.lora_target_modules.split(",")
    config = LoraConfig(
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Get PEFT model
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Training
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset["train"],
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Start training
    if training_args.resume_from_checkpoint:
        checkpoint = training_args.resume_from_checkpoint
    else:
        checkpoint = None

    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save model
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()
