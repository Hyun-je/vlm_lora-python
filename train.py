import sys, os
import importlib
import torch
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers import BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from accelerate import PartialState

from model import model_map
from dataset import dataset_map


@dataclass
class ModelArguments:
    model_name: str = field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "Path to pretrained model or identifier from huggingface.co/models"}
    )
    model_cache_dir: str = field(
        default="/emb_opt/huggingface_cache/models",
        metadata={"help": "Cache directory containing the model files"}
    )

@dataclass
class LoraArguments:
    alpha: float = field (
        default=16,
        metadata={"help": "Scaling factor for LoRA layers"}
    )
    dropout: float = field (
        default=0.05,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    rank: int = field (
        default=16,
        metadata={"help": "Rank of the LoRA matrices"}
    )

@dataclass
class DataArguments:
    dataset_name: list[str] = field(
        default_factory=list,
        metadata={"help": "The name of the dataset to use (via the datasets library) or local path"}
    )
    dataset_cache_dir: str = field(
        default="/emb_opt/huggingface_cache/datasets",
        metadata={"help": "Cache directory containing the dataset files"}
    )

@dataclass
class TrainArguments:
    output_dir: str = field(
        default="results",
        metadata={"help": "The name of the dataset to use (via the datasets library) or local path"}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Number of mini batch"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "tensorboard / wandb"}
    )




def main():

    os.environ["WANDB_MODE"] = "offline"

    parser = HfArgumentParser((ModelArguments, LoraArguments, DataArguments, TrainArguments))
    model_args, lora_args, data_args, train_args = parser.parse_args_into_dataclasses()


    train_dataset_list = []
    eval_dataset_list = []
    for dataset_name in data_args.dataset_name:
        dataset_module = importlib.import_module(f'dataset.{dataset_map[dataset_name]}')
        train_dataset, eval_dataset, _ = dataset_module.load(cache_dir=data_args.dataset_cache_dir)
        print("type(train_dataset)", type(train_dataset))
        train_dataset_list.append(train_dataset)
        eval_dataset_list.append(eval_dataset)

    train_dataset = concatenate_datasets(train_dataset_list)
    eval_dataset = concatenate_datasets(eval_dataset_list)


    ### Load the Quantized Model for Training ###
    model_name = model_map[model_args.model_name]
    model_module = importlib.import_module(f'model.{model_name}')


    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_kwargs = dict(
        device_map={'':PartialState().process_index},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        cache_dir=model_args.model_cache_dir
    )

    model, processor, collate_fn = model_module.load(**model_kwargs)


    ### Set Up QLoRA and SFTConfig ###

    # Configure LoRA
    peft_config = LoraConfig(
        lora_alpha=lora_args.alpha,
        lora_dropout=lora_args.dropout,
        r=lora_args.rank,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
            # "k_proj",
            # "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj"
        ],
        task_type="CAUSAL_LM",
    )

    # Apply PEFT model adaptation
    peft_model = get_peft_model(model, peft_config)

    # Print trainable parameters
    peft_model.print_trainable_parameters()



    # Configure training arguments
    training_args = SFTConfig(
        output_dir=train_args.output_dir,  # Directory to save the model
        num_train_epochs=train_args.num_train_epochs,  # Number of training epochs
        per_device_train_batch_size=train_args.batch_size,  # Batch size for training
        per_device_eval_batch_size=train_args.batch_size,  # Batch size for evaluation
        gradient_accumulation_steps=8,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=10,  # Steps interval for logging
        eval_steps=100,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=100,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        fp16=True,  # Use bfloat16 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # # Hub and reporting
        report_to=train_args.report_to,  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=1024  # Maximum sequence length for input
    )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()