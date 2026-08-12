"""LoRA fine-tuning for Qwen3.6-35B-A3B on RunPod.

Base: Qwen/Qwen3.6-35B-A3B (35B total, 3B active MoE, 256K context)
Uses QLoRA (4-bit NF4) for memory efficiency — fits on single A100 80GB.
For RTX 4090 (24GB), use LoRA r=32 and max_seq_length=8192.

Usage:
    python src/train/train_lora.py --config configs/training_agentic.json
    python src/train/train_lora.py --model Qwen/Qwen3.6-35B-A3B --dataset data/formatted/agentic_sft.jsonl
"""

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen3.6-35B-A3B",
    "dataset_path": "data/formatted/agentic_sft.jsonl",
    "output_dir": "output/lora_qwen36_agentic",
    "lora_r": 128,
    "lora_alpha": 256,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "max_seq_length": 32768,
    "logging_steps": 10,
    "save_steps": 200,
    "save_total_limit": 3,
    "bf16": True,
    "gradient_checkpointing": True,
    "optim": "adamw_torch_fused",
    "seed": 42,
    "report_to": "wandb",
    "dataset_num_proc": 8,
    "packing": True,
}


def load_training_config(config_path=None):
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return {**DEFAULT_CONFIG, **json.load(f)}
    return DEFAULT_CONFIG


def print_config_table(config):
    table = Table(title="Training Configuration", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for k, v in config.items():
        table.add_row(str(k), str(v))
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen3.6-35B-A3B")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    parser.add_argument("--output", type=str, default=None, help="Override output dir")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    args = parser.parse_args()

    config = load_training_config(args.config)
    if args.model: config["model_name"] = args.model
    if args.dataset: config["dataset_path"] = args.dataset
    if args.output: config["output_dir"] = args.output
    if args.epochs: config["num_train_epochs"] = args.epochs
    if args.lr: config["learning_rate"] = args.lr
    if args.lora_r:
        config["lora_r"] = args.lora_r
        config["lora_alpha"] = args.lora_r * 2
    if args.no_wandb: config["report_to"] = "none"
    if args.max_seq_len: config["max_seq_length"] = args.max_seq_len

    print_config_table(config)

    console.print(f"\n[bold cyan]Loading tokenizer: {config['model_name']}[/bold cyan]")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[bold cyan]Loading model with 4-bit NF4 quantization...[/bold cyan]")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    console.print("[bold cyan]Setting up LoRA...[/bold cyan]")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    console.print(f"[bold cyan]Loading dataset: {config['dataset_path']}[/bold cyan]")
    dataset = load_dataset("json", data_files=config["dataset_path"], split="train")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    console.print(f"Dataset size: {len(dataset)}")

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}
    dataset = dataset.map(format_example, remove_columns=dataset.column_names, num_proc=config["dataset_num_proc"])

    console.print(f"\n[bold green]Sample (first 500 chars):[/bold green]")
    console.print(dataset[0]["text"][:500] + "...")

    console.print("\n[bold cyan]Setting up SFT trainer...[/bold cyan]")
    sft_config = SFTConfig(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        max_seq_length=config["max_seq_length"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=config["optim"],
        seed=config["seed"],
        report_to=config["report_to"],
        packing=config["packing"],
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer
    )

    console.print("\n[bold yellow]Starting training...[/bold yellow]\n")
    trainer.train()

    console.print("\n[bold green]Saving LoRA adapter...[/bold green]")
    final_dir = os.path.join(config["output_dir"], "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    console.print(f"\n[bold green]Training complete! Output: {final_dir}[/bold green]")
    console.print(f"Merge: python src/train/merge_lora.py --base {config['model_name']} --adapter {final_dir} --output {config['output_dir']}/merged")
    console.print(f"Deploy: vllm serve {config['output_dir']}/merged --tool-call-parser qwen3_xml --reasoning-parser qwen3 --max-model-len 262144")


if __name__ == "__main__":
    main()
