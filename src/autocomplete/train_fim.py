"""Train a custom autocomplete/edit-prediction model for Zed.

Base: zed-industries/zeta-2 (fine-tuned Seed-Coder-8B, Apache-2.0)
This is Zed's own model, purpose-built for edit prediction.
We LoRA fine-tune it on YOUR codebase for personalized completions.

For lighter setup use sweepai/sweep-next-edit-1.5B instead.

Usage:
    python src/autocomplete/train_fim.py --config configs/training_fim.json
"""

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from rich.console import Console

console = Console()

FIM_CONFIG = {
    "model_name": "zed-industries/zeta-2",
    "dataset_path": "data/formatted/fim_combined.jsonl",
    "output_dir": "output/fim_autocomplete",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 2,
    "learning_rate": 5e-4,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "max_seq_length": 4096,
    "logging_steps": 20,
    "save_steps": 500,
    "save_total_limit": 2,
    "bf16": True,
    "gradient_checkpointing": True,
    "optim": "adamw_torch_fused",
    "seed": 42,
    "report_to": "wandb",
    "packing": True,
}


def main():
    parser = argparse.ArgumentParser(description="Train FIM autocomplete model for Zed")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="Override base model")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = dict(FIM_CONFIG)
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config.update(json.load(f))
    if args.model: config["model_name"] = args.model
    if args.dataset: config["dataset_path"] = args.dataset
    if args.output: config["output_dir"] = args.output
    if args.no_wandb: config["report_to"] = "none"

    console.print(f"[bold cyan]Loading tokenizer: {config['model_name']}[/bold cyan]")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[bold cyan]Loading model with 4-bit quantization...[/bold cyan]")
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

    console.print("\n[bold yellow]Starting FIM training...[/bold yellow]\n")
    trainer.train()

    final_dir = os.path.join(config["output_dir"], "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    console.print(f"\n[bold green]FIM model saved: {final_dir}[/bold green]")
    console.print(f"Merge: python src/train/merge_lora.py --base {config['model_name']} --adapter {final_dir} --output {config['output_dir']}/merged")
    console.print(f"Serve: vllm serve {config['output_dir']}/merged --served-model-name zeta-custom --enable-prefix-caching --enable-chunked-prefill")


if __name__ == "__main__":
    main()
