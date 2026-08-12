"""Unsloth QLoRA fine-tuning for the FIM / edit-prediction model (zed-industries/zeta-2).

Replaces the plain HF/peft path in train_fim.py. That path materialised fp32 logits
over zeta-2's ~155k vocab, which OOM'd at batch 4 / seq 4096 on a 24GB card and forced
seq 2048 / batch 2 at ~22.6 s/it. Unsloth's fused cross-entropy never materialises those
logits, so the same card fits longer sequences and larger batches.

The dataset already carries FIM markers in its "text" field — do NOT re-template it.

Usage:
    python src/autocomplete/train_fim_unsloth.py \
        --config configs/training_fim_unsloth.json \
        --dataset data/formatted/fim_combined_dedup.jsonl --no-wandb
"""

import os
import json
import argparse
import dataclasses
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CONFIG = {
    "model_name": "zed-industries/zeta-2",
    "dataset_path": "data/formatted/fim_combined_dedup.jsonl",
    "output_dir": "output/fim_autocomplete",
    "load_in_4bit": True,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 2,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "max_seq_length": 4096,
    "logging_steps": 5,
    "save_steps": 500,
    "save_total_limit": 2,
    "bf16": True,
    "gradient_checkpointing": True,
    "optim": "adamw_8bit",
    "seed": 42,
    "report_to": "wandb",
    "dataset_num_proc": 8,
    "packing": True,
}


def prepack(dataset, tokenizer, seq_len, num_proc, console):
    """Concatenate examples and chunk into exact seq_len blocks.

    TRL's own packing=True silently no-ops on some versions (observed: 0.15.2 returned
    a 1.00x reduction), which triples step count and cost. Doing it here makes the
    reduction explicit and identical across TRL versions.
    """
    eos = tokenizer.eos_token_id

    def _tok(batch):
        ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
        return {"ids": [x + [eos] for x in ids]}

    ds = dataset.map(_tok, batched=True, batch_size=1000,
                     remove_columns=dataset.column_names, num_proc=num_proc,
                     desc="tokenizing")

    def _group(batch):
        buf = []
        for ids in batch["ids"]:
            buf.extend(ids)
        n = (len(buf) // seq_len) * seq_len
        chunks = [buf[i:i + seq_len] for i in range(0, n, seq_len)]
        return {"input_ids": chunks, "attention_mask": [[1] * seq_len for _ in chunks]}

    ds = ds.map(_group, batched=True, batch_size=1000,
                remove_columns=["ids"], num_proc=num_proc, desc="packing")
    console.print(f"[green]Pre-packed into {len(ds):,} sequences of {seq_len} tokens[/green]")
    return ds


def main():
    parser = argparse.ArgumentParser(description="Unsloth QLoRA FIM training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config.update(json.load(f))
    if args.model: config["model_name"] = args.model
    if args.dataset: config["dataset_path"] = args.dataset
    if args.output: config["output_dir"] = args.output
    if args.epochs: config["num_train_epochs"] = args.epochs
    if args.lr: config["learning_rate"] = args.lr
    if args.lora_r:
        config["lora_r"] = args.lora_r
        config["lora_alpha"] = args.lora_r * 2
    if args.max_seq_len: config["max_seq_length"] = args.max_seq_len
    if args.batch_size: config["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum: config["gradient_accumulation_steps"] = args.grad_accum
    if args.no_wandb: config["report_to"] = "none"

    table = Table(title="FIM Training Configuration (Unsloth + QLoRA 4-bit)")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for k, v in config.items():
        table.add_row(str(k), str(v))
    console.print(table)

    console.print("\n[bold cyan]Loading Unsloth...[/bold cyan]")
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    # full_finetuning= only exists on newer Unsloth; this pod pins an older build to
    # match transformers 4.48 / trl 0.15, so pass it only if the signature accepts it.
    import inspect
    console.print(f"[bold cyan]Loading model: {config['model_name']} (4-bit NF4)[/bold cyan]")
    load_kwargs = dict(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
    )
    if "full_finetuning" in inspect.signature(FastLanguageModel.from_pretrained).parameters:
        load_kwargs["full_finetuning"] = False
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[bold cyan]Adding LoRA adapters...[/bold cyan]")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    console.print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # Dataset already contains FIM markers in "text" — no chat template, no re-templating.
    console.print(f"[bold cyan]Loading dataset: {config['dataset_path']}[/bold cyan]")
    dataset = load_dataset("json", data_files=config["dataset_path"], split="train")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    keep = [c for c in dataset.column_names if c != "text"]
    if keep:
        dataset = dataset.remove_columns(keep)
    console.print(f"  Size: {len(dataset):,}")
    console.print("[bold green]Sample (first 200 chars):[/bold green]")
    console.print(repr(dataset[0]["text"][:200]))

    raw_n = len(dataset)
    dataset = prepack(dataset, tokenizer, config["max_seq_length"],
                      config["dataset_num_proc"], console)

    # TRL renamed max_seq_length -> max_length around 0.20 and silently drops the
    # wrong one, which is how packing previously no-op'd. Build against the real fields.
    _fields = {f.name for f in dataclasses.fields(SFTConfig)}
    sft_kwargs = {
        "output_dir": config["output_dir"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "num_train_epochs": config["num_train_epochs"],
        "learning_rate": config["learning_rate"],
        "warmup_ratio": config["warmup_ratio"],
        "lr_scheduler_type": config["lr_scheduler_type"],
        "logging_steps": config["logging_steps"],
        "save_steps": config["save_steps"],
        "save_total_limit": config["save_total_limit"],
        "bf16": config["bf16"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": config["optim"],
        "seed": config["seed"],
        "report_to": config["report_to"],
        # Dataset is already tokenized and packed above; TRL must not re-process it.
        "packing": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "dataset_num_proc": config["dataset_num_proc"],
        "remove_unused_columns": False,
    }
    _seq_field = "max_length" if "max_length" in _fields else "max_seq_length"
    sft_kwargs[_seq_field] = config["max_seq_length"]

    dropped = sorted(k for k in sft_kwargs if k not in _fields)
    if dropped:
        console.print(f"[yellow]SFTConfig does not accept (dropping): {dropped}[/yellow]")
        sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in _fields}
    console.print(f"[cyan]Sequence length passed as [bold]{_seq_field}={config['max_seq_length']}[/bold][/cyan]")

    # Explicit collator: builds labels from input_ids for causal LM. All sequences are
    # exactly max_seq_length, so no padding occurs.
    from transformers import DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer_kwargs = dict(
        model=model,
        args=SFTConfig(**sft_kwargs),
        train_dataset=dataset,
        data_collator=collator,
    )
    _sig = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in _sig else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    _packed = len(trainer.train_dataset)
    _ratio = raw_n / max(_packed, 1)
    _eff = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    console.print(
        f"[bold]Packing check:[/bold] {raw_n:,} raw -> {_packed:,} packed "
        f"sequences ({_ratio:.2f}x reduction)"
    )
    console.print(
        f"[bold]Expected steps:[/bold] {_packed * config['num_train_epochs'] // _eff:,} "
        f"({config['num_train_epochs']} epochs, effective batch {_eff})"
    )
    if _ratio < 1.05:
        console.print("[bold red]WARNING: packing had no effect — steps/cost will be far higher than planned.[/bold red]")

    console.print("\n[bold yellow]Starting FIM training...[/bold yellow]\n")
    trainer.train()

    # Defensive save: adapter first (tiny, always), then the q4 merge.
    final_dir = os.path.join(config["output_dir"], "final")
    merged_dir = os.path.join(config["output_dir"], "merged_4bit")

    console.print("\n[bold green]Saving LoRA adapter...[/bold green]")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    console.print(f"  [green]LoRA adapter saved:[/green] {final_dir}")

    try:
        console.print("[bold cyan]Saving merged 4-bit model...[/bold cyan]")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_4bit_forced")
        console.print(f"  [green]Merged 4-bit model saved:[/green] {merged_dir}")
    except Exception as e:
        console.print(f"  [red]Merged 4-bit save FAILED (adapter is safe):[/red] {e}")

    console.print("\n[bold green]FIM training complete![/bold green]")
    console.print(f"  LoRA adapter: {final_dir}")
    console.print(f"  Merged 4-bit: {merged_dir}")


if __name__ == "__main__":
    main()
