"""UnsLoRA fine-tuning for Qwen3.6-35B-A3B on RunPod RTX Pro 6000.

Uses Unsloth's FastLanguageModel with bf16 LoRA (NOT QLoRA — MoE QLoRA not recommended).
Requires: transformers >= 5.2.0, unsloth, RTX Pro 6000 96GB or A100 80GB.

Usage:
    python src/train/train_unsloth.py --config configs/training_agentic.json
"""

import os
import json
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CONFIG = {
    "model_name": "unsloth/Qwen3.6-35B-A3B",
    "dataset_path": "data/formatted/agentic_sft.jsonl",
    "output_dir": "output/lora_qwen36_agentic",
    "qat_scheme": None,
    "load_in_4bit": False,
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
    "optim": "adamw_8bit",
    "seed": 42,
    "report_to": "wandb",
    "dataset_num_proc": 8,
    "packing": True,
}


def prepack(dataset, tokenizer, seq_len, num_proc, console):
    """Concatenate examples and chunk into exact seq_len blocks.

    TRL's packing=True silently no-ops on some versions (observed 1.00x on 0.15.2),
    which doubles step count and cost. Doing it here is explicit and version-proof.
    """
    eos = tokenizer.eos_token_id

    def _tok(batch):
        # transformers>=5 loads this model's tokenizer as a multimodal Processor, where a
        # positional arg is treated as an image source. text= forces the text path.
        ids = tokenizer(text=batch["text"], add_special_tokens=False)["input_ids"]
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
    parser = argparse.ArgumentParser(description="UnsLoRA fine-tuning for Qwen3.6-35B-A3B")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--qat-scheme", type=str, default=None,
                        help="QAT scheme: int4, int8-int4, fp8-int4, fp8-fp8. Enables quantization-aware training.")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="QLoRA: load base in 4-bit NF4. Matches a q4 deployment target and frees VRAM for larger batches.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
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
    if args.no_wandb: config["report_to"] = "none"
    if args.max_seq_len: config["max_seq_length"] = args.max_seq_len
    if args.qat_scheme: config["qat_scheme"] = args.qat_scheme
    if args.load_in_4bit: config["load_in_4bit"] = True
    if args.batch_size: config["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum: config["gradient_accumulation_steps"] = args.grad_accum

    qat_scheme = config.get("qat_scheme") or None
    load_in_4bit = bool(config.get("load_in_4bit"))
    if qat_scheme and load_in_4bit:
        raise SystemExit("--qat-scheme and --load-in-4bit are mutually exclusive: QAT fake-quantizes a bf16 base.")

    # Print config
    _mode = f"QAT+LoRA ({qat_scheme})" if qat_scheme else ("QLoRA 4-bit" if load_in_4bit else "bf16 LoRA")
    table = Table(title=f"Training Configuration (Unsloth + {_mode})")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for k, v in config.items():
        table.add_row(str(k), str(v))
    console.print(table)

    # Import Unsloth
    console.print("\n[bold cyan]Loading Unsloth...[/bold cyan]")
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    console.print(f"[bold cyan]Loading model: {config['model_name']} ({_mode})[/bold cyan]")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=load_in_4bit,
        load_in_16bit=not load_in_4bit,
        full_finetuning=False,
    )

    # Add LoRA (+ QAT fake-quantization when qat_scheme is set)
    if qat_scheme:
        console.print(f"[bold magenta]Adding LoRA adapters with QAT (scheme={qat_scheme})...[/bold magenta]")
    else:
        console.print("[bold cyan]Adding LoRA adapters...[/bold cyan]")
    _peft_kwargs = dict(
        r=config["lora_r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )
    if qat_scheme:
        _peft_kwargs["qat_scheme"] = qat_scheme
    model = FastLanguageModel.get_peft_model(model, **_peft_kwargs)

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    console.print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    console.print(f"  Total:     {total:,}")

    # Load dataset. Multiple comma-separated paths are concatenated, which is how the
    # agentic mix combines conversation data (a "messages" column) with document corpora
    # (a "text" column) -- the two schemas are reconciled below.
    paths = [p.strip() for p in str(config["dataset_path"]).split(",") if p.strip()]
    console.print(f"[bold cyan]Loading dataset(s): {paths}[/bold cyan]")
    if len(paths) == 1:
        dataset = load_dataset("json", data_files=paths[0], split="train")
    else:
        from datasets import concatenate_datasets
        parts = []
        for p in paths:
            ds = load_dataset("json", data_files=p, split="train")
            if "messages" in ds.column_names:
                def _render(ex):
                    return {"text": tokenizer.apply_chat_template(
                        ex["messages"], tokenize=False, add_generation_prompt=False)}
                ds = ds.map(_render, remove_columns=ds.column_names,
                            num_proc=config["dataset_num_proc"], desc=f"templating {Path(p).name}")
            else:
                ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
            console.print(f"  {Path(p).name}: {len(ds):,} rows")
            parts.append(ds)
        dataset = concatenate_datasets(parts)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    console.print(f"  Size: {len(dataset):,}")

    # Conversational datasets carry "messages" and need the chat template. Pre-rendered
    # datasets (e.g. next-edit prediction) already carry "text" and must be left alone --
    # re-templating them would wrap plain merge-marker text in chat turns.
    if "messages" in dataset.column_names:
        def format_example(example):
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        dataset = dataset.map(format_example, remove_columns=dataset.column_names,
                              num_proc=config["dataset_num_proc"])
    elif "text" in dataset.column_names:
        console.print("[cyan]Dataset already has 'text' — skipping chat template.[/cyan]")
        extra = [c for c in dataset.column_names if c != "text"]
        if extra:
            dataset = dataset.remove_columns(extra)
    else:
        raise SystemExit(f"Dataset needs a 'messages' or 'text' column; got {dataset.column_names}")
    console.print(f"\n[bold green]Sample (first 300 chars):[/bold green]")
    console.print(dataset[0]["text"][:300] + "...", markup=False)

    raw_n = len(dataset)
    dataset = prepack(dataset, tokenizer, config["max_seq_length"],
                      config["dataset_num_proc"], console)

    # Training config.
    # TRL renamed max_seq_length -> max_length around 0.20; passing the wrong one is
    # silently dropped, which is how a previous run trained unpacked at the default
    # length. Build kwargs against the installed dataclass instead of guessing.
    import dataclasses
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
        "optim": config["optim"],
        "seed": config["seed"],
        "report_to": config["report_to"],
        # Dataset is pre-tokenized and packed above; TRL must not re-process it.
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
    console.print(f"[cyan]Sequence length passed as [bold]{_seq_field}={config['max_seq_length']}[/bold] (trl fields verified)[/cyan]")

    sft_config = SFTConfig(**sft_kwargs)

    import inspect
    from transformers import DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        data_collator=collator,
    )
    _tsig = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in _tsig else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    # Verify packing actually took effect before burning GPU hours on it.
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
        console.print("[bold red]WARNING: packing had no effect — steps/cost will be ~2x higher than planned.[/bold red]")

    # Train
    console.print("\n[bold yellow]Starting training...[/bold yellow]\n")
    trainer.train()

    # === Defensive save: cheapest+most-important artifact first, each step
    # isolated so a late failure never discards a multi-hour run. ===
    final_dir = os.path.join(config["output_dir"], "final")
    merged_dir = os.path.join(config["output_dir"], "merged")
    qat_dir = os.path.join(config["output_dir"], "qat_int4")

    # 1) LoRA adapter — tiny, always save first.
    console.print("\n[bold green]Saving LoRA adapter...[/bold green]")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    console.print(f"  [green]LoRA adapter saved:[/green] {final_dir}")

    # 2) Merged weights. With a 4-bit base the deployment target is q4, so merge to
    #    4-bit — dequantizing to bf16 would write ~70GB we would only re-quantize.
    if load_in_4bit:
        try:
            console.print("[bold cyan]Saving merged 4-bit model...[/bold cyan]")
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_4bit_forced")
            console.print(f"  [green]Merged 4-bit model saved:[/green] {merged_dir}")
        except Exception as e:
            console.print(f"  [red]Merged 4-bit save FAILED (adapter is safe):[/red] {e}")
    else:
        try:
            console.print("[bold cyan]Saving merged bf16 model for vLLM...[/bold cyan]")
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            console.print(f"  [green]Merged bf16 model saved:[/green] {merged_dir}")
        except Exception as e:
            console.print(f"  [red]Merged bf16 save FAILED (adapter is safe):[/red] {e}")

    # 3) QAT int4 export — only when QAT was enabled. TorchAO PTQ format.
    if qat_scheme:
        try:
            console.print(f"[bold magenta]Exporting QAT int4 model (scheme={qat_scheme})...[/bold magenta]")
            from torchao.quantization import Int4WeightOnlyConfig
            model.save_pretrained_torchao(
                qat_dir, tokenizer, torchao_config=Int4WeightOnlyConfig(),
            )
            console.print(f"  [green]QAT int4 model saved:[/green] {qat_dir}")
        except Exception as e:
            console.print(f"  [red]QAT int4 export FAILED (adapter + bf16 merged are safe):[/red] {e}")
            console.print("  [yellow]Convert manually later from the saved adapter/merged model.[/yellow]")

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  LoRA adapter: {final_dir}")
    console.print(f"  Merged model: {merged_dir}")
    if qat_scheme:
        console.print(f"  QAT int4 model: {qat_dir}")
    # Deploy flags depend on what was trained. The agentic model needs Qwen3 tool-call and
    # reasoning parsers; a completion/edit model needs neither and would be broken by them.
    is_chat = "messages" in str(config.get("dataset_path", "")) or "agentic" in config["output_dir"]
    # ...and on which base model: the parser names are per-family, so printing qwen3_xml for a
    # MiniCPM5 run would hand over a command that fails at startup.
    _model = str(config.get("model_name", "")).lower()
    tool_parser, reasoning_parser = ("minicpm5", None) if "minicpm5" in _model else ("qwen3_xml", "qwen3")
    console.print(f"\nDeploy:")
    console.print(f"  vllm serve {merged_dir} \\")
    console.print(f"    --served-model-name {Path(config['output_dir']).name} \\")
    if is_chat:
        console.print(f"    --tool-call-parser {tool_parser} \\")
        console.print(f"    --enable-auto-tool-choice \\")
        if reasoning_parser:
            console.print(f"    --reasoning-parser {reasoning_parser} \\")
    console.print(f"    --max-model-len {config['max_seq_length']} \\")
    console.print(f"    --enable-prefix-caching \\")
    console.print(f"    --enable-chunked-prefill \\")
    console.print(f"    --trust-remote-code")
    console.print("\n[yellow]Note:[/yellow] this tokenizer loads as a multimodal processor -- "
                  "call it as tokenizer(text=...), never positionally, or the text is parsed as an image.")


if __name__ == "__main__":
    main()
