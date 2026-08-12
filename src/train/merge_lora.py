"""Merge LoRA adapter with base model and save full model for vLLM serving."""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base", type=str, required=True, help="Base model name or path")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA adapter path")
    parser.add_argument("--output", type=str, required=True, help="Output merged model path")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push to")
    args = parser.parse_args()

    console.print(f"[bold cyan]Loading base model: {args.base}[/bold cyan]")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    console.print(f"[bold cyan]Loading LoRA adapter: {args.adapter}[/bold cyan]")
    model = PeftModel.from_pretrained(model, args.adapter)

    console.print("[bold cyan]Merging adapter weights...[/bold cyan]")
    model = model.merge_and_unload()

    console.print(f"[bold green]Saving merged model: {args.output}[/bold green]")
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(args.output)

    if args.push_to_hub:
        console.print(f"[bold cyan]Pushing to HuggingFace Hub: {args.push_to_hub}[/bold cyan]")
        model.push_to_hub(args.push_to_hub, safe_serialization=True)
        tokenizer.push_to_hub(args.push_to_hub)

    console.print(f"\n[bold green]Done! Merged model at: {args.output}[/bold green]")
    console.print(f"Deploy with: vllm serve {args.output} --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
