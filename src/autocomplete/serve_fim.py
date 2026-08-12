"""Serve FIM autocomplete model via HTTP API compatible with Zed editor.

Zed supports custom completion providers via its settings.
This server exposes a /v1/completions endpoint that accepts
prefix + suffix and returns the middle (completion).

Run: python src/autocomplete/serve_fim.py --model path/to/model --port 8080
"""

import argparse
import json
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="FIM Autocomplete Server")

tokenizer = None
model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CompletionRequest(BaseModel):
    prefix: str
    suffix: str = ""
    language: str = "text"
    max_tokens: int = 64
    temperature: float = 0.2
    top_p: float = 0.95
    stop: list[str] = ["\n\n", "<|fim_hole|>", "<|endoftext|>"]


class CompletionResponse(BaseModel):
    text: str
    tokens: int
    latency_ms: float


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    start = time.time()
    
    # Format as Qwen FIM
    prompt = f"<|fim_prefix|>{req.prefix}<|fim_suffix|>{req.suffix}<|fim_middle|>"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|fim_hole|>"),
        )
    
    # Extract only the generated middle
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    
    # Clean up: remove fim_hole and other special tokens
    for stop_token in ["<|fim_hole|>", "<|endoftext|>", "<|im_end|>"]:
        generated = generated.split(stop_token)[0]
    
    latency = (time.time() - start) * 1000
    
    return CompletionResponse(
        text=generated,
        tokens=len(outputs[0]) - inputs["input_ids"].shape[1],
        latency_ms=latency
    )


@app.get("/v1/models")
async def list_models():
    return {"models": [{"id": "fim-autocomplete", "name": "FIM Autocomplete"}]}


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "model_loaded": model is not None}


def main():
    global tokenizer, model
    
    parser = argparse.ArgumentParser(description="Serve FIM autocomplete model")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    
    print(f"Model loaded on {DEVICE}")
    print(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
