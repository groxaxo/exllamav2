#!/usr/bin/env python3
"""
OpenAI-compatible FastAPI server for Qwen 3.5 EXL2 model.
Serves on GPU 0 + GPU 1 (two RTX 3090s) with 262144-token context.

Uses ExLlamaV2StreamingGenerator (DynamicGenerator is unsupported for
recurrent GatedDeltaNet layers; see exllamav2/cache.py).

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions   (streaming + non-streaming)
  POST /v1/completions        (streaming + non-streaming)

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python serve_qwen35_exl2.py
"""

import sys, os, time, json, uuid, argparse, asyncio, threading
from typing import Optional, List, AsyncIterator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler

# --------------------------------------------------------------------------- #
# Defaults (overridable via CLI)
# --------------------------------------------------------------------------- #
MODEL_DIR    = "/home/op/models/Qwen3.5-2B-Gorgona-EXL2"
MAX_SEQ_LEN  = 262144          # model hard max; ~12 GB KV cache on full 262144 ctx
GPU_SPLIT_GB = [23.0, 23.0]   # GB of model weight allowed per GPU; cache fills rest
MODEL_NAME   = "qwen3.5-2b-gorgona-exl2"

# runtime globals (populated in load_model_sync)
model     = None
cfg       = None
cache     = None
tokenizer = None
generator = None
gen_lock  = threading.Lock()   # StreamingGenerator is not re-entrant

app = FastAPI(title="Qwen 3.5 EXL2 Server", version="1.0.0")

# --------------------------------------------------------------------------- #
# Pydantic request/response schemas
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    min_p: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    enable_thinking: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    min_p: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

# --------------------------------------------------------------------------- #
# Prompt formatting  (Qwen3.5 ChatML)
# --------------------------------------------------------------------------- #
def format_chat_prompt(messages: List[ChatMessage], enable_thinking: bool = False) -> str:
    buf = []
    for msg in messages:
        content = msg.content
        if msg.role == "assistant" and "</think>" in content:
            content = content.split("</think>")[-1].lstrip("\n")
        buf.append(f"<|im_start|>{msg.role}\n{content}<|im_end|>\n")
    if enable_thinking:
        buf.append("<|im_start|>assistant\n<think>\n")
    else:
        buf.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(buf)

# --------------------------------------------------------------------------- #
# Sampler builder
# --------------------------------------------------------------------------- #
def make_sampler(req) -> ExLlamaV2Sampler.Settings:
    s = ExLlamaV2Sampler.Settings()
    s.temperature              = float(req.temperature or 0.7)
    s.top_p                    = float(req.top_p or 0.9)
    s.top_k                    = int(req.top_k or 50)
    s.min_p                    = float(req.min_p or 0.0)
    s.token_repetition_penalty = float(req.repetition_penalty or 1.0)
    return s

# --------------------------------------------------------------------------- #
# Generation helpers
# --------------------------------------------------------------------------- #
def _run_generate(prompt: str, max_new_tokens: int, settings, stop=None) -> str:
    with gen_lock:
        stop_conds = [tokenizer.eos_token_id, "<|im_end|>"]
        if stop:
            stop_conds.extend(stop)
        generator.set_stop_conditions(stop_conds)
        cache.current_seq_len = 0
        input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
        generator.begin_stream_ex(input_ids, settings, token_healing=False)
        chunks = []
        for _ in range(max_new_tokens):
            chunk, eos, *_ = generator.stream()
            chunks.append(chunk)
            if eos:
                break
        return "".join(chunks)

async def _stream_generate(prompt: str, max_new_tokens: int, settings, stop=None) -> AsyncIterator[str]:
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _worker():
        with gen_lock:
            stop_conds = [tokenizer.eos_token_id, "<|im_end|>"]
            if stop:
                stop_conds.extend(stop)
            generator.set_stop_conditions(stop_conds)
            cache.current_seq_len = 0
            input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
            generator.begin_stream_ex(input_ids, settings, token_healing=False)
            for _ in range(max_new_tokens):
                chunk, eos, *_ = generator.stream()
                if chunk:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                if eos:
                    break
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item

# --------------------------------------------------------------------------- #
# SSE helpers
# --------------------------------------------------------------------------- #
def _sse_chat(content: str, cid: str, ts: int) -> str:
    return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": ts,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}) + "\n\n"

def _sse_chat_done(cid: str, ts: int) -> str:
    return ("data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": ts,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
        + "data: [DONE]\n\n")

def _sse_comp(text: str, cid: str, ts: int) -> str:
    return "data: " + json.dumps({"id": cid, "object": "text_completion", "created": ts,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "text": text, "finish_reason": None}]}) + "\n\n"

def _sse_comp_done(cid: str, ts: int) -> str:
    return ("data: " + json.dumps({"id": cid, "object": "text_completion", "created": ts,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]}) + "\n\n"
        + "data: [DONE]\n\n")

# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "max_seq_len": MAX_SEQ_LEN,
            "cache_seq_len": cache.current_seq_len if cache else 0}

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model",
        "created": int(time.time()), "owned_by": "groxaxo",
        "max_context_length": MAX_SEQ_LEN}]}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prompt   = format_chat_prompt(req.messages, enable_thinking=bool(req.enable_thinking))
    n_tokens = req.max_tokens or 512
    settings = make_sampler(req)
    cid, ts  = "chatcmpl-" + uuid.uuid4().hex[:12], int(time.time())

    if req.stream:
        async def event_stream():
            yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk",
                "created": ts, "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
            async for chunk in _stream_generate(prompt, n_tokens, settings, req.stop):
                yield _sse_chat(chunk, cid, ts)
            yield _sse_chat_done(cid, ts)
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text = await asyncio.get_event_loop().run_in_executor(
        None, _run_generate, prompt, n_tokens, settings, req.stop)
    pt = tokenizer.encode(prompt, encode_special_tokens=True).shape[-1]
    ct = tokenizer.encode(text,   encode_special_tokens=True).shape[-1]
    return {"id": cid, "object": "chat.completion", "created": ts, "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    n_tokens = req.max_tokens or 512
    settings = make_sampler(req)
    cid, ts  = "cmpl-" + uuid.uuid4().hex[:12], int(time.time())

    if req.stream:
        async def event_stream():
            async for chunk in _stream_generate(req.prompt, n_tokens, settings, req.stop):
                yield _sse_comp(chunk, cid, ts)
            yield _sse_comp_done(cid, ts)
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text = await asyncio.get_event_loop().run_in_executor(
        None, _run_generate, req.prompt, n_tokens, settings, req.stop)
    pt = tokenizer.encode(req.prompt, encode_special_tokens=True).shape[-1]
    ct = tokenizer.encode(text,       encode_special_tokens=True).shape[-1]
    return {"id": cid, "object": "text_completion", "created": ts, "model": MODEL_NAME,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}

# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model_sync():
    global model, cfg, cache, tokenizer, generator
    print("\n" + "="*60)
    print("  Qwen 3.5 EXL2 Server  --  GPU 0 + GPU 1")
    print("  Model  :", MODEL_DIR)
    print("  Context:", f"{MAX_SEQ_LEN:,} tokens")
    print("="*60 + "\n")

    cfg             = ExLlamaV2Config(MODEL_DIR)
    cfg.max_seq_len = MAX_SEQ_LEN          # honour --max-seq-len flag
    cfg.prepare()
    model = ExLlamaV2(cfg)

    print("Allocating cache...")
    cache = ExLlamaV2Cache(model, max_seq_len=MAX_SEQ_LEN, lazy=True)

    print("Loading weights (GPU 0 + GPU 1, auto-split)...")
    model.load_autosplit(cache, progress=True)

    tokenizer = ExLlamaV2Tokenizer(cfg)
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    generator.set_stop_conditions([tokenizer.eos_token_id, "<|im_end|>"])

    print("Warming up CUDA kernels...")
    warmup_ids = tokenizer.encode("Hello")
    cache.current_seq_len = 0
    with torch.no_grad():
        model.forward(warmup_ids, cache=None)
    cache.current_seq_len = 0

    for i in range(torch.cuda.device_count()):
        used  = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {used:.2f} GB / {total:.2f} GB")
    print(f"\n  Ready -- {MAX_SEQ_LEN:,} token context on GPU 0+1\n")

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    global MODEL_DIR, MAX_SEQ_LEN, GPU_SPLIT_GB
    parser = argparse.ArgumentParser(description="Qwen 3.5 EXL2 OpenAI-compatible server")
    parser.add_argument("--host",        default="0.0.0.0")
    parser.add_argument("--port",        type=int,   default=8080)
    parser.add_argument("--model-dir",   default=MODEL_DIR)
    parser.add_argument("--max-seq-len", type=int,   default=MAX_SEQ_LEN)
    args = parser.parse_args()

    MODEL_DIR   = args.model_dir
    MAX_SEQ_LEN = args.max_seq_len

    load_model_sync()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
