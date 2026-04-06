"""GhostLM FastAPI inference server — serves model predictions via REST API."""

import os
import sys
import time
from dataclasses import fields
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("pip install fastapi uvicorn pydantic")
    sys.exit(1)

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


class GenerateRequest(BaseModel):
    """Request body for text generation endpoint."""

    prompt: str = Field(
        ..., min_length=1, max_length=1000, example="A SQL injection attack works by"
    )
    max_tokens: int = Field(default=150, ge=10, le=500)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=0, le=100)


class GenerateResponse(BaseModel):
    """Response body for text generation endpoint."""

    prompt: str
    generated: str
    tokens_generated: int
    time_seconds: float
    model_version: str


class HealthResponse(BaseModel):
    """Response body for health check endpoint."""

    status: str
    model_loaded: bool
    model_params: int
    device: str
    version: str = "0.1.0"


# Global model state
model = None
tokenizer = None
config = None
device = "cpu"


def load_model_on_startup(checkpoint_path: str = None):
    """Load GhostLM model on server startup.

    Attempts to load from the provided checkpoint path. Falls back to
    a randomly initialized ghost-tiny model if no checkpoint is found.
    Sets the global model, tokenizer, config, and device variables.

    Args:
        checkpoint_path: Path to .pt checkpoint file, or None for random init.
    """
    global model, tokenizer, config, device

    tokenizer = GhostTokenizer()

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading GhostLM from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        saved_config = checkpoint["config"]
        config = GhostLMConfig(**{
            f.name: saved_config[f.name]
            for f in fields(GhostLMConfig)
            if f.name in saved_config
        })

        model = GhostLM(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        device = "cpu"
        print(f"  Loaded: {model.num_params():,} params, step={checkpoint.get('step', 0)}")
    else:
        print("No checkpoint found — using random ghost-tiny weights.")
        config = GhostLMConfig.from_preset("ghost-tiny")
        config.vocab_size = 50261
        config.context_length = 128
        model = GhostLM(config)
        model.eval()
        device = "cpu"


app = FastAPI(
    title="GhostLM API",
    description="Open-source cybersecurity language model inference API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Load the GhostLM model when the server starts."""
    checkpoint = os.environ.get("GHOSTLM_CHECKPOINT") or "checkpoints/best_model.pt"
    load_model_on_startup(checkpoint)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Return the health status of the GhostLM API server.

    Returns:
        HealthResponse with model load status, parameter count, and device.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_params=model.num_params() if model else 0,
        device=device,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a user prompt using the loaded GhostLM model.

    Args:
        request: GenerateRequest with prompt, max_tokens, temperature, and top_k.

    Returns:
        GenerateResponse with the generated text, token count, and timing.

    Raises:
        HTTPException: 503 if model is not loaded, 500 on generation error.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        ids = tokenizer.encode(request.prompt)
        input_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

        top_k_val = request.top_k if request.top_k > 0 else None

        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=top_k_val,
            )
        elapsed = time.time() - t0

        generated_text = tokenizer.decode(output[0].tolist())

        # Strip prompt from output
        if generated_text.startswith(request.prompt):
            generated_text = generated_text[len(request.prompt):].strip()

        tokens_generated = output.size(1) - input_tensor.size(1)

        return GenerateResponse(
            prompt=request.prompt,
            generated=generated_text,
            tokens_generated=tokens_generated,
            time_seconds=round(elapsed, 3),
            model_version="0.1.0",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/")
async def root():
    """Return API root information.

    Returns:
        Dictionary with API name, docs URL, and health endpoint.
    """
    return {"message": "GhostLM API", "docs": "/docs", "health": "/health"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GhostLM FastAPI Inference Server")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    load_model_on_startup(args.checkpoint)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
