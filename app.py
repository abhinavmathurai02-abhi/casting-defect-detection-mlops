"""app.py — Stage 3: FastAPI inference service.

Implement /health (liveness + loaded model info) and POST /predict (multipart image upload
→ {label, prob_defect, confidence}). Load the model once at startup; log every prediction
to artifacts/predictions.log.   Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io, json, time
from contextlib import asynccontextmanager
from pathlib import Path

import torch, torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
from datetime import datetime

import config
from src import data_prep
from src.model import load_model

_state = {"model": None, "tf": None, "meta": {}}


def _load():
    """
    Load the trained model and inference resources.
    """

    if not config.MODEL_PATH.exists():
        return

    _state["model"] = load_model()

    _state["model"].eval()

    _state["tf"] = data_prep.get_transforms(
        train=False
    )
    _state["loaded_at"] = time.time()

    if config.MODEL_META_PATH.exists():

        _state["meta"] = json.loads(
            config.MODEL_META_PATH.read_text()
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load(); yield


app = FastAPI(title="Casting Defect Detection API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """
    Health endpoint for service monitoring.
    """

    model_loaded = _state["model"] is not None

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "classes": config.CLASS_TO_IDX,
        "positive_class": config.POSITIVE_CLASS,
        "model_info": _state["meta"],
        "registered_model": config.REGISTERED_MODEL,
        "loaded_at": _state.get("loaded_at"),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """
    Predict whether the uploaded casting image is defective.
    """

    # -------------------------------------------------
    # Step 1: Check model availability
    # -------------------------------------------------
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )
    
    # -------------------------------------------------
    # Start timing
    # -------------------------------------------------
    start_time = time.time()

    # -------------------------------------------------
    # Step 2: Read and validate uploaded image
    # -------------------------------------------------
    try:
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("L")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )

    # -------------------------------------------------
    # Step 3: Preprocess image
    # -------------------------------------------------
    x = _state["tf"](image)

    x = x.unsqueeze(0)

    # -------------------------------------------------
    # Step 4: Model inference
    # -------------------------------------------------
    with torch.no_grad():

        logits = _state["model"](x)

        probs = F.softmax(
            logits,
            dim=1,
        )

        prob_defect = float(
            probs[0, config.POSITIVE_IDX]
        )

        pred = int(
            probs.argmax(dim=1)
        )

    # -------------------------------------------------
    # Step 5: Prepare response
    # -------------------------------------------------
    label = config.IDX_TO_CLASS[pred]

    confidence = float(
        probs.max()
    )

    response = {
        "label": label,
        "is_defective": pred == config.POSITIVE_IDX,
        "prob_defect": round(prob_defect, 4),
        "confidence": round(confidence, 4),
    }

    inference_time_ms = round(
    (time.time() - start_time) * 1000,
    2,
    )

    log_record = {
    "timestamp": datetime.now().isoformat(),
    "filename": file.filename,
    "label": response["label"],
    "prob_defect": response["prob_defect"],
    "confidence": response["confidence"],
    "inference_time_ms": inference_time_ms,
     }
    
    # -------------------------------------------------
    # Step 6: Log prediction
    # -------------------------------------------------
    with open(
        config.PREDICTIONS_LOG,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(log_record) + "\n"
        )

    # -------------------------------------------------
    # Step 7: Return response
    # -------------------------------------------------
    return response
