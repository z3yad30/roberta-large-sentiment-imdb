import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline,
)

# Local imports (assuming schemas.py is in the same package)
from .schemas import (
    SentimentRequest,
    SentimentResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
    ModelMetadata,
)


# =============================================================================
# Project paths – resolved relative to this file's location
# =============================================================================

# Location of this file: sentiment-api/app/main.py
# → go two levels up → sentiment-api/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "roberta-large-imdb")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

# Normalize paths and make them absolute
MODEL_PATH = os.path.normpath(os.path.abspath(MODEL_DIR))
STATIC_PATH = os.path.normpath(os.path.abspath(STATIC_DIR))


# =============================================================================
# Model & label configuration
# =============================================================================

LABEL_MAP: Dict[str, str] = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE",
}

MODEL_METADATA = ModelMetadata(
    name="roberta-large-imdb",
    huggingface_id="roberta-large",
    fine_tuned_on="IMDB",
    language="en",
    classes=["NEGATIVE", "POSITIVE"],
    max_length=512,
    api_version="1.0",
)


# =============================================================================
# Model loading (executed once at startup)
# =============================================================================

def _load_sentiment_pipeline() -> Pipeline:
    """Load tokenizer + model + pipeline once during application startup."""
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(
            f"Model directory not found at: {MODEL_PATH}\n"
            "Expected structure: models/roberta-large-imdb/ with config.json, model.safetensors, etc."
        )

    print(f"Loading model from: {MODEL_PATH}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

        return pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=-1,              # CPU = -1, first GPU = 0, etc.
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        raise RuntimeError(f"Model loading failed:\n{exc}") from exc


# Load once (module-level)
sentiment_pipeline: Pipeline = _load_sentiment_pipeline()


# =============================================================================
# FastAPI application
# =============================================================================

app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description=(
        "Binary sentiment classification (positive / negative) using "
        "a fine-tuned RoBERTa-large model on the IMDB movie reviews dataset."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Serve static frontend files
if not os.path.isdir(STATIC_PATH):
    print(f"Warning: Static directory not found: {STATIC_PATH}")
else:
    app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=Dict[str, str])
async def health_check() -> Dict[str, str]:
    """Simple health endpoint confirming API and model availability."""
    return {
        "status": "healthy",
        "model": MODEL_METADATA.name,
        "version": MODEL_METADATA.api_version,
    }


@app.post("/api/v1/predict", response_model=SentimentResponse)
async def predict_single(request: SentimentRequest) -> SentimentResponse:
    """Single text sentiment prediction."""
    try:
        result = sentiment_pipeline(request.text)[0]
        raw_label = result["label"]
        score = result["score"]

        label = LABEL_MAP.get(raw_label, raw_label)
        confidence = score if label == "POSITIVE" else 1.0 - score

        probabilities = None
        if request.return_probabilities:
            probabilities = {
                "POSITIVE": confidence,
                "NEGATIVE": 1.0 - confidence,
            }

        return SentimentResponse(
            text=request.text,
            label=label,
            confidence=confidence,
            probabilities=probabilities,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        ) from exc


@app.post("/api/v1/batch-predict", response_model=BatchSentimentResponse)
async def predict_batch(request: BatchSentimentRequest) -> BatchSentimentResponse:
    """Batch sentiment prediction for multiple texts."""
    try:
        results = sentiment_pipeline(request.texts)
        predictions: List[SentimentResponse] = []

        for text, res in zip(request.texts, results):
            raw_label = res["label"]
            score = res["score"]

            label = LABEL_MAP.get(raw_label, raw_label)
            confidence = score if label == "POSITIVE" else 1.0 - score

            probabilities = None
            if request.return_probabilities:
                probabilities = {
                    "POSITIVE": confidence,
                    "NEGATIVE": 1.0 - confidence,
                }

            predictions.append(
                SentimentResponse(
                    text=text,
                    label=label,
                    confidence=confidence,
                    probabilities=probabilities,
                )
            )

        return BatchSentimentResponse(predictions=predictions)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(exc)}"
        ) from exc


@app.get("/api/v1/model-info", response_model=ModelMetadata)
async def get_model_info() -> ModelMetadata:
    """Return metadata about the currently loaded model."""
    return MODEL_METADATA


# =============================================================================
# Development server (only used when file is executed directly)
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )