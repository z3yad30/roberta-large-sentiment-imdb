from typing import List, Optional
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    """
    Request model for single text sentiment prediction.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The input text for sentiment analysis (e.g., a movie review)."
    )
    return_probabilities: Optional[bool] = Field(
        False,
        description="Whether to include detailed probabilities in the response."
    )


class SentimentResponse(BaseModel):
    """
    Response model for a single sentiment prediction.
    """
    text: str = Field(..., description="The original input text.")
    label: str = Field(..., description="Predicted sentiment label ('POSITIVE' or 'NEGATIVE').")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (higher is more certain)."
    )
    probabilities: Optional[dict] = Field(
        None,
        description="Detailed probabilities {'POSITIVE': float, 'NEGATIVE': float} (if requested)."
    )


class BatchSentimentRequest(BaseModel):
    """
    Request model for batch text sentiment prediction.
    """
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of input texts for batch sentiment analysis."
    )
    return_probabilities: Optional[bool] = Field(
        False,
        description="Whether to include detailed probabilities for each prediction."
    )


class BatchSentimentResponse(BaseModel):
    """
    Response model for batch sentiment predictions.
    """
    predictions: List[SentimentResponse] = Field(
        ...,
        description="List of sentiment predictions corresponding to the input texts."
    )


class ModelMetadata(BaseModel):
    """
    Metadata about the loaded model.
    """
    name: str = Field(..., description="Model name.")
    huggingface_id: str = Field(..., description="Base Hugging Face model ID.")
    fine_tuned_on: str = Field(..., description="Dataset used for fine-tuning (e.g., 'IMDB').")
    language: str = Field(..., description="Supported language (e.g., 'en').")
    classes: List[str] = Field(..., description="Possible sentiment classes.")
    max_length: int = Field(..., description="Maximum token length supported by the model.")
    api_version: str = Field(..., description="Current API version.")