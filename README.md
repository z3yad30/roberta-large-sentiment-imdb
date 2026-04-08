# Sentiment Analysis API

This API uses a fine-tuned RoBERTa-large model for IMDB sentiment prediction.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the API: `uvicorn app.main:app --reload`

## Endpoints
- `/health`: Check if the API is running.