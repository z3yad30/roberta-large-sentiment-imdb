# IMDB Sentiment Analysis API

**A production-ready FastAPI application for binary sentiment classification of movie reviews using a fine-tuned RoBERTa-large model.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-latest-orange)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Overview

This project delivers a complete, end-to-end sentiment analysis solution for English-language IMDB movie reviews. The core model is **siebert/sentiment-roberta-large-english** (RoBERTa-large fine-tuned on the IMDB dataset), achieving **~93–95% accuracy** on the IMDB test set in zero-shot transfer.

The application consists of:
- A **FastAPI** RESTful backend with single and batch prediction endpoints
- A responsive, modern **HTML/CSS/JavaScript** frontend for interactive use
- Comprehensive **Pydantic** request/response validation and model metadata
- A **Colab-optimized Jupyter notebook** for model evaluation and testing

The system supports both individual review analysis and high-volume batch processing, with optional probability scores and confidence metrics.

## ✨ Key Features

- **Dual-mode interface**: Single review analysis and batch processing (up to 100 reviews)
- **High-performance inference** using the full RoBERTa-large architecture
- **Detailed outputs**: Sentiment label, confidence score, and optional probability distribution
- **Robust error handling** and input validation (text length, batch size limits)
- **Static frontend** served directly by FastAPI for zero-configuration deployment
- **Health check** and model metadata endpoints for monitoring
- **GPU/CPU agnostic** deployment (CPU by default; GPU supported via device mapping)

## 🛠️ Technology Stack

| Layer       | Technology                          | Purpose |
|-------------|-------------------------------------|---------|
| Backend     | FastAPI + Uvicorn                   | REST API framework |
| Model       | Hugging Face `transformers` + PyTorch | RoBERTa-large sentiment pipeline |
| Validation  | Pydantic v2                         | Request/response schemas |
| Frontend    | HTML5, CSS3 (Inter font), Vanilla JS | Responsive user interface |
| Evaluation  | Jupyter Notebook + scikit-learn     | Model testing and metrics |
| Packaging   | `requirements.txt`                  | Reproducible environment |

## 📁 Project Structure

```bash
sentiment-api/
├── static/                     # Frontend assets (served at /static)
│   ├── index.html
│   ├── style.css
│   └── script.js
├── models/
│   └── roberta-large-imdb/     # Pre-downloaded model files
│       ├── config.json
│       ├── merges.txt
│       ├── model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.json
├── app/
│   ├── main.py                 # FastAPI application
│   └── schemas.py              # Pydantic models
├── requirements.txt
├── 11_Task_05.ipynb            # Colab-optimized evaluation notebook
├── Final structure.txt
└── README.md
```
🚀 Installation & Setup

Clone or download the project files.
Create and activate a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

Install dependencies:
```
pip install -r requirements.txt
```

Verify model location: Ensure the models/roberta-large-imdb/ directory contains all required model files (as shown in the structure above).

🏃‍♂️ Running the Application
Start the development server:
Bashcd sentiment-api
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
The application will be available at:
```
Web UI: http://localhost:8000/static/index.html
API Documentation:
Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc


Health check: GET /health
📡 API Endpoints

Method,Endpoint,Description
GET,/health,Service and model status
GET,/api/v1/model-info,Model metadata
POST,/api/v1/predict,Single review prediction
POST,/api/v1/batch-predict,Batch review prediction

MethodEndpointDescriptionGET/healthService and model statusGET/api/v1/model-infoModel metadataPOST/api/v1/predictSingle review predictionPOST/api/v1/batch-predictBatch review prediction
Example request body (single):
```
{
  "text": "This was an absolutely fantastic movie...",
  "return_probabilities": true
}
```
Response:
```
JSON{
  "text": "...",
  "label": "POSITIVE",
  "confidence": 0.9876,
  "probabilities": {
    "POSITIVE": 0.9876,
    "NEGATIVE": 0.0124
  }
}
```
📊 Model Evaluation
The included 11_Task_05.ipynb notebook provides:

GPU availability verification
Loading of the IMDB test set (test_texts.pkl, test_labels.pkl)
Full inference pipeline using the same RoBERTa-large model
Comprehensive metrics (accuracy, classification report, confusion matrix, ROC-AUC, etc.)

Expected performance on the IMDB test set (25,000 examples): 93–95% accuracy.
🔧 Configuration Notes

The model runs on CPU by default (device=-1). Change to device=0 in app/main.py for GPU acceleration.
Maximum input length: 512 tokens.
Batch size limit: 100 reviews (configurable in schemas.py).
Text length limit per review: 5000 characters.

📄 Frontend Usage
The web interface (index.html) offers:

Clean, professional design with gradient background
Real-time single review analysis with probability toggle
Batch mode supporting multiple reviews (one per line)
Responsive layout for desktop and mobile
Instant visual feedback with color-coded results (green = positive, red = negative)

🤝 Contributing
This project was developed as a complete demonstration of modern NLP deployment. Feel free to fork and extend it with additional features such as model quantization, async inference, or expanded evaluation dashboards.
📜 License
This project is licensed under the MIT License. See the LICENSE file for details (if included) or apply the standard MIT terms.

Built with ❤️ for accurate, fast, and production-ready sentiment analysis.
