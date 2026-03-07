"""
Movie Sentiment Analysis API
Predicts Rotten Tomatoes sentiment (Fresh/Rotten) for movie reviews
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Download NLTK data
nltk.download('stopwords', quiet=True)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Sentiment Analysis API",
    description="Predicts whether movie reviews are Fresh or Rotten based on Rotten Tomatoes critic data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
print("🔄 Loading sentiment model...")
start_time = time.time()

try:
    pipeline = joblib.load('traditional_models.pkl')
    model = pipeline['LR']
    vectorizer = pipeline['vectorizer']
    custom_stopwords = pipeline['custom_stopwords']
    threshold = pipeline.get('threshold', 0.45)
    load_time = time.time() - start_time
    print(f"✅ Model loaded successfully in {load_time:.2f}s")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    raise

# Preprocessing function
def clean_text(text: str) -> str:
    """Clean and preprocess text for prediction"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in custom_stopwords])
    return text

# Request/Response Models
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=5000, description="Movie review text")
    return_details: bool = Field(False, description="Include additional details in response")

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    fresh_probability: float
    message: Optional[str] = None
    processing_time_ms: Optional[float] = None

class BatchRequest(BaseModel):
    reviews: List[str] = Field(..., max_length=100)

class BatchResponse(BaseModel):
    results: List[SentimentResponse]

# Health Check Endpoint
@app.get("/health")
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "model": "Logistic Regression",
        "version": "1.0.0",
        "threshold": threshold
    }

# Single Prediction Endpoint
@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment for a single movie review
    
    - **text**: The movie review text (5-5000 characters)
    - **return_details**: Include additional details in response
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not request.text or len(request.text.strip()) < 5:
            raise HTTPException(
                status_code=400, 
                detail="Review text must be at least 5 characters"
            )
        
        if len(request.text) > 5000:
            raise HTTPException(
                status_code=400, 
                detail="Review text must be less than 5000 characters"
            )
        
        # Preprocess
        cleaned = clean_text(request.text)
        
        # Predict
        vec = vectorizer.transform([cleaned])
        fresh_proba = float(model.predict_proba(vec)[0][1])
        sentiment = "Fresh" if fresh_proba > threshold else "Rotten"
        confidence = fresh_proba if sentiment == "Fresh" else (1 - fresh_proba)
        
        # Generate message
        message = None
        if request.return_details:
            if confidence > 0.8:
                message = f"Very confident this is {sentiment}"
            elif confidence > 0.6:
                message = f"Likely {sentiment}"
            else:
                message = f"Uncertain, leaning towards {sentiment}"
        
        processing_time = (time.time() - start_time) * 1000
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=round(confidence, 4),
            fresh_probability=round(fresh_proba, 4),
            message=message,
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

# Batch Prediction Endpoint
@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """
    Predict sentiment for multiple reviews (max 100)
    """
    if len(request.reviews) > 100:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 100 reviews per batch"
        )
    
    results = []
    for text in request.reviews:
        try:
            cleaned = clean_text(text)
            vec = vectorizer.transform([cleaned])
            fresh_proba = float(model.predict_proba(vec)[0][1])
            sentiment = "Fresh" if fresh_proba > threshold else "Rotten"
            confidence = fresh_proba if sentiment == "Fresh" else (1 - fresh_proba)
            
            results.append(SentimentResponse(
                sentiment=sentiment,
                confidence=round(confidence, 4),
                fresh_probability=round(fresh_proba, 4),
                message=None,
                processing_time_ms=None
            ))
        except:
            results.append(SentimentResponse(
                sentiment="Error",
                confidence=0.0,
                fresh_probability=0.0,
                message="Failed to process",
                processing_time_ms=None
            ))
    
    return BatchResponse(results=results)

# Model Info Endpoint
@app.get("/model/info")
async def model_info():
    """Get information about the deployed model"""
    return {
        "model_type": "Logistic Regression",
        "features": "TF-IDF (10,000 features, unigrams + bigrams)",
        "training_data": "Rotten Tomatoes Critic Reviews (Top Critics)",
        "training_samples": "~278,000 reviews",
        "accuracy": "79.6%",
        "roc_auc": "87.9%",
        "threshold": threshold,
        "classes": ["Rotten", "Fresh"]
    }

# Root Endpoint
@app.get("/")
async def root():
    """API welcome message"""
    return {
        "message": "Welcome to Movie Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)