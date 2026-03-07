"""
API Test Suite
Run with: pytest tests/test_api.py
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_fresh():
    """Test positive review prediction"""
    response = client.post("/predict", json={
        "text": "This movie was absolutely fantastic and thrilling!",
        "return_details": True
    })
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["Fresh", "Rotten"]
    assert 0 <= data["confidence"] <= 1
    assert 0 <= data["fresh_probability"] <= 1

def test_predict_rotten():
    """Test negative review prediction"""
    response = client.post("/predict", json={
        "text": "Terrible waste of time, boring and poorly acted.",
        "return_details": True
    })
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["Fresh", "Rotten"]

def test_predict_short_text():
    """Test validation for short text"""
    response = client.post("/predict", json={"text": "Bad"})
    assert response.status_code == 400

def test_predict_empty_text():
    """Test validation for empty text"""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_batch_prediction():
    """Test batch prediction endpoint"""
    response = client.post("/predict/batch", json={
        "reviews": [
            "Great movie!",
            "Terrible film."
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2

def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model/info")
    assert response.status_code == 200
    assert "model_type" in response.json()