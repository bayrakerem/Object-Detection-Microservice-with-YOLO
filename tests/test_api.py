import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Object Detection API is running"

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()

def test_detect_endpoint_no_file():
    """Test detect endpoint without file"""
    response = client.post("/detect")
    assert response.status_code == 422  # Validation error

def test_detect_label_endpoint_no_file():
    """Test detect with label endpoint without file"""
    response = client.post("/detect/person")
    assert response.status_code == 422  # Validation error 