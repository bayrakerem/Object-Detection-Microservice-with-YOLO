import pytest
import asyncio
import httpx
import base64
import json
from pathlib import Path
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_IMAGES_DIR = Path("test_images")

class TestObjectDetectionAPI:
    """Test suite for Object Detection API"""
    
    @pytest.fixture(scope="session")
    def event_loop(self):
        """Create an instance of the default event loop for the test session."""
        loop = asyncio.get_event_loop_policy().new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture(scope="session")
    async def client(self):
        """Create HTTP client for testing"""
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
            yield client

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Object Detection Microservice" in data["message"]

    @pytest.mark.asyncio
    async def test_invalid_file_type(self, client):
        """Test API with invalid file type"""
        # Create a text file
        text_content = b"This is not an image"
        files = {"file": ("test.txt", text_content, "text/plain")}
        
        response = await client.post("/detect", files=files)
        assert response.status_code == 400
        data = response.json()
        assert "must be an image" in data["detail"]

    @pytest.mark.asyncio
    async def test_no_file_provided(self, client):
        """Test API with no file provided"""
        response = await client.post("/detect")
        assert response.status_code == 422  # Unprocessable Entity

    @pytest.mark.asyncio
    async def test_empty_file(self, client):
        """Test API with empty file"""
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        response = await client.post("/detect", files=files)
        assert response.status_code == 500  # Internal server error due to invalid image


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests() 