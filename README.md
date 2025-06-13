# Object Detection Microservice with YOLO

A high-performance, Dockerized microservice for object detection using YOLO (You Only Look Once) models converted to ONNX format. This microservice provides a REST API for detecting objects in images with optional label filtering.

## 🚀 Features

- **Fast Object Detection**: Uses optimized YOLO model in ONNX format
- **REST API**: Simple HTTP endpoints for image upload and detection
- **Label Filtering**: Optional filtering by specific object labels
- **Dockerized**: Fully containerized for easy deployment
- **Docker Compose**: Orchestrated setup with auxiliary services
- **High Performance**: Optimized for concurrent requests
- **Comprehensive Testing**: Full test suite with multiple scenarios
- **Production Ready**: Includes health checks, logging, and monitoring

## 📋 API Endpoints

### Core Endpoints

- `POST /detect` - Detect all objects in uploaded image
- `POST /detect/{label}` - Detect objects filtered by specific label
- `GET /health` - Health check endpoint
- `GET /` - Service status
- `GET /docs` - Interactive API documentation

### Response Format

```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
  "objects": [
    {
      "label": "person",
      "x": 12,
      "y": 453,
      "width": 10,
      "height": 40,
      "confidence": 0.6
    },
    {
      "label": "person",
      "x": 33,
      "y": 25,
      "width": 8,
      "height": 26,
      "confidence": 0.82
    }
  ],
  "count": 2
}
```

## 🛠️ Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for development)
- Git

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd TractusTechnologyTask
```

### 2. Build and Start Services

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs

# Test with an image
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_images/sample.jpg"
```

## 🧪 Testing

### Automated Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Run API integration tests
python scripts/test_api.py

# Run specific test with custom host/port
python scripts/test_api.py --host localhost --port 8000
```

### Manual Testing

The project includes comprehensive test images and scenarios:

1. **People Detection**: Test with images containing people
2. **Multiple Objects**: Test with complex scenes
3. **Label Filtering**: Test specific object type detection
4. **Edge Cases**: Test with invalid inputs
5. **Performance**: Test concurrent requests

### Test Images Provided

- `test_images/people.jpg` - Image with people for person detection
- `test_images/street_scene.jpg` - Complex scene with multiple objects
- `test_images/animals.jpg` - Image with various animals
- `test_images/vehicles.jpg` - Cars, trucks, and other vehicles

### Expected Test Outcomes

| Test Case | Expected Result |
|-----------|----------------|
| Health Check | `{"status": "healthy", "model_loaded": true}` |
| Person Detection | Objects with `"label": "person"` |
| Car Detection | Objects with `"label": "car"` |
| Invalid File | HTTP 400 error |
| No File | HTTP 422 error |
| Performance | < 2 seconds per request |

## 🏗️ Architecture & Design Decisions

### Technology Stack

- **FastAPI**: Chosen for async support, automatic documentation, and high performance
- **ONNX Runtime**: For optimized cross-platform inference
- **OpenCV**: Image processing and computer vision operations
- **Docker**: Containerization for consistency and deployment
- **Nginx**: Reverse proxy for load balancing (optional)
- **Redis**: Caching layer for improved performance (optional)

### Model Choice: YOLO + ONNX

#### Why YOLO?
- **Speed**: Single-pass detection algorithm
- **Accuracy**: Good balance of speed vs accuracy
- **Versatility**: Detects 80 different object classes (COCO dataset)
- **Mature**: Well-established with extensive community support

#### Why ONNX Format?

**ONNX (Open Neural Network Exchange) Benefits:**

1. **Cross-Platform Compatibility**
   - Runs on Windows, Linux, macOS
   - Works across different ML frameworks (PyTorch, TensorFlow, etc.)

2. **Performance Optimization**
   - Optimized inference engine
   - Hardware acceleration support (CPU, GPU, TPU)
   - Reduced memory footprint

3. **Production Deployment**
   - Smaller model files
   - Faster loading times
   - Better resource utilization

4. **Hardware Flexibility**
   - CPU optimization for cloud deployment
   - GPU acceleration when available
   - Edge device compatibility

5. **Standardization**
   - Industry-standard format
   - Version compatibility
   - Ecosystem support

### Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │    FastAPI      │    │     Redis       │
│  (Load Balancer)│───▶│   Application   │───▶│    (Cache)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   ONNX Runtime  │
                       │   (YOLO Model)  │
                       └─────────────────┘
```

## 🔧 Development Setup

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Convert YOLO model to ONNX
python scripts/convert_yolo_to_onnx.py --model-size n

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Model Conversion

The YOLO to ONNX conversion process:

```bash
# Convert different model sizes
python scripts/convert_yolo_to_onnx.py --model-size n  # Nano (fastest)
python scripts/convert_yolo_to_onnx.py --model-size s  # Small
python scripts/convert_yolo_to_onnx.py --model-size m  # Medium
python scripts/convert_yolo_to_onnx.py --model-size l  # Large
python scripts/convert_yolo_to_onnx.py --model-size x  # Extra Large

# Custom output path
python scripts/convert_yolo_to_onnx.py --output-path custom/path/model.onnx
```

**Conversion Process:**
1. Download pre-trained YOLOv8 model from Ultralytics
2. Load model in PyTorch format
3. Export to ONNX with optimization flags
4. Verify model compatibility
5. Test inference capabilities

**Conversion Benefits:**
- 📉 Reduced model size (typically 10-30% smaller)
- ⚡ Faster inference (optimized operations)
- 🔧 Better deployment compatibility
- 🎯 Consistent results across platforms

## 🐳 Docker Configuration

### Dockerfile Optimizations

- **Multi-stage builds**: Separate build and runtime environments
- **Minimal base image**: Python 3.11-slim for smaller size
- **Layer caching**: Optimized layer order for faster builds
- **Security**: Non-root user for container execution
- **Health checks**: Built-in container health monitoring

### Docker Compose Services

- **object-detection**: Main API service
- **redis-cache**: Optional caching layer
- **nginx**: Optional reverse proxy and load balancer

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 2G
    reservations:
      cpus: '0.5'
      memory: 512M
```

## 📊 Performance Considerations

### Optimization Strategies

1. **Model Optimization**
   - ONNX format for faster inference
   - FP32 precision for compatibility
   - Static input shapes for optimization

2. **API Optimization**
   - Async FastAPI for concurrent requests
   - Efficient image preprocessing
   - Memory management for large images

3. **Caching Strategy**
   - Redis for result caching
   - Image hash-based cache keys
   - Configurable TTL settings

4. **Resource Management**
   - Connection pooling
   - Memory limits in Docker
   - CPU allocation optimization

### Performance Metrics

- **Cold Start**: ~2-3 seconds (model loading)
- **Inference Time**: ~0.1-0.5 seconds per image
- **Concurrent Requests**: Supports 10+ simultaneous requests
- **Memory Usage**: ~500MB-1GB depending on model size

## 🔍 Monitoring & Logging

### Health Checks

- **Application Health**: `/health` endpoint
- **Model Status**: Validates ONNX model loading
- **Dependencies**: Checks external service connectivity

### Logging

- **Structured Logging**: JSON format for log aggregation
- **Log Levels**: Configurable (DEBUG, INFO, WARNING, ERROR)
- **Request Tracking**: Unique request IDs for tracing
- **Performance Metrics**: Response times and error rates

## 🚀 Deployment

### Production Deployment

```bash
# Production build
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d

# Scale services
docker-compose up --scale object-detection=3

# Monitor services
docker-compose logs -f
```

### Cloud Deployment

The application is ready for deployment on:
- **AWS**: ECS, EKS, or EC2
- **Google Cloud**: GKE or Cloud Run
- **Azure**: AKS or Container Instances
- **DigitalOcean**: Kubernetes or Droplets

### Environment Variables

```bash
# Application settings
PYTHONPATH=/app
LOG_LEVEL=INFO
MODEL_PATH=models/yolo.onnx

# Performance tuning
MAX_WORKERS=4
TIMEOUT=30
MAX_FILE_SIZE=50MB
```

## 🧩 API Usage Examples

### Python Example

```python
import requests

# Upload image for detection
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f}
    )

result = response.json()
print(f"Found {result['count']} objects")

# Filter by label
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect/person',
        files={'file': f}
    )
```

### cURL Examples

```bash
# Detect all objects
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"

# Detect only cars
curl -X POST "http://localhost:8000/detect/car" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@street.jpg"

# Health check
curl http://localhost:8000/health
```

### JavaScript Example

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/detect', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log(`Detected ${data.count} objects`);
    // Display results
});
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found**
   ```bash
   # Convert model manually
   python scripts/convert_yolo_to_onnx.py
   ```

2. **Out of Memory**
   ```bash
   # Increase Docker memory limit
   docker-compose up --memory=2g
   ```

3. **Slow Performance**
   ```bash
   # Use smaller model
   python scripts/convert_yolo_to_onnx.py --model-size n
   ```

4. **Connection Refused**
   ```bash
   # Check service status
   docker-compose ps
   docker-compose logs object-detection
   ```

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG docker-compose up

# Access container shell
docker-compose exec object-detection bash

# View logs
docker-compose logs -f object-detection
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For questions and support:
- Create an issue in the GitHub repository
- Check the troubleshooting section
- Review the API documentation at `/docs`

## 🔄 Version History

- **v1.0.0**: Initial release with YOLO object detection
- **v1.1.0**: Added ONNX optimization and Docker improvements
- **v1.2.0**: Enhanced testing and monitoring capabilities

---

**Built with ❤️ for efficient object detection** 