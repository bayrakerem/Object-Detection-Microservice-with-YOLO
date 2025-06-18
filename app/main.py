import os
import io
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Path
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict
import onnxruntime as ort
from PIL import Image
import cv2

app = FastAPI(
    title="Object Detection API",
    description="A microservice for object detection using YOLO",
    version="1.0.0"
)

# Global model session
model_session = None

@app.on_event("startup")
async def startup_event():
    """Load the ONNX model on startup"""
    global model_session
    model_path = "models/yolo.onnx"
    if os.path.exists(model_path):
        model_session = ort.InferenceSession(model_path)
        print("Model loaded successfully")

@app.get("/")
async def root():
    return {"message": "Object Detection API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_session is not None}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Detect all objects in image"""
    if model_session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Basic response for now
    return {"message": "Detection endpoint working", "filename": file.filename}

@app.post("/detect/{label}")
async def detect_objects_with_label(
    label: str = Path(..., description="Object label to filter"),
    file: UploadFile = File(...)
):
    """Detect specific objects in image"""
    if model_session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Basic response for now
    return {"message": f"Detection for {label}", "filename": file.filename} 