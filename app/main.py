import os
import io
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Path
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import onnxruntime as ort
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Object Detection Microservice",
    description="A microservice for object detection using YOLO model",
    version="1.0.0"
)

# Global variables for model and session
model_session = None
class_names = []

def load_model():
    """Load the ONNX model and class names"""
    global model_session, class_names
    
    model_path = "models/yolo.onnx"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load ONNX model
    model_session = ort.InferenceSession(model_path)
    logger.info("ONNX model loaded successfully")
    
    # Load class names (COCO dataset classes)
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

def preprocess_image(image: np.ndarray, input_size: tuple = (640, 640)) -> np.ndarray:
    """Preprocess image for YOLO inference"""
    # Resize image while maintaining aspect ratio
    height, width = image.shape[:2]
    scale = min(input_size[0] / width, input_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Create padded image
    padded_image = np.full((input_size[1], input_size[0], 3), 114, dtype=np.uint8)
    padded_image[:new_height, :new_width] = resized_image
    
    # Convert to RGB and normalize
    padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    padded_image = padded_image.astype(np.float32) / 255.0
    
    # Transpose to CHW format and add batch dimension
    padded_image = np.transpose(padded_image, (2, 0, 1))
    padded_image = np.expand_dims(padded_image, axis=0)
    
    return padded_image, scale

def postprocess_detections(outputs: np.ndarray, original_shape: tuple, scale: float, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> List[Dict]:
    """Post-process YOLO outputs to get final detections"""
    detections = []
    
    # Assuming YOLOv8 output format: [batch, num_detections, 84] where 84 = 4 (bbox) + 80 (classes)
    if len(outputs.shape) == 3:
        outputs = outputs[0]  # Remove batch dimension
    
    # Filter by confidence threshold
    class_scores = outputs[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    valid_detections = max_scores > conf_threshold
    
    if not np.any(valid_detections):
        return detections
    
    valid_outputs = outputs[valid_detections]
    valid_class_scores = class_scores[valid_detections]
    
    # Get class predictions
    class_ids = np.argmax(valid_class_scores, axis=1)
    confidences = np.max(valid_class_scores, axis=1)
    
    # Convert bbox coordinates
    original_height, original_width = original_shape[:2]
    
    boxes = []
    for i, bbox in enumerate(valid_outputs[:, :4]):
        # Convert from center format to corner format
        center_x, center_y, width, height = bbox
        x1 = (center_x - width / 2) / scale
        y1 = (center_y - height / 2) / scale
        x2 = (center_x + width / 2) / scale
        y2 = (center_y + height / 2) / scale
        
        # Clip to image bounds
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))
        
        boxes.append([x1, y1, x2, y2])
    
    # Apply NMS
    boxes = np.array(boxes)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold)
    
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            
            if class_id < len(class_names):
                detections.append({
                    "label": class_names[class_id],
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "confidence": float(confidence)
                })
    
    return detections

def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on the image"""
    result_image = image.copy()
    
    for detection in detections:
        x = detection["x"]
        y = detection["y"]
        width = detection["width"]
        height = detection["height"]
        label = detection["label"]
        confidence = detection["confidence"]
        
        # Draw bounding box
        cv2.rectangle(result_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # Draw label and confidence
        label_text = f"{label}: {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result_image, (x, y - text_height - 10), (x + text_width, y), (0, 255, 0), -1)
        cv2.putText(result_image, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result_image

def image_to_base64(image: np.ndarray) -> str:
    """Convert image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    try:
        load_model()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Object Detection Microservice is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_session is not None}

@app.post("/detect/{label}")
async def detect_objects_with_label(
    label: str = Path(..., description="Object label to filter detections"),
    file: UploadFile = File(..., description="Image file for object detection")
):
    """Detect objects in image and filter by specific label"""
    return await detect_objects(file, label)

@app.post("/detect")
async def detect_all_objects(
    file: UploadFile = File(..., description="Image file for object detection")
):
    """Detect all objects in image"""
    return await detect_objects(file)

async def detect_objects(file: UploadFile, filter_label: Optional[str] = None):
    """Main detection function"""
    if model_session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        original_shape = image_array.shape
        
        # Preprocess image
        input_tensor, scale = preprocess_image(image_array)
        
        # Run inference
        input_name = model_session.get_inputs()[0].name
        outputs = model_session.run(None, {input_name: input_tensor})
        
        # Post-process detections
        detections = postprocess_detections(outputs[0], original_shape, scale)
        
        # Filter by label if specified
        if filter_label:
            detections = [det for det in detections if det["label"].lower() == filter_label.lower()]
        
        # Draw detections on image
        result_image = draw_detections(image_array, detections)
        
        # Convert to base64
        image_base64 = image_to_base64(result_image)
        
        # Prepare response
        response = {
            "image": image_base64,
            "objects": detections,
            "count": len(detections)
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 