#!/usr/bin/env python3
"""
Convert YOLOv8 model to ONNX format for deployment
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def convert_yolo_to_onnx(model_size='n', output_path=None):
    """Convert YOLOv8 model to ONNX format"""
    
    model_name = f'yolov8{model_size}.pt'
    print(f"Loading YOLOv8 model: {model_name}")
    
    # Load the model (downloads if needed)
    model = YOLO(model_name)
    
    # Set output path
    if output_path is None:
        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "yolo.onnx"
    
    print(f"Converting to ONNX format...")
    
    # Export to ONNX
    model.export(
        format='onnx',
        imgsz=640,
        optimize=True,
        half=False,
        dynamic=False,
        simplify=True
    )
    
    print(f"Model converted successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv8 to ONNX")
    parser.add_argument('--model-size', default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--output-path', default=None)
    
    args = parser.parse_args()
    convert_yolo_to_onnx(args.model_size, args.output_path) 