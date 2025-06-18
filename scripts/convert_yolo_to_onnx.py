#!/usr/bin/env python3
"""
Convert YOLOv8 model to ONNX format for deployment

ONNX Benefits:
- Cross-platform compatibility
- Optimized inference performance  
- Hardware acceleration support
- Reduced model size
- Framework interoperability
"""

import argparse
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_yolo_to_onnx(model_size='n', output_path=None):
    """Convert YOLOv8 model to ONNX format"""
    
    model_name = f'yolov8{model_size}.pt'
    logger.info(f"Loading YOLOv8 model: {model_name}")
    
    # Load the model (downloads if needed)
    model = YOLO(model_name)
    
    # Set output path
    if output_path is None:
        output_dir = Path("models")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "yolo.onnx"
    
    logger.info(f"Converting to ONNX format...")
    
    # Export to ONNX with optimizations
    success = model.export(
        format='onnx',
        imgsz=640,
        optimize=True,
        half=False,
        dynamic=False,
        simplify=True,
        opset=11
    )
    
    if success:
        # Move exported file to desired location
        exported_file = Path(model_name.replace('.pt', '.onnx'))
        if exported_file.exists() and exported_file != output_path:
            exported_file.rename(output_path)
        
        logger.info(f"Model converted successfully to: {output_path}")
        
        # Print model info
        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Model size: {file_size:.2f} MB")
    else:
        logger.error("Conversion failed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv8 to ONNX")
    parser.add_argument('--model-size', default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--output-path', default=None)
    
    args = parser.parse_args()
    convert_yolo_to_onnx(args.model_size, args.output_path) 