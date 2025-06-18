#!/usr/bin/env python3
"""
YOLO to ONNX Conversion Script

This script converts a pre-trained YOLOv8 model from PyTorch to ONNX format.

Usage:
    python scripts/convert_yolo_to_onnx.py [--model-size n] [--output-path path]

Example:
    python scripts/convert_yolo_to_onnx.py --model-size n --output-path models/yolo.onnx
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_yolo_to_onnx(model_size: str = 'n', output_path: str = None) -> None:
    """
    Convert YOLOv8 model to ONNX format
    
    Args:
        model_size (str): Size of the YOLO model ('n', 's', 'm', 'l', 'x')
        output_path (str): Path where the ONNX model will be saved
    """
    try:
        # Define model sizes and their characteristics
        model_info = {
            'n': {'name': 'yolov8n.pt', 'description': 'Nano - fastest, least accurate'},
            's': {'name': 'yolov8s.pt', 'description': 'Small - good balance'},
            'm': {'name': 'yolov8m.pt', 'description': 'Medium - better accuracy'},
            'l': {'name': 'yolov8l.pt', 'description': 'Large - high accuracy'},
            'x': {'name': 'yolov8x.pt', 'description': 'Extra Large - highest accuracy'}
        }
        
        if model_size not in model_info:
            raise ValueError(f"Invalid model size. Choose from: {list(model_info.keys())}")
        
        model_name = model_info[model_size]['name']
        model_desc = model_info[model_size]['description']
        
        logger.info(f"Loading YOLOv8 model: {model_name} ({model_desc})")
        
        # Load the pre-trained YOLO model
        # This will download the model if it doesn't exist
        model = YOLO(model_name)
        
        # Set output path
        if output_path is None:
            output_dir = Path("models")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "yolo.onnx"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting model to ONNX format...")
        logger.info(f"Output path: {output_path}")
        
        # Export to ONNX
        # The export method handles the conversion automatically
        success = model.export(
            format='onnx',
            imgsz=640,  # Image size for inference
            optimize=True,  # Optimize for inference
            half=False,  # Use FP32 precision (more compatible)
            dynamic=False,  # Static input shape for better performance
            simplify=True,  # Simplify the model
            opset=11  # ONNX opset version for compatibility
        )
        
        if success:
            # Move the exported file to the desired location
            exported_file = Path(model_name.replace('.pt', '.onnx'))
            if exported_file.exists() and exported_file != output_path:
                exported_file.rename(output_path)
                logger.info(f"Model moved to: {output_path}")
            
            # Verify the converted model
            verify_onnx_model(output_path)
            
            logger.info("SUCCESS: YOLO to ONNX conversion completed successfully!")
            logger.info(f"Model saved at: {output_path}")
            
            # Print model information
            print_model_info(output_path)
            
        else:
            logger.error("FAILED: Failed to export model to ONNX")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"ERROR: Error during conversion: {e}")
        sys.exit(1)

def verify_onnx_model(model_path: Path) -> None:
    """Verify that the ONNX model is valid"""
    try:
        import onnx
        import onnxruntime as ort
        
        logger.info("Verifying ONNX model...")
        
        # Load and check the ONNX model
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
        
        # Test with ONNX Runtime
        session = ort.InferenceSession(str(model_path))
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        
        logger.info(f"SUCCESS: ONNX model verification successful")
        logger.info(f"Input shape: {input_shape}")
        logger.info(f"Output shape: {output_shape}")
        
    except Exception as e:
        logger.warning(f"WARNING: ONNX model verification failed: {e}")

def print_model_info(model_path: Path) -> None:
    """Print information about the converted model"""
    try:
        file_size = model_path.stat().st_size / (1024 * 1024)  # Size in MB
        
        print("\n" + "="*50)
        print("MODEL CONVERSION SUMMARY")
        print("="*50)
        print(f"Model Path: {model_path}")
        print(f"File Size: {file_size:.2f} MB")
        print(f"Format: ONNX")
        print(f"Input Size: 640x640 (default)")
        print(f"Classes: 80 (COCO dataset)")
        print("\nONNX Benefits:")
        print("* Cross-platform compatibility")
        print("* Optimized for inference")
        print("* Hardware acceleration support")
        print("* Reduced model size")
        print("* Framework interoperability")
        print("="*50)
        
    except Exception as e:
        logger.warning(f"Could not print model info: {e}")

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/convert_yolo_to_onnx.py
  python scripts/convert_yolo_to_onnx.py --model-size s
  python scripts/convert_yolo_to_onnx.py --model-size m --output-path custom/path/model.onnx

Model Sizes:
  n - Nano (fastest, smallest)
  s - Small (balanced)
  m - Medium (good accuracy)
  l - Large (high accuracy)
  x - Extra Large (highest accuracy)
        """
    )
    
    parser.add_argument(
        '--model-size',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='Size of the YOLO model to convert (default: n)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output path for the ONNX model (default: models/yolo.onnx)'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting YOLO to ONNX conversion...")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Output path: {args.output_path or 'models/yolo.onnx'}")
    
    convert_yolo_to_onnx(args.model_size, args.output_path)

if __name__ == "__main__":
    main() 