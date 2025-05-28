"""
Minimal script to test YOLOv8 model with a test image.
No dependencies on Streamlit or complex preprocessing.

Usage:
    python direct_test.py path/to/your/image.jpg
"""

import sys
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import os

# Define class names
CLASS_NAMES = ["deer", "fox", "beaver", "raccoon", "rabbit", "squirrel", 
               "goat", "chicken", "skunk", "coyote", "armadillo", "cat", "dog"]

def main():
    # Basic argument check
    if len(sys.argv) < 2:
        print("Usage: python direct_test.py path/to/your/image.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Testing with image: {image_path}")
    
    # Load model
    try:
        print("Loading model...")
        model_path = "animal_detector_best.pt"
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return
            
        model = YOLO(model_path)
        print(f"Model loaded successfully: {type(model)}")
        
        # Check device
        device = next(model.parameters()).device if hasattr(model, 'parameters') else "cpu"
        print(f"Model is on device: {device}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Load and preprocess image
    try:
        print("Loading image...")
        # Load with PIL first
        pil_image = Image.open(image_path)
        print(f"Image details: format={pil_image.format}, size={pil_image.size}, mode={pil_image.mode}")
        
        # Convert to numpy and ensure RGB
        img = np.array(pil_image)
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            print("Converted grayscale to RGB")
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            print("Converted RGBA to RGB")
            
        print(f"Numpy image shape: {img.shape}")
        
        # Simple resize to 640x640 (simplest approach)
        resized_img = cv2.resize(img, (640, 640))
        print(f"Resized to: {resized_img.shape}")
        
        # Save a copy of the resized image for debugging
        cv2.imwrite("debug_resized.jpg", cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
        print("Saved debug_resized.jpg for inspection")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return
    
    # Run inference
    try:
        print("Running inference...")
        results = model(resized_img)
        print(f"Got results: {type(results)}")
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            print(f"Result attributes: {dir(result)[:10]}...")
            
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                print(f"Detected {len(result.boxes)} objects")
                
                # Extract and display predictions
                predictions = []
                for i, (box, score, cls) in enumerate(zip(
                    result.boxes.xyxy, 
                    result.boxes.conf, 
                    result.boxes.cls
                )):
                    # Convert tensor to numpy
                    if isinstance(box, torch.Tensor):
                        box = box.cpu().numpy()
                    if isinstance(score, torch.Tensor):
                        score = score.item()
                    if isinstance(cls, torch.Tensor):
                        cls = cls.item()
                    
                    # Get class name
                    class_id = int(cls)
                    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
                    
                    # Add to predictions
                    predictions.append({
                        'box': box.tolist() if isinstance(box, np.ndarray) else box,
                        'label': class_name,
                        'score': float(score) * 100  # Convert to percentage
                    })
                
                # Sort by confidence
                predictions.sort(key=lambda x: x['score'], reverse=True)
                
                # Display predictions
                print("\nDetections:")
                for i, pred in enumerate(predictions):
                    print(f"  {i+1}. {pred['label']}: {pred['score']:.2f}%")
                
                # Draw on image
                output_img = img.copy()
                for pred in predictions:
                    # Draw bounding box
                    x1, y1, x2, y2 = [int(round(coord)) for coord in pred['box']]
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    text = f"{pred['label']}: {pred['score']:.1f}%"
                    cv2.putText(output_img, text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save result
                cv2.imwrite("result.jpg", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
                print("Saved result.jpg with detections")
                
            else:
                print("No objects detected in the image")
        else:
            print("Model returned no results")
            
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()