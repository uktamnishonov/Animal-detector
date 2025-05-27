import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import io

class YOLOPreprocessor:
    """
    Handles image preprocessing for YOLOv8 model inference.
    Ensures images are properly formatted for the model based on training parameters.
    """
    
    def __init__(self, img_size=640):
        """
        Initialize the preprocessor with the same image size used during training.
        
        Args:
            img_size (int): The target image size (default: 640, matching training)
        """
        self.img_size = img_size
        # Class names from your training data
        self.class_names = [
            "deer", "fox", "beaver", "raccoon", "rabbit", "squirrel", 
            "goat", "chicken", "skunk", "coyote", "armadillo", "cat", "dog"
        ]
        
    def preprocess_image(self, image, return_orig=True):
        """
        Preprocess an image for YOLOv8 inference.
        
        Args:
            image: Can be a file path, PIL Image, numpy array, or bytes
            return_orig (bool): Whether to return the original image alongside the processed one
            
        Returns:
            processed_img: The preprocessed image ready for model input
            orig_img: The original image (if return_orig=True)
        """
        orig_img = None
        
        # Handle different input types
        if isinstance(image, str) or isinstance(image, Path):
            # Image is a file path
            orig_img = cv2.imread(str(image))
            if orig_img is None:
                raise ValueError(f"Failed to load image from path: {image}")
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
        elif isinstance(image, bytes) or isinstance(image, bytearray):
            # Image is bytes (e.g., from file upload)
            nparr = np.frombuffer(image, np.uint8)
            orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if orig_img is None:
                raise ValueError("Failed to decode image bytes")
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
        elif isinstance(image, Image.Image):
            # Image is a PIL Image
            orig_img = np.array(image)
            if len(orig_img.shape) == 2:  # Grayscale
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
            elif orig_img.shape[2] == 4:  # RGBA
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGBA2RGB)
                
        elif isinstance(image, np.ndarray):
            # Image is already a numpy array
            orig_img = image.copy()
            if len(orig_img.shape) == 2:  # Grayscale
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
            elif orig_img.shape[2] == 4:  # RGBA
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGBA2RGB)
            elif orig_img.shape[2] == 3:
                # Check if the image is BGR (from OpenCV) and convert to RGB if needed
                if self._is_likely_bgr(orig_img):
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Create preprocessed version (letterboxed and resized)
        processed_img = self._letterbox(orig_img)
        
        if return_orig:
            return processed_img, orig_img
        else:
            return processed_img
    
    def _letterbox(self, img):
        """
        Resize and pad image to create a square letterboxed image.
        This maintains the aspect ratio by adding padding.
        
        Args:
            img: Original image (numpy array)
            
        Returns:
            canvas: Resized and padded image
        """
        # Get current dimensions
        h, w = img.shape[:2]
        
        # Calculate scale to maintain aspect ratio
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize the image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create a square canvas with the target size
        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Calculate offsets to center the image on the canvas
        offset_x = (self.img_size - new_w) // 2
        offset_y = (self.img_size - new_h) // 2
        
        # Place the resized image on the canvas
        canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
        
        return canvas
        
    def _is_likely_bgr(self, img):
        """
        Heuristic to guess if an image is in BGR format (from OpenCV) rather than RGB.
        This is not foolproof but can help in some cases.
        
        Args:
            img: Image as numpy array
            
        Returns:
            bool: True if the image is likely in BGR format
        """
        # Simple heuristic: in natural images, the blue channel typically has
        # less intensity than red and green. If we find the opposite, it might be BGR.
        if img.shape[2] != 3:
            return False
            
        # Sample the image (don't process every pixel for speed)
        sample = img[::10, ::10, :]
        avg_b = np.mean(sample[:, :, 0])
        avg_g = np.mean(sample[:, :, 1])
        avg_r = np.mean(sample[:, :, 2])
        
        # If the first channel (assumed to be blue in BGR) has significantly higher
        # value than the third channel (assumed to be red in BGR), it might be BGR
        return avg_b > avg_r * 1.1
    
    def process_predictions(self, results, confidence_threshold=0.3):
        """
        Process YOLOv8 results into a more usable format.
        
        Args:
            results: Raw detection results from the YOLO model
            confidence_threshold: Minimum confidence score threshold (0-1)
            
        Returns:
            list: Processed predictions with bounding boxes, labels, and scores
        """
        predictions = []
        
        # Check if any detection
        if results and len(results) > 0:
            # Get the first result (batch index 0)
            result = results[0]
            
            # Convert boxes to pixel values and extract all data
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for i, (box, score, cls) in enumerate(zip(
                    result.boxes.xyxy, 
                    result.boxes.conf, 
                    result.boxes.cls
                )):
                    if score >= confidence_threshold:
                        # Convert tensor to numpy if needed
                        if isinstance(box, torch.Tensor):
                            box = box.cpu().numpy()
                        if isinstance(score, torch.Tensor):
                            score = score.item()  # Get scalar value
                        if isinstance(cls, torch.Tensor):
                            cls = cls.item()      # Get scalar value
                        
                        # Get class name
                        class_id = int(cls)
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                        
                        # Create prediction entry
                        predictions.append({
                            'box': box.tolist() if isinstance(box, np.ndarray) else box,
                            'label': class_name,
                            'score': float(score) * 100  # Convert to percentage
                        })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return predictions
    
    def draw_detections(self, image, predictions):
        """
        Draw bounding boxes and labels on the image based on predictions.
        
        Args:
            image: Original image (numpy array)
            predictions: List of predictions from process_predictions()
            
        Returns:
            numpy.ndarray: Image with drawn bounding boxes and labels
        """
        img_copy = image.copy()
        
        for pred in predictions:
            # Extract prediction data
            box = pred['box']
            label = pred['label']
            score = pred['score']
            
            # Ensure box coordinates are integers
            x1, y1, x2, y2 = [int(round(coord)) for coord in box]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Create label text
            text = f"{label}: {score:.1f}%"
            
            # Get text size for better positioning
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(
                img_copy,
                (x1, y1 - text_size[1] - 8),
                (x1 + text_size[0] + 8, y1),
                color,
                -1  # Filled rectangle
            )
            
            # Draw label text (white on the green background)
            cv2.putText(
                img_copy,
                text,
                (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1
            )
        
        return img_copy

# Example usage:
if __name__ == "__main__":
    # Create preprocessor
    preprocessor = YOLOPreprocessor(img_size=640)
    
    # Example: Process an image (replace with your image path)
    try:
        # Load image
        img_path = "sample_image.jpg"
        processed_img, orig_img = preprocessor.preprocess_image(img_path)
        
        # Visualize the preprocessed image
        cv2.imshow("Original", cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        cv2.imshow("Preprocessed", cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("Image preprocessing successful!")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")