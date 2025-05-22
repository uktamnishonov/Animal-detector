import ssl
import urllib.request
import os
from pathlib import Path

def download_yolo_model():
    # Disable SSL verification (only for downloading the model)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    model_path = "yolov8n.pt"
    
    if not os.path.exists(model_path):
        print(f"Downloading YOLOv8n model from {model_url}")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully!")
    else:
        print("Model already exists!")

if __name__ == "__main__":
    download_yolo_model()