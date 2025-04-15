from ultralytics import YOLO
import os
import torch

def train_model():
    # Load the YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Define training arguments optimized for CPU training
    args = {
        'data': 'animal-dataset-yolo/data.yaml',  # Path to data.yaml file
        'epochs': 30,                             # Reduced epochs for initial training
        'imgsz': 416,                            # Reduced image size
        'batch': 8,                              # Smaller batch size for CPU
        'device': 'cpu',                         # Force CPU training
        'workers': 4,                            # Reduced number of workers
        'project': 'runs/train',                 # Project name
        'name': 'animal_detector',               # Experiment name
        'exist_ok': True,                        # Overwrite existing experiment
        'pretrained': True,                      # Use pretrained weights
        'optimizer': 'Adam',                     # Using Adam optimizer
        'patience': 10,                          # Early stopping patience
        'cache': True,                          # Cache images for faster training
        'verbose': True                         # Show training progress
    }
    
    # Start training
    try:
        results = model.train(**args)
        
        # Evaluate the model on validation set
        results = model.val()
        
        print("\nTraining completed successfully!")
        print(f"Results saved to {os.path.join('runs/train/animal_detector')}")
        print("\nTo continue training from this checkpoint, you can run:")
        print("model.train(resume=True)")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted! You can resume training using:")
        print("model.train(resume=True)")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Check the error message above and adjust parameters if needed.")

if __name__ == "__main__":
    train_model()