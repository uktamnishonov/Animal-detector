from ultralytics import YOLO
import os
import torch
import wandb
from pathlib import Path
import shutil

def train_model():
    # Initialize wandb
    wandb_run = wandb.init(project="animal-detector", name="yolov8n_training_efficient")
    
    # Load the YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Determine device (use GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Define training arguments with reduced resource usage
    args = {
        'data': 'animal-dataset-yolo/data.yaml',  # Path to dataset
        'epochs': 40,                             # Reduced epochs
        'imgsz': 640,                             # Image size
        'batch': 16 if device == 'cuda' else 4,    # Reduced batch size
        'device': device,                         # Device
        'workers': 4 if device == 'cuda' else 2,  # Reduced workers
        'project': 'runs/train',                  # Project directory
        'name': 'animal_detector',                # Experiment name
        'exist_ok': True,                         # Overwrite existing experiment
        'pretrained': True,                       # Use pretrained weights
        'optimizer': 'Adam',                      # Optimizer
        'lr0': 0.001,                             # Initial learning rate
        'weight_decay': 0.0005,                   # Weight decay
        'patience': 10,                           # Early stopping patience
        'cache': False,                           # Disable caching to save RAM
        'verbose': True,                          # Show verbose output
        'close_mosaic': 5,                        # Disable mosaic in final epochs
        'save': True,                             # Save checkpoints
        'save_period': 5                          # Save frequency
    }
    
    # Start training with W&B integration
    try:
        # Enable W&B logging
        os.environ["WANDB_PROJECT"] = "animal-detector"
        
        # Set the save directory path
        save_dir = Path('runs/train/animal_detector')
        
        # Train the model
        results = model.train(**args)
        
        print("\nTraining completed successfully!")
        
        # Directly access the saved model paths
        weights_dir = save_dir / 'weights'
        best_model_path = weights_dir / 'best.pt'
        last_model_path = weights_dir / 'last.pt'
        
        # Copy the best model to the current directory
        current_dir_best_path = Path('./animal_detector_best.pt')
        if best_model_path.exists():
            shutil.copy(best_model_path, current_dir_best_path)
            print(f"Best model saved to {current_dir_best_path}")
        else:
            print(f"Best model not found at {best_model_path}")
        
        # Copy the last model to the current directory
        current_dir_last_path = Path('./animal_detector_last.pt')
        if last_model_path.exists():
            shutil.copy(last_model_path, current_dir_last_path)
            print(f"Last model saved to {current_dir_last_path}")
        else:
            print(f"Last model not found at {last_model_path}")
        
        # Evaluate the model on validation set (after copying the models)
        val_results = model.val()
        
        # Export model to ONNX format
        print("Exporting model to ONNX format...")
        try:
            model.export(format="onnx", imgsz=640)
            
            # Move the exported ONNX model to current directory
            onnx_path = Path('./animal_detector.onnx')
            default_onnx_path = weights_dir / 'best.onnx'
            if default_onnx_path.exists():
                shutil.copy(default_onnx_path, onnx_path)
                print(f"Model exported to ONNX format at {onnx_path}")
            else:
                print(f"ONNX model not found at {default_onnx_path}")
        except Exception as export_error:
            print(f"Error exporting to ONNX: {str(export_error)}")
        
        print("\nTo continue training from this checkpoint, you can run:")
        print("model = YOLO('./animal_detector_best.pt')")
        print("model.train(resume=True)")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted! You can resume training using:")
        print("model = YOLO('runs/train/animal_detector/weights/last.pt')")
        print("model.train(resume=True)")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Check the error message above and adjust parameters if needed.")
    finally:
        # Always close wandb run
        wandb.finish()

if __name__ == "__main__":
    train_model()