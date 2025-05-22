from ultralytics import YOLO
import os
import torch
import wandb
from pathlib import Path
import shutil
import re


def get_next_run_index():
    base_dir = Path("runs/train")
    if not base_dir.exists():
        return 1

    existing_runs = [
        d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run-")
    ]
    if not existing_runs:
        return 1

    indices = [
        int(re.search(r"run-(\d+)", d.name).group(1))
        for d in existing_runs
        if re.search(r"run-(\d+)", d.name)
    ]
    return max(indices, default=0) + 1


def get_models_path():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    return models_dir


def train_model():
    # Initialize wandb
    wandb_run = wandb.init(project="animal-detector", name="yolov8n_training_efficient")

    # Get next run index and create run directory name
    run_index = get_next_run_index()
    run_name = f"run-{run_index}"

    # Load the YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Determine device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define training arguments with reduced resource usage
    args = {
        "data": "animal-dataset-yolo/data.yaml",  # Path to dataset
        "epochs": 40,  # Reduced epochs
        "imgsz": 640,  # Image size
        "batch": 16 if device == "cuda" else 4,  # Reduced batch size
        "device": device,  # Device
        "workers": 4 if device == "cuda" else 2,  # Reduced workers
        "project": "runs/train",  # Project directory
        "name": run_name,  # Use indexed run name
        "exist_ok": True,  # Overwrite existing experiment
        "pretrained": True,  # Use pretrained weights
        "optimizer": "Adam",  # Optimizer
        "lr0": 0.001,  # Initial learning rate
        "weight_decay": 0.0005,  # Weight decay
        "patience": 10,  # Early stopping patience
        "cache": False,  # Disable caching to save RAM
        "verbose": True,  # Show verbose output
        "close_mosaic": 5,  # Disable mosaic in final epochs
        "save": True,  # Save checkpoints
        "save_period": -1,  # Disable periodic saving
        "save_model_only_best": True,  # Save only best model
    }

    # Start training with W&B integration
    try:
        # Enable W&B logging
        os.environ["WANDB_PROJECT"] = "animal-detector"

        # Set the save directory path with indexed name
        save_dir = Path("runs/train") / run_name

        # Train the model
        results = model.train(**args)

        print("\nTraining completed successfully!")

        # Save best model with index
        models_dir = get_models_path()
        best_model_name = f"best-{run_index}.pt"
        best_model_path = save_dir / "weights" / "best.pt"

        # Copy the best model to the models directory with indexed name
        if best_model_path.exists():
            shutil.copy(best_model_path, models_dir / best_model_name)
            print(f"Best model saved to {models_dir / best_model_name}")
        else:
            print(f"Best model not found at {best_model_path}")

        # Evaluate the model
        val_results = model.val()

        # Export model to ONNX format
        print("Exporting model to ONNX format...")
        try:
            onnx_path = models_dir / f"best-{run_index}.onnx"
            model.export(format="onnx", imgsz=640)

            default_onnx_path = save_dir / "weights" / "best.onnx"
            if default_onnx_path.exists():
                shutil.copy(default_onnx_path, onnx_path)
                print(f"Model exported to ONNX format at {onnx_path}")
            else:
                print(f"ONNX model not found at {default_onnx_path}")
        except Exception as export_error:
            print(f"Error exporting to ONNX: {str(export_error)}")

        print("\nTo continue training from this checkpoint, you can run:")
        print(f"model = YOLO('models/{best_model_name}')")
        print("model.train(resume=True)")

    except KeyboardInterrupt:
        print("\nTraining interrupted! You can resume training using:")
        print(f"model = YOLO('runs/train/{run_name}/weights/last.pt')")
        print("model.train(resume=True)")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Check the error message above and adjust parameters if needed.")
    finally:
        # Always close wandb run
        wandb.finish()


if __name__ == "__main__":
    train_model()
