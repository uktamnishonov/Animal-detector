from ultralytics import YOLO
import os
import torch
from pathlib import Path
import shutil
import re
import ssl
import urllib.request


def download_yolo_model():
    # Disable SSL verification (only for downloading the model)
    ssl._create_default_https_context = ssl._create_unverified_context

    model_url = (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    )
    model_path = "yolov8n.pt"

    if not os.path.exists(model_path):
        print(f"Downloading YOLOv8n model from {model_url}")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully!")
    else:
        print("Model already exists!")
    return model_path


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


def check_gpu_memory():
    if torch.cuda.is_available():
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        print(
            f"GPU Memory Available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB"
        )


def train_model():
    # Disable wandb completely
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"

    # Download/verify model exists before training
    model_path = download_yolo_model()

    # Get next run index and create run directory name
    run_index = get_next_run_index()
    run_name = f"run-{run_index}"

    # Load the YOLOv8n model from downloaded path
    model = YOLO(model_path)

    # Determine device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        check_gpu_memory()

    # Maximum performance training arguments
    args = {
        "data": "animal-dataset-yolo/data.yaml",
        "epochs": 100,  # Maximum epochs for best convergence
        "imgsz": 640,
        "batch": 24 if device == "cuda" else 6,
        "device": device,
        "workers": 6 if device == "cuda" else 3,
        "project": "runs/train",
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        # Maximum performance learning parameters
        "optimizer": "AdamW",
        "lr0": 0.001,  # Very conservative learning rate for stability
        "lrf": 0.01,  # Final learning rate (lr0 * lrf)
        "weight_decay": 0.0005,
        "momentum": 0.937,
        # Training schedule for maximum performance
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "patience": 30,  # High patience for maximum convergence
        # Loss weights fine-tuned for better performance
        "box": 7.5,  # Box regression loss
        "cls": 1,  # Classification loss weight
        "dfl": 1.5,  # Distribution focal loss
        # Performance and memory optimization
        "cache": "ram" if device == "cuda" else False,  # Cache to RAM for speed
        "amp": True,  # Automatic Mixed Precision for faster training
        "verbose": True,
        "close_mosaic": 15,  # Disable mosaic in final 15 epochs
        "save": True,
        "save_period": 10,  # Save checkpoint every 10 epochs
        "plots": True,  # Generate all training plots
        "val": True,  # Validate during training
        # Maximum performance optimization parameters
        "cos_lr": True,  # Cosine learning rate schedule for better convergence
        "dropout": 0.0,  # No dropout
        "label_smoothing": 0.1,  # Label smoothing for better generalization
    }

    # Start training
    try:
        # Set the save directory path with indexed name
        save_dir = Path("runs/train") / run_name
        print(f"Training will be saved to: {save_dir}")

        # Train the model
        print("Starting training...")
        results = model.train(**args)

        print("\nTraining completed successfully!")
        print(f"Results saved in: {save_dir}")

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

        # Display final results
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)

        # Load and display metrics from results.csv if available
        results_csv = save_dir / "results.csv"
        if results_csv.exists():
            import pandas as pd

            df = pd.read_csv(results_csv)
            if not df.empty:
                final_metrics = df.iloc[-1]
                print(
                    f"Final mAP50: {final_metrics.get('metrics/mAP50(B)', 'N/A'):.5f}"
                )
                print(
                    f"Final mAP50-95: {final_metrics.get('metrics/mAP50-95(B)', 'N/A'):.5f}"
                )
                print(
                    f"Final Box Loss: {final_metrics.get('train/box_loss', 'N/A'):.5f}"
                )
                print(
                    f"Final Class Loss: {final_metrics.get('train/cls_loss', 'N/A'):.5f}"
                )

        # Evaluate the model on validation set
        print("\nRunning final validation...")
        val_results = model.val()

        # Export model to ONNX format
        print("\nExporting model to ONNX format...")
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

        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Best model: models/{best_model_name}")
        print(f"Training logs: {save_dir}")
        print("\nTo continue training from this checkpoint:")
        print(f"model = YOLO('models/{best_model_name}')")
        print("model.train(resume=True)")

    except KeyboardInterrupt:
        print("\nTraining interrupted! You can resume training using:")
        print(f"model = YOLO('runs/train/{run_name}/weights/last.pt')")
        print("model.train(resume=True)")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Check the error message above and adjust parameters if needed.")


if __name__ == "__main__":
    train_model()
