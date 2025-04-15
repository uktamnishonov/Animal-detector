import os
import shutil
import random
from pathlib import Path
import numpy as np
from PIL import Image

# Define paths and settings
source_dir = "animal-dataset"
dest_dir = "animal-dataset-yolo"
train_ratio = 0.8
MAX_IMAGES_PER_CLASS = 1000  # For dataset balancing

# Define class mapping including base YOLO classes and new ones
base_classes = ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
additional_classes = ["deer", "fox", "beaver", "raccoon", "rabbit", "squirrel", "goat", "chicken", "skunk", "coyote"]
class_names = base_classes + additional_classes
class_map = {name.upper(): idx for idx, name in enumerate(class_names)}

def create_directories():
    """Create necessary directories for YOLO format"""
    os.makedirs(f"{dest_dir}/images/train", exist_ok=True)
    os.makedirs(f"{dest_dir}/images/val", exist_ok=True)
    os.makedirs(f"{dest_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{dest_dir}/labels/val", exist_ok=True)

def balance_dataset(image_paths, max_images):
    """Balance dataset by randomly selecting up to max_images"""
    if len(image_paths) > max_images:
        return random.sample(image_paths, max_images)
    return image_paths

def process_dataset():
    """Process and organize the dataset"""
    create_directories()
    all_images = {cls.upper(): [] for cls in class_names}
    
    # Collect and balance images
    for class_name in os.listdir(source_dir):
        if not os.path.isdir(os.path.join(source_dir, class_name)):
            continue
            
        if class_name not in class_map:
            print(f"Skipping unknown class: {class_name}")
            continue
            
        class_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Get full paths
        image_paths = [os.path.join(class_dir, img) for img in images]
        
        # Balance dataset
        balanced_images = balance_dataset(image_paths, MAX_IMAGES_PER_CLASS)
        all_images[class_name] = balanced_images
        print(f"Collected {len(balanced_images)} images for {class_name}")
    
    # Split and copy images/labels
    for class_name, images in all_images.items():
        if not images:
            continue
            
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Process train and validation sets
        for img_set, subset in [(train_images, "train"), (val_images, "val")]:
            for img_path in img_set:
                # Copy image
                img_filename = os.path.basename(img_path)
                new_img_path = os.path.join(dest_dir, "images", subset, f"{class_name}_{img_filename}")
                shutil.copy2(img_path, new_img_path)
                
                # Create YOLO format label
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                # Create label with centered bounding box (80% of image size)
                box_width = 0.8
                box_height = 0.8
                x_center = 0.5
                y_center = 0.5
                
                label_filename = os.path.splitext(img_filename)[0] + ".txt"
                label_path = os.path.join(dest_dir, "labels", subset, f"{class_name}_{label_filename}")
                
                with open(label_path, "w") as f:
                    f.write(f"{class_map[class_name]} {x_center} {y_center} {box_width} {box_height}\n")

def create_yaml():
    """Create YAML configuration file for YOLOv8"""
    yaml_path = os.path.join(dest_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(dest_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name.lower()}\n")

if __name__ == "__main__":
    process_dataset()
    create_yaml()
    print("\nDataset preparation completed!")
    print(f"Dataset structure created at {dest_dir}")
    print("Note: Simple centered bounding boxes were created. For better accuracy, consider using actual annotations.")