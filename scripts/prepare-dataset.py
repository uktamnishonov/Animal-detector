import os
import shutil
import yaml
import subprocess
import random
from collections import defaultdict
import json

# Dataset configurations with direct download URLs
datasets = [
    {
        "name": "canine-detection",
        "class_name": "coyote",
        "url": "https://universe.roboflow.com/ds/g2TEXqVOWz?key=vj45X0pWsn",
    },
    {
        "name": "big-data",
        "class_name": "big-data",
        "url": "https://universe.roboflow.com/ds/8j0QSafOFN?key=wTGo3jXLVg",
    },
    # {
    #     "name": "deervision",
    #     "class_name": "deer",
    #     "url": "https://universe.roboflow.com/ds/bvyRMSzhpb?key=VBi6YEe6Xz",
    # },
    # {
    #     "name": "firefox2",
    #     "class_name": "fox",
    #     "url": "https://universe.roboflow.com/ds/Swp66vZE4l?key=KlDZ1tPMrh",
    # },
    # {
    #     "name": "racoonsfinder",
    #     "class_name": "raccoon",
    #     "url": "https://universe.roboflow.com/ds/633vWMmscg?key=QKEtXLvDvM",
    # },
    # {
    #     "name": "deteksi-kelinci",
    #     "class_name": "rabbit",
    #     "url": "https://universe.roboflow.com/ds/31Y3zyQL0i?key=loSj8JbokZ",
    # },
    # {
    #     "name": "rabbit-2",
    #     "class_name": "rabbit",
    #     "url": "https://universe.roboflow.com/ds/5Tgre9cLMJ?key=kpqloiqXtY",
    # },
    # {
    #     "name": "squirrel-annotation",
    #     "class_name": "squirrel",
    #     "url": "https://universe.roboflow.com/ds/eOVSFDCPj5?key=bJ5iTdlosp",
    # },
    # {
    #     "name": "goat-swaod",
    #     "class_name": "goat",
    #     "url": "https://universe.roboflow.com/ds/3AWT3yB66c?key=HJk21w7ToM",
    # },
    # {
    #     "name": "goat-2",
    #     "class_name": "goat",
    #     "url": "https://universe.roboflow.com/ds/v9VG7G4Yki?key=BqAQmBLVdA",
    # },
    # {
    #     "name": "outdoor-cats",
    #     "class_name": "cat",
    #     "url": "https://universe.roboflow.com/ds/uRX8CEcWdM?key=uNSsg7hkUc",
    # },
    # {
    #     "name": "outdoor-dogs",
    #     "class_name": "dog",
    #     "url": "https://universe.roboflow.com/ds/Onfj1F481G?key=JoW2uui0cI",
    # },
    # {
    #     "name": "dogs-2",
    #     "class_name": "dog",
    #     "url": "https://universe.roboflow.com/ds/Q4qc1CBObs?key=lZAs9Mz9R5",
    # },
    # {
    #     "name": "cats-2",
    #     "class_name": "cat",
    #     "url": "https://universe.roboflow.com/ds/Kmr1hMYxjc?key=MvUFNYThf3",
    # },
]


def download_datasets():
    """Download all datasets using curl with better error handling"""
    base_dir = "animal-dataset"

    # Clean up and create base directory
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    success_count = 0
    failed_datasets = []

    for dataset in datasets:
        print(f"\nDownloading {dataset['name']} dataset...")
        dataset_dir = os.path.join(base_dir, dataset["name"])
        os.makedirs(dataset_dir, exist_ok=True)

        try:
            # Download zip file with better error handling
            zip_path = os.path.join(dataset_dir, "dataset.zip")
            curl_command = f"curl -L --fail --connect-timeout 30 --max-time 300 '{dataset['url']}' -o {zip_path}"
            result = subprocess.run(
                curl_command, shell=True, capture_output=True, text=True
            )

            if result.returncode != 0:
                print(f"Failed to download {dataset['name']}: {result.stderr}")
                failed_datasets.append(dataset["name"])
                continue

            # Check if file was actually downloaded and has content
            if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
                print(
                    f"Downloaded file for {dataset['name']} is too small or doesn't exist"
                )
                failed_datasets.append(dataset["name"])
                continue

            # Extract zip file
            unzip_command = f"unzip -q {zip_path} -d {dataset_dir}"
            result = subprocess.run(
                unzip_command, shell=True, capture_output=True, text=True
            )

            if result.returncode != 0:
                print(f"Failed to extract {dataset['name']}: {result.stderr}")
                failed_datasets.append(dataset["name"])
                continue

            # Clean up zip file
            os.remove(zip_path)
            success_count += 1
            print(f"✓ Successfully downloaded and extracted {dataset['name']}")

        except Exception as e:
            print(f"✗ Error processing {dataset['name']}: {str(e)}")
            failed_datasets.append(dataset["name"])
            continue

    print(f"\n=== Download Summary ===")
    print(f"Successfully downloaded: {success_count}/{len(datasets)} datasets")
    if failed_datasets:
        print(f"Failed datasets: {', '.join(failed_datasets)}")

    return failed_datasets


def analyze_dataset_structure():
    """Analyze the structure of downloaded datasets"""
    base_dir = "animal-dataset"
    analysis = {}

    for dataset in datasets:
        dataset_dir = os.path.join(base_dir, dataset["name"])
        if not os.path.exists(dataset_dir):
            continue

        analysis[dataset["name"]] = {
            "class_name": dataset["class_name"],
            "all_files": [],  # Store all image files regardless of original split
            "total_images": 0,
        }

        # Collect ALL images from all splits (don't preserve original splits)
        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(dataset_dir, split, "images")
            label_dir = os.path.join(dataset_dir, split, "labels")

            if os.path.exists(img_dir):
                images = [
                    f
                    for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                for img in images:
                    img_path = os.path.join(img_dir, img)
                    label_file = img.rsplit(".", 1)[0] + ".txt"
                    label_path = os.path.join(label_dir, label_file)

                    # Only include if label exists
                    if os.path.exists(label_path):
                        analysis[dataset["name"]]["all_files"].append(
                            (img_path, label_path)
                        )

                analysis[dataset["name"]]["total_images"] += len(images)

    return analysis


def create_combined_dataset():
    """Combine all downloaded datasets with guaranteed train/valid split per class"""
    source_dir = "animal-dataset"
    target_dir = "animal-dataset-yolo"

    # Analyze dataset structure
    print("Analyzing dataset structure...")
    analysis = analyze_dataset_structure()

    if not analysis:
        print("No datasets found to process!")
        return

    # Remove existing target directory
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Create directories
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(target_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, "labels"), exist_ok=True)

    # Group datasets by class and collect all files
    class_files = defaultdict(list)
    class_names = []

    for dataset_name, info in analysis.items():
        class_name = info["class_name"]
        if class_name not in class_names:
            class_names.append(class_name)

        # Add all files for this class
        for img_path, label_path in info["all_files"]:
            class_files[class_name].append((dataset_name, img_path, label_path))

    print(f"\n=== Class File Analysis ===")
    for class_name in class_names:
        print(f"{class_name}: {len(class_files[class_name])} total files")

    # Process each class with guaranteed train/valid split
    class_stats = defaultdict(lambda: {"train": 0, "valid": 0})
    min_valid_samples = 20  # Minimum samples for validation per class
    max_samples_per_class = 600  # Cap each class at 600 samples

    for class_idx, class_name in enumerate(class_names):
        print(f"\nProcessing class: {class_name}")
        
        # if class_name == "coyote":
        #     max_samples_per_class = 300

        all_files = class_files[class_name]
        if not all_files:
            print(f"No files found for {class_name}")
            continue

        # Shuffle files for random distribution
        random.shuffle(all_files)

        # Limit to max samples per class
        if len(all_files) > max_samples_per_class:
            all_files = all_files[:max_samples_per_class]
            print(
                f"  Limited to {max_samples_per_class} samples (was {len(class_files[class_name])})"
            )

        # Calculate train/valid split (80/20 but ensure minimum validation samples)
        total_files = len(all_files)
        valid_count = max(min_valid_samples, int(total_files * 0.2))
        valid_count = min(valid_count, total_files)  # Don't exceed available files
        train_count = total_files - valid_count

        print(f"  Total: {total_files}, Train: {train_count}, Valid: {valid_count}")

        # Split files
        valid_files = all_files[:valid_count]
        train_files = all_files[valid_count:]

        # Process training files
        for dataset_name, img_path, label_path in train_files:
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)

            # Copy image
            dst_img_path = os.path.join(
                target_dir, "train", "images", f"{dataset_name}_{img_name}"
            )
            shutil.copy2(img_path, dst_img_path)

            # Copy and update label
            dst_label_path = os.path.join(
                target_dir, "train", "labels", f"{dataset_name}_{label_name}"
            )
            update_label_file(label_path, dst_label_path, class_idx)
            class_stats[class_name]["train"] += 1

        # Process validation files
        for dataset_name, img_path, label_path in valid_files:
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)

            # Copy image
            dst_img_path = os.path.join(
                target_dir, "valid", "images", f"{dataset_name}_{img_name}"
            )
            shutil.copy2(img_path, dst_img_path)

            # Copy and update label
            dst_label_path = os.path.join(
                target_dir, "valid", "labels", f"{dataset_name}_{label_name}"
            )
            update_label_file(label_path, dst_label_path, class_idx)
            class_stats[class_name]["valid"] += 1

    # Create data.yaml file
    create_data_yaml(target_dir, class_names)

    # Print final statistics
    print(f"\n=== Final Dataset Statistics ===")
    print(f"Total classes: {len(class_names)}")
    print(f"Classes: {', '.join(class_names)}")
    print("\nSamples per class:")

    # Verify every class has validation samples
    validation_issues = []
    for class_name in class_names:
        train_count = class_stats[class_name]["train"]
        valid_count = class_stats[class_name]["valid"]
        total = train_count + valid_count
        print(
            f"  {class_name}: {total} total (train: {train_count}, valid: {valid_count})"
        )

        if valid_count == 0:
            validation_issues.append(class_name)

    if validation_issues:
        print(f"\n⚠️  WARNING: Classes with no validation samples: {validation_issues}")
    else:
        print(f"\n✓ All classes have validation samples")

    # Save statistics
    with open(os.path.join(target_dir, "dataset_stats.json"), "w") as f:
        json.dump(dict(class_stats), f, indent=2)

    print("\nDataset preparation completed!")
    return len(validation_issues) == 0


def update_label_file(src_path, dst_path, new_class_idx):
    """Update label file with new class index and validate format"""
    try:
        valid_lines = 0
        with open(src_path, "r") as src, open(dst_path, "w") as dst:
            for line_num, line in enumerate(src, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    print(
                        f"Warning: Invalid label format in {src_path} line {line_num}: {line}"
                    )
                    continue

                try:
                    # Validate YOLO format (class x y w h)
                    old_class = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])

                    # Check if values are in valid range
                    if not (
                        0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1
                    ):
                        print(
                            f"Warning: Invalid bbox coordinates in {src_path} line {line_num}"
                        )
                        continue

                    # Replace class index with new index
                    parts[0] = str(new_class_idx)
                    dst.write(" ".join(parts) + "\n")
                    valid_lines += 1

                except ValueError:
                    print(
                        f"Warning: Invalid number format in {src_path} line {line_num}: {line}"
                    )
                    continue

        # Check if any valid labels were written
        if valid_lines == 0:
            print(f"ERROR: No valid labels in {src_path}")

    except Exception as e:
        print(f"Error processing label file {src_path}: {str(e)}")


def create_data_yaml(base_dir, class_names):
    """Create data.yaml file for YOLOv8 training"""
    data = {
        "path": os.path.abspath(base_dir),
        "train": os.path.join("train", "images"),
        "val": os.path.join("valid", "images"),
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }

    with open(os.path.join(base_dir, "data.yaml"), "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Created data.yaml with {len(class_names)} classes")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Download datasets
    failed_datasets = download_datasets()

    if failed_datasets:
        print(f"\nWarning: {len(failed_datasets)} datasets failed to download.")
        print("Continuing with available datasets...")

    # Create combined dataset with guaranteed splits
    success = create_combined_dataset()

    if success:
        print("\n✅ Dataset preparation completed successfully!")
        print("All classes have both training and validation samples.")
    else:
        print("\n⚠️  Dataset preparation completed with warnings.")
        print("Some classes may be missing validation samples.")
