import os
import shutil
import yaml
import subprocess
import random

# Dataset configurations with direct download URLs
datasets = [
    {
        "name": "deervision",
        "class_name": "deer",
        "url": "https://universe.roboflow.com/ds/bvyRMSzhpb?key=VBi6YEe6Xz",
    },
    {
        "name": "firefox2",
        "class_name": "fox",
        "url": "https://universe.roboflow.com/ds/Swp66vZE4l?key=KlDZ1tPMrh",
    },
    {
        "name": "racoonsfinder",
        "class_name": "raccoon",
        "url": "https://universe.roboflow.com/ds/633vWMmscg?key=QKEtXLvDvM",
    },
    {
        "name": "deteksi-kelinci",
        "class_name": "rabbit",
        "url": "https://universe.roboflow.com/ds/31Y3zyQL0i?key=loSj8JbokZ",
    },
    {
        "name": "rabbit-2",
        "class_name": "rabbit",
        "url": "https://universe.roboflow.com/ds/5Tgre9cLMJ?key=kpqloiqXtY",
    },
    {
        "name": "squirrel-annotation",
        "class_name": "squirrel",
        "url": "https://universe.roboflow.com/ds/eOVSFDCPj5?key=bJ5iTdlosp",
    },
    {
        "name": "goat-swaod",
        "class_name": "goat",
        "url": "https://universe.roboflow.com/ds/3AWT3yB66c?key=HJk21w7ToM",
    },
    {
        "name": "goat-2",
        "class_name": "goat",
        "url": "https://universe.roboflow.com/ds/v9VG7G4Yki?key=BqAQmBLVdA",
    },
    {
        "name": "canine-detection",
        "class_name": "coyote",
        "url": "https://universe.roboflow.com/ds/g2TEXqVOWz?key=vj45X0pWsn",
    },
    {
        "name": "outdoor-cats",
        "class_name": "cat",
        "url": "https://universe.roboflow.com/ds/uRX8CEcWdM?key=uNSsg7hkUc",
    },
    {
        "name": "outdoor-dogs",
        "class_name": "dog",
        "url": "https://universe.roboflow.com/ds/Onfj1F481G?key=JoW2uui0cI",
    },
    {
        "name": "dogs-2",
        "class_name": "dog",
        "url": "https://universe.roboflow.com/ds/Q4qc1CBObs?key=lZAs9Mz9R5",
    },
    {
        "name": "cats-2",
        "class_name": "cat",
        "url": "https://universe.roboflow.com/ds/Kmr1hMYxjc?key=MvUFNYThf3",
    },
]


def download_datasets():
    """Download all datasets using curl"""
    base_dir = "animal-dataset"

    # Clean up and create base directory
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    for dataset in datasets:
        print(f"\nDownloading {dataset['name']} dataset...")
        dataset_dir = os.path.join(base_dir, dataset["name"])
        os.makedirs(dataset_dir, exist_ok=True)

        # Download and extract dataset
        try:
            # Download zip file
            zip_path = os.path.join(dataset_dir, "dataset.zip")
            curl_command = f"curl -L '{dataset['url']}' > {zip_path}"
            subprocess.run(curl_command, shell=True, check=True)

            # Extract zip file
            unzip_command = f"unzip {zip_path} -d {dataset_dir}"
            subprocess.run(unzip_command, shell=True, check=True)

            # Clean up zip file
            os.remove(zip_path)
            print(f"Successfully downloaded and extracted {dataset['name']}")
        except Exception as e:
            print(f"Error processing {dataset['name']}: {str(e)}")
            continue


def get_limited_files(file_list, limit=1000):
    """Limit the number of files to process"""
    if len(file_list) > limit:
        return random.sample(file_list, limit)
    return file_list


def create_combined_dataset():
    """Combine all downloaded datasets into one YOLO format dataset"""
    source_dir = "animal-dataset"
    target_dir = "animal-dataset-yolo"

    # Remove existing target directory if it exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # Create main dataset directory with correct structure
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(target_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, "labels"), exist_ok=True)

    # Keep track of class names and their indices
    class_names = []

    # Process each downloaded dataset
    for dataset in datasets:
        print(f"\nProcessing {dataset['name']} dataset...")
        try:
            dataset_dir = os.path.join(source_dir, dataset["name"])
            if not os.path.exists(dataset_dir):
                print(f"Skipping {dataset['name']} - directory not found")
                continue

            # Add class name if not already in list
            if dataset["class_name"] not in class_names:
                class_names.append(dataset["class_name"])

            class_idx = class_names.index(dataset["class_name"])

            # Move files to combined dataset
            for split in ["train", "valid"]:
                src_img_dir = os.path.join(dataset_dir, split, "images")
                src_label_dir = os.path.join(dataset_dir, split, "labels")

                if os.path.exists(src_img_dir):
                    # Get list of images and limit them
                    all_images = os.listdir(src_img_dir)
                    selected_images = get_limited_files(all_images, limit=1000)

                    # Copy limited images and labels
                    for img in selected_images:
                        shutil.copy2(
                            os.path.join(src_img_dir, img),
                            os.path.join(
                                target_dir, split, "images", f"{dataset['name']}_{img}"
                            ),
                        )

                        # Copy and update label file if it exists
                        label_file = (
                            img.replace(".jpg", ".txt")
                            .replace(".jpeg", ".txt")
                            .replace(".png", ".txt")
                        )
                        if os.path.exists(os.path.join(src_label_dir, label_file)):
                            update_label_file(
                                os.path.join(src_label_dir, label_file),
                                os.path.join(
                                    target_dir,
                                    split,
                                    "labels",
                                    f"{dataset['name']}_{label_file}",
                                ),
                                class_idx,
                            )

            print(f"Successfully processed {dataset['name']}")

        except Exception as e:
            print(f"Error processing {dataset['name']}: {str(e)}")
            continue

    # Create data.yaml file
    create_data_yaml(target_dir, class_names)
    print("\nDataset preparation completed!")
    print(f"Total number of classes: {len(class_names)}")
    print(f"Classes: {', '.join(class_names)}")


def update_label_file(src_path, dst_path, new_class_idx):
    """Update label file with new class index"""
    with open(src_path, "r") as src, open(dst_path, "w") as dst:
        for line in src:
            parts = line.strip().split()
            if parts:
                # Replace class index with new index
                parts[0] = str(new_class_idx)
                dst.write(" ".join(parts) + "\n")


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


if __name__ == "__main__":
    # First download all datasets using curl
    download_datasets()
    # Then combine them
    create_combined_dataset()
