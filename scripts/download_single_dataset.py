import os
import requests
from pathlib import Path
import zipfile
import sys


def download_and_extract_dataset(url, output_dir):
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Temporary zip file path
    zip_path = output_dir / "dataset.zip"

    try:
        # Download the dataset
        print("Downloading dataset...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the zip file
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        # Remove the temporary zip file
        zip_path.unlink()
        print(f"Dataset successfully downloaded and extracted to {output_dir}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")
        sys.exit(1)
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file")
        zip_path.unlink()
        sys.exit(1)


if __name__ == "__main__":
    DATASET_URL = "https://universe.roboflow.com/ds/g2TEXqVOWz?key=vj45X0pWsn"
    # "https://universe.roboflow.com/ds/8j0QSafOFN?key=wTGo3jXLVg" # 3k dataset
    # "https://universe.roboflow.com/ds/g2TEXqVOWz?key=vj45X0pWsn" # coyote
    OUTPUT_DIR = (
        "dataset-2/coyote"  # You can modify this to your desired output directory
    )

    download_and_extract_dataset(DATASET_URL, OUTPUT_DIR)
