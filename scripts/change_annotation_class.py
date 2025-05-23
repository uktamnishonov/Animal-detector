import os

# Define the base directory and splits
base_dir = "dataset-2/coyote"
splits = ["train", "test", "valid"]

# Process each split
for split in splits:
    labels_dir = os.path.join(base_dir, split, "labels")

    # Check if the labels directory exists
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found for {split}")
        continue

    # Process each txt file in the labels directory
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_dir, filename)

            # Read the file content
            with open(file_path, "r") as file:
                lines = file.readlines()

            # Modify the class index
            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts:  # Check if line is not empty
                    parts[0] = "10"  # Change class index to 10
                    modified_lines.append(" ".join(parts) + "\n")

            # Write back to file
            with open(file_path, "w") as file:
                file.writelines(modified_lines)

    print(f"Processed all files in {split} split")

print("Completed changing class indices to 10")
