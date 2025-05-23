#!/bin/bash

# Create predictions directory if it doesn't exist
mkdir -p test/predictions

# Loop through all images in test/images
for img in test/images/*; do
    # Get the filename without path
    filename=$(basename "$img")
    echo "Processing $filename..."
    
    # Run the prediction script and save directly to predictions folder
    python scripts/test.py --image "$img" --output test/predictions
done

echo "All predictions completed! Check test/predictions folder for results."
