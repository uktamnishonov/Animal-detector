from ultralytics import YOLO
import cv2
import argparse
import os


def predict_image(model_path, image_path, output_dir="."):
    # Load the trained model
    model = YOLO(model_path)

    # Load and predict on the image
    results = model.predict(image_path, conf=0.25)

    # Get the first result (since we're processing one image)
    result = results[0]

    # Load the image for visualization
    img = cv2.imread(image_path)

    # Plot results on the image
    for box in result.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get confidence and class
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = result.names[class_id]

        # Draw rectangle and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # Get the filename and create output filename
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{image_name}_prediction.jpg")

    # Save the annotated image
    cv2.imwrite(output_path, img)
    print(f"Prediction saved to {output_path}")

    # Print detections
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]
        print(f"Detected {class_name} with confidence: {confidence:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict animals in an image using trained YOLO model"
    )
    parser.add_argument(
        "--model", default="models/best-13.pt", help="Path to the trained model"
    )
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument(
        "--output", default=".", help="Output directory for predictions"
    )

    args = parser.parse_args()
    predict_image(args.model, args.image, args.output)
