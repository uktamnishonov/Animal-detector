import cv2
import requests
from ultralytics import YOLO
import numpy as np

# Load your YOLOv8 model
model = YOLO("yolov8n.pt")

# Raspberry Pi MJPEG stream
url = "http://192.168.216.225:5000/video-feed"

# Start the stream
stream = requests.get(url, stream=True)

bytes_buffer = b""

for chunk in stream.iter_content(chunk_size=1024):
    bytes_buffer += chunk
    a = bytes_buffer.find(b"\xff\xd8")  # JPEG start
    b = bytes_buffer.find(b"\xff\xd9")  # JPEG end

    if a != -1 and b != -1:
        jpg = bytes_buffer[a : b + 2]
        bytes_buffer = bytes_buffer[b + 2 :]

        # Decode JPEG to numpy array
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Run YOLOv8 inference
        results = model(frame)
        annotated = results[0].plot()

        # Show result
        cv2.imshow("YOLOv8 Stream", annotated)
        if cv2.waitKey(1) == ord("q"):
            break

cv2.destroyAllWindows()
