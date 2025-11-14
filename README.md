# Animal Detector

A real-time animal detection system powered by YOLOv8, designed to work with Raspberry Pi camera streams and support image/video uploads. This application provides intelligent distance-based warnings for autonomous vehicle applications.

## Features

- **🔴 Live Camera Stream**: Real-time object detection from Raspberry Pi camera via socket connection
- **🖼️ Image Upload**: Detect animals in static images with bounding boxes
- **📹 Video Processing**: Process video files with frame-by-frame animal detection
- **⚠️ Distance Warnings**: Intelligent proximity alerts based on object size
  - "Object is too close, stop the car" (objects >15% of frame)
  - "Caution! Object detected in front of the car" (smaller objects)
- **🎯 Adjustable Confidence**: Real-time confidence threshold control (0.1 - 1.0)

## Supported Animals

The model can detect 11 different animal classes:
- Bird
- Boar
- Cat
- Deer
- Dog
- Opossum
- Person
- Raccoon
- Skunk
- Squirrel
- Coyote

## Requirements

### Python Dependencies
```bash
pip install streamlit opencv-python pillow numpy ultralytics torch
```

### System Requirements
- Python 3.8+
- CUDA-capable GPU (optional, for faster processing)
- Network connectivity for Raspberry Pi streaming

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Animal-detector.git
cd Animal-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your YOLOv8 model file:
```bash
mkdir -p models
# Copy your trained model to: models/best-13.pt
```

## Usage

### Starting the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Modes of Operation

#### 1. Live Camera Stream (Raspberry Pi)
- Click **"🔴 Live Camera (Pi)"**
- Click **"Start Stream"** to begin receiving frames
- The app listens on port **9999** for incoming camera streams
- Detections appear in real-time with distance warnings
- Click **"Stop Stream"** to end the session

#### 2. Image Upload
- Click **"🖼️ Upload Photo"**
- Upload JPG, JPEG, or PNG files
- Instant detection results with bounding boxes
- Distance warning appears in the right panel

#### 3. Video Processing
- Click **"📹 Upload Video"**
- Upload MP4, AVI, MOV, or MKV files
- Automatic processing with progress tracking
- Download processed video with detection overlays

## Raspberry Pi Camera Setup

To stream from Raspberry Pi to this application:

```python
import socket
import pickle
import struct
import cv2

# Connect to the server
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('SERVER_IP', 9999))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Serialize frame
    data = pickle.dumps(frame)
    message_size = struct.pack("L", len(data))
    
    # Send frame
    client_socket.sendall(message_size + data)

cap.release()
client_socket.close()
```

Replace `SERVER_IP` with the IP address of the machine running the Streamlit app.

## Configuration

Edit `app.py` to customize settings:

```python
MODEL_PATH = "models/best-13.pt"  # Path to YOLOv8 model
SOCKET_PORT = 9999                 # Port for Raspberry Pi stream
CLOSE_THRESHOLD = 0.15             # Distance warning threshold (15% of frame)
```

## Project Structure

```
Animal-detector/
├── app.py              # Main Streamlit application
├── models/             # Directory for YOLOv8 models
│   └── best-13.pt     # Trained model file
├── README.md          # This file
└── requirements.txt   # Python dependencies
```

## How It Works

### Detection Pipeline
1. **Input**: Live stream, image, or video frame
2. **Preprocessing**: Resize to 640x640 with padding, maintaining aspect ratio
3. **Inference**: YOLOv8 model processes the frame
4. **Post-processing**: Filter by confidence threshold, scale coordinates back
5. **Distance Calculation**: Compute bounding box area ratio to determine proximity
6. **Visualization**: Draw bounding boxes and labels on frame
7. **Output**: Display results with warnings

### Distance Warning Logic
```python
area_ratio = (box_width * box_height) / (image_width * image_height)

if area_ratio > 0.15:  # 15% of frame
    warning = "Object is too close, stop the car"
else:
    warning = "Caution! Object detected in front of the car"
```

## Performance Tips

- **Live Stream**: Frames are queued (max 10) to prevent lag
- **Video Processing**: Every Nth frame is processed for speed (adaptive sampling)
- **GPU Acceleration**: Automatically uses CUDA if available
- **Caching**: Model is cached to prevent reloading

## Troubleshooting

### Model Not Found
```
Error: Model file not found: models/best-13.pt
```
**Solution**: Ensure your trained YOLOv8 model is in the `models/` directory

### Socket Connection Failed
```
Error: Connection error
```
**Solution**: Check network connectivity and firewall settings on port 9999

### Video Processing Issues
```
Error: Cannot open video file
```
**Solution**: Try converting video to MP4 format with H.264 codec

## Use Cases

- **Wildlife Monitoring**: Track animal movements in natural habitats
- **Autonomous Vehicles**: Detect obstacles and wildlife on roads
- **Security Systems**: Alert when animals approach restricted areas
- **Research**: Analyze animal behavior from video footage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics
- UI powered by [Streamlit](https://streamlit.io/)
- Computer vision processing with [OpenCV](https://opencv.org/)

## Contact

For questions or support, please open an issue on GitHub.