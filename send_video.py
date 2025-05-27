import cv2
import socket
import struct
import pickle
import time

# Replace with your laptop's IP address where Streamlit is running
LAPTOP_IP = 'YOUR_LAPTOP_IP'  # e.g., '192.168.1.100'
LAPTOP_PORT = 9999

def connect_to_streamlit():
    """Connect to the Streamlit app"""
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((LAPTOP_IP, LAPTOP_PORT))
            print(f"Connected to Streamlit app at {LAPTOP_IP}:{LAPTOP_PORT}")
            return s
        except Exception as e:
            print(f"Streamlit app not available: {e}")
            print("Retrying in 3 seconds...")
            time.sleep(3)

def find_usb_camera():
    """Find available USB camera"""
    print("Searching for USB cameras...")
    
    # Check what video devices exist
    import os
    video_devices = []
    for i in range(10):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            video_devices.append(i)
            print(f"Found video device: {device_path}")
    
    if not video_devices:
        print("No /dev/video* devices found")
        print("Check if USB camera is connected: lsusb")
        return None
    
    # Try each video device with different methods
    for index in video_devices:
        print(f"Testing video device {index}...")
        
        # Method 1: Default OpenCV
        try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Try to read a frame to make sure it really works
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ USB camera found at index {index} (default method)")
                    cap.release()
                    return index, None
                else:
                    print(f"  Camera {index} opens but can't read frames")
            cap.release()
        except Exception as e:
            print(f"  Error with default method: {e}")
        
        # Method 2: V4L2 backend (most common for USB cameras on Linux)
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ USB camera found at index {index} (V4L2 backend)")
                    cap.release()
                    return index, cv2.CAP_V4L2
                else:
                    print(f"  Camera {index} with V4L2 opens but can't read frames")
            cap.release()
        except Exception as e:
            print(f"  Error with V4L2 backend: {e}")
    
    # If nothing worked, try forcing common indices
    print("Trying common camera indices...")
    common_indices = [0, 1, 2]
    for index in common_indices:
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ Found working camera at index {index} (forced)")
                    cap.release()
                    return index, cv2.CAP_V4L2
            cap.release()
        except:
            continue
    
    return None

def initialize_camera():
    """Initialize USB camera with proper settings"""
    camera_info = find_usb_camera()
    
    if camera_info is None:
        print("No USB camera found!")
        print("Troubleshooting steps:")
        print("1. Check if camera is connected: lsusb")
        print("2. Check video devices: ls -la /dev/video*")
        print("3. Try different USB ports")
        print("4. Check if camera works with other software: cheese, vlc, etc.")
        return None
    
    # Handle different return types from find_usb_camera
    if isinstance(camera_info, tuple):
        index, backend = camera_info
        if backend is not None:
            cap = cv2.VideoCapture(index, backend)
            print(f"Using camera {index} with backend {backend}")
        else:
            cap = cv2.VideoCapture(index)
            print(f"Using camera {index} with default backend")
    else:
        index = camera_info
        cap = cv2.VideoCapture(index)
        print(f"Using camera {index}")
    
    if not cap.isOpened():
        print(f"Failed to open camera at index {index}")
        return None
    
    # Set camera properties (some USB cameras might not support all properties)
    print("Setting camera properties...")
    
    # Try to set properties, but don't fail if they're not supported
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
    except Exception as e:
        print(f"Warning: Could not set some camera properties: {e}")
    
    # Test frame capture
    print("Testing frame capture...")
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Cannot capture frames from camera")
        cap.release()
        return None
    
    # Get actual settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"USB Camera initialized successfully:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frame shape: {frame.shape}")
    
    return cap

def main():
    # Initialize camera
    cap = initialize_camera()
    if cap is None:
        return
    
    while True:
        sock = connect_to_streamlit()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                # Optional: resize frame to reduce bandwidth
                # frame = cv2.resize(frame, (640, 480))
                
                # Serialize frame
                data = pickle.dumps(frame)
                message = struct.pack("L", len(data)) + data
                
                try:
                    sock.sendall(message)
                except BrokenPipeError:
                    print("Connection broken, reconnecting...")
                    break
                except Exception as e:
                    print(f"Send error: {e}")
                    break
                    
                # Small delay to control frame rate
                time.sleep(0.05)  # ~20 FPS

        except Exception as e:
            print(f"Connection lost: {e}")
        finally:
            sock.close()
            print("Reconnecting in 2 seconds...")
            time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down camera stream...")
    finally:
        cv2.destroyAllWindows()


'''
[Unit]
Description=Video Stream Sender to Laptop
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Desktop/capstone
ExecStart=/home/ubuntu/capstone/bin/python /home/ubuntu/Desktop/capstone/send_video.py
Restart=always

[Install]
WantedBy=multi-user.target
'''
