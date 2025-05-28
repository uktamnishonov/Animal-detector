import cv2
import socket
import struct
import pickle
import time

# Replace with your laptop's IP address where Streamlit is running
server_ip = 'YOUR_LAPTOP_IP'  # e.g., '192.168.1.100'
server_port = 9999

def connect_to_streamlit():
    """Connect to the Streamlit app"""
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_ip, server_port))
            print("Connected to Streamlit app.")
            return s
        except Exception as e:
            print(f"Streamlit app not available: {e}")
            print("Retrying in 3 seconds...")
            time.sleep(3)

def initialize_camera():
    """Initialize camera with error checking"""
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 25)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
    
    # Test if we can actually capture a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Camera opened but cannot capture frames")
        cap.release()
        return None
    
    print(f"Camera initialized successfully - Frame shape: {frame.shape}")
    return cap

# Main streaming loop
while True:
    # Initialize camera for each connection cycle
    cap = initialize_camera()
    if cap is None:
        print("Camera initialization failed, retrying in 5 seconds...")
        time.sleep(5)
        continue
    
    # Connect to Streamlit
    sock = connect_to_streamlit()
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame (frame #{frame_count})")
                # Try to reinitialize camera
                cap.release()
                time.sleep(1)
                cap = initialize_camera()
                if cap is None:
                    print("Camera reinitialization failed")
                    break
                continue

            frame_count += 1
            if frame_count % 100 == 0:  # Progress indicator
                print(f"Streamed {frame_count} frames")

            # Serialize and send frame
            try:
                data = pickle.dumps(frame)
                message = struct.pack("L", len(data)) + data
                sock.sendall(message)
            except BrokenPipeError:
                print("Connection broken")
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
        cap.release()
        print("Reconnecting in 2 seconds...")
        time.sleep(2)

# Cleanup (won't be reached due to infinite loop, but good practice)
print("Shutting down...")
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
