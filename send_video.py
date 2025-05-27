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

# Initialize camera (using your working method)
cap = cv2.VideoCapture(0)

# Optional: Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 20)

print("Camera initialized successfully")

while True:
    sock = connect_to_streamlit()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Serialize and send frame
            data = pickle.dumps(frame)
            message = struct.pack("L", len(data)) + data
            sock.sendall(message)
            
            # Small delay to control frame rate
            time.sleep(0.05)  # ~20 FPS

    except Exception as e:
        print(f"Connection lost: {e}")
        sock.close()
        print("Reconnecting in 2 seconds...")
        time.sleep(2)

# Cleanup (this won't be reached due to infinite loop, but good practice)
cap.release()
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
