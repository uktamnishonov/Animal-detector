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

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera initialized successfully")
    
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
