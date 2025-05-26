import cv2
import socket
import struct
import pickle
import time

server_ip = 'YOUR_LAPTOP_IP'  # Replace this
server_port = 9999

def connect_to_receiver():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_ip, server_port))
            print("Connected to receiver.")
            return s
        except:
            print("Receiver not available, retrying in 3 seconds...")
            time.sleep(3)

cap = cv2.VideoCapture(0)

while True:
    sock = connect_to_receiver()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            data = pickle.dumps(frame)
            message = struct.pack("L", len(data)) + data
            sock.sendall(message)

    except Exception as e:
        print(f"Connection lost: {e}")
        sock.close()
        print("Reconnecting...")
        time.sleep(2)


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
