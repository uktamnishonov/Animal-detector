import cv2
import socket
import struct
import pickle

# Setup camera
cap = cv2.VideoCapture(0)

# Setup socket
server_ip = 'YOUR_LAPTOP_IP'  # Replace with your laptop's local IP
server_port = 9999
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))
connection = client_socket.makefile('wb')

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Serialize frame
        data = pickle.dumps(frame)
        # Send size first
        client_socket.sendall(struct.pack("L", len(data)) + data)

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    client_socket.close()
