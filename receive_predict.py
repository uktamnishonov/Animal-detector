import socket
import struct
import pickle
import cv2
from ultralytics import YOLO

model = YOLO("models/best-13.pt")

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 9999))
server_socket.listen(1)
print("Waiting for camera stream...")

conn, addr = server_socket.accept()
print(f"Connected to {addr}")
data = b""
payload_size = struct.calcsize("L")

while True:
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data)
    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("YOLOv8 Inference", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

conn.close()
cv2.destroyAllWindows()
