"""
Optimized Streamlit animal detection app with socket-based camera stream from Raspberry Pi.
"""

import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time
from ultralytics import YOLO
import io
import os
import torch
import socket
import struct
import pickle
import threading
from queue import Queue
import atexit
import psutil
import signal

# Configuration
MODEL_PATH = "models/best-13.pt"
SOCKET_PORT = 9999
CLASS_NAMES = [
    "bird",
    "boar",
    "cat",
    "deer",
    "dog",
    "opossum",
    "person",
    "raccoon",
    "skunk",
    "squirrel",
    "coyote",
]

# Page setup
st.set_page_config(
    page_title="Animal Detection", layout="wide", initial_sidebar_state="collapsed"
)

# Styling
st.markdown(
    """
<style>
    .main { background-color: #0E1117; color: white; }
    .stButton button { background-color: white; color: black; border-radius: 20px; border: none; padding: 8px 16px; }
    .detection-info { background-color: #262730; border-radius: 10px; padding: 20px; margin-top: 20px; color: white; }
    .red-dot { color: #ff6b6b; font-size: 20px; }
    .title, .stream-title { font-weight: bold; font-size: 24px; margin-bottom: 20px; color: white; }
    .stSlider > div > div > div > div { background-color: #262730; color: white; }
</style>
""",
    unsafe_allow_html=True,
)


# Enhanced session state initialization
def init_session_state():
    defaults = {
        "socket_connected": False,
        "frame_queue": Queue(maxsize=10),
        "socket_thread": None,
        "stop_streaming": False,
        "model": None,
        "current_mode": None,
        "last_uploaded_image": None,
        "stream_active": False,
        "stop_event": None,  # Add this
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Model loading
@st.cache_resource
def load_model():
    """Load YOLOv8 model with error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found: {MODEL_PATH}")
            return None
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Enhanced socket receiver with better connection stability
def socket_receiver(frame_queue, stop_event):
    """Receive frames from Raspberry Pi via socket with improved stability"""
    server_socket = None
    conn = None
    reconnect_count = 0

    try:
        # Create socket with proper options
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Set socket to non-blocking mode for better control
        server_socket.settimeout(1.0)

        try:
            server_socket.bind(("0.0.0.0", SOCKET_PORT))
        except OSError as e:
            if e.errno == 48:
                print(f"Port {SOCKET_PORT} is busy. Waiting for it to be freed...")
                time.sleep(3)
                try:
                    server_socket.bind(("0.0.0.0", SOCKET_PORT))
                except OSError:
                    print(
                        f"Could not bind to port {SOCKET_PORT}. Trying alternative port..."
                    )
                    # Try a few alternative ports
                    for alt_port in range(SOCKET_PORT + 1, SOCKET_PORT + 10):
                        try:
                            server_socket.bind(("0.0.0.0", alt_port))
                            print(f"Using alternative port: {alt_port}")
                            break
                        except OSError:
                            continue
                    else:
                        raise Exception("No available ports found")
            else:
                raise e

        server_socket.listen(1)
        bound_port = server_socket.getsockname()[1]
        print(f"Socket server listening on port {bound_port}")

        while not stop_event.is_set():
            try:
                # Accept connection with timeout
                conn, addr = server_socket.accept()
                reconnect_count += 1
                print(f"Connection #{reconnect_count} established with {addr}")

                # Set connection timeout
                conn.settimeout(5.0)

                # Connection handling with better error recovery
                data = b""
                payload_size = struct.calcsize("L")
                consecutive_errors = 0
                frames_received = 0

                while not stop_event.is_set():
                    try:
                        # Receive payload size with timeout handling
                        while len(data) < payload_size and not stop_event.is_set():
                            try:
                                packet = conn.recv(4096)
                                if not packet:
                                    raise ConnectionError("No data received")
                                data += packet
                                consecutive_errors = (
                                    0  # Reset error count on successful receive
                                )
                            except socket.timeout:
                                # Check if we should continue waiting
                                if consecutive_errors < 3:
                                    consecutive_errors += 1
                                    continue
                                else:
                                    raise ConnectionError("Timeout receiving data")

                        if len(data) < payload_size:
                            break  # Stop event was set

                        # Extract message size
                        msg_size = struct.unpack("L", data[:payload_size])[0]

                        # Validate message size (prevent memory issues)
                        if msg_size > 10 * 1024 * 1024:  # 10MB limit
                            print(f"Warning: Large message size {msg_size}, skipping")
                            data = b""
                            continue

                        data = data[payload_size:]

                        # Receive frame data
                        while len(data) < msg_size and not stop_event.is_set():
                            try:
                                remaining = msg_size - len(data)
                                packet = conn.recv(min(4096, remaining))
                                if not packet:
                                    raise ConnectionError(
                                        "Connection lost during frame receive"
                                    )
                                data += packet
                            except socket.timeout:
                                if consecutive_errors < 3:
                                    consecutive_errors += 1
                                    continue
                                else:
                                    raise ConnectionError(
                                        "Timeout receiving frame data"
                                    )

                        if len(data) < msg_size:
                            break  # Stop event was set

                        # Process frame
                        try:
                            frame_data = data[:msg_size]
                            frame = pickle.loads(frame_data)
                            data = data[msg_size:]

                            # Validate frame
                            if isinstance(frame, np.ndarray) and frame.size > 0:
                                frames_received += 1

                                # Add to queue (non-blocking)
                                try:
                                    if frame_queue.full():
                                        frame_queue.get_nowait()  # Remove oldest frame
                                    frame_queue.put_nowait(frame)
                                except:
                                    pass  # Queue operations failed, continue

                                # Log progress periodically
                                if frames_received % 100 == 0:
                                    print(
                                        f"Received {frames_received} frames from {addr}"
                                    )
                            else:
                                print("Warning: Invalid frame received")

                        except (pickle.UnpicklingError, TypeError) as e:
                            print(f"Frame unpickling error: {e}")
                            data = b""  # Reset buffer on unpickling error
                            continue

                    except ConnectionError as e:
                        print(f"Connection error: {e}")
                        break
                    except Exception as e:
                        print(f"Unexpected error: {e}")
                        consecutive_errors += 1
                        if consecutive_errors > 5:
                            print("Too many consecutive errors, dropping connection")
                            break

                print(
                    f"Connection with {addr} ended. Received {frames_received} frames total."
                )

            except socket.timeout:
                # Normal timeout, continue listening
                continue
            except Exception as e:
                print(f"Accept error: {e}")
                time.sleep(1)  # Brief pause before retrying
            finally:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
                    conn = None

    except Exception as e:
        print(f"Socket server error: {e}")
    finally:
        # Cleanup
        if conn:
            try:
                conn.close()
            except:
                pass
        if server_socket:
            try:
                server_socket.close()
            except:
                pass
        print("Socket server cleaned up")


def kill_process_on_port(port):
    """Kill any process using the specified port - Fixed version"""
    try:
        import psutil

        killed = False
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                # Use net_connections() instead of deprecated connections()
                for conn in proc.net_connections():
                    if (
                        hasattr(conn, "laddr")
                        and conn.laddr
                        and conn.laddr.port == port
                    ):
                        print(
                            f"Found process {proc.info['pid']} ({proc.info['name']}) using port {port}"
                        )
                        try:
                            proc.terminate()  # Try graceful termination first
                            proc.wait(timeout=3)  # Wait up to 3 seconds
                            killed = True
                        except psutil.TimeoutExpired:
                            proc.kill()  # Force kill if graceful termination fails
                            killed = True
                        except psutil.AccessDenied:
                            print(
                                f"Access denied when trying to kill process {proc.info['pid']}"
                            )
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return killed
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
        return False


def find_available_port(start_port=9999, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
            sock.close()
            return port
        except OSError:
            continue
    return None


def cleanup_socket_resources():
    """Improved cleanup function"""
    try:
        if hasattr(st.session_state, "stop_event") and st.session_state.stop_event:
            st.session_state.stop_event.set()

        if (
            hasattr(st.session_state, "socket_thread")
            and st.session_state.socket_thread
        ):
            if st.session_state.socket_thread.is_alive():
                st.session_state.socket_thread.join(timeout=5)

        # Don't automatically kill processes - let them close gracefully
        print("Socket resources cleaned up")
    except Exception as e:
        print(f"Error during cleanup: {e}")


# Detection processing
def process_predictions(results, confidence_threshold=0.3):
    """Process YOLO results into usable format"""
    predictions = []

    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, "boxes") and len(result.boxes) > 0:
            for box, score, cls in zip(
                result.boxes.xyxy, result.boxes.conf, result.boxes.cls
            ):
                # Convert tensors to Python types
                if isinstance(box, torch.Tensor):
                    box = box.cpu().numpy()
                score = float(
                    score.item() if isinstance(score, torch.Tensor) else score
                )
                cls = int(cls.item() if isinstance(cls, torch.Tensor) else cls)

                if score >= confidence_threshold:
                    class_name = (
                        CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
                    )
                    predictions.append(
                        {
                            "box": box.tolist() if isinstance(box, np.ndarray) else box,
                            "label": class_name,
                            "score": score * 100,
                        }
                    )

    return sorted(predictions, key=lambda x: x["score"], reverse=True)


def draw_detections(image, predictions):
    """Draw bounding boxes and labels on image"""
    img_copy = image.copy()
    for pred in predictions:
        x1, y1, x2, y2 = [int(coord) for coord in pred["box"]]
        color = (0, 255, 0)

        # Draw box and label
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        text = f"{pred['label']}: {pred['score']:.1f}%"
        cv2.putText(
            img_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return img_copy


def resize_for_detection(image, target_size=640):
    """Resize image maintaining aspect ratio with padding"""
    h, w = image.shape[:2]
    if w > h:
        new_w, new_h = target_size, int(target_size * h / w)
    else:
        new_w, new_h = int(target_size * w / h), target_size

    resized = cv2.resize(image, (new_w, new_h))

    # Add padding to make it square
    if new_w != target_size or new_h != target_size:
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2
        resized = cv2.copyMakeBorder(
            resized,
            pad_h,
            target_size - new_h - pad_h,
            pad_w,
            target_size - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

    return resized, (new_w, new_h), (pad_w, pad_h)


def scale_predictions_back(predictions, original_size, resized_size, padding):
    """Scale prediction coordinates back to original image size"""
    orig_w, orig_h = original_size
    new_w, new_h = resized_size
    pad_w, pad_h = padding

    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

    for pred in predictions:
        x1, y1, x2, y2 = pred["box"]
        # Remove padding and scale back
        x1 = max(0, int((x1 - pad_w) * scale_x))
        y1 = max(0, int((y1 - pad_h) * scale_y))
        x2 = min(orig_w, int((x2 - pad_w) * scale_x))
        y2 = min(orig_h, int((y2 - pad_h) * scale_y))
        pred["box"] = [x1, y1, x2, y2]

    return predictions


def detect_objects(model, image, confidence_threshold):
    """Run object detection on image"""
    original_size = (image.shape[1], image.shape[0])  # (width, height)

    # Resize for detection
    resized_img, resized_size, padding = resize_for_detection(image)

    # Run inference
    results = model(resized_img, verbose=False)
    predictions = process_predictions(results, confidence_threshold)

    # Scale coordinates back to original size
    if predictions:
        predictions = scale_predictions_back(
            predictions, original_size, resized_size, padding
        )

    return predictions


def calculate_distance_warning(box, image_shape):
    """Calculate relative distance based on bounding box size"""
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1

    # Calculate box area relative to image area
    box_area = box_width * box_height
    image_area = image_width * image_height
    area_ratio = box_area / image_area

    # Define threshold for "close" objects (adjust as needed)
    CLOSE_THRESHOLD = 0.15  # 15% of image area

    if area_ratio > CLOSE_THRESHOLD:
        return "Object is too close, stop the car", True
    else:
        return "Caution! Object detected in front of the car", False


def update_detection_info(animal_text, accuracy_text, predictions, image_shape=None):
    """Update detection info display with distance warning"""
    if predictions:
        animal_text.write(f"Detected: {predictions[0]['label']}")
        accuracy_text.write(f"Accuracy: {predictions[0]['score']:.2f}%")
    else:
        animal_text.write("Detected: No detection")
        accuracy_text.write("Accuracy: -")


# Main app
def main():
    init_session_state()

    # Load model
    if st.session_state.model is None:
        with st.spinner("Loading animal detection model..."):
            st.session_state.model = load_model()

    model = st.session_state.model

    # Header
    st.markdown("<h1>Animal Detection</h1>", unsafe_allow_html=True)

    # Control buttons - only show once at the top
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("🔴 Live Camera (Pi)", key="main_live_camera"):
            st.session_state.current_mode = "live_camera"
    with col2:
        if st.button("🖼️ Upload Photo", key="main_upload_photo"):
            st.session_state.current_mode = "upload_photo"
    with col3:
        if st.button("📹 Upload Video", key="main_upload_video"):
            st.session_state.current_mode = "upload_video"
    with col4:
        if st.button("⚫ Dark Mode", key="main_dark_mode"):
            pass  # Dark mode functionality can be added here

    # Layout
    left_col, right_col = st.columns([2.5, 1])

    # Detection info panel - always visible
    with right_col:
        st.markdown('<div class="detection-info">', unsafe_allow_html=True)
        st.markdown('<div class="title">Detection Info</div>', unsafe_allow_html=True)
        animal_text = st.empty()
        accuracy_text = st.empty()
        animal_text.write("Detected: -")
        accuracy_text.write("Accuracy: -")

        st.markdown("### Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 0.1, 1.0, 0.3, 0.05, key="confidence_slider"
        )

        # Add warning text section
        st.markdown("### Distance Warning")
        warning_text = st.empty()

        # Update the warning display function to use the new warning_text
        def update_warning(predictions, image_shape):
            if predictions and image_shape is not None:
                warning_msg, is_close = calculate_distance_warning(
                    predictions[0]["box"], image_shape
                )
                warning_color = "red" if is_close else "yellow"
                warning_text.markdown(
                    f"<div style='color: {warning_color}; font-weight: bold;'>{warning_msg}</div>",
                    unsafe_allow_html=True,
                )
            else:
                warning_text.markdown(
                    "<div style='color: gray;'>No objects detected</div>",
                    unsafe_allow_html=True,
                )

        st.session_state.update_warning = update_warning
        st.markdown("</div>", unsafe_allow_html=True)

    # Handle different modes without showing duplicate buttons
    with left_col:
        if st.session_state.current_mode == "live_camera" and model:
            handle_live_camera_content(
                animal_text, accuracy_text, confidence_threshold, model
            )

        elif st.session_state.current_mode == "upload_photo" and model:
            handle_photo_upload_content(
                animal_text, accuracy_text, confidence_threshold, model
            )

        elif st.session_state.current_mode == "upload_video" and model:
            handle_video_upload_content(
                animal_text, accuracy_text, confidence_threshold, model
            )

        elif st.session_state.current_mode is None:
            st.markdown("### Welcome to Animal Detection")
            st.markdown("Select a mode above to get started:")
            st.markdown("- **🔴 Live Camera**: Stream from Raspberry Pi")
            st.markdown("- **🖼️ Upload Photo**: Detect animals in images")
            st.markdown("- **📹 Upload Video**: Process video files")

    # Footer
    st.markdown("---")
    st.markdown("Animal detection powered by YOLOv8 | Raspberry Pi Camera Stream")


# Modified handle_live_camera_content with better connection management
def handle_live_camera_content(animal_text, accuracy_text, confidence_threshold, model):
    """Handle live camera stream with improved connection stability"""
    st.markdown(
        '<span class="red-dot">●</span> <span class="stream-title">Live Stream from Raspberry Pi</span>',
        unsafe_allow_html=True,
    )

    # Control buttons
    col_start, col_stop, col_reset = st.columns([1, 1, 1])
    with col_start:
        start_stream = st.button("Start Stream", key="start_stream")
    with col_stop:
        stop_stream = st.button("Stop Stream", key="stop_stream")
    with col_reset:
        reset_connection = st.button("Reset Connection", key="reset_connection")

    # Handle reset connection (more conservative approach)
    if reset_connection:
        # Stop current streaming
        if hasattr(st.session_state, "stop_event") and st.session_state.stop_event:
            st.session_state.stop_event.set()

        # Wait for thread to finish
        if (
            hasattr(st.session_state, "socket_thread")
            and st.session_state.socket_thread
        ):
            if st.session_state.socket_thread.is_alive():
                st.session_state.socket_thread.join(timeout=3)

        # Clear queue
        while not st.session_state.frame_queue.empty():
            try:
                st.session_state.frame_queue.get_nowait()
            except:
                break

        # Reset state
        st.session_state.socket_connected = False
        st.session_state.stream_active = False
        st.session_state.socket_thread = None
        st.session_state.stop_event = None

        st.success("Connection reset successfully!")
        time.sleep(1)
        st.rerun()

    # Handle start stream
    if start_stream and not st.session_state.socket_connected:
        try:
            st.session_state.stop_streaming = False

            if (
                st.session_state.socket_thread is None
                or not st.session_state.socket_thread.is_alive()
            ):

                stop_event = threading.Event()
                st.session_state.socket_thread = threading.Thread(
                    target=socket_receiver,
                    args=(st.session_state.frame_queue, stop_event),
                    daemon=True,
                )
                st.session_state.socket_thread.start()
                st.session_state.socket_connected = True
                st.session_state.stream_active = True
                st.session_state.stop_event = stop_event

                st.success("Stream started! Waiting for Raspberry Pi connection...")

        except Exception as e:
            st.error(f"Failed to start stream: {e}")

    # Handle stop stream
    if stop_stream:
        st.session_state.stop_streaming = True
        st.session_state.stream_active = False
        if hasattr(st.session_state, "stop_event") and st.session_state.stop_event:
            st.session_state.stop_event.set()
        st.session_state.socket_connected = False
        st.success("Stream stopped")
        st.rerun()

    # Enhanced status display
    status_placeholder = st.empty()
    connection_info = st.empty()

    if st.session_state.socket_connected and st.session_state.stream_active:
        status_placeholder.success(f"🟢 Socket server running on port {SOCKET_PORT}")

        # Check queue size for connection health
        # queue_size = st.session_state.frame_queue.qsize()
        # if queue_size > 0:
        #     connection_info.info(f"📡 Receiving frames - Queue: {queue_size}/10")
        # else:
        #     connection_info.warning("⏳ Waiting for frames from Raspberry Pi...")
    else:
        status_placeholder.info("🔵 Click 'Start Stream' to begin")
        connection_info.empty()

    # Video feed
    video_feed = st.empty()

    # Process stream if active
    if (
        st.session_state.socket_connected
        and st.session_state.stream_active
        and not st.session_state.stop_streaming
    ):

        frames_processed = 0
        last_detection_time = time.time()
        last_frame_time = time.time()

        while st.session_state.socket_connected and not st.session_state.stop_streaming:
            try:
                if not st.session_state.frame_queue.empty():
                    frame = st.session_state.frame_queue.get(block=False)
                    frames_processed += 1
                    last_frame_time = time.time()

                    # Update status every 30 frames
                    # if frames_processed % 30 == 0:
                    #     status_placeholder.success(
                    #         f"🟢 Streaming active - {frames_processed} frames processed"
                    #     )

                    # Run detection
                    predictions = detect_objects(model, frame, confidence_threshold)

                    if predictions:
                        frame = draw_detections(frame, predictions)
                        update_detection_info(animal_text, accuracy_text, predictions)
                        if hasattr(st.session_state, "update_warning"):
                            st.session_state.update_warning(predictions, frame.shape)
                        last_detection_time = time.time()
                    elif time.time() - last_detection_time > 3:
                        update_detection_info(animal_text, accuracy_text, [])
                        if hasattr(st.session_state, "update_warning"):
                            st.session_state.update_warning([], None)

                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_feed.image(frame_rgb, width=900)

                    # Update connection info
                    queue_size = st.session_state.frame_queue.qsize()
                    # connection_info.success(
                    #     f"📡 Live feed active - Queue: {queue_size}/10"
                    # )

                else:
                    # No frames available, check if connection is stale
                    if time.time() - last_frame_time > 10:  # 10 seconds without frames
                        connection_info.warning(
                            "⚠️ No frames received for 10 seconds. Connection may be lost."
                        )

                    time.sleep(0.1)

            except Exception as e:
                st.error(f"Error processing stream: {e}")
                break

        # Clean up when loop ends
        video_feed.empty()
        connection_info.info("Stream processing stopped")


def handle_photo_upload_content(
    animal_text, accuracy_text, confidence_threshold, model
):
    """Handle photo upload content only"""
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="photo_uploader",
    )

    # Store uploaded image in session state
    if uploaded_file is not None:
        st.session_state.last_uploaded_image = uploaded_file

    # Process image if available
    if st.session_state.last_uploaded_image is not None:
        try:
            # Load and process image
            pil_image = Image.open(
                io.BytesIO(st.session_state.last_uploaded_image.getvalue())
            )
            image_np = np.array(pil_image)

            # Handle different color formats
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

            # Run detection with current confidence threshold
            predictions = detect_objects(model, image_np, confidence_threshold)

            if predictions:
                annotated_img = draw_detections(image_np, predictions)
                st.image(
                    annotated_img, caption="Detection Results", use_container_width=True
                )
                update_detection_info(animal_text, accuracy_text, predictions)
                st.session_state.update_warning(predictions, image_np.shape)
            else:
                st.image(image_np, caption="No Detections", use_container_width=True)
                st.info("No detections in this image.")
                update_detection_info(animal_text, accuracy_text, [])
                st.session_state.update_warning([], None)

        except Exception as e:
            st.error(f"Error processing image: {e}")


def handle_video_upload_content(
    animal_text, accuracy_text, confidence_threshold, model
):
    """Handle video upload content only"""
    uploaded_file = st.file_uploader(
        "Choose a video...",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
        key="video_uploader",
    )

    if uploaded_file is not None:
        try:
            # Save temporary video with original extension
            file_extension = uploaded_file.name.split(".")[-1].lower()
            temp_input_path = f"temp_input_video.{file_extension}"

            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"✅ Video uploaded successfully! ({uploaded_file.name})")

            # Video info without showing the video
            cap_info = cv2.VideoCapture(temp_input_path)
            if cap_info.isOpened():
                total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap_info.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_info.release()

                st.info(
                    f"📹 Video Info: {width}x{height}, {duration:.1f}s, {total_frames} frames, {fps:.1f} FPS"
                )

            # Automatically start processing
            with st.spinner(
                "🔄 Processing video with animal detection... This may take a while for long videos."
            ):
                cap = cv2.VideoCapture(temp_input_path)

                if not cap.isOpened():
                    st.error(
                        "❌ Error: Cannot open video file. Please try a different format."
                    )
                    return

                # Video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))  # Ensure fps is at least 1
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if width == 0 or height == 0 or total_frames == 0:
                    st.error("❌ Error: Invalid video properties detected.")
                    cap.release()
                    return

                st.info(f"🎬 Processing {total_frames} frames at {fps} FPS...")

                # Try different codecs for better web compatibility
                output_path = "processed_video.mp4"

                # Use H.264 codec for better web compatibility
                try:
                    # Try H.264 first (best web compatibility)
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if not out.isOpened():
                        # Fallback to mp4v
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                except:
                    # Final fallback
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                if not out.isOpened():
                    st.error("❌ Error: Cannot create output video file.")
                    cap.release()
                    return

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Processing parameters
                sample_rate = max(
                    1, total_frames // 200
                )  # Process every Nth frame for speed
                frame_count = 0
                processed_frames = 0
                detections_found = []
                last_detections = (
                    []
                )  # Keep track of recent detections for smoother video

                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1
                        progress = min(frame_count / total_frames, 1.0)
                        progress_bar.progress(progress)

                        # Process every Nth frame for detection
                        original_frame = frame.copy()
                        current_detections = []

                        if frame_count % sample_rate == 0:
                            try:
                                predictions = detect_objects(
                                    model, frame, confidence_threshold
                                )
                                if predictions:
                                    detections_found.extend(predictions)
                                    processed_frames += 1
                                    current_detections = predictions
                                    last_detections = (
                                        predictions  # Update last known detections
                                    )

                                status_text.text(
                                    f"🔍 Processed {processed_frames} frames with detections | Frame {frame_count}/{total_frames}"
                                )
                            except Exception as detect_error:
                                st.warning(
                                    f"⚠️ Detection failed on frame {frame_count}: {detect_error}"
                                )

                        # Draw detections on frame (use last known detections for smoother video)
                        if current_detections:
                            frame = draw_detections(original_frame, current_detections)
                        elif (
                            last_detections
                            and frame_count % (sample_rate * 2) < sample_rate
                        ):
                            # Show last detections for a few frames for smoother appearance
                            frame = draw_detections(original_frame, last_detections)
                        else:
                            frame = original_frame

                        # Write frame to output
                        out.write(frame)

                        # Update status every 100 frames
                        if frame_count % 100 == 0:
                            status_text.text(
                                f"⏳ Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)"
                            )

                except Exception as processing_error:
                    st.error(f"❌ Error during processing: {processing_error}")
                finally:
                    cap.release()
                    out.release()

                # Final results
                if detections_found:
                    best_detection = max(detections_found, key=lambda x: x["score"])
                    update_detection_info(
                        animal_text, accuracy_text, [best_detection], (height, width, 3)
                    )

                    # Summary
                    unique_animals = list(set([d["label"] for d in detections_found]))
                    st.success(
                        f"✅ Processing complete! Found {len(detections_found)} detections across {processed_frames} frames."
                    )
                    st.info(f"Detections: {', '.join(unique_animals)}")

                else:
                    update_detection_info(animal_text, accuracy_text, [])
                    st.session_state.update_warning([], None)
                    st.info("ℹ️ No detections in this video.")

                # Display processed video with detection boxes
                try:
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        st.success(
                            "🎉 Video processing complete! Showing video with detection boxes:"
                        )

                        # Display the processed video
                        with open(output_path, "rb") as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes)

                        # Also provide download link
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=video_bytes,
                            file_name=f"processed_{uploaded_file.name}",
                            mime="video/mp4",
                        )
                    else:
                        st.error("❌ Processed video file is empty or not found.")

                except Exception as display_error:
                    st.error(f"❌ Cannot display processed video: {display_error}")

                    # Try alternative display method
                    try:
                        if os.path.exists(output_path):
                            st.info("🔄 Trying alternative display method...")
                            st.video(output_path)
                    except:
                        st.warning(
                            "⚠️ Video was processed but cannot be displayed. Try downloading it instead."
                        )

        except Exception as e:
            st.error(f"❌ Error with video upload: {e}")
            st.info(
                "💡 Try uploading a different video format (MP4 recommended) or check if the file is corrupted."
            )


if __name__ == "__main__":
    main()
