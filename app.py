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


# Initialize session state
def init_session_state():
    defaults = {
        "socket_connected": False,
        "frame_queue": Queue(maxsize=10),
        "socket_thread": None,
        "stop_streaming": False,
        "model": None,
        "current_mode": None,  # Track current mode to prevent duplicate buttons
        "last_uploaded_image": None,  # Store uploaded image for re-detection
        "stream_active": False,  # Track if stream is actively running
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


# Socket receiver for Raspberry Pi stream
def socket_receiver(frame_queue, stop_event):
    """Receive frames from Raspberry Pi via socket"""
    server_socket = None
    conn = None

    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("0.0.0.0", SOCKET_PORT))
        server_socket.listen(1)

        while not stop_event.is_set():
            try:
                server_socket.settimeout(1.0)
                conn, addr = server_socket.accept()

                data = b""
                payload_size = struct.calcsize("L")

                while not stop_event.is_set():
                    try:
                        # Receive payload size
                        while len(data) < payload_size:
                            packet = conn.recv(4096)
                            if not packet:
                                raise ConnectionError("Connection lost")
                            data += packet

                        # Extract and receive frame
                        msg_size = struct.unpack("L", data[:payload_size])[0]
                        data = data[payload_size:]

                        while len(data) < msg_size:
                            packet = conn.recv(4096)
                            if not packet:
                                raise ConnectionError("Connection lost")
                            data += packet

                        # Process frame
                        frame = pickle.loads(data[:msg_size])
                        data = data[msg_size:]

                        # Add to queue (replace old frames if full)
                        if frame_queue.full():
                            try:
                                frame_queue.get(block=False)
                            except:
                                pass
                        frame_queue.put(frame, block=False)

                    except ConnectionError:
                        break
                    except Exception as e:
                        print(f"Frame error: {e}")
                        break

            except socket.timeout:
                continue
            except Exception as e:
                print(f"Connection error: {e}")
                time.sleep(2)
            finally:
                if conn:
                    conn.close()
                    conn = None

    except Exception as e:
        print(f"Socket error: {e}")
    finally:
        if conn:
            conn.close()
        if server_socket:
            server_socket.close()


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


def update_detection_info(animal_text, accuracy_text, predictions):
    """Update detection info display"""
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


def handle_live_camera_content(animal_text, accuracy_text, confidence_threshold, model):
    """Handle live camera stream content only"""
    st.markdown(
        '<span class="red-dot">●</span> <span class="stream-title">Live Stream from Raspberry Pi</span>',
        unsafe_allow_html=True,
    )

    # Start/Stop buttons
    col_start, col_stop = st.columns([1, 1])
    with col_start:
        start_stream = st.button("Start Stream", key="start_stream")
    with col_stop:
        stop_stream = st.button("Stop Stream", key="stop_stream")

    # Handle start stream
    if start_stream and not st.session_state.socket_connected:
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

    # Handle stop stream
    if stop_stream:
        st.session_state.stop_streaming = True
        st.session_state.stream_active = False
        if hasattr(st.session_state, "stop_event"):
            st.session_state.stop_event.set()
        st.session_state.socket_connected = False
        st.rerun()

    # Status display
    status_placeholder = st.empty()
    if st.session_state.socket_connected and st.session_state.stream_active:
        status_placeholder.success(
            f"🟢 Socket server running on port {SOCKET_PORT}. Raspberry Pi connected!"
        )
    else:
        status_placeholder.info(
            "🔵 Click 'Start Stream' to begin receiving from Raspberry Pi"
        )

    # Video feed
    video_feed = st.empty()

    # Process stream if active
    if (
        st.session_state.socket_connected
        and st.session_state.stream_active
        and not st.session_state.stop_streaming
    ):
        frames_received = 0
        last_detection_time = time.time()

        while st.session_state.socket_connected and not st.session_state.stop_streaming:
            if not st.session_state.frame_queue.empty():
                frame = st.session_state.frame_queue.get(block=False)
                frames_received += 1

                if frames_received % 30 == 0:
                    status_placeholder.success(
                        f"🟢 Streaming active - {frames_received} frames received"
                    )

                # Run detection with current confidence threshold
                predictions = detect_objects(model, frame, confidence_threshold)

                if predictions:
                    frame = draw_detections(frame, predictions)
                    update_detection_info(animal_text, accuracy_text, predictions)
                    last_detection_time = time.time()
                elif time.time() - last_detection_time > 3:
                    update_detection_info(animal_text, accuracy_text, [])

                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_feed.image(frame_rgb, width=650)
            else:
                time.sleep(0.1)


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
                # Show only the image with detection boxes
                annotated_img = draw_detections(image_np, predictions)
                st.image(
                    annotated_img, caption="Detection Results", use_container_width=True
                )
                update_detection_info(animal_text, accuracy_text, predictions)
            else:
                # Show original image only when no detections found
                st.image(
                    image_np, caption="No Detections", use_container_width=True
                )
                st.info("No detections in this image.")
                update_detection_info(animal_text, accuracy_text, [])

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
                    # Get best detection
                    best_detection = max(detections_found, key=lambda x: x["score"])
                    update_detection_info(animal_text, accuracy_text, [best_detection])

                    # Summary
                    unique_animals = list(set([d["label"] for d in detections_found]))
                    st.success(
                        f"✅ Processing complete! Found {len(detections_found)} detections across {processed_frames} frames."
                    )
                    st.info(f"Detections: {', '.join(unique_animals)}")

                else:
                    update_detection_info(animal_text, accuracy_text, [])
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
