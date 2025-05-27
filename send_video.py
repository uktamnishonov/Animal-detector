#!/usr/bin/env python3
"""
Raspberry Pi Camera Test Script
Run this first to check if your camera is working
"""

import cv2
import os
import subprocess

def check_camera_devices():
    """Check for available camera devices"""
    print("=== Checking Camera Devices ===")
    
    # Check /dev/video* devices
    video_devices = []
    for i in range(10):
        device = f"/dev/video{i}"
        if os.path.exists(device):
            video_devices.append(device)
            print(f"Found device: {device}")
    
    if not video_devices:
        print("No /dev/video* devices found")
    
    return video_devices

def check_camera_modules():
    """Check if camera modules are loaded"""
    print("\n=== Checking Camera Modules ===")
    
    try:
        result = subprocess.run(['lsmod'], capture_output=True, text=True)
        modules = result.stdout
        
        camera_modules = ['bcm2835_v4l2', 'uvcvideo', 'videodev']
        for module in camera_modules:
            if module in modules:
                print(f"✓ {module} module is loaded")
            else:
                print(f"✗ {module} module is NOT loaded")
    except Exception as e:
        print(f"Error checking modules: {e}")

def test_opencv_camera():
    """Test OpenCV camera access"""
    print("\n=== Testing OpenCV Camera Access ===")
    
    # Test different camera indices
    for i in range(5):
        print(f"Testing camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            print(f"✓ Camera {i} opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"  ✓ Successfully captured frame: {frame.shape}")
                
                # Test a few more frames
                for j in range(3):
                    ret, frame = cap.read()
                    if ret:
                        print(f"  ✓ Frame {j+2}: {frame.shape}")
                    else:
                        print(f"  ✗ Failed to capture frame {j+2}")
                        break
            else:
                print(f"  ✗ Failed to capture frame from camera {i}")
            
            cap.release()
            return i  # Return working camera index
        else:
            print(f"✗ Camera {i} failed to open")
            cap.release()
    
    return None

def test_with_backends():
    """Test with different OpenCV backends"""
    print("\n=== Testing Different Backends ===")
    
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_FFMPEG, "FFmpeg"),
    ]
    
    for backend_id, backend_name in backends:
        print(f"Testing {backend_name} backend...")
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i, backend_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"  ✓ {backend_name} camera {i} works: {frame.shape}")
                        cap.release()
                        return i, backend_id
                    else:
                        print(f"  ✗ {backend_name} camera {i} opened but no frame")
                cap.release()
            except Exception as e:
                print(f"  ✗ {backend_name} camera {i} error: {e}")
    
    return None, None

def show_system_info():
    """Show system information"""
    print("\n=== System Information ===")
    
    try:
        # Check if we're on Raspberry Pi
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'BCM' in cpuinfo or 'Raspberry Pi' in cpuinfo:
                print("✓ Running on Raspberry Pi")
            else:
                print("? Not detected as Raspberry Pi")
    except:
        print("? Could not determine if running on Raspberry Pi")
    
    # Check OpenCV version
    print(f"OpenCV version: {cv2.__version__}")
    
    # Check available camera backends
    print("Available backends:")
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_GSTREAMER, "GStreamer"), 
        (cv2.CAP_FFMPEG, "FFmpeg"),
    ]
    
    for backend_id, name in backends:
        try:
            # Just try to create a VideoCapture with the backend
            cap = cv2.VideoCapture()
            if cap.open(0, backend_id):
                print(f"  ✓ {name}")
                cap.release()
            else:
                print(f"  ✗ {name}")
        except:
            print(f"  ✗ {name}")

def main():
    print("Raspberry Pi Camera Diagnostic Tool")
    print("=" * 50)
    
    show_system_info()
    check_camera_devices()
    check_camera_modules()
    
    # Test basic OpenCV access
    working_index = test_opencv_camera()
    
    if working_index is not None:
        print(f"\n🎉 SUCCESS: Camera {working_index} is working!")
        print(f"Use this in your code: cv2.VideoCapture({working_index})")
    else:
        print("\n❌ No working camera found with basic method")
        print("Trying different backends...")
        
        working_index, backend = test_with_backends()
        if working_index is not None:
            print(f"\n🎉 SUCCESS: Camera {working_index} works with backend {backend}")
            print(f"Use this in your code: cv2.VideoCapture({working_index}, {backend})")
        else:
            print("\n❌ No camera found with any method")
            print("\nTroubleshooting steps:")
            print("1. Enable camera: sudo raspi-config -> Interface Options -> Camera")
            print("2. Load camera module: sudo modprobe bcm2835-v4l2")
            print("3. Check connection: ls -la /dev/video*")
            print("4. Reboot: sudo reboot")
            print("5. For USB cameras: Check lsusb output")

if __name__ == "__main__":
    main()

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
