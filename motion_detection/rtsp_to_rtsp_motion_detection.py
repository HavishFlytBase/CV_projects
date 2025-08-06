"""
RTSP-to-RTSP Motion Detection
Reads from an RTSP input stream, performs motion detection, and publishes to an output RTSP stream
"""

import cv2
import numpy as np
import threading
import time
import argparse
import asyncio
from motion_detection_utils import *

class RTSPToRTSPMotionDetector:
    def __init__(self, input_rtsp_url, output_rtsp_url, method='frame_differencing'):
        """
        Initialize RTSP-to-RTSP Motion Detector
        
        Parameters:
        - input_rtsp_url: RTSP URL to read from
        - output_rtsp_url: RTSP URL to publish to
        - method: 'background_subtraction', 'frame_differencing', or 'optical_flow'
        """
        self.input_rtsp_url = input_rtsp_url
        self.output_rtsp_url = output_rtsp_url
        self.method = method
        
        # Initialize capture and output
        self.capture = None
        self.output_rtsp = None
        self.running = False
        
        # Video properties (will be auto-detected)
        self.width = 640
        self.height = 480
        self.frame_rate = 30.0
        self.frame_duration = 1.0 / self.frame_rate
        
        # Method-specific initialization
        self.prev_frame = None
        if method == 'background_subtraction':
            self.backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True)
            self.backSub.setShadowThreshold(0.5)
        
        self.kernel = np.array((9,9), dtype=np.uint8)
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None
        
        # Threading
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # GStreamer pipelines
        self.input_pipeline = None
        self.output_pipeline = None
        self._setup_pipelines()
    
    def _setup_pipelines(self):
        """Setup GStreamer pipelines for input and output"""
        # Input pipeline - RTSP source to OpenCV
        self.input_pipeline = (
            f"rtspsrc location={self.input_rtsp_url} buffer-mode=none latency=0 tcp-timeout=5000000 ! "
            "rtph264depay ! h264parse ! nvh264dec ! videoconvert ! appsink wait-on-eos=false drop=true"
        )
        
        # Output pipeline - OpenCV to RTSP sink
        self.output_pipeline = (
            "appsrc max-bytes=500000 max-latency=0 ! videoconvert ! video/x-raw,format=I420 ! videoscale ! "
            "queue max-size-buffers=0 flush-on-eos=true ! "
            "nvh264enc ! "
            f"rtspclientsink location={self.output_rtsp_url} latency=0"
        )
    
    def connect_input(self):
        """Connect to input RTSP stream using GStreamer"""
        print(f"Connecting to input RTSP: {self.input_rtsp_url}")
        self.capture = cv2.VideoCapture(self.input_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.capture.isOpened():
            raise Exception(f"Failed to open input RTSP stream: {self.input_rtsp_url}")
        
        # Get video properties
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = self.capture.get(cv2.CAP_PROP_FPS)
        
        if self.frame_rate <= 0:
            self.frame_rate = 30.0
        
        self.frame_duration = 1.0 / self.frame_rate
        
        print(f"Input stream connected: {self.width}x{self.height} @ {self.frame_rate} FPS")
        return True
    
    def connect_output(self):
        """Connect to output RTSP stream using GStreamer"""
        print(f"Connecting to output RTSP: {self.output_rtsp_url}")
        self.output_rtsp = cv2.VideoWriter(
            self.output_pipeline, 
            cv2.CAP_GSTREAMER,
            0, 
            self.frame_rate,
            (self.width, self.height), 
            True
        )
        
        if not self.output_rtsp.isOpened():
            raise Exception(f"Failed to open output RTSP stream: {self.output_rtsp_url}")
        
        print(f"Output stream connected: {self.width}x{self.height} @ {self.frame_rate} FPS")
        return True
    
    def detect_motion(self, frame):
        """
        Detect motion in frame using specified method
        Returns list of bounding boxes for detected motion
        """
        if self.method == 'background_subtraction':
            return self._detect_background_subtraction(frame)
        elif self.method == 'frame_differencing':
            return self._detect_frame_differencing(frame)
        elif self.method == 'optical_flow':
            return self._detect_optical_flow(frame)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def _detect_background_subtraction(self, frame):
        """Background subtraction method"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.backSub.apply(gray)
        mask = self._get_background_subtraction_mask(mask)
        return get_bounding_boxes(mask, min_area=500)
    
    def _detect_frame_differencing(self, frame):
        """Frame differencing method"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return []
        
        mask = self._get_frame_differencing_mask(self.prev_frame, gray)
        self.prev_frame = gray
        return get_bounding_boxes(mask, min_area=500)
    
    def _detect_optical_flow(self, frame):
        """Optical flow method"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return []
        
        flow = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray, None, None)
        flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mask = self._get_optical_flow_mask(flow_mag, motion_thresh=2.0)
        self.prev_frame = gray
        return get_bounding_boxes(mask, min_area=500)
    
    def _get_background_subtraction_mask(self, mask):
        """Clean up background subtraction mask"""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        return mask
    
    def _get_frame_differencing_mask(self, prev_frame, curr_frame):
        """Get motion mask from frame differencing"""
        frame_diff = cv2.absdiff(prev_frame, curr_frame)
        frame_diff = cv2.medianBlur(frame_diff, 3)
        mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 3)
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        return mask
    
    def _get_optical_flow_mask(self, flow_mag, motion_thresh):
        """Get motion mask from optical flow"""
        kernel = np.ones((7,7))
        motion_mask = np.uint8(flow_mag > motion_thresh) * 255
        motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return motion_mask
    
    def _add_overlay(self, frame, detection_count):
        """Add information overlay to frame"""
        # Add method name
        cv2.putText(frame, f"Method: {self.method.replace('_', ' ').title()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add detection count
        cv2.putText(frame, f"Detections: {detection_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, 
                   (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add FPS
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", 
                       (frame.shape[1] - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def release_resources(self, input_only=False, output_only=False):
        """Release video capture and writer resources"""
        try:
            if self.capture is not None and not output_only:
                self.capture.release()
                self.capture = None
            if self.output_rtsp is not None and not input_only:
                self.output_rtsp.release()
                self.output_rtsp = None
        except Exception as e:
            print(f"Error releasing resources: {e}")
    
    def process_stream(self):
        """Main processing loop for RTSP-to-RTSP streaming"""
        self.running = True
        self.start_time = time.time()
        
        print(f"Starting RTSP-to-RTSP motion detection with method: {self.method}")
        print("Press Ctrl+C to stop...")
        
        try:
            # Connect to input and output streams
            self.connect_input()
            self.connect_output()
            
            while self.running and not self.stop_event.is_set():
                start_frame_time = time.time()
                
                # Read frame from input stream
                ret, frame = self.capture.read()
                if not ret:
                    print("Failed to read frame, attempting to reconnect...")
                    self.release_resources(output_only=True)
                    time.sleep(1)
                    try:
                        self.connect_input()
                        continue
                    except Exception as e:
                        print(f"Failed to reconnect input: {e}")
                        break
                
                self.frame_count += 1
                
                # Check if video properties have changed
                height, width = frame.shape[:2]
                if height != self.height or width != self.width:
                    print(f"Video resolution changed: {width}x{height}")
                    self.width, self.height = width, height
                    
                    # Recreate output stream with new dimensions
                    self.release_resources(input_only=True)
                    self.connect_output()
                
                # Detect motion
                detections = self.detect_motion(frame)
                
                # Draw bounding boxes
                if len(detections) > 0:
                    self.detection_count += len(detections)
                    draw_bboxes(frame, detections)
                
                # Add overlay info
                self._add_overlay(frame, len(detections))
                
                # Write frame to output stream
                if self.output_rtsp and self.output_rtsp.isOpened():
                    self.output_rtsp.write(frame)
                else:
                    print("Output stream not available, attempting to reconnect...")
                    try:
                        self.connect_output()
                        self.output_rtsp.write(frame)
                    except Exception as e:
                        print(f"Failed to reconnect output: {e}")
                
                # Print statistics every 100 frames
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    print(f"Processed {self.frame_count} frames, "
                          f"{self.detection_count} total detections, "
                          f"FPS: {fps:.1f}")
                
                # Frame rate control
                elapsed_frame_time = time.time() - start_frame_time
                sleep_time = self.frame_duration - elapsed_frame_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        except Exception as e:
            print(f"Error in processing loop: {e}")
        finally:
            self.stop()
    
    def start_async(self):
        """Start processing in a separate thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            print("Processing already running")
            return
        
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self.process_stream, daemon=True)
        self.processing_thread.start()
        print("Started RTSP-to-RTSP processing in background thread")
    
    def stop(self):
        """Stop processing and cleanup"""
        print("Stopping RTSP-to-RTSP motion detection...")
        self.running = False
        self.stop_event.set()
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Release resources
        self.release_resources()
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"Final statistics: {self.frame_count} frames processed, "
                  f"{self.detection_count} total detections, "
                  f"Average FPS: {fps:.1f}")
        
        print("RTSP-to-RTSP motion detection stopped")

def main():
    parser = argparse.ArgumentParser(description='RTSP-to-RTSP Motion Detection')
    parser.add_argument('--input-rtsp', required=True, 
                       help='Input RTSP stream URL (e.g., rtsp://192.168.1.100:554/stream)')
    parser.add_argument('--output-rtsp', required=True,
                       help='Output RTSP stream URL (e.g., rtsp://192.168.1.200:554/output)')
    parser.add_argument('--method', choices=['background_subtraction', 'frame_differencing', 'optical_flow'],
                       default='frame_differencing', help='Motion detection method')
    
    args = parser.parse_args()
    
    print(f"Input RTSP: {args.input_rtsp}")
    print(f"Output RTSP: {args.output_rtsp}")
    print(f"Detection method: {args.method}")
    
    # Create and start detector
    detector = RTSPToRTSPMotionDetector(
        input_rtsp_url=args.input_rtsp,
        output_rtsp_url=args.output_rtsp,
        method=args.method
    )
    
    try:
        detector.process_stream()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        detector.stop()

if __name__ == "__main__":
    main()
