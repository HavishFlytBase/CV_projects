"""
RTSP-to-RTSP Motion Detection
Reads from an RTSP input stream, performs motion detection, and outputs to another RTSP stream
"""

import cv2
import numpy as np
import threading
import time
import argparse
from motion_detection_utils import *

class RTSPMotionDetector:
    def __init__(self, input_rtsp_url, method='frame_differencing'):
        """
        Initialize RTSP Motion Detector
        
        Parameters:
        - input_rtsp_url: RTSP URL to read from
        - method: 'background_subtraction', 'frame_differencing', or 'optical_flow'
        """
        self.input_rtsp_url = input_rtsp_url
        self.method = method
        
        # Initialize capture
        self.cap = None
        self.running = False
        
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
        
    def connect_input(self):
        """Connect to input RTSP stream"""
        print(f"Connecting to input RTSP: {self.input_rtsp_url}")
        self.cap = cv2.VideoCapture(self.input_rtsp_url)
        
        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise Exception(f"Failed to connect to input RTSP stream: {self.input_rtsp_url}")
        
        # Get stream properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.input_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Input stream properties: {self.width}x{self.height} @ {self.input_fps} FPS")
        
    def setup_display(self):
        """Setup OpenCV display window"""
        print("Setting up OpenCV display window")
        
    def detect_motion(self, frame):
        """Detect motion in frame using selected method"""
        detections = []
        
        if self.method == 'background_subtraction':
            detections = self._detect_background_subtraction(frame)
            
        elif self.method == 'frame_differencing':
            if self.prev_frame is not None:
                detections = self._detect_frame_differencing(self.prev_frame, frame)
            self.prev_frame = frame.copy()
            
        elif self.method == 'optical_flow':
            if self.prev_frame is not None:
                detections = self._detect_optical_flow(self.prev_frame, frame)
            self.prev_frame = frame.copy()
        
        return detections
    
    def _detect_background_subtraction(self, frame):
        """Background subtraction detection"""
        fg_mask = self.backSub.apply(frame)
        motion_mask = self._get_motion_mask(fg_mask)
        detections = get_contour_detections(motion_mask, thresh=100)
        
        if len(detections) == 0:
            return []
        
        bboxes = detections[:, :4]
        scores = detections[:, -1]
        return non_max_suppression(bboxes, scores, threshold=0.1)
    
    def _detect_frame_differencing(self, frame1, frame2):
        """Frame differencing detection"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        mask = self._get_frame_diff_mask(gray1, gray2)
        detections = get_contour_detections(mask, thresh=400)
        
        if len(detections) == 0:
            return []
        
        bboxes = detections[:, :4]
        scores = detections[:, -1]
        return non_max_suppression(bboxes, scores, threshold=0.1)
    
    def _detect_optical_flow(self, frame1, frame2):
        """Optical flow detection"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        gray1 = cv2.GaussianBlur(gray1, (3,3), 5)
        gray2 = cv2.GaussianBlur(gray2, (3,3), 5)
        
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                            pyr_scale=0.75, levels=3, winsize=5,
                                            iterations=3, poly_n=10, poly_sigma=1.2,
                                            flags=0)
        
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        h, w = mag.shape
        motion_thresh = np.c_[np.linspace(0.3, 1, h)].repeat(w, axis=-1)
        
        motion_mask = self._get_optical_flow_mask(mag, motion_thresh)
        detections = get_contour_detections(motion_mask, thresh=400)
        
        if len(detections) == 0:
            return []
        
        bboxes = detections[:, :4]
        scores = detections[:, -1]
        return non_max_suppression(bboxes, scores, threshold=0.1)
    
    def _get_motion_mask(self, fg_mask):
        """Get motion mask from foreground mask"""
        _, thresh = cv2.threshold(fg_mask, 0, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.medianBlur(thresh, 3)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        return motion_mask
    
    def _get_frame_diff_mask(self, frame1, frame2):
        """Get motion mask from frame differencing"""
        frame_diff = cv2.subtract(frame2, frame1)
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
    
    def process_stream(self):
        """Main processing loop with OpenCV display"""
        self.running = True
        self.start_time = time.time()
        
        print(f"Starting motion detection with method: {self.method}")
        print("Press 'q' to quit or Ctrl+C to stop...")
        
        # Create window
        window_name = f"Motion Detection - {self.method.replace('_', ' ').title()}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame, reconnecting...")
                    self.cap.release()
                    time.sleep(1)
                    self.connect_input()
                    continue
                
                self.frame_count += 1
                
                # Detect motion
                detections = self.detect_motion(frame)
                
                # Draw bounding boxes
                if len(detections) > 0:
                    self.detection_count += len(detections)
                    draw_bboxes(frame, detections)
                
                # Add overlay info
                self._add_overlay(frame, len(detections))
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit key pressed")
                    break
                
                # Print statistics every 100 frames
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    print(f"Processed {self.frame_count} frames, "
                          f"{self.detection_count} total detections, "
                          f"FPS: {fps:.1f}")
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cv2.destroyAllWindows()
            self.stop()
    
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
    
    def stop(self):
        """Stop processing and cleanup"""
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"\nProcessing complete:")
            print(f"- Total frames processed: {self.frame_count}")
            print(f"- Total detections: {self.detection_count}")
            print(f"- Average FPS: {fps:.1f}")
            print(f"- Processing time: {elapsed:.1f} seconds")


def main():
    parser = argparse.ArgumentParser(description='RTSP Motion Detection with OpenCV Display')
    parser.add_argument('--input', '-i', required=True,
                        help='Input RTSP URL (e.g., rtsp://localhost:8554/input)')
    parser.add_argument('--method', '-m', 
                        choices=['background_subtraction', 'frame_differencing', 'optical_flow'],
                        default='frame_differencing',
                        help='Motion detection method (default: frame_differencing)')
    
    args = parser.parse_args()
    
    # Create and run detector
    detector = RTSPMotionDetector(
        input_rtsp_url=args.input,
        method=args.method
    )
    
    try:
        detector.connect_input()
        detector.setup_display()
        detector.process_stream()
    except Exception as e:
        print(f"Error: {e}")
        detector.stop()


if __name__ == "__main__":
    main()
