"""
Motion Detection from MP4 Video Files
Adapted from the existing notebook implementations to work with video files
"""

import cv2
import numpy as np
from motion_detection_utils import *
from motion_comp_utils import *
try:
    from constrained_ransac import *
    from scipy.stats import kurtosis
    from sklearn.cluster import DBSCAN
    ADVANCED_METHODS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced methods not available due to missing dependencies: {e}")
    print("Install scipy and sklearn for flow_based and unsupervised methods")
    ADVANCED_METHODS_AVAILABLE = False

def detect_motion_from_mp4(video_path, method='background_subtraction', output_path=None):
    """
    Detect moving objects from an MP4 video file
    
    Parameters:
    - video_path: path to input MP4 file
    - method: 'background_subtraction', 'frame_differencing', 'optical_flow', 
              'flow_based', or 'unsupervised'
    - output_path: path to save output video (optional)
    
    Returns:
    - List of detection results per frame
    """
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize video writer if output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize method-specific components
    if method == 'background_subtraction':
        # Initialize background subtractor
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True)
        backSub.setShadowThreshold(0.5)
        kernel = np.array((9,9), dtype=np.uint8)
        
    elif method == 'frame_differencing':
        prev_frame = None
        kernel = np.array((9,9), dtype=np.uint8)
        
    elif method == 'optical_flow':
        prev_frame = None
        
    elif method == 'flow_based':
        if not ADVANCED_METHODS_AVAILABLE:
            raise ValueError("flow_based method requires scipy and constrained_ransac module")
        prev_frame = None
        # Flow-based method uses CRA for background flow estimation
        
    elif method == 'unsupervised':
        if not ADVANCED_METHODS_AVAILABLE:
            raise ValueError("unsupervised method requires scipy and sklearn")
        prev_frame = None
        cluster_model = DBSCAN(eps=30.0, min_samples=3)
        
    # Store all detections
    all_detections = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}", end='\r')
        
        detections = []
        
        if method == 'background_subtraction':
            detections = get_detections_background_subtraction(backSub, frame, kernel)
            
        elif method == 'frame_differencing':
            if prev_frame is not None:
                detections = get_detections_frame_differencing(prev_frame, frame, kernel)
            prev_frame = frame.copy()
            
        elif method == 'optical_flow':
            if prev_frame is not None:
                detections = get_detections_optical_flow(prev_frame, frame)
            prev_frame = frame.copy()
            
        elif method == 'flow_based':
            if prev_frame is not None:
                detections = get_detections_flow_based(prev_frame, frame)
            prev_frame = frame.copy()
            
        elif method == 'unsupervised':
            if prev_frame is not None:
                detections = get_detections_unsupervised(prev_frame, frame, cluster_model)
            prev_frame = frame.copy()
        
        # Draw bounding boxes on frame
        if len(detections) > 0:
            draw_bboxes(frame, detections)
        
        # Save to output video
        if output_path and out:
            out.write(frame)
            
        # Store detections for this frame
        all_detections.append(detections)
    
    # Clean up
    cap.release()
    if output_path and out:
        out.release()
        print(f"\nOutput video saved to: {output_path}")
    
    print(f"\nProcessed {frame_count} frames")
    return all_detections


def get_detections_background_subtraction(backSub, frame, kernel):
    """Background subtraction detection for a single frame"""
    # Update Background Model and get foreground mask
    fg_mask = backSub.apply(frame)
    
    # Get clean motion mask
    motion_mask = get_motion_mask(fg_mask, kernel=kernel)
    
    # Get initially proposed detections from contours
    detections = get_contour_detections(motion_mask, thresh=100)
    
    if len(detections) == 0:
        return []
    
    # Separate bboxes and scores
    bboxes = detections[:, :4]
    scores = detections[:, -1]
    
    # Perform Non-Maximal Suppression on initial detections
    return non_max_suppression(bboxes, scores, threshold=0.1)


def get_detections_frame_differencing(frame1, frame2, kernel):
    """Frame differencing detection between two frames"""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Get motion mask
    mask = get_mask_frame_diff(gray1, gray2, kernel)
    
    # Get detections
    detections = get_contour_detections(mask, thresh=400)
    
    if len(detections) == 0:
        return []
    
    # Separate bboxes and scores
    bboxes = detections[:, :4]
    scores = detections[:, -1]
    
    # Perform Non-Maximal Suppression
    return non_max_suppression(bboxes, scores, threshold=0.1)


def get_detections_optical_flow(frame1, frame2):
    """Optical flow detection between two frames"""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Blur images
    gray1 = cv2.GaussianBlur(gray1, (3,3), 5)
    gray2 = cv2.GaussianBlur(gray2, (3,3), 5)
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                        pyr_scale=0.75,
                                        levels=3,
                                        winsize=5,
                                        iterations=3,
                                        poly_n=10,
                                        poly_sigma=1.2,
                                        flags=0)
    
    # Get magnitude
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create motion threshold (you may need to adjust this based on your video)
    h, w = mag.shape
    motion_thresh = np.c_[np.linspace(0.3, 1, h)].repeat(w, axis=-1)
    
    # Get motion mask
    motion_mask = get_motion_mask_optical_flow(mag, motion_thresh)
    
    # Get detections
    detections = get_contour_detections(motion_mask, thresh=400)
    
    if len(detections) == 0:
        return []
    
    # Separate bboxes and scores
    bboxes = detections[:, :4]
    scores = detections[:, -1]
    
    # Perform Non-Maximal Suppression
    return non_max_suppression(bboxes, scores, threshold=0.1)


def get_mask_frame_diff(frame1, frame2, kernel):
    """Get motion mask from frame differencing"""
    frame_diff = cv2.subtract(frame2, frame1)
    
    # Blur the frame difference
    frame_diff = cv2.medianBlur(frame_diff, 3)
    
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 3)
    
    mask = cv2.medianBlur(mask, 3)
    
    # Morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return mask


def get_motion_mask(fg_mask, min_thresh=0, kernel=np.array((9,9), dtype=np.uint8)):
    """Get motion mask from foreground mask"""
    _, thresh = cv2.threshold(fg_mask, min_thresh, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.medianBlur(thresh, 3)
    
    # Morphological operations
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return motion_mask


def get_motion_mask_optical_flow(flow_mag, motion_thresh, kernel=np.ones((7,7))):
    """Get motion mask from optical flow magnitude"""
    motion_mask = np.uint8(flow_mag > motion_thresh) * 255
    
    motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return motion_mask


def get_detections_flow_based(frame1, frame2):
    """Flow-based detection using constrained RANSAC for moving camera scenarios"""
    if not ADVANCED_METHODS_AVAILABLE:
        return []
    
    # Resize frames for better performance
    h, w = frame1.shape[:2]
    h2, w2 = h//2, w//2
    
    frame1_resized = cv2.resize(frame1, (w2, h2))
    frame2_resized = cv2.resize(frame2, (w2, h2))
    
    # Convert to RGB for flow computation
    frame1_rgb = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB)
    
    # Compute dense optical flow using Farneback
    gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
    
    gray1 = cv2.GaussianBlur(gray1, (3,3), 5)
    gray2 = cv2.GaussianBlur(gray2, (3,3), 5)
    
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                        pyr_scale=0.75, levels=3, winsize=5,
                                        iterations=3, poly_n=10, poly_sigma=1.2,
                                        flags=0)
    
    # Convert to proper format for CRA
    flow_np = flow.astype(np.float32)
    
    # Use RANSAC to obtain H matrix
    h_flow, w_flow = flow_np.shape[:2]
    
    # Get points P and polynomial expansion X
    P, X = get_px(w_flow, h_flow)
    
    # Get sample index
    index, n_ttl, n_s = get_sampling_index(w_flow, h_flow, s=50, p=0.5)
    
    # Obtain H matrix
    H, _ = cra(flow_np, P, X, index, n_ttl, n_s, thresh=0.01, min_inliers=10000, num_iters=50)
    
    # Use H matrix to get estimated background and foreground
    Fb = (X @ H) - P
    background_flow = Fb.reshape(flow_np.shape)
    
    foreground_flow = flow_np - background_flow
    mag_f, _ = cv2.cartToPolar(foreground_flow[:, :, 0], foreground_flow[:, :, 1])
    
    # Threshold foreground flow to get motion mask
    c = 0.5  # sensitivity parameter
    motion_mask = np.uint8(mag_f > (mag_f.mean() + c*mag_f.std(ddof=1))) * 255
    
    # Scale motion mask back to original size
    motion_mask = cv2.resize(motion_mask, (w, h))
    
    # Get detections
    detections = get_contour_detections(motion_mask, thresh=25)
    
    if len(detections) == 0:
        return []
    
    # Scale bounding boxes back to original size
    detections[:, :4] *= 2  # Scale coordinates back
    
    # Separate bboxes and scores
    bboxes = detections[:, :4]
    scores = detections[:, -1]
    
    # Perform Non-Maximal Suppression
    return non_max_suppression(bboxes, scores, threshold=0.1)


def get_detections_unsupervised(frame1, frame2, cluster_model):
    """Unsupervised motion detection using sparse optical flow and clustering"""
    if not ADVANCED_METHODS_AVAILABLE:
        return []
    
    # Get motion compensation and clusters
    clusters = get_motion_detections(frame1, frame2, cluster_model, 
                                   c=1, angle_thresh=0.1, max_cluster_size=50,
                                   distance_metric='l2', transform_type='affine')
    
    if len(clusters) == 0:
        return []
    
    # Convert point clusters to bounding boxes
    detections = []
    for cluster in clusters:
        if len(cluster) > 0:
            x_min, y_min = cluster.min(axis=0)
            x_max, y_max = cluster.max(axis=0)
            area = (x_max - x_min) * (y_max - y_min)
            detections.append([x_min, y_min, x_max, y_max, area])
    
    if len(detections) == 0:
        return []
    
    detections = np.array(detections)
    bboxes = detections[:, :4]
    scores = detections[:, -1]
    
    # Perform Non-Maximal Suppression
    return non_max_suppression(bboxes, scores, threshold=0.1)


def get_motion_detections(frame1, frame2, cluster_model, c=2, angle_thresh=0.1, 
                         edge_thresh=50, max_cluster_size=80, distance_metric='l2', 
                         transform_type='affine'):
    """Motion detection using sparse optical flow and clustering"""
    if not ADVANCED_METHODS_AVAILABLE:
        return []
    
    transform_type = transform_type.lower()
    assert(transform_type in ['affine', 'homography'])

    # get frame info
    h, w, _ = frame1.shape

    # get affine transformation matrix for motion compensation between frames
    A, prev_points, curr_points = motion_comp(frame1, frame2, num_points=10000, 
                                            points_to_use=5000, transform_type=transform_type)

    if A is None or prev_points is None or curr_points is None:
        return []

    # get transformed key points
    if transform_type == 'affine':
        A = np.vstack((A, np.zeros((3,)))) # get 3x3 matrix to xform points

    compensated_points = np.hstack((prev_points, np.ones((len(prev_points), 1)))) @ A.T
    compensated_points = compensated_points[:, :2]

    # get a distance metric for the current and previous keypoints
    if distance_metric == 'l1':
        x = np.sum(np.abs(curr_points - compensated_points), axis=1) # l1 norm
    else:
        x = np.linalg.norm(curr_points - compensated_points, ord=2, axis=1) # l2 norm
    
    # compute kurtosis of x to determine outlier hyperparameter c
    if kurtosis(x, bias=False) < 1:
        c /= 2 # reduce outlier hyperparameter

    # get outlier bound (only care about upper bound since lower values are not likely movers)
    upper_bound = np.mean(x) + c*np.std(x, ddof=1)

    # get motion points
    motion_idx = (x >= upper_bound)
    motion_points = curr_points[motion_idx]

    if len(motion_points) == 0:
        return []

    # add additional motion data for clustering
    motion = compensated_points[motion_idx] - motion_points
    magnitude = np.linalg.norm(motion, ord=2, axis=1)
    angle = np.arctan2(motion[:, 0], motion[:, 1]) # horizontal/vertical

    motion_data = np.hstack((motion_points, np.c_[magnitude], np.c_[angle]))

    # cluster motion data
    cluster_model.fit(motion_data)

    # filter clusters with large variation in angular motion
    clusters = []
    far_edge_array = np.array([w - edge_thresh, h - edge_thresh])
    for lbl in np.unique(cluster_model.labels_):
        
        cluster_idx = cluster_model.labels_ == lbl
        
        # get standard deviation of the angle of apparent motion 
        angle_std = angle[cluster_idx].std(ddof=1)
        if angle_std <= angle_thresh:
            cluster = motion_points[cluster_idx]

            # remove clusters that are too close to the edges and ones that are too large
            centroid = cluster.mean(axis=0)
            if (len(cluster) < max_cluster_size) \
                and not (np.any(centroid < edge_thresh) or np.any(centroid > far_edge_array)):
                clusters.append(cluster)

    return clusters


# Example usage
if __name__ == "__main__":
    # Specific video file path
    video_path = "/Users/havishbychapur/Movies/BAPSPerimeterParkedVehicles.mp4"
    
    # All available methods
    methods = ['unsupervised']
    
    print(f"Starting motion detection on: {video_path}")
    print(f"Running all {len(methods)} algorithms for comparison...")
    print("="*60)
    
    # Run each method
    for i, method in enumerate(methods, 1):
        output_path = f"/Users/havishbychapur/Movies/BAPSPerimeterParkedVehicles_{method}.mp4"
        
        print(f"\n[{i}/3] Processing with {method.upper().replace('_', ' ')}...")
        print(f"Output will be saved to: {output_path}")
        
        # Run detection
        detections = detect_motion_from_mp4(video_path, method=method, output_path=output_path)
        
        if detections:
            frames_with_objects = sum(len(d) > 0 for d in detections)
            total_detections = sum(len(d) for d in detections)
            print(f"✓ {method.upper().replace('_', ' ')} completed!")
            print(f"  - Found objects in {frames_with_objects}/{len(detections)} frames ({frames_with_objects/len(detections)*100:.1f}%)")
            print(f"  - Total detections: {total_detections}")
        else:
            print(f"✗ {method.upper().replace('_', ' ')} failed or no objects found")
    
    print("\n" + "="*60)
    print("All algorithms completed! Compare the results:")
    for method in methods:
        print(f"- {method.upper().replace('_', ' ')}: chandeRoadTwoWheelersFourWheelers_{method}.mp4")
