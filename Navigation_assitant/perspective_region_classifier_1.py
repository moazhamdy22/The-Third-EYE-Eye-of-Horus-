# --- START OF FILE perspective_region_classifier.py ---

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

@dataclass
class RegionClassificationResult:
    """Result of perspective-aware region classification."""
    is_path_obstacle: bool              # True if the object is considered within the walking path
    relative_direction: str             # "Left", "Center", "Right", "Off-Path-Left", "Off-Path-Right"
    relative_distance_zone: str         # "Close", "Mid", "Far", or "Unknown"
    location_description: str           # Combined description (e.g., "Center-Close", "Off-Path-Left")
    reference_point: Tuple[int, int]    # The (x, y) point used for classification (e.g., bottom-center)

class PerspectiveRegionClassifier:
    """
    Classifies detected objects based on their position relative to a
    perspective-aware walking path defined on the image plane.
    """
    def __init__(self,
                 frame_width: int,
                 frame_height: int,
                 horizon_y_factor: float = 0.5,
                 bottom_y_factor: float = 1.0,
                 horizon_x_offset_factor: float = 0.15,
                 bottom_x_offset_factor: float = 0.45,
                 close_zone_y_factor: float = 0.80,
                 mid_zone_y_factor: float = 0.65):
        """ Initializes the classifier with frame dimensions and path/zone parameters. """
        # --- Input Validation ---
        if not (0 <= horizon_y_factor < bottom_y_factor <= 1.0):
            raise ValueError("Y factors must satisfy 0 <= horizon < bottom <= 1.0")
        if not (0 <= horizon_x_offset_factor <= 0.5 and 0 <= bottom_x_offset_factor <= 0.5):
             raise ValueError("X offset factors must be between 0 and 0.5")

        self.frame_width = frame_width
        self.frame_height = frame_height

        # Calculate absolute pixel values
        self.horizon_y = int(horizon_y_factor * frame_height)
        self.bottom_y = int(bottom_y_factor * frame_height)
        self.bottom_y = max(self.bottom_y, self.horizon_y + 1) # Avoid division by zero

        self.horizon_x_offset = int(horizon_x_offset_factor * frame_width)
        self.bottom_x_offset = int(bottom_x_offset_factor * frame_width)

        self.close_zone_y = int(close_zone_y_factor * frame_height)
        self.mid_zone_y = int(mid_zone_y_factor * frame_height)

        # Sanity check and correct zone thresholds if necessary
        if not (self.horizon_y <= self.mid_zone_y < self.close_zone_y <= self.bottom_y):
            print(f"Warning: Review distance zone Y factors. Initial: H={self.horizon_y}, M={self.mid_zone_y}, C={self.close_zone_y}, B={self.bottom_y}")
            self.mid_zone_y = max(self.horizon_y + 1, self.mid_zone_y) # Ensure Mid is below Horizon
            self.close_zone_y = max(self.mid_zone_y + 1, self.close_zone_y) # Ensure Close is below Mid
            self.close_zone_y = min(self.close_zone_y, self.bottom_y) # Ensure Close doesn't exceed Bottom
            print(f"Corrected thresholds: Mid={self.mid_zone_y}, Close={self.close_zone_y}")

        # Define path corners
        self.path_corners = {
            'top_left': (self.frame_width // 2 - self.horizon_x_offset, self.horizon_y),
            'top_right': (self.frame_width // 2 + self.horizon_x_offset, self.horizon_y),
            'bottom_left': (self.frame_width // 2 - self.bottom_x_offset, self.bottom_y),
            'bottom_right': (self.frame_width // 2 + self.bottom_x_offset, self.bottom_y)
        }

        # Pre-calculate line equations (x = my + c)
        y1_b, y2_t = self.path_corners['bottom_left'][1], self.path_corners['top_left'][1]
        delta_y = y2_t - y1_b
        if delta_y == 0:
            self.m_left = self.m_right = float('inf')
            self.c_left = self.path_corners['bottom_left'][0]
            self.c_right = self.path_corners['bottom_right'][0]
        else:
            x1_bl, x2_tl = self.path_corners['bottom_left'][0], self.path_corners['top_left'][0]
            self.m_left = (x2_tl - x1_bl) / delta_y
            self.c_left = x1_bl - self.m_left * y1_b
            x1_br, x2_tr = self.path_corners['bottom_right'][0], self.path_corners['top_right'][0]
            self.m_right = (x2_tr - x1_br) / delta_y
            self.c_right = x1_br - self.m_right * y1_b

    def _get_path_boundaries_at_y(self, y: int) -> Tuple[int, int]:
        """Calculates the left and right X boundaries of the path at a given Y coord."""
        if not (self.horizon_y <= y <= self.bottom_y):
            center_x = self.frame_width // 2
            return center_x, center_x
        if self.m_left == float('inf'):
             x_left, x_right = self.c_left, self.c_right
        else:
             x_left = int(self.m_left * y + self.c_left)
             x_right = int(self.m_right * y + self.c_right)
        return x_left, x_right

    def classify(self, bbox: Tuple[float, float, float, float]) -> RegionClassificationResult:
        """ Classifies a bounding box based on its position relative to the perspective path. """
        x1, y1, x2, y2 = map(int, bbox)
        ref_x = (x1 + x2) // 2
        ref_y = y2 # Use bottom-center point

        if ref_y < self.horizon_y:
            return RegionClassificationResult(False, "Off-Path", "Unknown", "Off-Path (Above Horizon)", (ref_x, ref_y))

        clamped_y = max(self.horizon_y, min(ref_y, self.bottom_y))
        path_x_left, path_x_right = self._get_path_boundaries_at_y(clamped_y)
        is_within_path = (path_x_left <= ref_x <= path_x_right)

        is_obstacle = False
        direction = "Off-Path"
        distance_zone = "Unknown"
        location_desc = "Off-Path"

        if is_within_path:
            is_obstacle = True
            # Determine Distance Zone
            if ref_y >= self.close_zone_y: distance_zone = "Close"
            elif ref_y >= self.mid_zone_y: distance_zone = "Mid"
            else: distance_zone = "Far"
            # Determine Lateral Zone
            path_center_x = (path_x_left + path_x_right) // 2
            path_width_at_y = path_x_right - path_x_left
            center_zone_width = max(1, path_width_at_y // 3)
            center_start_x = path_center_x - center_zone_width // 2
            center_end_x = path_center_x + center_zone_width // 2
            if ref_x < center_start_x: direction = "Left"
            elif ref_x > center_end_x: direction = "Right"
            else: direction = "Center"
            location_desc = f"{direction}-{distance_zone}"
        else:
            # Determine rough off-path direction
            if ref_x < path_x_left: direction = "Off-Path-Left"
            else: direction = "Off-Path-Right"
            location_desc = direction

        return RegionClassificationResult(is_obstacle, direction, distance_zone, location_desc, (ref_x, ref_y))

    def draw_perspective_regions(self, frame: np.ndarray) -> np.ndarray:
        """Draws the perspective path and distance zones onto a frame for visualization."""
        overlay = frame.copy()
        path_color, close_color, mid_color, far_color = (0, 255, 255), (0, 0, 255), (0, 165, 255), (0, 255, 0)
        line_thickness = 2

        # Draw Path Lines
        cv2.line(overlay, self.path_corners['bottom_left'], self.path_corners['top_left'], path_color, line_thickness)
        cv2.line(overlay, self.path_corners['bottom_right'], self.path_corners['top_right'], path_color, line_thickness)

        # Draw Distance Zone Lines
        for y_val, color, name in [(self.close_zone_y, close_color, "Close"),
                                   (self.mid_zone_y, mid_color, "Mid"),
                                   (self.horizon_y, far_color, "Far")]:
            x_left, x_right = self._get_path_boundaries_at_y(y_val)
            cv2.line(overlay, (x_left, y_val), (x_right, y_val), color, line_thickness)
            cv2.putText(overlay, name, (x_right + 5, y_val + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        alpha = 0.6 # Transparency
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# --- Standalone Test Block ---
if __name__ == "__main__":
    import time
    import logging
    # Import the correct detector class and its config
    try:
        # *** Use the specific filename you provided ***
        from object_detection_obstacle_only_1 import ObstacleDetector, Config as DetectorConfig
        DETECTOR_AVAILABLE = True
    except ImportError as e:
        print(f"WARNING: Failed to import ObstacleDetector from 'object_detection_obstacleonly_enhanced.py': {e}")
        print("Standalone test will require manual bounding boxes if detector fails.")
        DETECTOR_AVAILABLE = False
        class ObstacleDetector: pass # Dummy class
        class DetectorConfig: MODEL_PATH="?"; DEVICE="?"; CONFIDENCE_THRESHOLD=0.5 # Dummy config


    logging.basicConfig(level=logging.INFO)

    # --- Test Configuration ---
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    VIDEO_SOURCE = 0 # Use 0 for webcam or path to video file

    # --- Initialize Classifier ---
    try:
        classifier = PerspectiveRegionClassifier(
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            # *** Tuned factors based on feedback ***
            horizon_y_factor=0.5,     # Keep horizon the same for now
            bottom_y_factor=1.0,      # Keep bottom the same
            horizon_x_offset_factor=0.08, # Slightly narrower path at horizon
            bottom_x_offset_factor=0.35,  # <<< Reduced: Make path significantly narrower at the bottom
            close_zone_y_factor=0.90,     # <<< Increased: Require object to be lower for 'Close'
            mid_zone_y_factor=0.65      # Keep Mid threshold, ensure close_zone_y > mid_zone_y
        )
        print("PerspectiveRegionClassifier initialized with tuned parameters.")
    except ValueError as e:
         print(f"Error initializing classifier: {e}")
         exit()
         

    # --- Initialize Detector (Using the imported class) ---
    detector: Optional[ObstacleDetector] = None
    if DETECTOR_AVAILABLE:
        try:
            print(f"Attempting to load detector: Model={DetectorConfig.MODEL_PATH}, Device={DetectorConfig.DEVICE}")
            detector = ObstacleDetector(
                model_path=DetectorConfig.MODEL_PATH,
                device=DetectorConfig.DEVICE,
                confidence_threshold=DetectorConfig.CONFIDENCE_THRESHOLD
                # Uses default obstacle classes from its own Config
            )
            print("Obstacle detector loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load detector: {e}. Running test without detector.")
            detector = None # Ensure detector is None if loading failed

    # --- Video Capture ---
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}.")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print(f"Video source {VIDEO_SOURCE} opened.")

    print("Starting Perspective Region Classifier Test...")
    print("Press 'q' to quit.")

    prev_time = time.monotonic() # Use monotonic time for FPS calculation
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream end or cannot read frame. Exiting...")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # --- Get Detections (from the non-tracking detector) ---
        raw_detections = [] # List[Tuple[x1, y1, x2, y2, confidence, class_name]]
        if detector:
            raw_detections = detector.detect(frame) # Use the .detect() method
        else:
             # Manual BBox for testing without detector: (x1, y1, x2, y2, conf, name)
            test_h, test_w = 80, 40
            test_y2 = int(FRAME_HEIGHT * 0.9)
            test_y1 = test_y2 - test_h
            test_x_center = FRAME_WIDTH // 2
            test_x1 = test_x_center - test_w // 2
            test_x2 = test_x_center + test_w // 2
            raw_detections = [(test_x1, test_y1, test_x2, test_y2, 0.99, "TestBox")]

        # --- Draw Perspective Guidelines ---
        vis_frame = classifier.draw_perspective_regions(frame)

        # --- Classify and Draw Results ---
        print(f"--- Frame @ {time.time():.2f} ---")
        # Extract just the bounding boxes for the classifier
        bboxes = [(det[0], det[1], det[2], det[3]) for det in raw_detections] # x1,y1,x2,y2

        for i, bbox in enumerate(bboxes):
            result = classifier.classify(bbox)
            x1, y1, x2, y2 = map(int, bbox)

            # Get corresponding full result data for labeling
            detection_data = raw_detections[i]
            object_label = f"{detection_data[5]}" # Just class name (index 5)

            print(f"  {object_label}: Obstacle={result.is_path_obstacle}, Loc={result.location_description}, RefPt={result.reference_point}")

            # Draw bounding box (color indicates if it's a path obstacle)
            color = (0, 0, 255) if result.is_path_obstacle else (0, 255, 0) # Red if path obstacle, Green if not
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 1)

            # Draw reference point (bottom-center) - helps debugging
            cv2.circle(vis_frame, result.reference_point, 4, color, -1)

            # Draw classification label near the top of the box
            label_text = f"{object_label}: {result.location_description}"
            (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            label_y = y1 - 5 if y1 - 5 > label_height else y1 + 15 # Adjust label position
            # Add a small background rectangle for the label text
            cv2.rectangle(vis_frame, (x1, label_y - label_height - baseline + 1), (x1 + label_width, label_y + baseline), (0,0,0), cv2.FILLED)
            cv2.putText(vis_frame, label_text, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)


        # Calculate and display FPS
        curr_time = time.monotonic()
        elapsed = curr_time - prev_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        prev_time = curr_time
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Perspective Region Classification Test', vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit key pressed.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test finished.")

# --- END OF FILE perspective_region_classifier.py ---