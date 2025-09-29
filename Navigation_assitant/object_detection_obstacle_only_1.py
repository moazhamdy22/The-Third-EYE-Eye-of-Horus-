# --- START OF FILE object_detection_obstacleonly_enhanced.py ---

import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
import argparse
from typing import List, Tuple, Dict, Any, Set
from collections import Counter # For counting obstacle types

# --- Logging Setup ---
def setup_logging():
    """Configure logging settings for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

# --- Configuration ---
class Config:
    """Configuration class for detection parameters."""
    # --- Model & Detection Parameters ---
    MODEL_PATH: str = "yolov8n.pt"
    DEVICE: str = "cpu"
    CONFIDENCE_THRESHOLD: float = 0.4
    IOU_THRESHOLD: float = 0.5

    # --- Organized Obstacle Class Indices (COCO 80 class IDs) ---
    OBSTACLE_CATEGORIES: Dict[str, List[int]] = {
        "Indoor Static": [
            56, # chair
            57, # couch
            58, # potted plant
            59,
            60, # dining table
            61,
            72, # refrigerator (can be large/immovable)
            73,
            # Add others like 'bed' (59), 'toilet' (61), 'sink' (73) if needed
        ],
        "Indoor Dynamic": [
            0,  # person
            15, # cat
            16, # dog
            14, # bird
            # Add 'bird' (14) if relevant indoors?
            # Consider robots if applicable model exists
        ],
        "Outdoor Static": [
            9,  # traffic light
            10, # fire hydrant
            11, # stop sign
            12, # parking meter
            13, # bench
            # Add 'pole' (often detected as part of signs/lights), 'bollard' if model supports
        ],
        "Outdoor Dynamic": [
            1,  # bicycle
            2,  # car
            3,  # motorcycle
            5,  # bus
            7,  # truck
            16, # dog (also appears outdoors)
            15, # cat (also appears outdoors)
            24, # backpack (often dropped/left)
            28, # suitcase (often dropped/left)
            36, # skateboard
        ]
    }

    # Combined list generated from categories for actual detection
    COMBINED_OBSTACLE_INDICES: List[int] = sorted(list(set(
        idx for category_indices in OBSTACLE_CATEGORIES.values() for idx in category_indices
    )))

    # --- Standalone Test Parameters ---
    TEST_VIDEO_SOURCE: Any = 0
    TEST_WINDOW_NAME: str = "Obstacle Detection Test"
    TEST_TARGET_FPS: int = 30
    TEST_AUDIO_COOLDOWN_SECONDS: float = 4.0 # Cooldown for announcing same obstacle type

# --- Type Hinting ---
DetectionResult = Tuple[int, int, int, float, str]

# --- Obstacle Detection Class ---
class ObstacleDetector:
    """
    Handles obstacle detection using a YOLOv8 model, focusing only on
    predefined obstacle classes.
    """
    def __init__(self,
                 model_path: str = Config.MODEL_PATH,
                 device: str = Config.DEVICE,
                 confidence_threshold: float = Config.CONFIDENCE_THRESHOLD,
                 iou_threshold: float = Config.IOU_THRESHOLD,
                 obstacle_class_indices: List[int] = Config.COMBINED_OBSTACLE_INDICES):
        """ Initializes the ObstacleDetector. """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.obstacle_class_indices = obstacle_class_indices
        self.model_names: Dict[int, str] = {}

        try:
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info(f"Moving model to device: {self.device}")
            self.model.to(self.device)
            self.model_names = self.model.names
            logger.info(f"YOLO model loaded successfully. Class names available: {len(self.model_names)}")

            # Log the targeted classes based on the combined list
            target_class_names = [self.model_names.get(idx, f"Unknown Index {idx}") for idx in self.obstacle_class_indices]
            logger.info(f"Targeting {len(self.obstacle_class_indices)} obstacle classes (combined): {', '.join(sorted(target_class_names))}")
            # Log the categories for clarity
            logger.info("Obstacle Categories Defined:")
            for category, indices in Config.OBSTACLE_CATEGORIES.items():
                cat_names = [self.model_names.get(idx, f"Idx {idx}") for idx in indices]
                logger.info(f"  - {category}: {', '.join(sorted(cat_names))}")

        except Exception as e:
            logger.exception(f"Failed to load YOLO model from '{self.model_path}' on device '{self.device}'.")
            raise RuntimeError("Model loading failed") from e

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """ Performs obstacle detection on a single frame. """
        detections: List[DetectionResult] = []
        if frame is None or frame.size == 0:
            logger.warning("Received an empty frame for detection.")
            return detections

        try:
            results = self.model(
                frame,
                classes=self.obstacle_class_indices,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

            if results and results[0].boxes:
                for box in results[0].boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = round(box.conf[0].item(), 2)
                        class_idx = int(box.cls[0].item())
                        class_name = self.model_names.get(class_idx, f"Unknown_{class_idx}")
                        detections.append((x1, y1, x2, y2, confidence, class_name))
                    except Exception as e:
                        logger.warning(f"Error processing a single detection box: {e}", exc_info=False)
                        continue
        except Exception as e:
            logger.exception(f"Error during model inference: {e}")
        return detections

    def get_class_name(self, class_index: int) -> str:
        """Safely gets the class name for a given index."""
        return self.model_names.get(class_index, f"Unknown_{class_index}")

# --- Standalone Test Helper Functions ---

def draw_visuals(frame: np.ndarray, detections: List[DetectionResult], fps: float, counts: Counter) -> np.ndarray:
    """Draws detections, FPS, and obstacle counts on the frame."""
    if frame is None: return np.zeros((100, 100, 3), dtype=np.uint8)
    processed_frame = frame.copy()
    obstacle_color = (0, 0, 255) # Red
    text_color = (255, 255, 255) # White
    info_color = (0, 255, 0)     # Green

    # --- Draw Detections ---
    for detection in detections:
        x1, y1, x2, y2, confidence, class_name = detection
        label = f"{class_name}: {confidence:.2f}"
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), obstacle_color, 2)
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 5
        cv2.rectangle(processed_frame, (x1, label_y - label_height - baseline), (x1 + label_width, label_y + baseline), obstacle_color, cv2.FILLED)
        cv2.putText(processed_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # --- Draw FPS ---
    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, info_color, 2, cv2.LINE_AA)

    # --- Draw Counts ---
    total_obstacles = sum(counts.values())
    cv2.putText(processed_frame, f"Total Obstacles: {total_obstacles}", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, obstacle_color, 2, cv2.LINE_AA)

    # Optional: Display top N counts on screen
    y_offset = 95
    for item, count in counts.most_common(3): # Display top 3
         count_text = f"- {item}: {count}"
         cv2.putText(processed_frame, count_text, (15, y_offset),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
         y_offset += 25

    return processed_frame

def print_detections_to_console(detections: List[DetectionResult], counts: Counter):
    """Prints detection details and counts to the console for the current frame."""
    print("-" * 40) # Separator
    total_obstacles = sum(counts.values())
    print(f"Frame Summary: Total Obstacles = {total_obstacles}")

    if not detections:
        print("  No obstacles detected in this frame.")
        return

    print("  Counts per Type:")
    for item, count in counts.items():
        print(f"    - {item}: {count}")

    print("  Individual Detections:")
    for i, (x1, y1, x2, y2, confidence, class_name) in enumerate(detections):
        print(f"    {i+1}. Class: {class_name}, Conf: {confidence:.2f}, Box: [{x1},{y1},{x2},{y2}]")
    print("-" * 40 + "\n") # End separator and newline

# --- Main Execution Block (Standalone Test) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standalone Obstacle Detection Test with Counts')
    parser.add_argument('--device', type=str, default=Config.DEVICE, help=f'Device (cpu, cuda, mps). Default: {Config.DEVICE}')
    parser.add_argument('--model', type=str, default=Config.MODEL_PATH, help=f'YOLOv8 model path. Default: {Config.MODEL_PATH}')
    parser.add_argument('--conf', type=float, default=Config.CONFIDENCE_THRESHOLD, help=f'Confidence threshold. Default: {Config.CONFIDENCE_THRESHOLD}')
    parser.add_argument('--source', type=str, default=str(Config.TEST_VIDEO_SOURCE), help=f'Video source (0/webcam or path). Default: {Config.TEST_VIDEO_SOURCE}')
    parser.add_argument('--fps', type=int, default=Config.TEST_TARGET_FPS, help=f'Target FPS. Default: {Config.TEST_TARGET_FPS}')
    args = parser.parse_args()

    logger.info("--- Starting Obstacle Detector Standalone Test ---")
    logger.info(f"Test Config: Device={args.device}, Model={args.model}, Conf={args.conf}, Source={args.source}, Target FPS={args.fps}")

    # --- Initialize Components ---
    detector: ObstacleDetector = None
    cap: cv2.VideoCapture = None

    try:
        # Initialize Detector
        detector = ObstacleDetector(
            model_path=args.model,
            device=args.device,
            confidence_threshold=args.conf
        )

        # Setup Video Capture
        try: video_source = int(args.source)
        except ValueError: video_source = args.source
        logger.info(f"Opening video source: {video_source}")
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened(): raise RuntimeError(f"Could not open video source: {video_source}")
        logger.info(f"Video source opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.2f} FPS")

        # --- Timing, State Variables ---
        target_frame_time = 1.0 / args.fps if args.fps > 0 else 0
        fps_calc_start_time = time.monotonic()
        frame_count = 0
        display_fps_value = 0.0

        logger.info(f"Processing started. Press 'q' in the '{Config.TEST_WINDOW_NAME}' window to quit.")

        # --- Main Loop ---
        while True:
            start_time = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                logger.info("End of video source or cannot read frame.")
                break

            # --- Detection ---
            detections = detector.detect(frame)

            # --- Counting ---
            obstacle_counts = Counter(det[5] for det in detections)

            # --- Console Output ---
            print_detections_to_console(detections, obstacle_counts)

            # --- Visualization ---
            processed_frame = draw_visuals(frame, detections, display_fps_value, obstacle_counts)

            # --- Display Frame ---
            cv2.imshow(Config.TEST_WINDOW_NAME, processed_frame)

            # --- Frame Rate Calculation & Control ---
            end_time = time.monotonic()
            processing_time = end_time - start_time
            wait_duration = max(0.001, target_frame_time - processing_time)

            frame_count += 1
            if end_time - fps_calc_start_time >= 1.0:
                display_fps_value = frame_count / (end_time - fps_calc_start_time)
                frame_count = 0
                fps_calc_start_time = end_time

            # --- Quit Condition ---
            if cv2.waitKey(int(wait_duration * 1000)) & 0xFF == ord('q'):
                logger.info("Quit key pressed.")
                break

    except Exception as e:
        logger.exception(f"An critical error occurred during the test run: {e}")
    finally:
        # --- Cleanup ---
        logger.info("--- Shutting down Obstacle Detector Standalone Test ---")
        if cap is not None and cap.isOpened():
            cap.release()
            logger.info("Video capture released.")
        cv2.destroyAllWindows()
        logger.info("OpenCV windows closed.")
        logger.info("Shutdown complete.")

# --- END OF FILE object_detection_obstacleonly_enhanced.py ---