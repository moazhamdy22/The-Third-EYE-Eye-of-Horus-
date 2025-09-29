# --- START OF FILE object_detection_capture.py ---

import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import logging
from collections import defaultdict
import os
import sys
import torch

# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from audio_feedback_vision_assitant import AudioFeedbackHandler # Use the dedicated handler

# Configure logging with timestamp format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Silence Ultralytics' own verbose logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class SpatialObjectAnalyzer:
    def __init__(self, model_path="yolov8n.pt", device="auto", confidence_threshold=0.5):
        """
        Initializes the SpatialObjectAnalyzer class for single-shot analysis.
        """
        # Auto-detect device if not specified
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("GPU detected and available - Using CUDA acceleration")
            else:
                self.device = "cpu"
                logger.info("GPU not available - Using CPU processing")
        else:
            self.device = device
            logger.info(f"Device manually set to: {self.device}")

        try:
            logger.info(f"Loading YOLO model from: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.labels = self.model.names
            logger.info(f"YOLO model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            raise

        self.confidence_threshold = confidence_threshold
        logger.info(f"Confidence threshold set to: {confidence_threshold}")

        # Frame dimensions (will be set when processing the selected frame)
        self.frame_width = 0
        self.frame_height = 0

        # --- Spatial Calculation Parameters (same as before) ---
        self.near_threshold = 0.40
        self.mid_threshold = 0.15
        self.h_center_threshold = 0.2
        self.v_center_threshold = 0.2
        self.v_upper_threshold = 0.4
        self.v_lower_threshold = 0.6
        self.side_threshold = 0.6

    def _calculate_spatial_properties(self, x1, y1, x2, y2, class_idx):
        """Calculates spatial properties for a single detection."""
        if self.frame_width == 0 or self.frame_height == 0:
             logger.error("Frame dimensions not set before calculating spatial properties.")
             return None # Cannot calculate without dimensions

        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        class_name = self.labels.get(class_idx, "Unknown")

        # Distance
        relative_height = height / self.frame_height
        if relative_height >= self.near_threshold: distance = "near"
        elif relative_height >= self.mid_threshold: distance = "mid-distance"
        else: distance = "far"

        # Position
        frame_center_x = self.frame_width / 2
        frame_center_y = self.frame_height / 2
        norm_pos_x = (center_x - frame_center_x) / frame_center_x
        norm_pos_y = (center_y - frame_center_y) / frame_center_y

        if abs(norm_pos_x) <= self.h_center_threshold: h_position = "front"
        elif norm_pos_x < 0: h_position = "left" if norm_pos_x > -self.side_threshold else "far left"
        else: h_position = "right" if norm_pos_x < self.side_threshold else "far right"

        if center_y < self.frame_height * self.v_upper_threshold: v_position = "high"
        elif center_y > self.frame_height * self.v_lower_threshold: v_position = "low"
        else: v_position = "center-level"

        # Position Description for feedback
        if h_position == "front" and v_position == "center-level": position_desc = "directly in front"
        elif h_position == "front": position_desc = f"in front and {v_position}"
        elif v_position == "center-level": position_desc = f"to your {h_position}"
        else: position_desc = f"to your {h_position} and {v_position}"

        # Guidance String - FIXED DIRECTION LOGIC
        is_reachable = (distance == "near")
        if distance == "near" and h_position == "front":
            guidance = "reachable directly in front of you"
        elif distance == "near":
            # CORRECTED: If object is to your left, move left to reach it
            move_dir = "left" if "left" in h_position else "right"
            guidance = f"reachable, move slightly {move_dir} to reach it"
        elif distance == "mid-distance":
            movement = ""
            if h_position != "front":
                # CORRECTED: Move toward the direction where object is located
                move_dir = "left" if "left" in h_position else "right"
                movement = f"move {move_dir} and "
            guidance = f"at medium distance, {movement}step forward to reach it"
        else:  # far
            movement = ""
            if h_position != "front":
                # CORRECTED: Move toward the direction where object is located
                move_dir = "left" if "left" in h_position else "right"
                movement = f"move {move_dir} and "
            guidance = f"far away, {movement}walk forward to reach it"

        return {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "class_name": class_name,
            "confidence": 0.0,
            "distance": distance,
            "h_position": h_position,
            "v_position": v_position,
            "position_desc": position_desc,
            "is_reachable": is_reachable,
            "guidance": guidance,
        }

    def detect_objects_in_frame(self, frame):
        """
        Detects objects in a single provided frame and calculates spatial properties.

        Args:
            frame (numpy.ndarray): The single frame to analyze.

        Returns:
            list: A list of dictionaries, each containing spatial information
                  about a detected object above the confidence threshold.
        """
        if frame is None:
            logger.warning("Received None frame for detection.")
            return []

        # Set frame dimensions for this analysis run
        self.frame_height, self.frame_width = frame.shape[:2]
        if self.frame_width == 0 or self.frame_height == 0:
             logger.error("Invalid frame dimensions for detection.")
             return []

        try:
            results = self.model(frame, verbose=False)
        except Exception as e:
            logger.error(f"Error during YOLO model inference: {e}", exc_info=True)
            return []

        spatial_detections = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                confidence = box.conf[0].item()
                if confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_idx = int(box.cls[0].item())
                    detection_info = self._calculate_spatial_properties(x1, y1, x2, y2, class_idx)
                    if detection_info:
                        detection_info["confidence"] = confidence
                        spatial_detections.append(detection_info)
        return spatial_detections

    def generate_summary_feedback(self, spatial_detections):
        """
        Generates a list of verbal feedback strings from detections, handling counts.

        Args:
            spatial_detections (list): List of detected objects with spatial info.

        Returns:
            list: A list of strings, each meant to be spoken separately.
                  Returns an empty list if no detections.
        """
        if not spatial_detections:
            return ["No objects detected in the scene."]

        # Group detections by class name
        grouped_detections = defaultdict(list)
        for det in spatial_detections:
            grouped_detections[det['class_name']].append(det)

        feedback_list = []

        # Sort class names for consistent announcement order (optional)
        sorted_class_names = sorted(grouped_detections.keys())

        if not sorted_class_names:
             return ["No objects identified above the confidence threshold."]

        for class_name in sorted_class_names:
            detections = grouped_detections[class_name]
            count = len(detections)

            if count == 1:
                # Single object: Use the detailed format
                det = detections[0] # Only one item in the list
                feedback = f"{det['class_name']} is {det['distance']} distance, {det['position_desc']}. {det['guidance']}."
                feedback_list.append(feedback)
            else:
                # Multiple objects: Announce count, then locations
                plural = "s" if count > 1 else ""
                count_feedback = f"{count} {class_name}{plural} detected."
                feedback_list.append(count_feedback)

                # Sort instances (e.g., near to far, or left to right) for consistent description
                detections.sort(key=lambda d: (
                    {'near': 0, 'mid-distance': 1, 'far': 2}.get(d['distance'], 3), # Sort by distance first
                    d['bbox'][0] # Then by left position
                ))

                # Provide brief location for each (up to a limit)
                max_detailed_count = 3
                for i, det in enumerate(detections[:max_detailed_count]):
                     # Use a slightly shorter format for each instance
                     instance_feedback = f"One {det['distance']}, {det['position_desc']}."
                     feedback_list.append(instance_feedback)
                if count > max_detailed_count:
                    feedback_list.append(f"And {count - max_detailed_count} more.")

        return feedback_list

    def get_color_for_distance(self, distance):
        if distance == "near": return (0, 255, 0)
        elif distance == "mid-distance": return (0, 165, 255)
        else: return (0, 0, 255)

    def draw_overlays(self, frame, spatial_detections):
        if frame is None: 
            logger.warning("Received None frame for overlay drawing")
            return None
        
        output_frame = frame.copy()
        h, w = output_frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        grouped_for_drawing = defaultdict(list)
        for det in spatial_detections:
            grouped_for_drawing[det['class_name']].append(det)

        logger.info(f"Drawing overlays for {len(spatial_detections)} detections")

        for class_name, detections in grouped_for_drawing.items():
             # Sort detections within group for consistent labeling if needed (e.g., left-to-right)
             detections.sort(key=lambda d: d['bbox'][0])
             count = len(detections)
             for i, det in enumerate(detections):
                x1, y1, x2, y2 = det["bbox"]
                color = self.get_color_for_distance(det["distance"])
                
                # Draw bounding box with thicker line
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 3)
                
                # Smaller, more compact text for multiple objects
                base_font_scale = 0.45 if count > 3 else 0.5
                thickness = 1
                
                # Create compact label
                if count > 1:
                    main_label = f"{class_name} {i+1}"
                else:
                    main_label = class_name
                
                # Shorter distance labels
                distance_short = {"near": "Near", "mid-distance": "Mid", "far": "Far"}[det['distance']]
                
                # Shorter position labels
                h_pos_short = {"front": "Front", "left": "Left", "right": "Right", 
                              "far left": "F.Left", "far right": "F.Right"}[det['h_position']]
                
                # Calculate text dimensions
                (tw1, th1), _ = cv2.getTextSize(main_label, font, base_font_scale, thickness)
                (tw2, th2), _ = cv2.getTextSize(distance_short, font, base_font_scale * 0.8, thickness)
                (tw3, th3), _ = cv2.getTextSize(h_pos_short, font, base_font_scale * 0.8, thickness)
                
                # Calculate compact background size
                max_width = max(tw1, tw2, tw3) + 6
                line_height = max(th1, th2, th3) + 2
                total_height = line_height * 3 + 4
                
                # Position text inside the bounding box (top-left corner)
                text_x = x1 + 3
                text_y_start = y1 + line_height + 2
                
                # Ensure text doesn't go outside frame
                if text_y_start + total_height > y2:
                    text_y_start = y1 + line_height + 2
                if text_x + max_width > x2:
                    text_x = max(x1 + 3, x2 - max_width - 3)
                
                # Draw semi-transparent background inside the box
                overlay = output_frame.copy()
                cv2.rectangle(overlay, 
                            (text_x - 2, text_y_start - line_height), 
                            (text_x + max_width, text_y_start + total_height - line_height), 
                            (0, 0, 0), cv2.FILLED)
                cv2.addWeighted(overlay, 0.7, output_frame, 0.3, 0, output_frame)
                
                # Draw text lines with white color for better contrast
                text_color = (255, 255, 255)
                
                # Line 1: Object name
                cv2.putText(output_frame, main_label, (text_x, text_y_start), 
                           font, base_font_scale, text_color, thickness, cv2.LINE_AA)
                
                # Line 2: Distance
                cv2.putText(output_frame, distance_short, (text_x, text_y_start + line_height), 
                           font, base_font_scale * 0.8, text_color, thickness, cv2.LINE_AA)
                
                # Line 3: Position
                cv2.putText(output_frame, h_pos_short, (text_x, text_y_start + line_height * 2), 
                           font, base_font_scale * 0.8, text_color, thickness, cv2.LINE_AA)

        # Smaller, cleaner legend
        legend_y = 25
        legend_x = 10
        box_size = 12
        text_offset = 4
        font_scale_legend = 0.4
        legend_thickness = 1
        
        # Near legend
        cv2.rectangle(output_frame, (legend_x, legend_y-box_size), (legend_x+box_size, legend_y), (0,255,0), -1)
        cv2.rectangle(output_frame, (legend_x, legend_y-box_size), (legend_x+box_size, legend_y), (0,0,0), 1)
        cv2.putText(output_frame, "Near", (legend_x+box_size+text_offset, legend_y-2), font, font_scale_legend, (255,255,255), legend_thickness, cv2.LINE_AA)
        legend_x += box_size + text_offset + 35
        
        # Mid legend
        cv2.rectangle(output_frame, (legend_x, legend_y-box_size), (legend_x+box_size, legend_y), (0,165,255), -1)
        cv2.rectangle(output_frame, (legend_x, legend_y-box_size), (legend_x+box_size, legend_y), (0,0,0), 1)
        cv2.putText(output_frame, "Mid", (legend_x+box_size+text_offset, legend_y-2), font, font_scale_legend, (255,255,255), legend_thickness, cv2.LINE_AA)
        legend_x += box_size + text_offset + 30
        
        # Far legend
        cv2.rectangle(output_frame, (legend_x, legend_y-box_size), (legend_x+box_size, legend_y), (0,0,255), -1)
        cv2.rectangle(output_frame, (legend_x, legend_y-box_size), (legend_x+box_size, legend_y), (0,0,0), 1)
        cv2.putText(output_frame, "Far", (legend_x+box_size+text_offset, legend_y-2), font, font_scale_legend, (255,255,255), legend_thickness, cv2.LINE_AA)
        
        # Cleaner status info
        status_y = h - 10
        device_info = f"Device: {self.device.upper()}"
        status_text = f"Press '0' to Quit | 'Enter' to Retake | {device_info}"
        cv2.putText(output_frame, status_text, (10, status_y), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        return output_frame


def calculate_blur_score(frame):
    """Calculates Laplacian variance as a measure of blur."""
    if frame is None: return 0
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Using CV_64F for higher precision, take absolute value before converting back
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = np.var(laplacian)
        return variance
    except cv2.error as e:
        return 0 # Return low score on error

def select_best_frame(frames, blur_threshold=50.0):
    """
    Selects the least blurry frame from a list based on Laplacian variance.

    Args:
        frames (list): List of captured OpenCV frames.
        blur_threshold (float): Minimum variance score to consider a frame "not blurry".

    Returns:
        numpy.ndarray: The selected best frame, or the middle frame as fallback.
    """
    if not frames:
        logger.warning("No frames provided to select_best_frame.")
        return None

    best_frame_index = -1 # Keep track of the index with max variance
    max_variance = -1.0
    sharp_frames_indices = []

    for i, frame in enumerate(frames):
        variance = calculate_blur_score(frame)
        if variance > max_variance:
            max_variance = variance
            best_frame_index = i
        if variance >= blur_threshold:
            sharp_frames_indices.append(i)

    if not sharp_frames_indices:
        logger.warning(f"All frames seem blurry (max variance {max_variance:.2f} < threshold {blur_threshold}).")
        return frames[best_frame_index] if best_frame_index != -1 else frames[len(frames) // 2]
    elif len(sharp_frames_indices) == 1:
        return frames[sharp_frames_indices[0]]
    else:
        middle_sharp_idx_in_list = len(sharp_frames_indices) // 2
        return frames[sharp_frames_indices[middle_sharp_idx_in_list]]


def run_object_detection_capture(
    audio_handler: AudioFeedbackHandler,
    duration=5,
    camera_id=1,
    model_path="yolov8n.pt",
    device="auto",
    conf=0.5,
    blur_thresh=50.0
):
    """
    Runs the capture-then-analyze object detection system using a shared audio handler.

    Args:
        audio_handler (AudioFeedbackHandler): The shared instance for TTS feedback.
        duration (int): Seconds to capture video frames.
        camera_id (int): Camera index.
        model_path (str): Path to YOLO model.
        device (str): 'cpu' or 'cuda'.
        conf (float): Confidence threshold.
        blur_thresh (float): Laplacian variance threshold for blur detection.
    """
    analyzer = None
    cap = None
    user_interrupted_flow = False # Flag to track if user quit early

    # Check if a valid audio handler was passed
    if not audio_handler or not isinstance(audio_handler, AudioFeedbackHandler):
        logger.error("Invalid or missing AudioFeedbackHandler provided")
        return

    try:
        logger.info("Starting object detection capture system")
        audio_handler.speak("Welcome to object detection system. Starting up, please wait.")
        logger.info("Audio: Welcome message played")
        
        analyzer = SpatialObjectAnalyzer(model_path=model_path, device=device, confidence_threshold=conf)
        
        logger.info(f"Attempting to open camera with ID: {camera_id}")
        cap = cv2.VideoCapture(camera_id,cv2.CAP_DSHOW)  # Use DirectShow for better compatibility on Windows
        if not cap.isOpened():
            if camera_id != 1:
                logger.warning(f"Primary camera {camera_id} failed, trying backup camera 0")
                audio_handler.speak("Main camera not responding. Trying backup camera.")
                logger.info("Audio: Backup camera message played")
                cap.open(0)
                if not cap.isOpened():
                    logger.error("Both primary and backup cameras failed to open")
                    audio_handler.speak("Error: Camera not found.")
                    logger.info("Audio: Camera error message played")
                    return
            else:
                logger.error("Default camera failed to open")
                audio_handler.speak("Please check your camera connection and try again.")
                logger.info("Audio: Camera connection error message played")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set to 1280x720 for better quality
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logger.info("Camera resolution set to 1280x720")

        # --- Capture Phase ---
        frames = []
        start_time = time.time()
        capture_window_name = "Capturing Scene..."
        logger.info(f"Starting capture phase for {duration} seconds")
        audio_handler.speak("Capturing scene.")
        logger.info("Audio: Capture start message played")

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                if not cap.isOpened():
                    logger.error("Camera disconnected during capture")
                    audio_handler.speak("Error: Camera disconnected.")
                    logger.info("Audio: Camera disconnection error played")
                    user_interrupted_flow = True
                    break 
                continue
            frames.append(frame.copy())

            display_frame = frame.copy()
            elapsed = int(time.time() - start_time)
            cv2.putText(display_frame, f"Capturing... {elapsed}/{duration}s (Press '0')", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(capture_window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('0'):
                logger.info("User interrupted capture phase")
                audio_handler.force_stop()
                user_interrupted_flow = True
                break
        
        if user_interrupted_flow:
            logger.info("Exiting due to user interruption during capture")
            try: cv2.destroyWindow(capture_window_name)
            except cv2.error: pass
            if cap and cap.isOpened(): cap.release()
            return

        try:
            cv2.destroyWindow(capture_window_name)
        except cv2.error:
            pass

        if cap and cap.isOpened():
             cap.release()

        if not frames:
            logger.warning("No frames captured during capture phase")
            audio_handler.speak("No images were captured. Please ensure there is adequate lighting.")
            logger.info("Audio: No frames captured message played")
            return

        logger.info(f"Captured {len(frames)} frames successfully")
        audio_handler.speak("Images captured.")
        logger.info("Audio: Images captured confirmation played")
        
        best_frame = select_best_frame(frames, blur_threshold=blur_thresh)
        del frames
        if best_frame is None:
            logger.error("Could not select a suitable frame from captured images")
            audio_handler.speak("Could not find a clear enough image. Please try to hold the camera more steady.")
            logger.info("Audio: Unclear image message played")
            return

        logger.info("Best frame selected, starting object detection analysis")
        processing_frame = best_frame.copy()
        cv2.putText(processing_frame, "Analyzing Frame...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        processing_window_name = "Processing..."
        cv2.imshow(processing_window_name, processing_frame)
        cv2.waitKey(1)

        spatial_detections = analyzer.detect_objects_in_frame(best_frame)
        logger.info(f"Object detection completed - Found {len(spatial_detections)} objects")

        try:
            cv2.destroyWindow(processing_window_name)
        except cv2.error:
            pass

        if not spatial_detections:
            logger.info("No objects detected above confidence threshold")
            audio_handler.speak("I don't see any objects I recognize in this view. Try pointing the camera in a different direction.")
            logger.info("Audio: No objects detected message played")
        else:
            logger.info("Objects detected, starting feedback generation")
            audio_handler.speak("I found some objects. Here's what I see:")
            logger.info("Audio: Objects found message played")
            
        feedback_list = analyzer.generate_summary_feedback(spatial_detections)
        result_window_name = "Object Detection Results - Press '0' to Quit"

        # Display results first so user can see while hearing feedback
        result_frame = analyzer.draw_overlays(best_frame, spatial_detections)
        cv2.imshow(result_window_name, result_frame)
        cv2.waitKey(1)

        logger.info(f"Starting audio feedback - {len(feedback_list)} messages to deliver")
        for i, message in enumerate(feedback_list):
            logger.info(f"Audio feedback {i+1}/{len(feedback_list)}: {message}")
            audio_handler.speak(message)
            
            # Check for interruption during feedback speaking
            key = cv2.waitKey(20) & 0xFF
            if key == ord('0'):
                logger.info("User interrupted feedback delivery")
                audio_handler.force_stop()
                user_interrupted_flow = True
                break
            try:
                if cv2.getWindowProperty(result_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed during feedback delivery")
                    audio_handler.force_stop()
                    user_interrupted_flow = True
                    break
            except cv2.error: 
                logger.info("Window error during feedback delivery")
                audio_handler.force_stop()
                user_interrupted_flow = True
                break
        
        if user_interrupted_flow:
            logger.info("Feedback interrupted, preparing to exit")
            audio_handler.speak("Exiting object detection.")
            logger.info("Audio: Exit message played")
            time.sleep(1.5)
            return

        # --- Wait for Quit or Retake (if feedback was not interrupted) ---
        logger.info("Analysis complete, waiting for user input")
        audio_handler.speak("Analysis complete. Press 0 to exit or Enter to take a new image.")
        logger.info("Audio: Analysis complete message played")
        
        # --- Retake Logic (Enter key) ---
        while True:
            key = cv2.waitKey(100) & 0xFF
            
            window_closed_by_x = False
            try:
                if cv2.getWindowProperty(result_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    window_closed_by_x = True
            except cv2.error:
                window_closed_by_x = True # Treat as closed

            if window_closed_by_x:
                if audio_handler.speaking: 
                    audio_handler.force_stop()
                    time.sleep(0.2) 
                audio_handler.speak("Exiting object detection.")
                time.sleep(1.5) 
                user_interrupted_flow = True 
                break

            if key == ord('0'):
                if audio_handler.speaking: 
                    audio_handler.force_stop()
                    time.sleep(0.2) 
                audio_handler.speak("Exiting object detection.")
                time.sleep(1.5) 
                user_interrupted_flow = True 
                break
            
            if key == 13:  # Enter key
                if audio_handler.speaking: 
                    audio_handler.force_stop()
                    time.sleep(0.2) 
                audio_handler.speak("Taking a new image.")
                cv2.destroyAllWindows()
                time.sleep(0.5)
                
                # Restart from capture phase without re-initializing
                cap = cv2.VideoCapture(camera_id,cv2.CAP_DSHOW)  # Use DirectShow for better compatibility on Windows
                if not cap.isOpened():
                    audio_handler.speak("Error: Camera not found.")
                    return
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # --- New Capture Phase ---
                frames = []
                start_time = time.time()
                capture_window_name = "Capturing Scene..."
                audio_handler.speak("Capturing scene.")

                while time.time() - start_time < duration:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue
                    frames.append(frame.copy())

                    display_frame = frame.copy()
                    elapsed = int(time.time() - start_time)
                    cv2.putText(display_frame, f"Capturing... {elapsed}/{duration}s (Press '0')", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(capture_window_name, display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('0'):
                        audio_handler.force_stop()
                        user_interrupted_flow = True
                        break
                
                if user_interrupted_flow:
                    try: cv2.destroyWindow(capture_window_name)
                    except cv2.error: pass
                    if cap and cap.isOpened(): cap.release()
                    return
                
                try: cv2.destroyWindow(capture_window_name)
                except cv2.error: pass
                
                if cap and cap.isOpened(): cap.release()
                
                if not frames:
                    audio_handler.speak("No images were captured.")
                    return
                
                # Process new frames
                audio_handler.speak("Images captured.")
                best_frame = select_best_frame(frames, blur_threshold=blur_thresh)
                del frames
                
                if best_frame is None:
                    audio_handler.speak("Could not find a clear enough image.")
                    return
                
                # Analyze new frame
                processing_frame = best_frame.copy()
                cv2.putText(processing_frame, "Analyzing Frame...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                processing_window_name = "Processing..."
                cv2.imshow(processing_window_name, processing_frame)
                cv2.waitKey(1)
                
                spatial_detections = analyzer.detect_objects_in_frame(best_frame)
                
                try: cv2.destroyWindow(processing_window_name)
                except cv2.error: pass
                
                if not spatial_detections:
                    audio_handler.speak("I don't see any objects I recognize in this view.")
                else:
                    audio_handler.speak("I found some objects. Here's what I see:")
                
                feedback_list = analyzer.generate_summary_feedback(spatial_detections)
                result_frame = analyzer.draw_overlays(best_frame, spatial_detections)
                cv2.imshow(result_window_name, result_frame)
                cv2.waitKey(1)
                
                # Provide feedback for new analysis
                for message in feedback_list:
                    audio_handler.speak(message)
                    key = cv2.waitKey(20) & 0xFF
                    if key == ord('0'):
                        audio_handler.force_stop()
                        user_interrupted_flow = True
                        break
                
                if user_interrupted_flow:
                    audio_handler.speak("Exiting object detection.")
                    time.sleep(1.5)
                    return
                
                # Continue with the wait loop for the new analysis
                audio_handler.speak("Analysis complete. Press 0 to exit or Enter to take a new image.")
                continue
            
            if not audio_handler.speaking and key == 0xFF:
                continue

    except Exception as e:
        logger.error(f"Unexpected error in object detection: {e}", exc_info=True)
        if audio_handler and audio_handler.speaking:
            audio_handler.force_stop()
            time.sleep(0.2) 
        
        if audio_handler: 
            audio_handler.speak("I encountered a problem and need to restart. Please try again.")
            logger.info("Audio: Error recovery message played")

    finally:
        if not user_interrupted_flow and audio_handler:
            audio_handler.speak("Exiting object detection.")
            logger.info("Audio: Final exit message played")

        if cap and cap.isOpened():
            cap.release()
            logger.info("Camera resources released")
        cv2.destroyAllWindows()
        logger.info("OpenCV windows closed")


if __name__ == "__main__":
    logger.info("Starting Object Detection Capture as main module")
    test_audio_handler = None
    try:
        test_audio_handler = AudioFeedbackHandler()
        if not test_audio_handler.engine:
            logger.error("AudioFeedbackHandler failed to initialize")
        else:
            logger.info("AudioFeedbackHandler initialized successfully")
            run_object_detection_capture(
                audio_handler=test_audio_handler,
                duration=4,
                camera_id=1,
                model_path="yolov8n.pt",
                device="auto",
                conf=0.5,
                blur_thresh=60.0
            )
    except Exception as e:
        logger.error(f"Test execution failed: {e}", exc_info=True)
    finally:
        if test_audio_handler:
            test_audio_handler.stop()
            logger.info("AudioFeedbackHandler stopped")
        cv2.destroyAllWindows()
        logger.info("Application cleanup completed")
