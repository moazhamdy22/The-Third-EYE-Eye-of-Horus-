# --- START OF FILE object_detection_realtime.py ---

import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import logging
from collections import defaultdict, deque
import os
import sys
import torch

# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from audio_feedback_vision_assitant import AudioFeedbackHandler # Use the dedicated handler

# Configure enhanced logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Silence Ultralytics' own verbose logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def detect_best_device():
    """
    Detects the best available device (GPU if available, otherwise CPU)
    Returns device string and logs the selection
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name}. Using GPU acceleration.")
        print(f"🚀 GPU ACCELERATION ENABLED: {gpu_name}")
    else:
        device = "cpu"
        logger.info("No GPU detected. Using CPU for inference.")
        print("💻 Using CPU for inference (GPU not available)")
    
    return device

class RealtimeSpatialAwareness:
    def __init__(self, model_path="yolov8n.pt", device=None, confidence_threshold=0.5):
        """
        Initializes the RealtimeSpatialAwareness class with state management.
        """
        # Auto-detect device if not specified
        if device is None:
            device = detect_best_device()
        
        self.device = device
        logger.info(f"Initializing YOLO model with device: {self.device}")
        
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.labels = self.model.names
            logger.info(f"YOLO model loaded successfully from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            raise

        self.confidence_threshold = confidence_threshold

        # Frame dimensions (will be set dynamically)
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

        # --- State Management for Audio Feedback ---
        self.last_announced_state = {}
        self.last_seen_time = {}
        self.state_timeout = 10.0 # Seconds after which to clear state if object not seen

    def _calculate_spatial_properties(self, x1, y1, x2, y2, class_idx):
        """Calculates spatial properties for a single detection. (Mostly unchanged)"""
        if self.frame_width == 0 or self.frame_height == 0: return None

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

        # --- Guidance String (FIXED direction logic) ---
        is_reachable = (distance == "near") # Keep simple reachability
        # Generate movement guidance based on distance and position
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

        state_string = f"{distance}_{h_position}_{v_position}"

        return {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "class_name": class_name,
            "confidence": 0.0,
            "distance": distance,
            "h_position": h_position,
            "v_position": v_position,
            "position_desc": position_desc, # For display/logging
            "is_reachable": is_reachable,
            "guidance": guidance, # The detailed guidance string
            "state_string": state_string # For state comparison
        }

    def process_frame(self, frame):
        """Detects objects, calculates properties. (Mostly unchanged)"""
        if frame is None: return []

        if self.frame_width == 0 or self.frame_height == 0:
            self.frame_height, self.frame_width = frame.shape[:2]
            if self.frame_width == 0 or self.frame_height == 0: return []

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

    def _cleanup_stale_state(self, current_time):
        """Removes state for objects not seen recently."""
        stale_keys = [
            key for key, last_time in self.last_seen_time.items()
            if current_time - last_time > self.state_timeout
        ]
        for key in stale_keys:
            if key in self.last_announced_state:
                del self.last_announced_state[key]
            if key in self.last_seen_time:
                del self.last_seen_time[key]

    def format_stateful_feedback(self, spatial_detections, current_time, force_announce=False):
        """
        Generates verbal feedback based on changes from the last announced state,
        counts objects, and uses the detailed format.

        Args:
            spatial_detections (list): List of detected objects with spatial info.
            current_time (float): The current time.

        Returns:
            str: A single string summarizing *new* or *changed* relevant detections,
                 or None if no new/changed info to announce.
        """
        self._cleanup_stale_state(current_time) # Remove old states first

        if not spatial_detections:
            return None

        grouped_detections = defaultdict(list)
        for det in spatial_detections:
            grouped_detections[det['class_name']].append(det)
            self.last_seen_time[det['class_name']] = current_time # Update last seen time

        messages_to_speak = []
        sorted_class_names = sorted(grouped_detections.keys())

        for class_name in sorted_class_names:
            detections = grouped_detections[class_name]
            count = len(detections)

            detections.sort(key=lambda d: {'near': 0, 'mid-distance': 1, 'far': 2}.get(d['distance'], 3))
            representative_detection = detections[0]
            current_state_tuple = (count, representative_detection['state_string'])

            last_state_tuple = self.last_announced_state.get(class_name)

            if force_announce or current_state_tuple != last_state_tuple:
                if count == 1:
                    det = representative_detection
                    feedback = f"{det['class_name']} is {det['distance']} distance, {det['position_desc']}. {det['guidance']}."
                    messages_to_speak.append(feedback)
                else:
                    plural = "s" if count > 1 else ""
                    count_feedback = f"{count} {class_name}{plural} detected."
                    messages_to_speak.append(count_feedback)
                    max_detailed_count = 3 # Can be adjusted
                    for i, det_item in enumerate(detections[:max_detailed_count]): # Renamed 'det' to 'det_item'
                         instance_feedback = f"One {det_item['distance']}, {det_item['position_desc']}."
                         messages_to_speak.append(instance_feedback)
                    if count > max_detailed_count:
                        messages_to_speak.append(f"And {count - max_detailed_count} more.")

                self.last_announced_state[class_name] = current_state_tuple

        if not messages_to_speak:
            return None

        return " ".join(messages_to_speak)

    def get_color_for_distance(self, distance):
        if distance == "near": return (0, 255, 0)
        elif distance == "mid-distance": return (0, 165, 255)
        else: return (0, 0, 255)

    def draw_overlays(self, frame, spatial_detections):
        if frame is None: return None
        output_frame = frame.copy()
        h, w = output_frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        grouped_for_drawing = defaultdict(list)
        for det in spatial_detections:
            grouped_for_drawing[det['class_name']].append(det)

        for class_name, detections in grouped_for_drawing.items():
            count = len(detections)
            detections.sort(key=lambda d: d['bbox'][0]) # Sort by x-coordinate for consistent numbering if multiple
            for i, det_item in enumerate(detections):
                x1, y1, x2, y2 = det_item["bbox"]
                color = self.get_color_for_distance(det_item["distance"])
                
                # Draw bounding box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                
                # Enhanced label with object name, distance, and position
                object_name = det_item['class_name']
                distance = det_item['distance']
                h_position = det_item['h_position']
                
                # Create multi-line label
                label_lines = [
                    f"{object_name}",
                    f"Distance: {distance}",
                    f"Position: {h_position}"
                ]
                
                if count > 1:
                    label_lines[0] += f" ({i+1}/{count})"
                
                font_scale = 0.4
                thickness = 1
                line_height = 15
                
                # Calculate background rectangle size
                max_width = 0
                total_height = len(label_lines) * line_height + 5
                
                for line in label_lines:
                    (tw, th), bl = cv2.getTextSize(line, font, font_scale, thickness)
                    max_width = max(max_width, tw)
                
                # Draw background rectangle
                label_y_start = max(y1 - total_height - 5, 0)
                cv2.rectangle(output_frame, 
                            (x1, label_y_start), 
                            (x1 + max_width + 10, label_y_start + total_height), 
                            color, cv2.FILLED)
                
                # Draw text lines
                for idx, line in enumerate(label_lines):
                    text_y = label_y_start + (idx + 1) * line_height
                    cv2.putText(output_frame, line, (x1 + 5, text_y), 
                              font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # Enhanced Legend with position information
        legend_y = 30
        legend_x = 10
        box_size = 15
        text_offset = 5
        font_scale_legend = 0.5
        
        # Distance legend
        cv2.rectangle(output_frame, (legend_x, legend_y-box_size), (legend_x+box_size, legend_y), (0,255,0), -1)
        cv2.putText(output_frame, "Near", (legend_x+box_size+text_offset, legend_y), font, font_scale_legend, (0,255,0), 1, cv2.LINE_AA)
        legend_x += box_size+text_offset+40
        
        cv2.rectangle(output_frame, (legend_x, legend_y-box_size), (legend_x+box_size, legend_y), (0,165,255), -1)
        cv2.putText(output_frame, "Mid", (legend_x+box_size+text_offset, legend_y), font, font_scale_legend, (0,165,255), 1, cv2.LINE_AA)
        legend_x += box_size+text_offset+30
        
        cv2.rectangle(output_frame, (legend_x, legend_y-box_size), (legend_x+box_size, legend_y), (0,0,255), -1)
        cv2.putText(output_frame, "Far", (legend_x+box_size+text_offset, legend_y), font, font_scale_legend, (0,0,255), 1, cv2.LINE_AA)
        
        # System info
        system_info = f"Device: {self.device.upper()}"
        cv2.putText(output_frame, system_info, (10, h - 35), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw Status Info
        status_y = h - 15
        cv2.putText(output_frame, "Press '0' to Quit", (10, status_y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return output_frame


def run_realtime_spatial_awareness(
    audio_handler: AudioFeedbackHandler,
    camera_id=1,
    model_path="yolov8n.pt",
    device=None,
    conf=0.5,
    feedback_interval=5.0
):
    """
    Runs the stateful real-time spatial awareness detection system using a shared audio handler.
    """
    detector = None
    cap = None
    user_initiated_exit = False 

    if not audio_handler or not isinstance(audio_handler, AudioFeedbackHandler):
        logger.error("Invalid or missing AudioFeedbackHandler provided.")
        return

    try:
        logger.info("Starting real-time spatial awareness system")
        audio_handler.speak("Welcome to real-time object detection. Starting system initialization.")
        
        # Auto-detect device if not specified
        if device is None:
            device = detect_best_device()
        
        detector = RealtimeSpatialAwareness(model_path=model_path, device=device, confidence_threshold=conf)
        
        logger.info(f"Attempting to open camera with ID: {camera_id}")
        cap = cv2.VideoCapture(camera_id,cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.error(f"Error: Could not open camera with ID {camera_id}.")
            if camera_id != 2:
                logger.info("Trying backup camera (ID: 0)")
                audio_handler.speak("Main camera not responding. Trying backup camera.")
                cap.open(0)
                if not cap.isOpened():
                    logger.error("No cameras found")
                    audio_handler.speak("No cameras found. Please connect a camera and try again.")
                    return
                else:
                    logger.info("Backup camera connected successfully")
                    audio_handler.speak("Backup camera connected successfully.")
            else:
                logger.error("Camera not found")
                audio_handler.speak("Camera not found. Please check your camera connection.")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        window_name = "Real-time Spatial Awareness (Enhanced) - Press '0' to Quit"
        logger.info("System initialization complete")
        audio_handler.speak("System ready. I will now describe objects as they appear. Press 0 to stop.")
        
        last_feedback_time = time.time() 
        initial_feedback_given = False
        MAX_INITIAL_FRAME_ATTEMPTS = 15 
        initial_frame_count = 0
        
        while not user_initiated_exit:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1) 
                if not cap.isOpened(): 
                    logger.error("Camera connection lost")
                    if audio_handler: audio_handler.speak("Camera connection lost. Please check your camera.")
                    user_initiated_exit = True 
                    break
                continue

            current_time = time.time()
            spatial_detections = detector.process_frame(frame)

            if not initial_feedback_given:
                initial_frame_count +=1
                if spatial_detections:
                    feedback_message = detector.format_stateful_feedback(spatial_detections, current_time)
                    if feedback_message:
                        logger.info(f"Initial detection feedback: {feedback_message}")
                        audio_handler.speak(feedback_message)
                        initial_feedback_given = True
                        last_feedback_time = current_time 
                elif initial_frame_count >= MAX_INITIAL_FRAME_ATTEMPTS:
                    logger.info("No initial detections found after maximum attempts")
                    initial_feedback_given = True 
                    last_feedback_time = current_time 
            
            elif initial_feedback_given and (current_time - last_feedback_time >= feedback_interval):
                feedback_message = detector.format_stateful_feedback(spatial_detections, current_time)
                if feedback_message:
                    if not audio_handler.speaking:
                        logger.info(f"Audio feedback: {feedback_message}")
                        audio_handler.speak(feedback_message)
                last_feedback_time = current_time

            display_frame = detector.draw_overlays(frame, spatial_detections)
            if display_frame is not None: 
                cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF 
            
            quit_by_q = (key == ord('0') or key == ord('0'))
            quit_by_x = False
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    quit_by_x = True
            except cv2.error: 
                quit_by_x = True

            if quit_by_q:
                logger.info("User initiated exit via '0' key")
                if audio_handler: audio_handler.speak("Exiting real-time detection.")
                time.sleep(2.0) 
                user_initiated_exit = True
            
            elif quit_by_x: 
                logger.info("User closed window")
                if audio_handler: audio_handler.speak("Window closed. Exiting real-time detection.")
                time.sleep(2.0) 
                user_initiated_exit = True

    except Exception as e:
        logger.error(f"An error occurred in real-time mode: {e}", exc_info=True)
        if audio_handler: 
            if audio_handler.speaking: 
                audio_handler.force_stop()
                time.sleep(0.2) 
            audio_handler.speak("I encountered a problem and need to stop. Please try again.")
            time.sleep(2.5) 
        user_initiated_exit = True 

    finally:
        logger.info("Shutting down real-time detection system")
        if audio_handler: 
            audio_handler.speak("Real-time detection system shutting down.")
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_audio_handler = None
    try:
        logger.info("Starting object detection application")
        test_audio_handler = AudioFeedbackHandler()
        if not test_audio_handler.engine:
             logger.error("Failed to initialize audio. Exiting.")
        else:
            run_realtime_spatial_awareness(
                audio_handler=test_audio_handler,
                camera_id=1,
                model_path="yolov8n.pt",
                device=None,  # Auto-detect
                conf=0.5,
                feedback_interval=3.0
            )
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        if test_audio_handler:
            test_audio_handler.stop()
        cv2.destroyAllWindows()
        logger.info("Application terminated")