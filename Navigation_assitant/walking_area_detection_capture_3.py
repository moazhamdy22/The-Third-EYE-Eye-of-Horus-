# --- START OF FILE walking_area_batch.py ---

import cv2
import torch
import numpy as np
import time
import os
import sys
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import logging
import threading
import queue
from collections import Counter, defaultdict # Import defaultdict
from typing import List, Tuple, Optional, Set # Added Set

# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- Import Audio Handler ---
try:
    from audio_feedback_vision_assitant import AudioFeedbackHandler
    AUDIO_AVAILABLE = True
except ImportError:
    logging.error("Failed to import AudioFeedbackHandler. Audio feedback disabled.")
    AudioFeedbackHandler = None
    AUDIO_AVAILABLE = False

# (Optional) For Arabic text handling
try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    ARABIC_AVAILABLE = True
except ImportError:
    ARABIC_AVAILABLE = False
    def reshape(text): return text
    def get_display(text): return text

# --- Configuration ---
# Model & Detection
MODEL_NAME = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
MIN_AREA_RATIO = 0.3  # Minimum area ratio for detected walking areas

# Batch Processing
CAPTURE_DURATION_SECONDS = 4.0 # How long to capture frames
NUM_FRAMES_TO_PROCESS = 1    # Max number of frames to process from the batch (use <= 0 to process all)
# Consider adding blur/sharpness selection later if needed

# Camera
CAMERA_INDEX = 1
FRAME_WIDTH = 1280  # Target processing width
FRAME_HEIGHT = 720 # Target processing height

# Audio Feedback
AUDIO_ENABLED = True
AUDIO_SPEAK_ENGLISH = True
AUDIO_SPEAK_ARABIC = True # Set to False if no Arabic needed
AUDIO_COOLDOWN_SECONDS = 6.0
AUDIO_INITIAL_DELAY_SECONDS = 2.0

# --- Walking Area Classes (Mapillary Vistas) ---
# (Adding Arabic names - adjust these as needed)
WALKING_AREA_CLASSES = {
    # Class Name EN: (Class ID, Class Name AR)
    "Sidewalk":      (13, "رصيف"),
    "Path":          (14, "مسار"), # Footpath/Trail
    "Road":          (12, "طريق"), # Includes pedestrian zones sometimes
    "Crosswalk":     (15, "معبر مشاة"),
    "Floor":         (16, "أرضية"), # Indoor
    "Stairs":        (17, "درج"),
    "Ramp":          (18, "منحدر"),
    "Manhole":       (30, "فتحة صرف"), # Often on walkable surfaces
    "Pothole":       (31, "حفرة"),      # Often on walkable surfaces
    # Consider adding 'terrain' (grass/soil) if relevant (class 22)
}
# Extract IDs for faster lookup during processing
WALKING_AREA_IDS = {v[0] for v in WALKING_AREA_CLASSES.values()}
ID_TO_NAME_EN = {v[0]: k for k, v in WALKING_AREA_CLASSES.items()}
ID_TO_NAME_AR = {v[0]: v[1] for k, v in WALKING_AREA_CLASSES.items()}


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Silence verbose logs from transformers if desired
logging.getLogger("transformers").setLevel(logging.WARNING)


class WalkingAreaDetectorBatch:
    def __init__(self, shared_audio_handler=None):
        logger.info("Initializing Walking Area Detection System...")
        self.frame_width = FRAME_WIDTH
        self.frame_height = FRAME_HEIGHT

        # --- Model Initialization ---
        logger.info(f"Loading walking area model: {MODEL_NAME}...")
        try:
            self.area_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
            self.area_model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME)
            self.area_model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.exception(f"FATAL: Failed to load model {MODEL_NAME}. Exiting.")
            raise RuntimeError("Model Loading Failed") from e

        # --- State Variables for Aggregated Results ---
        self.aggregated_guidance: List[str] = ["Initializing..."]
        self.aggregated_detected_areas_en: Set[str] = set()
        self.aggregated_detected_areas_ar: Set[str] = set()
        self.aggregated_position_grid = np.zeros((3, 3)) # Averaged grid

        # --- Visualization ---
        self.last_frame_display = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) # Placeholder
        # Text for overlay (similar to scene detection display)
        self.display_text_en = ["Initializing..."]
        self.display_text_ar = [self.process_arabic_text("جار التهيئة...")] if ARABIC_AVAILABLE else []
        # --- NEW: Store the mask used for the latest visualization ---
        self.last_visualization_mask: Optional[np.ndarray] = None

        # --- Audio Initialization (SIMPLIFIED like object detection) ---
        self.audio_handler = shared_audio_handler
        self.last_announced_guidance_key: str = ""
        self.last_announced_areas_key: str = ""
        self.last_announcement_time: float = 0.0
        self.start_time: float = time.monotonic()

        if self.audio_handler:
            logger.info("Walking Area Detection is using shared AudioFeedbackHandler instance.")
        else:
            if AUDIO_ENABLED:
                logger.warning("Audio feedback is disabled for Walking Area Detection (no handler provided).")

        logger.info("Walking Area Detection System Initialized.")

    def process_arabic_text(self, text):
        """Helper for processing Arabic text if libraries are available."""
        if not ARABIC_AVAILABLE or not text:
            return text # Return original if libs not available or empty
        try:
            return get_display(reshape(str(text)))
        except Exception as e:
            logger.error(f"Error processing Arabic text '{text}': {e}")
            return str(text)

    def detect_single_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Set[int]]: # Changed return type hint
        """Detects walking areas in a single frame, returns mask and detected class IDs."""
        try:
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = self.area_processor(images=image_pil, return_tensors="pt")

            with torch.no_grad():
                outputs = self.area_model(**inputs)

            predicted_map = self.area_processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image_pil.size[::-1]])[0]

            walking_area_binary_mask = torch.zeros_like(predicted_map, dtype=torch.uint8)
            detected_ids_in_frame = set()
            total_pixels = predicted_map.numel()

            # Iterate through unique values present in the map for efficiency
            unique_ids_in_map = torch.unique(predicted_map).tolist()

            for class_id in unique_ids_in_map:
                if class_id in WALKING_AREA_IDS:
                    mask = (predicted_map == class_id)
                    area_ratio = mask.sum().item() / total_pixels
                    if area_ratio > MIN_AREA_RATIO:
                        walking_area_binary_mask[mask] = 1 # Use 1 for binary mask
                        detected_ids_in_frame.add(class_id)

            return walking_area_binary_mask.cpu().numpy(), detected_ids_in_frame

        except Exception as e:
            logger.error(f"Error during single frame detection: {e}")
            return None, set()

    # --- analyze_grid_position, generate_guidance_from_grid, process_frame_batch ---
    # --- remain the same as the previous correct version ---
    def analyze_grid_position(self, walking_area_binary_mask: np.ndarray) -> np.ndarray:
        """Calculates the 3x3 grid occupancy from a binary mask."""
        if walking_area_binary_mask is None or walking_area_binary_mask.size == 0:
            return np.zeros((3, 3))

        h, w = walking_area_binary_mask.shape
        grid_h, grid_w = max(1, h // 3), max(1, w // 3) # Avoid division by zero
        grid = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                row_start, row_end = i * grid_h, (i + 1) * grid_h
                col_start, col_end = j * grid_w, (j + 1) * grid_w
                row_end = min(row_end, h); col_end = min(col_end, w)
                section = walking_area_binary_mask[row_start:row_end, col_start:col_end]
                section_pixels = section.size
                if section_pixels > 0:
                    grid[i, j] = np.count_nonzero(section) / section_pixels
        return grid

    def generate_guidance_from_grid(self, position_grid: np.ndarray) -> List[str]:
        """Generates navigation guidance based on the 3x3 grid."""
        guidance = []
        safe_threshold = 0.30; side_preference_threshold = 0.15; min_side_threshold = 0.20
        bottom_center = position_grid[2, 1]; mid_center = position_grid[1,1]; top_center = position_grid[0,1]
        center_path_confidence = (bottom_center * 0.6 + mid_center * 0.3 + top_center * 0.1)
        left_region_avg = np.mean(position_grid[:, 0]); right_region_avg = np.mean(position_grid[:, 2])

        if bottom_center < 0.15: guidance.append("Caution: Path directly ahead unclear.")
        elif center_path_confidence >= safe_threshold: guidance.append("Path forward seems clear.")
        else: guidance.append("Limited walking space ahead.")

        if left_region_avg > right_region_avg + side_preference_threshold and left_region_avg > min_side_threshold:
            guidance.append("More space appears available to the left.")
        elif right_region_avg > left_region_avg + side_preference_threshold and right_region_avg > min_side_threshold:
            guidance.append("More space appears available to the right.")
        elif left_region_avg < 0.1 and right_region_avg < 0.1 and center_path_confidence < safe_threshold:
             guidance.append("Warning: Very limited space detected overall.")

        if not guidance: guidance.append("Analyze surroundings.")
        return guidance

    def process_frame_batch(self, frames: List[np.ndarray]) -> Tuple[Set[int], np.ndarray]: # Changed hint
        """Processes a batch of frames, returning aggregated detected IDs and average grid."""
        if not frames: return set(), np.zeros((3,3))
        batch_detected_ids = set(); batch_grid_sum = np.zeros((3, 3)); valid_frames_processed = 0
        for frame in frames:
            mask, ids = self.detect_single_frame(frame)
            if mask is not None:
                batch_detected_ids.update(ids); grid = self.analyze_grid_position(mask)
                batch_grid_sum += grid; valid_frames_processed += 1
            else: logger.warning("Skipping a frame in batch due to detection error.")
        if valid_frames_processed > 0: average_grid = batch_grid_sum / valid_frames_processed
        else: average_grid = np.zeros((3,3)); logger.warning("No frames processed successfully in the batch.")
        return batch_detected_ids, average_grid

    # --- Audio Handling (_generate_audio_message, _handle_audio_feedback) ---
    # --- remain the same as the previous correct version ---
    def _generate_audio_message(self, detected_areas_en: Set[str], detected_areas_ar: Set[str], guidance: List[str]) -> str:
        """Creates the audio message."""
        message_parts_en = []
        message_parts_ar = []
        
        # English message construction
        if detected_areas_en:
            areas_str_en = ", ".join(sorted(list(detected_areas_en)))
            message_parts_en.append(f"Detected areas include: {areas_str_en}.")
        else:
            message_parts_en.append("No specific walking areas detected.")
        
        if guidance:
            guidance_str_en = " ".join(guidance)
            message_parts_en.append(f"Guidance: {guidance_str_en}")

        # Arabic message construction (separate from English to avoid duplication)
        if AUDIO_SPEAK_ARABIC and ARABIC_AVAILABLE:
            if detected_areas_ar:
                areas_str_ar = ", ".join(self.process_arabic_text(name) for name in sorted(list(detected_areas_ar)))
                detected_areas_prefix_ar = self.process_arabic_text("المناطق المكتشفة تشمل")
                message_parts_ar.append(f"{detected_areas_prefix_ar}: {areas_str_ar}.")
            else:
                message_parts_ar.append(self.process_arabic_text("لم يتم اكتشاف مناطق محددة للمشي."))
            
            if guidance:
                guidance_prefix_ar = self.process_arabic_text("التوجيه")
                # Translate key guidance phrases to Arabic instead of repeating English
                guidance_ar_parts = []
                for guide in guidance:
                    if "Path directly ahead unclear" in guide:
                        guidance_ar_parts.append(self.process_arabic_text("المسار أمامك غير واضح"))
                    elif "Path forward seems clear" in guide:
                        guidance_ar_parts.append(self.process_arabic_text("المسار أمامك يبدو واضحاً"))
                    elif "More space appears available to the left" in guide:
                        guidance_ar_parts.append(self.process_arabic_text("مساحة أكبر متاحة على اليسار"))
                    elif "More space appears available to the right" in guide:
                        guidance_ar_parts.append(self.process_arabic_text("مساحة أكبر متاحة على اليمين"))
                    elif "Very limited space detected" in guide:
                        guidance_ar_parts.append(self.process_arabic_text("مساحة محدودة جداً"))
                    else:
                        # Fallback for other guidance messages
                        guidance_ar_parts.append(self.process_arabic_text("تحليل المحيط"))
                
                if guidance_ar_parts:
                    guidance_str_ar = ". ".join(guidance_ar_parts)
                    message_parts_ar.append(f"{guidance_prefix_ar}: {guidance_str_ar}")

        # Combine messages
        final_message_parts = []
        if AUDIO_SPEAK_ENGLISH and message_parts_en:
            final_message_parts.append(" ".join(message_parts_en))
        if AUDIO_SPEAK_ARABIC and message_parts_ar:
            if final_message_parts:  # Add separator if English part exists
                final_message_parts.append(" " + " ".join(message_parts_ar))
            else:
                final_message_parts.append(" ".join(message_parts_ar))

        final_message = "".join(final_message_parts).strip()
        return final_message if final_message else "No navigation update."

    def _handle_audio_feedback(self):
        """Checks state, cooldown, and triggers audio."""
        if not self.audio_handler:
            return
            
        now = time.monotonic()
        current_areas_key = ",".join(sorted(list(self.aggregated_detected_areas_en)))
        current_guidance_key = ",".join(self.aggregated_guidance)
        
        areas_changed = (current_areas_key != self.last_announced_areas_key)
        guidance_changed = (current_guidance_key != self.last_announced_guidance_key)
        state_changed = areas_changed or guidance_changed
        
        cooldown_active = (now - self.last_announcement_time < AUDIO_COOLDOWN_SECONDS)
        is_initial_period = (now - self.start_time < AUDIO_INITIAL_DELAY_SECONDS)
        first_announcement_pending = (self.last_announcement_time == 0)
        
        should_speak = False
        if state_changed and not cooldown_active:
            should_speak = True
            logger.info("Walking area state change detected, triggering audio.")
        elif first_announcement_pending and not is_initial_period:
            should_speak = True
            logger.info("First walking area announcement after initial delay.")
            
        if should_speak:
            message = self._generate_audio_message(
                self.aggregated_detected_areas_en, 
                self.aggregated_detected_areas_ar, 
                self.aggregated_guidance
            )
            if message and message != "No navigation update.":
                logger.info(f"AUDIO (Walking): Speaking: '{message}'")
                self.audio_handler.speak(message)  # Remove priority parameter
                self.last_announced_areas_key = current_areas_key
                self.last_announced_guidance_key = current_guidance_key
                self.last_announcement_time = now
            else:
                logger.debug("Generated walking audio message was empty or default, not speaking.")

    # --- Visualization (MODIFIED) ---
    def create_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Creates visualization with walking area overlay and text."""
        h, w = frame.shape[:2]
        overlay_color = (0, 0, 0)
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        line_height = 25
        walking_area_overlay_color = [0, 255, 0] # Green

        # --- Apply Green Walking Area Overlay ---
        # Start with the original frame
        vis_frame = frame.copy()
        if self.last_visualization_mask is not None and self.last_visualization_mask.shape == frame.shape[:2]:
            try:
                # Create a colored layer for the mask
                walking_overlay_colored = np.zeros_like(frame, dtype=np.uint8)
                # Ensure mask is boolean or 0/1 for indexing
                mask_bool = self.last_visualization_mask > 0
                walking_overlay_colored[mask_bool] = walking_area_overlay_color

                # Blend the colored overlay with the frame
                alpha = 0.4 # Transparency of the green overlay
                vis_frame = cv2.addWeighted(walking_overlay_colored, alpha, vis_frame, 1 - alpha, 0)
            except Exception as e:
                 logger.error(f"Error applying walking area overlay: {e}")
                 # Fallback to original frame if error occurs
                 vis_frame = frame.copy()
        # else: # If no mask, vis_frame remains the original frame copy


        # --- Draw Text Overlay Background ---
        # Calculate height needed based on number of lines in EN and AR lists
        num_en_lines = len(self.display_text_en)
        num_ar_lines = len(self.display_text_ar) if AUDIO_SPEAK_ARABIC and ARABIC_AVAILABLE else 0
        total_lines = num_en_lines + num_ar_lines
        # Add extra space for padding top/bottom
        text_area_height = (total_lines * line_height) + (line_height // 2) # Add padding

        # Ensure text area doesn't exceed frame height
        text_area_height = min(text_area_height, h - 10)
        text_area_y_start = h - text_area_height - 10

        cv2.rectangle(vis_frame, (0, text_area_y_start), (w, h), overlay_color, -1) # Draw black background


        # --- Draw English Text ---
        y_pos = text_area_y_start + (line_height // 2) # Start with padding
        for i, line in enumerate(self.display_text_en):
            current_y = y_pos + i * line_height
            if current_y < h - (line_height // 4): # Ensure text fits vertically
                 cv2.putText(vis_frame, line, (10, current_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # --- Draw Arabic Text (if enabled and available) ---
        if AUDIO_SPEAK_ARABIC and ARABIC_AVAILABLE and num_ar_lines > 0:
            # Start Arabic text below English text
            y_pos += num_en_lines * line_height
            for i, line_ar in enumerate(self.display_text_ar):
                # line_ar should already be processed Arabic text
                try:
                    (text_width, _), _ = cv2.getTextSize(line_ar, font, font_scale, thickness)
                except Exception as e:
                    logger.warning(f"Could not get text size for Arabic: {e}")
                    text_width = 100 # Estimate width if error occurs

                current_y = y_pos + i * line_height
                if current_y < h - (line_height // 4): # Ensure text fits vertically
                     # Right-align Arabic text
                    cv2.putText(vis_frame, line_ar, (w - text_width - 10, current_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        return vis_frame


    # --- Main Loop (MODIFIED to store visualization mask) ---
    def run(self):
        logger.info("Starting Walking Area Detection loop...")
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Use DSHOW for better compatibility on Windows
        if not cap.isOpened():
            logger.error(f"FATAL: Could not open camera {CAMERA_INDEX}")
            if self.audio_handler:
                self.audio_handler.speak("Error: Camera not found. Walking area detection cannot start.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        logger.info(f"Camera {CAMERA_INDEX} opened.")
        print("Press '0' to quit.")

        if self.audio_handler:
            self.audio_handler.speak("Walking area detection is now active. I will analyze walking surfaces around you.")

        is_running = True
        user_initiated_quit = False
        
        try:
            while is_running:
                try:
                    # --- Capture Phase ---
                    captured_frames = []
                    start_capture_time = time.monotonic()
                    while time.monotonic() - start_capture_time < CAPTURE_DURATION_SECONDS:
                        ret, frame = cap.read()
                        if ret:
                            frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))
                            captured_frames.append(frame_resized)
                            self.last_frame_display = frame_resized
                        else:
                            logger.warning("Failed to capture frame during capture phase.")
                            time.sleep(0.05)

                        vis_frame = self.create_visualization(self.last_frame_display)
                        cv2.imshow("Walking Area Detection", vis_frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('0'):
                            logger.info("Quit signal received during capture.")
                            is_running = False
                            user_initiated_quit = True
                            break
                    if not is_running: break

                    # --- Selection Phase ---
                    if not captured_frames:
                        logger.warning("No frames captured in interval.")
                        continue
                    frames_to_process = captured_frames
                    if NUM_FRAMES_TO_PROCESS > 0 and len(frames_to_process) > NUM_FRAMES_TO_PROCESS:
                        frames_to_process = frames_to_process[:NUM_FRAMES_TO_PROCESS]
                        logger.info(f"Processing {len(frames_to_process)} frames (limited).")
                    if not frames_to_process:
                        logger.warning("No frames selected for processing.")
                        continue

                    # --- Processing & Aggregation Phase ---
                    logger.info(f"Processing batch of {len(frames_to_process)} frames...")
                    start_proc_time = time.monotonic()
                    batch_ids, avg_grid = self.process_frame_batch(frames_to_process)
                    proc_duration = time.monotonic() - start_proc_time

                    # --- Update Aggregated State ---
                    self.aggregated_detected_areas_en = {ID_TO_NAME_EN.get(id, "Unknown") for id in batch_ids}
                    self.aggregated_detected_areas_ar = {ID_TO_NAME_AR.get(id, "غير معروف") for id in batch_ids}
                    self.aggregated_position_grid = avg_grid
                    self.aggregated_guidance = self.generate_guidance_from_grid(avg_grid)

                    logger.info(f"Batch processed in {proc_duration:.3f}s. Areas: {self.aggregated_detected_areas_en}, Guidance: {self.aggregated_guidance}")

                    # --- Prepare Visualization Mask (Run detection on the LAST frame) ---
                    vis_mask, _ = self.detect_single_frame(self.last_frame_display)
                    self.last_visualization_mask = vis_mask # Store mask for display

                    # --- Update Display Text ---
                    self.display_text_en = []
                    self.display_text_ar = []
                    if self.aggregated_detected_areas_en: self.display_text_en.append(f"Areas: {', '.join(sorted(list(self.aggregated_detected_areas_en)))}")
                    else: self.display_text_en.append("Areas: None")
                    if self.aggregated_guidance: self.display_text_en.extend(self.aggregated_guidance)

                    if AUDIO_SPEAK_ARABIC and ARABIC_AVAILABLE:
                     if self.aggregated_detected_areas_ar:
                          areas_str_ar = ", ".join(self.process_arabic_text(name) for name in sorted(list(self.aggregated_detected_areas_ar)))
                          self.display_text_ar.append(f"{self.process_arabic_text('المناطق')}: {areas_str_ar}")
                     else: self.display_text_ar.append(self.process_arabic_text("المناطق: لا يوجد"))
                     if self.aggregated_guidance: self.display_text_ar.append(self.process_arabic_text("التوجيه: جار التحليل"))

                    # --- Audio Feedback Phase ---
                    self._handle_audio_feedback()

                    # --- Display Update ---
                    vis_frame = self.create_visualization(self.last_frame_display) # Now uses the updated mask
                    cv2.imshow("Walking Area Detection", vis_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('0'):
                        logger.info("Quit signal received after processing.")
                        is_running = False
                        user_initiated_quit = True
                        break
                   
                except KeyboardInterrupt:
                    logger.info("KeyboardInterrupt received during main loop. Stopping...")
                    is_running = False
                    user_initiated_quit = True
                    break
                except Exception as e:
                    logger.exception(f"An unexpected error occurred in the main loop: {e}")
                    is_running = False
                    break

        finally:
            # --- Cleanup (SIMPLIFIED like object detection) ---
            logger.info("Starting cleanup...")
            
            if self.audio_handler and user_initiated_quit:
                exit_message = "Walking area detection has been stopped. Thank you for using this feature."
                logger.info("User initiated quit. Playing exit message.")
                try:
                    self.audio_handler.speak(exit_message)
                    time.sleep(2.0)
                except Exception as e:
                    logger.error(f"Error playing exit message: {e}")

            if cap is not None:
                cap.release()
                logger.info("Video capture released.")
            
            try:
                cv2.destroyAllWindows()
                logger.info("OpenCV windows destroyed.")
            except Exception as e:
                logger.error(f"Error destroying OpenCV windows: {e}")
            
            logger.info("Walking Area Detection stopped.")

# --- Main Execution (SIMPLIFIED like object detection) ---
def main(shared_audio_handler_external=None):
    """
    Main function for walking area detection.
    
    Args:
        shared_audio_handler_external: Optional AudioFeedbackHandler instance
    """
    logger.info("Starting Walking Area Detection Application...")

    if not shared_audio_handler_external:
        print("❌ AudioFeedbackHandler is required for walking area detection.")
        logger.error("AudioFeedbackHandler is required for walking area detection.")
        return

    # Pass the shared audio handler directly to the detector
    try:
        detector = WalkingAreaDetectorBatch(shared_audio_handler=shared_audio_handler_external)
        detector.run()
    except RuntimeError as e:
        logger.error(f"Runtime error during WalkingAreaDetectorBatch initialization or run: {e}")
        if shared_audio_handler_external:
             shared_audio_handler_external.speak("Critical error starting walking area system. Please check logs.")
    except Exception as e:
        logger.exception("Unhandled exception during WalkingAreaDetectorBatch execution.")
        if shared_audio_handler_external:
             shared_audio_handler_external.speak("An unexpected error occurred in the walking area system.")
    finally:
        logger.info("Walking Area Detection Application finished.")

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    # Create audio handler for standalone mode
    test_audio_handler = None
    try:
        if AUDIO_ENABLED and AUDIO_AVAILABLE:
            test_audio_handler = AudioFeedbackHandler()
            if not test_audio_handler.engine:
                print("❌ Failed to initialize audio. Walking area detection cannot start.")
                exit()
        else:
            print("❌ Audio not available. Walking area detection requires audio feedback.")
            exit()
            
        main(shared_audio_handler_external=test_audio_handler)
        
    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
    finally:
        if test_audio_handler:
            test_audio_handler.stop()
        cv2.destroyAllWindows()
        print("Standalone test finished.")

# --- END OF FILE walking_area_batch.py ---