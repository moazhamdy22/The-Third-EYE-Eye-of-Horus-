# --- START OF FILE main_full_obstacle_integrated_1.py ---

import cv2
import time
import logging
import argparse
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from dataclasses import dataclass
import os
import sys

# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- Import Core Modules ---
try:
    from object_detection_obstacle_only_1 import ObstacleDetector, Config as DetectorConfigBase
    from perspective_region_classifier_1 import PerspectiveRegionClassifier, RegionClassificationResult
    from new_distance_measurement_region_aware_1 import DistanceMeasurement, DistanceResult, get_default_object_heights
    from danger_level_points_1 import DangerLevelPointSystem, DangerAssessment
    from movement_suggestion_1 import MovementAdvisor, MovementSuggestion
    from audio_feedback_vision_assitant import AudioFeedbackHandler
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Failed to import one or more core modules: {e}")
    print("Please ensure all required .py files are in the same directory or Python path.")
    MODULES_AVAILABLE = False
    class ObstacleDetector: pass
    class DetectorConfigBase: pass
    class PerspectiveRegionClassifier: pass
    class RegionClassificationResult: pass
    class DistanceMeasurement: pass
    class DistanceResult: pass
    def get_default_object_heights(): return {}
    class DangerLevelPointSystem: pass
    class DangerAssessment: pass
    class MovementAdvisor: pass
    class MovementSuggestion: pass
    class AudioFeedbackHandler: pass

# --- Logging Setup ---
def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO, # Set to DEBUG for more verbose output
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("ultralytics").setLevel(logging.ERROR) # Quieter YOLO logs
    logger = logging.getLogger("UnifiedSystem")
    return logger

logger = setup_logging()

# --- Configuration Class ---
class Config:
    """Central configuration for the integrated system."""
    # --- Video / Frame ---
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    VIDEO_SOURCE: Any = 1 # 0 for webcam, or path to video file
    TARGET_FPS: int = 15 # Target FPS for processing loop

    # --- Detector ---
    # Inherit from detector's base config if available, provide defaults otherwise
    DETECTOR_MODEL: str = getattr(DetectorConfigBase, 'MODEL_PATH', "yolov8n.pt")
    DETECTOR_DEVICE: str = getattr(DetectorConfigBase, 'DEVICE', "cpu")
    DETECTOR_CONF: float = getattr(DetectorConfigBase, 'CONFIDENCE_THRESHOLD', 0.4)
    # IMPORTANT: Get obstacle categories from the detector's config source
    OBSTACLE_CATEGORIES: Dict = getattr(DetectorConfigBase, 'OBSTACLE_CATEGORIES', {})

    # --- Classifier ---
    # TUNE THESE perspective parameters based on camera placement!
    HORIZON_Y_FACTOR: float = 0.5
    BOTTOM_Y_FACTOR: float = 1.0
    HORIZON_X_OFFSET_FACTOR: float = 0.1
    BOTTOM_X_OFFSET_FACTOR: float = 0.4
    CLOSE_ZONE_Y_FACTOR: float = 0.85
    MID_ZONE_Y_FACTOR: float = 0.65

    # --- Distance ---
    # CRITICAL: Calibrate this focal length!
    FOCAL_LENGTH_PIXELS: float = 200.0
    USE_ULTRASONIC: bool = True # Set to False to disable
    ULTRASONIC_PORT: Optional[str] = "COM10" # Set port ('COM3', '/dev/ttyACM0') or None to auto-detect
    ULTRASONIC_UNKNOWN_THRESHOLD_M: float = 1.5  # Max distance for ultrasonic to trigger 'unknown'

    # --- Audio ---
    AUDIO_CHANGE_THRESHOLD_DISTANCE: float = 0.75 # Meters change needed to re-announce distance
    AUDIO_COOLDOWN_SECONDS: float = 4.0 # Min time between announcing same object type
    AUDIO_RATE_OBSTACLE = 170  # Words per minute (slightly faster than scene system)

# --- Main System Class ---
class UnifiedSystem:
    def __init__(self, config: Config, shared_audio_handler: Optional[AudioFeedbackHandler] = None):
        self.config = config
        self.frame_width = config.FRAME_WIDTH
        self.frame_height = config.FRAME_HEIGHT
        logger.info("Initializing Unified Navigation System (Obstacles)...")
        
        if not MODULES_AVAILABLE:
            raise ImportError("One or more required modules failed to import for UnifiedSystem.")

        # Store loaded components
        self.detector: Optional[ObstacleDetector] = None
        self.classifier: Optional[PerspectiveRegionClassifier] = None
        self.measurer: Optional[DistanceMeasurement] = None
        self.danger_assessor: Optional[DangerLevelPointSystem] = None
        self.movement_advisor: Optional[MovementAdvisor] = None
        
        # --- Audio Initialization (SIMPLIFIED FURTHER) ---
        self.audio_handler = shared_audio_handler

        # SIMPLIFIED audio state tracking
        self.last_announced_class = None
        self.last_announcement_time = 0.0
        self.announcement_cooldown = 2.0  # Reduced cooldown for more responsive updates
        
        # Track obstacles - SIMPLIFIED
        self.current_obstacles = {}  # {class_name: {"region": str, "distance": float, "level": int, "first_detected": float}}
        self.last_announced_states = {}  # {class_name: {"region": str, "distance": float, "level": int}}
        self.min_detection_time = 1.0  # Reduced to 1 second for faster response
        self.update_threshold_distance = 0.5  # Reduced threshold for more sensitive distance updates
        
        # Remove complex noise reduction - keep it simple
        self.consecutive_detection_threshold = 2  # Only need 2 consecutive detections
        self.frame_skip_count = 0
        self.audio_frame_skip = 1  # Process audio every 2nd frame instead of every 3rd
        
        # Track if we need to announce "continue" when obstacles disappear
        self.had_obstacles_last_frame = False
        
        # FPS calculation
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.fps_start_time: float = time.monotonic()

        # --- Initialize Modules ---
        try:
            # 1. Detector
            logger.info("Initializing Detector...")
            self.detector = ObstacleDetector(
                model_path=config.DETECTOR_MODEL,
                device=config.DETECTOR_DEVICE,
                confidence_threshold=config.DETECTOR_CONF
            )
            # Critical: Get data needed by other modules
            self.detector_class_names = self.detector.model_names
            self.obstacle_categories = config.OBSTACLE_CATEGORIES # Use config value directly
            logger.info("Detector Initialized.")

            # 2. Classifier
            logger.info("Initializing Classifier...")
            self.classifier = PerspectiveRegionClassifier(
                frame_width=config.FRAME_WIDTH,
                frame_height=config.FRAME_HEIGHT,
                horizon_y_factor=config.HORIZON_Y_FACTOR,
                bottom_y_factor=config.BOTTOM_Y_FACTOR,
                horizon_x_offset_factor=config.HORIZON_X_OFFSET_FACTOR,
                bottom_x_offset_factor=config.BOTTOM_X_OFFSET_FACTOR,
                close_zone_y_factor=config.CLOSE_ZONE_Y_FACTOR,
                mid_zone_y_factor=config.MID_ZONE_Y_FACTOR
            )
            logger.info("Classifier Initialized.")

            # 3. Distance Measurer
            logger.info("Initializing Distance Measurer...")
            obj_heights = get_default_object_heights()
            if 'default' not in obj_heights: obj_heights['default'] = 1.0 # Ensure fallback
            self.measurer = DistanceMeasurement(
                focal_length=config.FOCAL_LENGTH_PIXELS,
                class_names=list(self.detector_class_names.values()),
                object_heights_m=obj_heights,
                use_ultrasonic=config.USE_ULTRASONIC,
                ultrasonic_port=config.ULTRASONIC_PORT
            )
            logger.info("Distance Measurer Initialized.")

            # 4. Danger Assessor
            logger.info("Initializing Danger Assessor...")
            self.danger_assessor = DangerLevelPointSystem(
                obstacle_categories=self.obstacle_categories,
                detector_class_names=self.detector_class_names
            )
            logger.info("Danger Assessor Initialized.")

            # 5. Movement Advisor
            logger.info("Initializing Movement Advisor...")
            self.movement_advisor = MovementAdvisor()
            logger.info("Movement Advisor Initialized.")

            # 6. Audio Handler Setup (SIMPLIFIED)
            if self.audio_handler:
                logger.info("Using shared AudioFeedbackHandler for UnifiedSystem (Obstacles).")
            else:
                logger.warning("No shared_audio_handler provided to UnifiedSystem. Audio will be disabled.")

        except Exception as e:
            logger.exception(f"CRITICAL ERROR during module initialization: {e}")
            raise RuntimeError("System initialization failed.") from e

        logger.info("Unified System Initialized Successfully.")
        if shared_audio_handler:
            complete_msg = "Obstacle detection system is ready."
            shared_audio_handler.speak(complete_msg)

    def _calculate_fps(self):
        """Updates the FPS calculation."""
        self.frame_count += 1
        now = time.monotonic()
        elapsed = now - self.fps_start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = now

    def _get_default_safe_assessment(self) -> DangerAssessment:
        """Returns a default safe assessment when no obstacles are relevant."""
        return DangerAssessment(
            level=1,
            description=DangerLevelPointSystem.DESCRIPTIONS[1],
            color=DangerLevelPointSystem.COLORS[1],
            requires_action=False,
            total_points=0
        )

    def _update_current_obstacles(self, current_frame_states: Dict):
        """Track only currently detected obstacles in this frame with noise reduction."""
        current_time = time.monotonic()
        
        # Update current obstacles with frame data
        frame_obstacle_names = set(current_frame_states.keys())
        
        # Add new obstacles or update existing ones
        for class_name, state in current_frame_states.items():
            if class_name in self.current_obstacles:
                # Update existing obstacle and increment consistency counter
                self.current_obstacles[class_name].update({
                    "region": state.get("region"),
                    "distance": state.get("distance"),
                    "level": state.get("level"),
                    "last_seen": current_time,
                    "consecutive_detections": self.current_obstacles[class_name].get("consecutive_detections", 0) + 1
                })
            else:
                # New obstacle detected
                self.current_obstacles[class_name] = {
                    "region": state.get("region"),
                    "distance": state.get("distance"),
                    "level": state.get("level"),
                    "first_detected": current_time,
                    "last_seen": current_time,
                    "consecutive_detections": 1
                }
        
        # Remove obstacles not seen in current frame
        to_remove = []
        for class_name in self.current_obstacles:
            if class_name not in frame_obstacle_names:
                to_remove.append(class_name)
        
        for class_name in to_remove:
            logger.debug(f"Removing obstacle no longer detected: {class_name}")
            del self.current_obstacles[class_name]
            # Also remove from last announced states
            if class_name in self.last_announced_states:
                del self.last_announced_states[class_name]

    def _should_announce_obstacle(self, class_name: str, current_state: Dict) -> Tuple[bool, str]:
        """Simplified logic: announce new obstacles and significant updates only."""
        if not self.audio_handler:
            return False, "none"
            
        current_time = time.monotonic()
        
        # Basic cooldown to prevent spam
        if current_time - self.last_announcement_time < self.announcement_cooldown:
            return False, "cooldown"
        
        # Check if this is a completely new obstacle
        if class_name not in self.last_announced_states:
            return True, "new"
        
        # Check for significant changes in existing obstacles
        last_state = self.last_announced_states[class_name]
        
        # Check for region change (left/right/center change)
        current_region = current_state.get("region", "")
        last_region = last_state.get("region", "")
        if current_region != last_region:
            return True, "location_change"
        
        # Check for significant distance change
        current_distance = current_state.get("distance")
        last_distance = last_state.get("distance")
        if (current_distance is not None and last_distance is not None and 
            abs(current_distance - last_distance) >= self.update_threshold_distance):
            return True, "distance_change"
        
        # Check for danger level increase (more urgent)
        current_level = current_state.get("level", 0)
        last_level = last_state.get("level", 0)
        if current_level > last_level:
            return True, "danger_increase"
        
        return False, "no_change"

    def _trigger_audio_update(self, current_frame_states: Dict):
        """Simplified audio update - announce once and update on changes."""
        if not self.audio_handler:
            return
        
        # Simple frame skipping
        self.frame_skip_count += 1
        if self.frame_skip_count <= self.audio_frame_skip:
            self._update_current_obstacles(current_frame_states)
            return
        
        self.frame_skip_count = 0
        
        # Update obstacle tracking
        self._update_current_obstacles(current_frame_states)
        
        # Check if obstacles disappeared and we should announce "continue"
        current_has_obstacles = len(current_frame_states) > 0
        if self.had_obstacles_last_frame and not current_has_obstacles:
            # Obstacles disappeared - announce continue
            current_time = time.monotonic()
            if current_time - self.last_announcement_time >= self.announcement_cooldown:
                self.audio_handler.speak("Path clear. Continue.")
                self.last_announcement_time = current_time
                logger.info("AUDIO: Path cleared - announced continue")
        
        self.had_obstacles_last_frame = current_has_obstacles
        
        # Get ready obstacles (simplified criteria)
        ready_obstacles = self._get_ready_to_announce_obstacles()
        if not ready_obstacles:
            return

        # Find the most important obstacle to announce
        highest_class = None
        highest_level = 0
        highest_state = None
        announcement_type = "none"
        
        for class_name, state in ready_obstacles.items():
            level = state.get("level", 0)
            should_announce, msg_type = self._should_announce_obstacle(class_name, state)
            
            # Prioritize by urgency level, then by announcement type
            if should_announce and (level > highest_level or 
                                  (level == highest_level and msg_type in ["new", "danger_increase"])):
                highest_level = level
                highest_class = class_name
                highest_state = state
                announcement_type = msg_type
        
        # Make announcement
        if highest_class and highest_state and announcement_type != "none":
            message = self._generate_audio_message(
                highest_class, 
                highest_state.get("region"), 
                highest_state.get("distance"), 
                highest_level, 
                getattr(self, 'last_suggestion', None),
                announcement_type
            )
            
            logger.info(f"AUDIO ({announcement_type.upper()}): {highest_class} - Level {highest_level}")
            self.audio_handler.speak(message)
            
            # Update tracking
            self.last_announced_class = highest_class
            self.last_announcement_time = time.monotonic()
            self.last_announced_states[highest_class] = {
                "region": highest_state.get("region"),
                "distance": highest_state.get("distance"),
                "level": highest_state.get("level")
            }

    def _generate_audio_message(self, class_name: str, region: Optional[str],
                               distance: Optional[float], level: int,
                               suggestion: Optional[MovementSuggestion],
                               announcement_type: str = "new") -> str:
        """Generate concise audio message based on announcement type."""
        if not class_name:
            class_name = "Obstacle"

        # Location description
        if region and "Center" in region:
            location = "ahead"
        elif region and "Left" in region:
            location = "to your left"
        elif region and "Right" in region:
            location = "to your right"
        else:
            location = "nearby"

        # Distance info
        dist_info = ""
        if distance is not None and distance < 3.0:
            dist_info = f" at {distance:.1f} meters"

        # Generate message based on announcement type
        if announcement_type == "new":
            # First time detection
            if level >= 4:
                message = f"Critical! {class_name} {location}{dist_info}"
            elif level >= 3:
                message = f"Warning! {class_name} {location}{dist_info}"
            else:
                message = f"{class_name} detected {location}{dist_info}"
                
        elif announcement_type == "location_change":
            message = f"{class_name} moved {location}{dist_info}"
            
        elif announcement_type == "distance_change":
            message = f"{class_name} now {location}{dist_info}"
            
        elif announcement_type == "danger_increase":
            if level >= 4:
                message = f"Critical! {class_name} closer {location}{dist_info}"
            else:
                message = f"Warning! {class_name} approaching {location}{dist_info}"
        else:
            message = f"{class_name} {location}{dist_info}"

        # Add movement suggestion for important obstacles
        if suggestion and level >= 3:  # Only for warning/critical levels
            action = suggestion.primary_action.lower()
            
            if action == "stop":
                message += ". Stop immediately"
                if suggestion.direction:
                    message += f", then turn {suggestion.direction.lower()}"
            elif action == "turn":
                if suggestion.direction:
                    message += f". Turn {suggestion.direction.lower()} now"
            elif action == "slow":
                message += ". Slow down"
                if suggestion.direction:
                    message += f" and move {suggestion.direction.lower()}"

        return message.strip()

    def _get_ready_to_announce_obstacles(self) -> Dict:
        """Simplified criteria for ready obstacles."""
        current_time = time.monotonic()
        ready_obstacles = {}
        
        for class_name, data in self.current_obstacles.items():
            time_since_first = current_time - data["first_detected"]
            consecutive_detections = data.get("consecutive_detections", 0)
            
            # Simplified criteria: present for min time AND consistent detections
            if (time_since_first >= self.min_detection_time and 
                consecutive_detections >= self.consecutive_detection_threshold):
                ready_obstacles[class_name] = data
        
        return ready_obstacles

    def _print_frame_summary(self, results: List[Dict], suggestion: Optional[MovementSuggestion]):
        """Enhanced frame summary with noise reduction info."""
        if not results and suggestion is None:
            return

        print(f"\n--- Frame Summary (Time: {time.time():.2f}, FPS: {self.fps:.1f}) ---")

        if not results:
            print("  No obstacles detected in this frame.")
        else:
            print("  Detected Obstacles:")
            ready_obstacles = self._get_ready_to_announce_obstacles()
            
            for i, res in enumerate(results):
                class_name = res.get('class_name', 'N/A')
                confidence = res.get('confidence', 0.0)
                region_res = res.get('region')
                dist_res = res.get('distance')
                danger_res = res.get('danger')

                conf_str = f"{confidence:.2f}"
                type_str = self.danger_assessor.class_to_type_map.get(class_name.lower(), "Unknown") if self.danger_assessor else "N/A"
                
                dist_str = "Dist:? m"
                dist_method = ""
                if dist_res:
                    dist_val = dist_res.distance
                    dist_method = f"({dist_res.method[0]})" if dist_res.method != 'none' else ""
                    dist_str = f"Dist:{dist_val:.1f}m" if dist_val != float('inf') else "Dist:? m"

                region_str = f"Reg:{region_res.location_description}" if region_res else "Reg:N/A"
                is_path_str = f"(Path:{'Y' if region_res and region_res.is_path_obstacle else 'N'})" if region_res else ""

                danger_str = f"Danger:L{danger_res.level}" if danger_res else "Danger:N/A"
                points_str = f"(P:{danger_res.total_points})" if danger_res else ""
                
                # Add current detection info with consistency tracking
                detection_info = ""
                if class_name in self.current_obstacles:
                    time_present = time.monotonic() - self.current_obstacles[class_name]["first_detected"]
                    consecutive_det = self.current_obstacles[class_name].get("consecutive_detections", 0)
                    is_ready = class_name in ready_obstacles
                    detection_info = f" [Present:{time_present:.1f}s, Consecutive:{consecutive_det}, Ready:{'Y' if is_ready else 'N'}]"

                print(f"    {i+1}. {class_name} ({conf_str}) | Type:{type_str} | {dist_str}{dist_method} | {region_str}{is_path_str} | {danger_str}{points_str}{detection_info}")

        # Show currently tracked obstacles
        if self.current_obstacles:
            print(f"  Currently Tracked: {list(self.current_obstacles.keys())}")
            ready_obstacles = self._get_ready_to_announce_obstacles()
            if ready_obstacles:
                print(f"  Ready for Audio: {list(ready_obstacles.keys())}")

        if suggestion:
            print("  Movement Suggestion:")
            sugg_str = f"    Action: {suggestion.primary_action.upper()}"
            if suggestion.direction: sugg_str += f" {suggestion.direction.upper()}"
            if suggestion.secondary_action: sugg_str += f" -> {suggestion.secondary_action.upper()}"
            print(sugg_str)
            print(f"    Urgency: {suggestion.urgency} | Desc: {suggestion.description}")
        elif not results:
            print("  Movement Suggestion: Proceed.")

        print("----------------------------------------------------")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes a single frame through the entire pipeline."""
        self._calculate_fps()
        vis_frame = frame.copy() # For drawing

        # --- Step 1: Detection ---
        # Returns List[Tuple[x1, y1, x2, y2, confidence, class_name]]
        raw_detections = self.detector.detect(frame) if self.detector else []

        # --- Step 2-4: Classification, Measurement, Assessment per detection ---
        frame_results = [] # Store results for each detected object
        current_frame_states = {} # Track state of path obstacles in this frame
        self.highest_danger_assessment_frame = self._get_default_safe_assessment() # Reset for frame

        # --- Check Ultrasonic Independently (FIXED) ---
        ultrasonic_only_triggered = False
        unknown_obstacle_assessment = None
        ultrasonic_distance = None
        
        if self.measurer and self.measurer.ultrasonic:
            ultrasonic_distance = self.measurer.ultrasonic.get_distance()
            
        # Only trigger unknown obstacle if ultrasonic detects AND it's within threshold AND no center YOLO detection
        if (ultrasonic_distance is not None and 
            ultrasonic_distance < self.config.ULTRASONIC_UNKNOWN_THRESHOLD_M):
            
            logger.debug(f"Ultrasonic reading: {ultrasonic_distance:.2f}m")
            center_yolo_detected = False
            
            # Check if there's already a YOLO detection in center region
            if raw_detections and self.classifier:
                for i, det in enumerate(raw_detections):
                    region_res_check = self.classifier.classify(det[:4])
                    if "Center" in region_res_check.location_description:
                        center_yolo_detected = True
                        logger.debug(f"Center YOLO detection found: {det[5]}")
                        break
            
            # Only create unknown obstacle if no center YOLO detection exists
            if not center_yolo_detected:
                logger.info(f"Ultrasonic detected unknown obstacle at {ultrasonic_distance:.2f}m (no center YOLO detection)")
                ultrasonic_only_triggered = True
                if self.danger_assessor:
                    unknown_obstacle_assessment = self.danger_assessor.assess_danger(
                        distance=ultrasonic_distance,
                        region_name="Center-Close",
                        class_name="Unknown Obstacle"
                    )
                    frame_results.append({
                        "bbox": None,
                        "confidence": 1.0,
                        "class_name": "Unknown Obstacle",
                        "region": RegionClassificationResult(True, "Center", "Close", "Center-Close", 
                                 (self.frame_width//2, int(self.frame_height*0.9))),
                        "distance": DistanceResult(distance=ultrasonic_distance, confidence=0.8, 
                                  method='ultrasonic', ultrasonic_distance=ultrasonic_distance),
                        "danger": unknown_obstacle_assessment
                    })
                    if unknown_obstacle_assessment.level > self.highest_danger_assessment_frame.level:
                        self.highest_danger_assessment_frame = unknown_obstacle_assessment
                    
                    # Add unknown obstacle to current frame states for proper tracking
                    current_frame_states["Unknown Obstacle"] = {
                        "region": "Center-Close",
                        "distance": ultrasonic_distance,
                        "level": unknown_obstacle_assessment.level
                    }
            else:
                logger.debug("Ultrasonic reading ignored - center YOLO detection exists")
        else:
            if ultrasonic_distance is not None:
                logger.debug(f"Ultrasonic reading {ultrasonic_distance:.2f}m is above threshold {self.config.ULTRASONIC_UNKNOWN_THRESHOLD_M}m")

        # Continue with regular YOLO detections processing...
        for det in raw_detections:
            x1, y1, x2, y2, confidence, class_name = det
            bbox = (x1, y1, x2, y2)
            pixel_height = float(y2 - y1)

            # --- Classify Region ---
            region_result: Optional[RegionClassificationResult] = None
            region_desc = "Unknown"
            is_path_obstacle = False
            if self.classifier:
                region_result = self.classifier.classify(bbox)
                region_desc = region_result.location_description
                is_path_obstacle = region_result.is_path_obstacle

            # --- Measure Distance ---
            distance_result: Optional[DistanceResult] = None
            distance_val = None # Store finite distance value
            if self.measurer:
                distance_result = self.measurer.measure_distance(pixel_height, class_name, region_desc)
                if distance_result and distance_result.distance != float('inf'):
                     distance_val = distance_result.distance

            # --- Assess Danger ---
            danger_assessment: Optional[DangerAssessment] = None
            if self.danger_assessor:
                danger_assessment = self.danger_assessor.assess_danger(distance_val, region_desc, class_name)

            # Store results temporarily
            frame_results.append({
                "bbox": bbox, "confidence": confidence, "class_name": class_name,
                "region": region_result, "distance": distance_result, "danger": danger_assessment
            })

            # --- Process only PATH OBSTACLES for navigation/audio ---
            if is_path_obstacle and danger_assessment:
                logger.debug(f"Path Obstacle: {class_name} in {region_desc}, Level {danger_assessment.level}, Dist {distance_val}")
                # Update highest danger for the frame
                if danger_assessment.level > self.highest_danger_assessment_frame.level:
                    self.highest_danger_assessment_frame = danger_assessment

                # Store current state for audio comparison
                current_state = {
                    "region": region_desc,
                    "distance": distance_val, # Store actual distance
                    "level": danger_assessment.level
                }
                current_frame_states[class_name] = current_state


        # --- Step 5: Determine Overall Movement Suggestion ---
        highest_danger = self.highest_danger_assessment_frame
        self.last_suggestion = self.movement_advisor.get_suggestion(
            danger_level=highest_danger.level,
            # Use region/distance/class from the specific object causing highest danger
            is_path_obstacle=True if highest_danger.level > 1 else False, # Assume level > 1 means a path obstacle caused it
            region_name=highest_danger.region,
            distance=highest_danger.distance,
            class_name=highest_danger.class_name
        ) if self.movement_advisor and highest_danger else None

        # --- Step 6: Generate Audio Feedback (SIMPLIFIED) ---
        self._trigger_audio_update(current_frame_states)

        # Call new frame summary printing
        self._print_frame_summary(frame_results, self.last_suggestion)

        # --- Step 7: Visualization ---
        if self.classifier:
            vis_frame = self.classifier.draw_perspective_regions(vis_frame) # Draw guidelines

        for res in frame_results: # Draw all detections, color by path status
            self._draw_detection(vis_frame, res)

        if self.last_suggestion: # Draw movement overlay
            vis_frame = self._add_movement_overlay(vis_frame)

        cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Draw FPS

        return vis_frame

    def _draw_detection(self, frame: np.ndarray, result_dict: Dict):
        """Draws a single detection with region/distance/danger info."""
        bbox = result_dict.get('bbox')
        if bbox is None and result_dict.get('class_name') == "Unknown Obstacle":
            danger_res = result_dict.get('danger')
            color = danger_res.color if danger_res else (0,0,255)
            text = f"UNKNOWN OBSTACLE (US)"
            dist_res = result_dict.get('distance')
            if dist_res and dist_res.distance != float('inf'):
                text += f" @ {dist_res.distance:.1f}m"
            if danger_res:
                text += f" L:{danger_res.level}"
            
            (lw, lh), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x = self.frame_width // 2 - lw // 2
            text_y = self.frame_height - 30
            cv2.rectangle(frame, (text_x - 5, text_y - lh - base - 5),
                         (text_x + lw + 5, text_y + base + 5), (0,0,0), cv2.FILLED)
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return

        x1, y1, x2, y2 = map(int, result_dict['bbox'])
        class_name = result_dict['class_name']
        region_res: Optional[RegionClassificationResult] = result_dict['region']
        dist_res: Optional[DistanceResult] = result_dict['distance']
        danger_res: Optional[DangerAssessment] = result_dict['danger']

        color = (0, 255, 0) # Default Green (off-path)
        is_obstacle = False
        if region_res and region_res.is_path_obstacle and danger_res:
            color = danger_res.color # Use danger color for path obstacles
            is_obstacle = True

        # Draw BBox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_obstacle else 1)

        # Create Labels (only show details for path obstacles or if no danger info available)
        label_lines = [f"{class_name}"]
        if is_obstacle or danger_res is None: # Show details if it's a path obstacle or if danger assessment failed
            if region_res: label_lines.append(f"{region_res.location_description}")
            if dist_res:
                 dist_str = f"{dist_res.distance:.1f}m" if dist_res.distance != float('inf') else "? m"
                 label_lines.append(f"{dist_str} ({dist_res.method[0]})")
            if danger_res: label_lines.append(f"L:{danger_res.level} P:{danger_res.total_points}") # Show Level/Points

        # Draw multi-line label
        label_y = y1 - 7
        for i, line in enumerate(reversed(label_lines)):
             (lw, lh), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
             current_y = label_y - i * (lh + 5)
             cv2.rectangle(frame, (x1, current_y - lh - base + 1), (x1 + lw, current_y + base), (0,0,0), cv2.FILLED)
             cv2.putText(frame, line, (x1, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    def _add_movement_overlay(self, frame: np.ndarray):
        """Adds the movement suggestion overlay panel."""
        if not self.last_suggestion: return frame

        height, width = frame.shape[:2]
        panel_height = 80
        panel_y = height - panel_height

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (width, height), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Get color based on urgency
        color = self.movement_advisor.get_urgency_color(self.last_suggestion.urgency)

        # Format suggestion text
        primary_text = f"ACTION: {self.last_suggestion.primary_action.upper()}"
        if self.last_suggestion.direction: primary_text += f" {self.last_suggestion.direction.upper()}"
        if self.last_suggestion.secondary_action: primary_text += f" -> {self.last_suggestion.secondary_action.upper()}"
        secondary_text = f"Urgency: {self.last_suggestion.urgency} | Desc: {self.last_suggestion.description}"

        # Draw suggestion text
        cv2.putText(frame, primary_text, (10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(frame, secondary_text, (10, panel_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def close(self):
        """Cleanup resources."""
        logger.info("Shutting down Unified System (Obstacles)...")
        try:
            if hasattr(self, 'measurer') and self.measurer:
                self.measurer.close()
                logger.info("Distance measurer closed.")
        except Exception as e:
            logger.error(f"Error during UnifiedSystem (Obstacles) shutdown: {e}")
        finally:
            logger.info("Unified System (Obstacles) instance cleanup complete.")

# --- Main Execution (SIMPLIFIED like object detection) ---
def main(shared_audio_handler: Optional[AudioFeedbackHandler] = None):
    parser = argparse.ArgumentParser(description="Unified Navigation Assistant System")
    parser.add_argument('--source', default=Config.VIDEO_SOURCE, help='Video source (2 for webcam or path)')
    parser.add_argument('--device', default=Config.DETECTOR_DEVICE, help='Computation device (cpu, cuda)')
    parser.add_argument('--focal', type=float, default=Config.FOCAL_LENGTH_PIXELS, help='Camera focal length (pixels)')
    parser.add_argument('--no-ultrasonic', action='store_true', help='Disable ultrasonic sensor')

    args = parser.parse_args()

    if not shared_audio_handler:
        print("❌ AudioFeedbackHandler is required for obstacle detection system.")
        logger.error("AudioFeedbackHandler is required for obstacle detection system.")
        return

    # Create config object
    config = Config()
    
    video_source_arg = args.source
    try:
        config.VIDEO_SOURCE = int(video_source_arg)
        logger.info(f"Interpreted video source '{video_source_arg}' as camera index {config.VIDEO_SOURCE}.")
    except ValueError:
        config.VIDEO_SOURCE = str(video_source_arg)
        logger.info(f"Interpreted video source '{video_source_arg}' as file path.")

    config.DETECTOR_DEVICE = args.device
    config.FOCAL_LENGTH_PIXELS = args.focal
    config.USE_ULTRASONIC = not args.no_ultrasonic

    system: Optional[UnifiedSystem] = None
    cap: Optional[cv2.VideoCapture] = None
    user_quit = False

    if not MODULES_AVAILABLE:
        logger.error("Cannot run UnifiedSystem (Obstacles): Core modules are missing.")
        if shared_audio_handler:
            shared_audio_handler.speak("Error: Obstacle detection system modules are missing.")
        return

    try:
        if shared_audio_handler:
            shared_audio_handler.speak("Initializing obstacle detection system. Please wait.")
            
        system = UnifiedSystem(config, shared_audio_handler=shared_audio_handler)

        logger.info(f"Initializing video capture from source: {config.VIDEO_SOURCE}")
        cap = cv2.VideoCapture(config.VIDEO_SOURCE,cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {config.VIDEO_SOURCE}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        logger.info("Video capture initialized.")

        target_frame_time = 1.0 / config.TARGET_FPS if config.TARGET_FPS > 0 else 0

        while True:
            loop_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or cannot read frame.")
                break

            frame_resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            processed_frame = system.process_frame(frame_resized)
            cv2.imshow("Navigation Assistant", processed_frame)

            loop_end = time.monotonic()
            elapsed = loop_end - loop_start
            wait_time_ms = max(1, int((target_frame_time - elapsed) * 1000)) if target_frame_time > 0 else 1

            if cv2.waitKey(wait_time_ms) & 0xFF == ord('0'):
                logger.info("Quit signal received in Obstacle System.")
                user_quit = True
                break

    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
        if shared_audio_handler:
            shared_audio_handler.speak("Error in obstacle detection system.")
            time.sleep(1.0)
    finally:
        logger.info("Starting cleanup for Obstacle System...")
        
        # Clean exit message like other systems
        if shared_audio_handler and user_quit:
            shared_audio_handler.speak("Obstacle detection system has been stopped.")
            time.sleep(2.0)  # Allow message to complete
        
        if cap is not None:
            cap.release()
            logger.info("Video capture released.")
        if system is not None:
            system.close()
        
        cv2.destroyAllWindows()
        logger.info("Obstacle System cleanup complete.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    test_audio_handler = None
    try:
        if AudioFeedbackHandler:
            test_audio_handler = AudioFeedbackHandler()
            if not test_audio_handler.engine:
                print("❌ Failed to initialize audio. Obstacle detection cannot start.")
                exit()
        else:
            print("❌ Audio not available. Obstacle detection requires audio feedback.")
            exit()
            
        main(shared_audio_handler=test_audio_handler)
        
    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
    finally:
        if test_audio_handler:
            test_audio_handler.stop()
        cv2.destroyAllWindows()
        print("Standalone test finished.")

# --- END OF FILE main_full_obstacle_integrated_1.py ---