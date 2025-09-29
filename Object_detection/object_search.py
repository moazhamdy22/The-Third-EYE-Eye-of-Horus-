# --- START OF FILE object_search.py ---

import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import logging
from collections import defaultdict
import os
import sys

# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from object_detection_realtime import RealtimeSpatialAwareness # Ensure AudioFeedbackHandler is accessible if not explicitly imported
from audio_feedback_vision_assitant import AudioFeedbackHandler # Explicit import for clarity

# Configure basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configured in main
logger = logging.getLogger(__name__)

# Silence Ultralytics' own verbose logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)


def _confirm_selection(candidate_name, audio_handler: AudioFeedbackHandler, window_name="Object Selection"):
    """Asks user to confirm a selection with '5' in CV window."""
    prompt_message = f"Did you mean '{candidate_name}'?"
    confirmation_prompt = f"{prompt_message} Press '5' to confirm, or any other key to search again"
    
    logger.info(prompt_message)
    audio_handler.speak(prompt_message + " Press 5 to confirm, or any other key to search again.")
    
    # Create a blank image for the confirmation window
    confirm_img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv2.putText(confirm_img, confirmation_prompt, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.imshow(window_name, confirm_img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('5'):
            logger.info(f"User confirmed selection: {candidate_name}")
            return candidate_name
        elif key != 255:  # Any other key
            logger.info("User rejected selection.")
            audio_handler.speak("Okay, let's try again.")
            return None

def _get_target_object_from_user(available_objects, audio_handler: AudioFeedbackHandler):
    """Handles object selection through CV window interface."""
    target_object_name = None
    current_input = ""
    window_name = "Object Selection"
    
    # Sort objects for display
    common_object_order = [
        "person", "cell phone", "bottle", "cup", "chair", "table", "laptop",
        "keyboard", "mouse", "book", "tv", "remote", "sofa", "bed",
        "dining table", "refrigerator", "oven", "microwave", "sink", "toilet",
        "door", "window", "backpack", "handbag", "suitcase", "umbrella",
        "car", "bicycle", "motorcycle", "bus", "train", "truck", "cat", "dog",
        "stop sign", "traffic light", "fire hydrant", "parking meter", "bench",
        "apple", "banana", "orange", "broccoli", "carrot", "pizza", "donut", "cake",
        "tie", "scissors", "toothbrush", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "wine glass",
        "fork", "knife", "spoon", "bowl", "vase", "potted plant", "clock",
        "teddy bear", "hair drier", "bird", "horse", "sheep", "cow", "elephant",
        "bear", "zebra", "giraffe", "frisbee", "skis", "snowboard", "airplane", "boat",
    ]
    order_map = {name: i for i, name in enumerate(common_object_order)}
    def sort_key(item_name): return order_map.get(item_name, len(common_object_order)), item_name
    display_sorted_objects = sorted(available_objects, key=sort_key)
    
    def create_selection_window(input_text="", message=""):
        img = np.ones((720, 1000, 3), dtype=np.uint8) * 255
        y_offset = 50
        
        # Title
        cv2.putText(img, "Object Search - Available Objects", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Input box
        cv2.rectangle(img, (20, y_offset+20), (980, y_offset+60), (200, 200, 200), -1)
        cv2.putText(img, f"Input: {input_text}", (30, y_offset+50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Message/Error display
        if message:
            cv2.putText(img, message, (20, y_offset+90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Object list
        items_per_row = 3
        max_width = 300
        y_start = y_offset + 120
        
        for i, obj in enumerate(display_sorted_objects):
            x = 20 + (i % items_per_row) * max_width
            y = y_start + (i // items_per_row) * 30
            text = f"{i+1}. {obj}"
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Instructions
        instructions = [
            "Type numbers to select object",
            "Backspace to delete",
            "Enter to confirm selection",
            "ESC to clear input",
            "0 to quit"
        ]
        
        for i, instr in enumerate(instructions):
            cv2.putText(img, instr, (20, 680 + i*20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return img

    audio_handler.speak("Please select an object by typing its number or name.")
    
    while True:
        img = create_selection_window(current_input)
        cv2.imshow(window_name, img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('0') or key == ord('0'):
            return None
            
        elif key == 13:  # Enter key
            if current_input:
                # Process selection similarly to original code
                try:
                    # Try as number first
                    idx = int(current_input) - 1
                    if 0 <= idx < len(display_sorted_objects):
                        candidate = display_sorted_objects[idx]
                        confirmed = _confirm_selection(candidate, audio_handler, window_name)
                        if confirmed:
                            return confirmed
                except ValueError:
                    # Try as text search
                    matches = [obj for obj in display_sorted_objects 
                             if current_input.lower() in obj.lower()]
                    if len(matches) == 1:
                        confirmed = _confirm_selection(matches[0], audio_handler, window_name)
                        if confirmed:
                            return confirmed
                    elif len(matches) > 1:
                        # Show matches in new window for selection
                        current_input = ""  # Clear input
                        audio_handler.speak(f"Found {len(matches)} matches. Please type the number of your choice.")
                        
            current_input = ""
            
        elif key == 8:  # Backspace
            current_input = current_input[:-1]
        elif key == 27:  # ESC
            current_input = ""
        elif 32 <= key <= 126:  # Printable characters
            current_input += chr(key)


def run_object_search(
    audio_handler: AudioFeedbackHandler,
    camera_id=1,
    model_path="yolov8n.pt",
    device="cpu",
    conf=0.5,
    feedback_interval=4.0
):
    """
    Runs the targeted object search system using a shared audio handler.
    """
    logger.info("Initializing Object Search System...")
    detector = None
    cap = None

    if not audio_handler or not isinstance(audio_handler, AudioFeedbackHandler):
        logger.error("Invalid or missing AudioFeedbackHandler provided to run_object_search.")
        print("ERROR: AudioFeedbackHandler not available for object search.")
        return

    try:
        audio_handler.speak("Welcome to object search mode. Please select what you'd like to search for.")
        detector = RealtimeSpatialAwareness(
            model_path=model_path,
            device=device,
            confidence_threshold=conf
        )
        logger.info("RealtimeSpatialAwareness detector initialized for search.")

        # First get the target object before camera initialization
        all_model_objects = list(detector.labels.values())

        # Get target object first
        target_object_name = _get_target_object_from_user(all_model_objects, audio_handler)

        if target_object_name is None: # User cancelled or critical error during selection
            logger.info("No target object selected or selection cancelled. Exiting object search.")
            return

        logger.info(f"Target object confirmed: {target_object_name}")
        audio_handler.speak(f"Starting search for {target_object_name}. Press 1 key on the keyboard when you think you have reached it, or 0 to stop searching.")
        time.sleep(1) # Give a bit of time for the longer instruction to play

        # --- Initialize Video Capture ---
        logger.info(f"Attempting to open camera ID: {camera_id}")
        cap = cv2.VideoCapture(camera_id,cv2.CAP_DSHOW)  # Use DSHOW for better compatibility on Windows
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}.")
            if camera_id != 2:
                logger.warning("Trying default camera (ID 0)...")
                audio_handler.speak("Main camera not found. Trying default camera.")
                cap.open(0)
                if not cap.isOpened():
                    logger.critical("Default camera also failed.")
                    audio_handler.speak("Error: No camera found. Please check your camera connection.")
                    return
                else:
                    logger.info("Default camera opened successfully.")
                    audio_handler.speak("Default camera connected.")
            else:
                audio_handler.speak("Error: Camera not found. Please check connection.")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {w}x{h} @ {fps:.2f} FPS")
        # audio_handler.speak("Camera connected.") # Covered by previous messages

        # --- Search Loop ---
        last_feedback_time = 0
        window_name = f"Object Search: {target_object_name} - Press '1' if found, '0' to Quit"
        logger.info(f"Starting object search loop for: {target_object_name}")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Could not read frame from camera.")
                time.sleep(0.5) # Wait a bit before retrying
                if not cap.isOpened(): # Check if camera got disconnected
                    logger.error("Camera disconnected during search.")
                    audio_handler.speak("Error: Camera has been disconnected.")
                    break
                continue

            current_time = time.time()
            all_detections = detector.process_frame(frame)
            target_detections = [det for det in all_detections if det['class_name'].lower() == target_object_name.lower()]

            if current_time - last_feedback_time >= feedback_interval:
                if not target_detections:
                    if detector.last_announced_state.get(target_object_name): # If it was previously seen but now gone
                        audio_handler.speak(f"Lost sight of the {target_object_name}. Keep looking.")
                        detector.last_announced_state.pop(target_object_name, None) # Clear its state
                        detector.last_seen_time.pop(target_object_name, None)
                else: # Target is detected
                    feedback_message = detector.format_stateful_feedback(target_detections, current_time)
                    if feedback_message:
                        if not audio_handler.speaking: # Avoid interrupting itself for rapid updates
                            audio_handler.speak(feedback_message)
                last_feedback_time = current_time

            # --- Draw Overlays ---
            display_frame = frame.copy() # Start with a fresh frame copy
            target_found_this_frame = bool(target_detections)

            if target_found_this_frame:
                target_detections.sort(key=lambda d: {'near': 0, 'mid-distance': 1, 'far': 2}.get(d['distance'], 3))
                count = len(target_detections)
                for i, det in enumerate(target_detections):
                    x1, y1, x2, y2 = det["bbox"]
                    instance_color = detector.get_color_for_distance(det['distance'])
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), instance_color, 3)
                    label_text = f"TARGET: {det['class_name']}"
                    if count > 1: label_text += f" ({i+1}/{count})"
                    label_text += f" ({det['distance']})"
                    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; thickness = 1
                    (tw, th), bl = cv2.getTextSize(label_text, font, font_scale, thickness)
                    label_y_pos = max(y1 - 5, th + bl)
                    cv2.rectangle(display_frame, (x1, label_y_pos - th - bl), (x1 + tw + 4, label_y_pos), instance_color, cv2.FILLED)
                    cv2.putText(display_frame, label_text, (x1 + 2, label_y_pos - bl // 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            status_text = f"Searching for: {target_object_name}"
            status_color = (0, 220, 0) if target_found_this_frame else (200, 200, 200)
            if target_found_this_frame: status_text += f" - DETECTED ({len(target_detections)})"
            
            cv2.putText(display_frame, status_text, (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_frame, "Press '1' if Reached | '0' to Quit", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            quit_pressed = (key == ord('0') or key == ord('0'))
            found_pressed = (key == ord('1') or key == ord('1'))
            
            window_closed = False
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    window_closed = True
            except cv2.error:
                logger.warning("Could not get window property, assuming closed if error.")
                window_closed = True

            if quit_pressed or window_closed:
                msg = "Search window closed by user." if window_closed else "Quitting search."
                logger.info(msg)
                audio_handler.speak(msg + " Thank you for using object search.")
                if audio_handler.speaking: time.sleep(4)
                audio_handler.force_stop()
                break
            elif found_pressed:
                if target_found_this_frame and target_detections:
                    det = target_detections[0]
                    final_msg = f"Great! You've indicated you reached the {target_object_name}. It is currently seen as {det['distance']} and {det['position_desc']}. Search completed."
                    audio_handler.speak(final_msg)
                    while audio_handler.speaking:
                        time.sleep(0.1)
                        cv2.imshow(window_name, display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('0'):
                            break
                else:
                    audio_handler.speak(f"Okay, search marked as complete for {target_object_name}. I don't currently see it, but I hope you found it.")
                    while audio_handler.speaking:
                        time.sleep(0.1)
                        cv2.imshow(window_name, display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('0'):
                            break
                break
    
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt: Exiting object search mode.")
        if audio_handler: audio_handler.speak("Search cancelled by user.")
        time.sleep(4)
    except Exception as e:
        logger.critical(f"Unhandled error during object search: {e}", exc_info=True)
        if audio_handler:
            audio_handler.speak("I encountered a problem in search mode and need to stop. Please try again.")
            time.sleep(1)
        
    finally:
        if audio_handler:
            final_message = "Object search system shutting down."
            print(f"System: {final_message}")
            audio_handler.speak(final_message)
            
            start_time = time.time()
            max_wait = 5.0
            
            while audio_handler.speaking and (time.time() - start_time) < max_wait:
                time.sleep(0.1)
                try:
                    cv2.waitKey(1)
                except:
                    pass
            
            time.sleep(0.5)
            
        logger.info("Shutting down object search...")
        if cap and cap.isOpened():
            cap.release()
        
        # Properly destroy all windows
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Process window events
        time.sleep(0.5)  # Give time for windows to close

# --- Standalone Test Block ---
if __name__ == "__main__":
    print("\nRunning Object Search Standalone Test...")
    test_audio_handler = None
    try:
        test_audio_handler = AudioFeedbackHandler()
        run_object_search(
            audio_handler=test_audio_handler,
            camera_id=1,
            model_path="yolov8n.pt",
            device="cpu",
            conf=0.5,
            feedback_interval=3.0
        )
    except NameError:
        print("ERROR: AudioFeedbackHandler not defined. Cannot run standalone test without it.")
        print("Ensure audio_feedback.py is in the same directory or PYTHONPATH.")
    except Exception as e:
        print(f"Standalone test failed: {e}")
        logger.error(f"Standalone test exception: {e}", exc_info=True)
    finally:
        if test_audio_handler:
            print("\nStopping standalone test audio handler...")
            
            if test_audio_handler.speaking:
                print("Waiting for final messages to complete...")
                start_time = time.time()
                max_wait = 3.0
                
                while test_audio_handler.speaking and (time.time() - start_time) < max_wait:
                    time.sleep(0.1)
                
                if test_audio_handler.speaking:
                    print("Audio completion timed out, forcing stop...")
                
            test_audio_handler.stop()
            time.sleep(0.5)
            print("Audio handler stopped successfully.")
        
        cv2.destroyAllWindows()
        print("Standalone test finished.\n")