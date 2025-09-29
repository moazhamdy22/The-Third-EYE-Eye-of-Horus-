# --- START OF FILE smart_asstant_for_BVI.py ---

# Main Integration Script for Smart Assistant for BVI
# Provides unified access to all subsystems

import os
import logging
import time
import cv2
import numpy as np
import sys
import threading

# --- Configuration ---
MENU_WINDOW_NAME = "Smart Assistant for BVI - Main Menu"
MENU_BG_COLOR = (20, 20, 40)       # Dark blue-gray
MENU_TEXT_COLOR = (230, 230, 230)  # Light gray
MENU_HIGHLIGHT_COLOR = (100, 180, 255)  # Light blue
MENU_TITLE_COLOR = (255, 255, 255)  # White
MENU_FONT = cv2.FONT_HERSHEY_SIMPLEX
MENU_FONT_SCALE_TITLE = 1.0
MENU_FONT_SCALE_OPTIONS = 0.8
MENU_FONT_SCALE_INSTRUCTIONS = 0.6
MENU_LINE_THICKNESS = 2
MENU_LINE_SPACING = 50
MENU_TOP_MARGIN = 60
MENU_LEFT_MARGIN = 80
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700



# Key codes for arrow navigation
UP_KEYS = {2490368, 65362, 82}
DOWN_KEYS = {2621440, 65364, 84}
LEFT_KEYS = {2424832, 65361, 81}
RIGHT_KEYS = {2555904, 65363, 83}
ENTER_KEY = 13
ESC_KEY = 27

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmartAssistantBVI")

# Quieten verbose loggers
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("pyttsx3.engine").setLevel(logging.WARNING)
logging.getLogger("comtypes").setLevel(logging.WARNING)

# --- NEW: Import the SOS listener ---
try:
    from sos.sos_listener import listen_for_sos_trigger
    SOS_LISTENER_AVAILABLE = True
except ImportError as e:
    print(f"CRITICAL WARNING: Could not import SOS listener: {e}. SOS button will NOT function.")
    logger.critical(f"Failed to import SOS listener: {e}. SOS button will be disabled.")
    SOS_LISTENER_AVAILABLE = False
# --- END NEW IMPORT ---

# --- Import Audio Feedback Handler ---
AUDIO_HANDLER_AVAILABLE = False
try:
    from audio_feedback_vision_assitant import AudioFeedbackHandler
    AUDIO_HANDLER_AVAILABLE = True
    logger.info("AudioFeedbackHandler imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import AudioFeedbackHandler: {e}. Audio feedback will be disabled.")
    AudioFeedbackHandler = None

# --- Import Subsystem Modules ---
NAVIGATION_AVAILABLE = False
try:
    # We now rename the import to avoid confusion
    from Navigation_assitant.main_nav import main_menu_loop as run_navigation_system
    NAVIGATION_AVAILABLE = True
    logger.info("Navigation System imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import Navigation System: {e}")
    def run_navigation_system(shared_audio_handler=None):
        logger.error("Navigation System not available.")
        print("Navigation System is not available.")
        if shared_audio_handler: shared_audio_handler.speak("Navigation System is not available.")
        time.sleep(2)

OBJECT_DETECTION_AVAILABLE = False
try:
    # Rename for clarity
    from Object_detection.main_modd2_object_detection import main as run_object_detection_system
    OBJECT_DETECTION_AVAILABLE = True
    logger.info("Object Detection System imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import Object Detection System: {e}")
    def run_object_detection_system(shared_audio_handler=None):
        logger.error("Object Detection System not available.")
        print("Object Detection System is not available.")
        if shared_audio_handler: shared_audio_handler.speak("Object Detection System is not available.")
        time.sleep(2)

# --- Import OCR Processor ---
OCR_PROCESSOR = None
OCR_AVAILABLE = False
try:
    from OCR.main_ocr import main as run_ocr_system
    from OCR.OCR import OCRProcessor
    OCR_AVAILABLE = True
    logger.info("OCR System imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import OCR System: {e}")
    def run_ocr_system(shared_audio_handler=None, ocr_processor=None):
        logger.error("OCR System not available.")
        if shared_audio_handler: shared_audio_handler.speak("OCR System is not available.")
        print("OCR System is not available.")
        time.sleep(2)

# --- Import Face Recognition System ---
FACE_RECOGNITION_PROCESSOR = None
FACE_RECOGNITION_AVAILABLE = False
try:
    from Face_recognation.main_face import main as run_face_recognition_system
    from Face_recognation.Face_Recognition_System import UnifiedFaceRecognition
    FACE_RECOGNITION_AVAILABLE = True
    logger.info("Face Recognition System imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import Face Recognition System: {e}")
    def run_face_recognition_system(shared_audio_handler=None, face_system=None):
        logger.error("Face Recognition System not available.")
        print("Face Recognition System is not available.")
        time.sleep(2)

def create_menu_image(title: str, options: dict, instructions: str = "", current_selection_index: int = -1):
    """Creates an OpenCV image for the main menu with a selector."""
    img = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), MENU_BG_COLOR, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (WINDOW_WIDTH-10, WINDOW_HEIGHT-10), MENU_HIGHLIGHT_COLOR, 2)
    
    title_size, _ = cv2.getTextSize(title, MENU_FONT, MENU_FONT_SCALE_TITLE, 3)
    title_x = (WINDOW_WIDTH - title_size[0]) // 2
    cv2.putText(img, title, (title_x, MENU_TOP_MARGIN), MENU_FONT, MENU_FONT_SCALE_TITLE, MENU_TITLE_COLOR, 3, cv2.LINE_AA)
    
    subtitle = "Comprehensive Assistive Technology Suite"
    subtitle_size, _ = cv2.getTextSize(subtitle, MENU_FONT, 0.5, 1)
    subtitle_x = (WINDOW_WIDTH - subtitle_size[0]) // 2
    cv2.putText(img, subtitle, (subtitle_x, MENU_TOP_MARGIN + 35), MENU_FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    y_pos = MENU_TOP_MARGIN + 110
    for i, (key, text) in enumerate(options.items()):
        color = MENU_TEXT_COLOR
        display_text = f"  {text}" # Indent for selector
        
        if i == current_selection_index:
            color = MENU_HIGHLIGHT_COLOR
            cv2.putText(img, ">", (MENU_LEFT_MARGIN - 30, y_pos), MENU_FONT, MENU_FONT_SCALE_OPTIONS, color, 3, cv2.LINE_AA)
        
        cv2.putText(img, display_text, (MENU_LEFT_MARGIN, y_pos), MENU_FONT, MENU_FONT_SCALE_OPTIONS, color, MENU_LINE_THICKNESS, cv2.LINE_AA)
        y_pos += MENU_LINE_SPACING

    if instructions:
        inst_y = WINDOW_HEIGHT - 80
        cv2.putText(img, instructions, (MENU_LEFT_MARGIN, inst_y), MENU_FONT, MENU_FONT_SCALE_INSTRUCTIONS, (180, 180, 180), 1, cv2.LINE_AA)
    
    return img

def get_menu_choice_cv(audio_handler, title, options):
    """Displays an OpenCV menu and gets user choice via arrow key navigation."""
    instructions = "Up/Down: Navigate | Right/Enter: Select | ESC: Shutdown"
    current_selection_idx = 0
    option_keys = list(options.keys())
    option_values = list(options.values())

    if audio_handler:
        audio_handler.speak(f"{title}. Use the arrow keys to navigate. Currently selected: {option_values[current_selection_idx]}")

    while True:
        menu_img = create_menu_image(title, options, instructions, current_selection_idx)
        cv2.imshow(MENU_WINDOW_NAME, menu_img)
        
        key_pressed = cv2.waitKeyEx(0)
        
        if key_pressed == ESC_KEY:
            return 'esc'
        elif key_pressed in UP_KEYS:
            current_selection_idx = (current_selection_idx - 1 + len(options)) % len(options)
            if audio_handler: audio_handler.speak(option_values[current_selection_idx])
        elif key_pressed in DOWN_KEYS:
            current_selection_idx = (current_selection_idx + 1) % len(options)
            if audio_handler: audio_handler.speak(option_values[current_selection_idx])
        elif key_pressed in RIGHT_KEYS or key_pressed == ENTER_KEY:
            return option_keys[current_selection_idx]
        # Left key does nothing in the main menu, only ESC quits

def confirm_action(audio_handler, action_name):
    """Confirms final action using Enter (confirm) or Left/ESC (cancel)."""
    message = f"Start {action_name}?"
    if audio_handler:
        audio_handler.speak(f"{message} Press Enter to confirm, or Left Arrow to cancel.")

    confirm_options = {'confirm': "Confirm", 'cancel': "Cancel"}
    instructions = "Enter: Confirm | Left Arrow or ESC: Cancel"
    
    # We will alternate the selection to draw attention
    selection_idx = 0
    last_switch_time = time.time()

    while True:
        # Blinking/alternating selector for confirmation
        if time.time() - last_switch_time > 0.7:
             selection_idx = 1 - selection_idx
             last_switch_time = time.time()

        menu_img = create_menu_image("Confirmation", confirm_options, instructions, current_selection_index=selection_idx)
        cv2.imshow(MENU_WINDOW_NAME, menu_img)

        key_pressed = cv2.waitKeyEx(10) # Use a timeout to allow the blink effect
        
        if key_pressed == ENTER_KEY: return True
        elif key_pressed in LEFT_KEYS or key_pressed == ESC_KEY: return False


def main():
    """Main menu loop for the Smart Assistant for BVI."""
    logger.info("Starting Smart Assistant for BVI - Main Integration System")
    global OCR_PROCESSOR, FACE_RECOGNITION_PROCESSOR

    if SOS_LISTENER_AVAILABLE:
        logger.info("SOS Listener is available. Starting in a background thread.")
        sos_thread = threading.Thread(target=listen_for_sos_trigger, name="SOS_Listener_Thread", daemon=True)
        sos_thread.start()
    else:
        logger.error("SOS Listener is not available. The physical emergency button will not function.")

    main_audio_handler = None
    if AUDIO_HANDLER_AVAILABLE and AudioFeedbackHandler:
        try:
            main_audio_handler = AudioFeedbackHandler()
            if main_audio_handler and main_audio_handler.engine:
                logger.info("Main AudioFeedbackHandler created successfully.")
                main_audio_handler.speak("Welcome to the Smart Assistant for Blind and Visually Impaired individuals.")
            else:
                main_audio_handler = None
        except Exception as e:
            logger.error(f"Failed to initialize Main AudioFeedbackHandler: {e}")
            main_audio_handler = None

    # Preload OCR System
    if OCR_AVAILABLE:
        try:
            logger.info("Preloading OCR system...")
            OCR_PROCESSOR = OCRProcessor(shared_audio_handler=main_audio_handler)
            logger.info("OCR system preloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to preload OCR system: {e}")
            OCR_PROCESSOR = None

    # Preload Face Recognition System
    if FACE_RECOGNITION_AVAILABLE:
        try:
            logger.info("Preloading Face Recognition system...")
            FACE_RECOGNITION_PROCESSOR = UnifiedFaceRecognition()
            logger.info("Face Recognition system preloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to preload Face Recognition system: {e}")
            FACE_RECOGNITION_PROCESSOR = None

    cv2.namedWindow(MENU_WINDOW_NAME)
    cv2.setWindowProperty(MENU_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    
    system_availability = {
        '1': NAVIGATION_AVAILABLE, '2': OBJECT_DETECTION_AVAILABLE,
        '3': OCR_AVAILABLE, '4': FACE_RECOGNITION_AVAILABLE,
        '0': True
    }
    
    if main_audio_handler:
        system_names = {
            '1': "Navigation Assistant", '2': "Object Detection and Scene Analysis",
            '3': "OCR Text Recognition", '4': "Face Recognition"
        }
        available_systems = [system_names[k] for k, v in system_availability.items() if k != '0' and v]
        unavailable_systems = [system_names[k] for k, v in system_availability.items() if k != '0' and not v]
        
        if available_systems: main_audio_handler.speak(f"Available systems: {', '.join(available_systems)}.")
        if unavailable_systems: main_audio_handler.speak(f"Unavailable systems: {', '.join(unavailable_systems)}.")
    
    menu_options = {
        '1': f"Navigation Assistant{'' if system_availability['1'] else ' (Not Available)'}",
        '2': f"Object Detection & Scene Analysis{'' if system_availability['2'] else ' (Not Available)'}",
        '3': f"OCR Text Recognition{'' if system_availability['3'] else ' (Not Available)'}",
        '4': f"Face Recognition{'' if system_availability['4'] else ' (Not Available)'}",
        '0': "Shutdown System"
    }
    
    running = True
    print("Welcome to Smart Assistant for BVI")
    print("===================================")
    for desc in menu_options.values(): print(f"- {desc}")
    print("===================================")
    
    while running:
        choice = get_menu_choice_cv(main_audio_handler, "Smart Assistant Main Menu", menu_options)
        
        if choice == 'esc' or choice == '0':
            if confirm_action(main_audio_handler, "shutdown the system"):
                if main_audio_handler:
                    main_audio_handler.speak("Shutting down Smart Assistant. Goodbye.")
                print("Shutting down Smart Assistant for BVI...")
                logger.info("System shutdown requested by user.")
                running = False
            continue

        if choice in ['1', '2', '3', '4']:
            option_names = {
                '1': "Navigation Assistant", '2': "Object Detection & Scene Analysis",
                '3': "OCR Text Recognition", '4': "Face Recognition"
            }
            if not system_availability[choice]:
                logger.warning(f"Attempted to select unavailable option: {option_names[choice]}")
                if main_audio_handler: main_audio_handler.speak(f"{option_names[choice]} is not available.")
                continue

            selected_option_name = option_names[choice]
            
            if confirm_action(main_audio_handler, selected_option_name):
                logger.info(f"Starting: {selected_option_name}")
                if main_audio_handler: main_audio_handler.speak(f"Starting {selected_option_name}.")
                
                cv2.destroyWindow(MENU_WINDOW_NAME)
                cv2.waitKey(1)

                try:
                    if choice == '1': 
                        run_navigation_system(shared_audio_handler=main_audio_handler)
                    elif choice == '2': 
                        run_object_detection_system(shared_audio_handler=main_audio_handler)
                    elif choice == '3': 
                        switch_mode = run_ocr_system(shared_audio_handler=main_audio_handler, ocr_processor=OCR_PROCESSOR)
                        if switch_mode:
                            # Handle mode switching if needed
                            pass
                    elif choice == '4': 
                        if FACE_RECOGNITION_PROCESSOR:
                            FACE_RECOGNITION_PROCESSOR.run_Face_Recognition()
                        else:
                            logger.error("Face Recognition System is not available.")
                    logger.info(f"Finished running: {selected_option_name}")

                except Exception as e:
                    logger.error(f"Error while running {selected_option_name}: {e}", exc_info=True)
                    error_msg = f"An error occurred in {selected_option_name}. Returning to main menu."
                    if main_audio_handler: main_audio_handler.speak(error_msg)
                
                cv2.destroyAllWindows(); cv2.waitKey(100)
                
                if running:
                    cv2.namedWindow(MENU_WINDOW_NAME)
                    cv2.setWindowProperty(MENU_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
                    if main_audio_handler: main_audio_handler.speak("Returning to main menu.")
    
    logger.info("Performing system cleanup...")
    if main_audio_handler:
        try:
            main_audio_handler.stop()
            logger.info("Main audio handler stopped during cleanup.")
        except Exception as e:
            logger.error(f"Error during audio cleanup: {e}")
    
    # Clean up Face Recognition system
    if FACE_RECOGNITION_PROCESSOR:
        try:
            logger.info("Cleaning up Face Recognition system...")
            FACE_RECOGNITION_PROCESSOR.exit_flag = True
            if hasattr(FACE_RECOGNITION_PROCESSOR, 'camera') and FACE_RECOGNITION_PROCESSOR.camera is not None:
                FACE_RECOGNITION_PROCESSOR.camera.release()
        except Exception as e:
            logger.error(f"Error during Face Recognition cleanup: {e}")

    cv2.destroyAllWindows()
    print("Smart Assistant for BVI has been shut down.")
    logger.info("System shutdown complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program interrupted by user (Ctrl+C).")
        print("\nProgram interrupted. Shutting down...")
    except Exception as e:
        logger.critical(f"Critical error in main application: {e}", exc_info=True)
        print(f"Critical error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        logger.info("Application cleanup completed.")
        sys.exit(0)

