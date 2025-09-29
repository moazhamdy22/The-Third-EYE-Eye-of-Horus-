# main_navigation_assistant.py

import os
import logging
import time
import cv2
import numpy as np
import sys

# --- Configuration for Main Menu ---
MENU_WINDOW_NAME = "Navigation Assistant Menu"
MENU_BG_COLOR = (30, 30, 30)      # Dark gray
MENU_TEXT_COLOR = (230, 230, 230) # Light gray
MENU_HIGHLIGHT_COLOR = (100, 180, 255) # Light blue
MENU_TITLE_COLOR = (255, 255, 255) # White
MENU_FONT = cv2.FONT_HERSHEY_SIMPLEX
MENU_FONT_SCALE_TITLE = 0.9
MENU_FONT_SCALE_OPTIONS = 0.7
MENU_FONT_SCALE_INSTRUCTIONS = 0.6 # Slightly larger for readability
MENU_LINE_THICKNESS = 1
MENU_LINE_SPACING = 40
MENU_TOP_MARGIN = 40
MENU_LEFT_MARGIN = 50
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MAIN_MENU_AUDIO_RATE = 160  # Moderate pace for general navigation

# Key codes for arrow navigation
UP_KEYS = {2490368, 65362, 82}
DOWN_KEYS = {2621440, 65364, 84}
LEFT_KEYS = {2424832, 65361, 81}
RIGHT_KEYS = {2555904, 65363, 83}
ENTER_KEY = 13
ESC_KEY = 27

# --- Logging Setup for Main Menu ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s'
)
main_logger = logging.getLogger("NavigationAssistantMain")

# Quieten overly verbose loggers from subsystems if necessary
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("pyttsx3.engine").setLevel(logging.INFO)
logging.getLogger("comtypes").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- Import Core Components & Subsystems ---
AUDIO_HANDLER_MODULE_AVAILABLE = False
try:
    from audio_feedback_vision_assitant import AudioFeedbackHandler # Consistent name
    AUDIO_HANDLER_MODULE_AVAILABLE = True
    main_logger.info("AudioFeedbackHandler imported successfully.")
except ImportError as e:
    main_logger.error(f"Failed to import AudioFeedbackHandler: {e}. Audio feedback will be limited.")
    AudioFeedbackHandler = None # Define as None if import fails

OBSTACLE_SYSTEM_AVAILABLE = False
try:
    from main_full_obstacle_integrated_1 import main as run_obstacle_detection_system
    OBSTACLE_SYSTEM_AVAILABLE = True
    main_logger.info("Obstacle Detection System imported successfully.")
except ImportError as e:
    main_logger.error(f"Failed to import Obstacle Detection System: {e}")
    def run_obstacle_detection_system(shared_audio_handler=None): # Dummy
        main_logger.error("Obstacle Detection System not available.")
        if shared_audio_handler: shared_audio_handler.speak("Obstacle Detection System is not available.", priority=0)
        time.sleep(2)

SCENE_SYSTEM_AVAILABLE = False
try:
    # Assuming arabic_new_resnet50_grouped_capture_with_audio.py contains the SceneClassifierBatch and a main()
    from arabic_new_resnet50_grouped_capture_2 import main as run_scene_localization_system
    SCENE_SYSTEM_AVAILABLE = True
    main_logger.info("Scene Localization System imported successfully.")
except ImportError as e:
    main_logger.error(f"Failed to import Scene Localization System: {e}")
    def run_scene_localization_system(shared_audio_handler_external=None): # Dummy
        main_logger.error("Scene Localization System not available.")
        if shared_audio_handler_external: shared_audio_handler_external.speak("Scene Localization System is not available.", priority=0)
        time.sleep(2)


WALKING_AREA_SYSTEM_AVAILABLE = False
try:
    from walking_area_detection_capture_3 import main as run_walking_area_system
    WALKING_AREA_SYSTEM_AVAILABLE = True
    main_logger.info("Walking Area Detection System imported successfully.")
except ImportError as e:
    main_logger.error(f"Failed to import Walking Area Detection System: {e}")
    def run_walking_area_system(shared_audio_handler_external=None): # Dummy
        main_logger.error("Walking Area Detection System not available.")
        if shared_audio_handler_external: shared_audio_handler_external.speak("Walking Area Detection System is not available.", priority=0)
        time.sleep(2)

GPS_SYSTEM_AVAILABLE = False
try:
    from GPS_openstreet_audio_cv import main as run_gps_directions_system
    GPS_SYSTEM_AVAILABLE = True
    main_logger.info("GPS Directions System imported successfully.")
except ImportError as e:
    main_logger.error(f"Failed to import GPS Directions System: {e}")
    def run_gps_directions_system(shared_audio_handler=None): # Updated dummy function
        main_logger.error("GPS Directions System not available.")
        if shared_audio_handler: 
            shared_audio_handler.speak("GPS Directions System is not available.")
        time.sleep(2)

# Add Google Maps GPS import
GOOGLE_GPS_SYSTEM_AVAILABLE = False
try:
    from GPS_googlemaps_audio_cv import main as run_google_gps_directions_system
    GOOGLE_GPS_SYSTEM_AVAILABLE = True
    main_logger.info("Google Maps GPS Directions System imported successfully.")
except ImportError as e:
    main_logger.error(f"Failed to import Google Maps GPS Directions System: {e}")
    def run_google_gps_directions_system(shared_audio_handler=None): # Dummy function
        main_logger.error("Google Maps GPS Directions System not available.")
        if shared_audio_handler: 
            shared_audio_handler.speak("Google Maps GPS Directions System is not available.")
        time.sleep(2)


def create_menu_image(title: str, options: dict, instructions: str = "", current_selection_index: int = -1):
    """Creates an OpenCV image for the menu with an arrow selector."""
    img = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), MENU_BG_COLOR, dtype=np.uint8)

    # Draw Title
    title_size, _ = cv2.getTextSize(title, MENU_FONT, MENU_FONT_SCALE_TITLE, 2)
    title_x = (WINDOW_WIDTH - title_size[0]) // 2
    cv2.putText(img, title, (title_x, MENU_TOP_MARGIN + 20), MENU_FONT, MENU_FONT_SCALE_TITLE, MENU_TITLE_COLOR, 2, cv2.LINE_AA)

    # Draw Options
    y_pos = MENU_TOP_MARGIN + 20 + title_size[1] + MENU_LINE_SPACING
    for i, (key, text) in enumerate(options.items()):
        color = MENU_TEXT_COLOR
        display_text = f"  {text}"  # Indent text for the selector
        
        if i == current_selection_index:
            color = MENU_HIGHLIGHT_COLOR
            # Draw selector arrow
            cv2.putText(img, ">", (MENU_LEFT_MARGIN - 20, y_pos), MENU_FONT, MENU_FONT_SCALE_OPTIONS, color, 2, cv2.LINE_AA)
        
        cv2.putText(img, display_text, (MENU_LEFT_MARGIN, y_pos), MENU_FONT, MENU_FONT_SCALE_OPTIONS, color, MENU_LINE_THICKNESS, cv2.LINE_AA)
        y_pos += MENU_LINE_SPACING

    # Draw Instructions
    if instructions:
        inst_y = WINDOW_HEIGHT - 60
        cv2.putText(img, instructions, (MENU_LEFT_MARGIN, inst_y), MENU_FONT, MENU_FONT_SCALE_INSTRUCTIONS, MENU_TEXT_COLOR, 1, cv2.LINE_AA)
    
    return img

def get_menu_choice_cv(audio_handler, title, options):
    """Displays an OpenCV menu and gets user choice via arrow key navigation."""
    instructions = "Up/Down: Navigate | Right/Enter: Select | Left: Back | ESC: Quit"
    current_selection_idx = 0
    option_keys = list(options.keys())
    option_values = list(options.values())

    if audio_handler:
        audio_handler.speak(f"{title}. Use arrow keys to navigate. Currently on: {option_values[current_selection_idx]}")

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
        elif key_pressed in LEFT_KEYS:
            if '0' in option_keys: return '0' # Return to previous menu
            else:
                 if audio_handler: audio_handler.speak("No back option here.")

def confirm_final_action(audio_handler, action_name):
    """Confirms final action using Enter (confirm) or Left/ESC (cancel)."""
    message = f"Start {action_name}?"
    
    if audio_handler:
        audio_handler.speak(f"{message} Press Enter to confirm, or Left Arrow to cancel.")

    confirm_options = {'confirm': "Confirm", 'cancel': "Cancel"}
    instructions = "Enter: Confirm | Left Arrow or ESC: Cancel"
    menu_img = create_menu_image("Confirmation", confirm_options, instructions, current_selection_index=0)
    cv2.imshow(MENU_WINDOW_NAME, menu_img)

    while True:
        key_pressed = cv2.waitKeyEx(0)
        if key_pressed == ENTER_KEY: return True
        elif key_pressed in LEFT_KEYS or key_pressed == ESC_KEY: return False

def select_gps_service(master_audio_handler):
    """Handle GPS service selection submenu using arrow key navigation."""
    gps_service_options = {
        '1': f"Google Maps Directions{'' if GOOGLE_GPS_SYSTEM_AVAILABLE else ' (Not Available)'}",
        '2': f"OpenStreet Maps{'' if GPS_SYSTEM_AVAILABLE else ' (Not Available)'}",
        '0': "Back to Main Menu"
    }
    
    gps_choice = get_menu_choice_cv(master_audio_handler, "Select GPS Service", gps_service_options)
    
    if gps_choice in ['0', 'esc']:
        return None
    
    # Check availability and get service info
    if gps_choice == '1':
        if not GOOGLE_GPS_SYSTEM_AVAILABLE:
            if master_audio_handler: master_audio_handler.speak("Google Maps Directions service is not available.")
            return None
        service_name = "Google Maps Directions"
        service_function = run_google_gps_directions_system
    elif gps_choice == '2':
        if not GPS_SYSTEM_AVAILABLE:
            if master_audio_handler: master_audio_handler.speak("OpenStreet Maps GPS service is not available.")
            return None
        service_name = "OpenStreet Maps GPS"
        service_function = run_gps_directions_system
    else:
        return None
    
    if confirm_final_action(master_audio_handler, service_name):
        return service_function
    
    return None

def main_menu_loop(shared_audio_handler=None):
    """Displays and handles the main navigation assistant menu."""
    main_logger.info("Initializing Navigation Assistant Main Menu...")
    
    master_audio_handler = shared_audio_handler
    
    if master_audio_handler is None:
        main_logger.warning("No shared audio handler provided. Attempting to create a local one.")
        if AUDIO_HANDLER_MODULE_AVAILABLE and AudioFeedbackHandler:
            try:
                master_audio_handler = AudioFeedbackHandler()
                if not (master_audio_handler and master_audio_handler.engine):
                    main_logger.error("Failed to initialize LOCAL AudioFeedbackHandler. Audio will be disabled.")
                    master_audio_handler = None
            except Exception as e:
                main_logger.error(f"Failed to initialize LOCAL AudioFeedbackHandler: {e}")
                master_audio_handler = None
        else:
            main_logger.warning("AudioFeedbackHandler module not available. Audio will be disabled.")
    else:
        main_logger.info("Using the shared AudioFeedbackHandler provided by the main application.")

    if master_audio_handler:
        master_audio_handler.speak("Welcome to the Navigation Assistant.")

    cv2.namedWindow(MENU_WINDOW_NAME)
    cv2.setWindowProperty(MENU_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

    system_availability = {
        '1': OBSTACLE_SYSTEM_AVAILABLE, '2': SCENE_SYSTEM_AVAILABLE,
        '3': WALKING_AREA_SYSTEM_AVAILABLE, '4': GPS_SYSTEM_AVAILABLE or GOOGLE_GPS_SYSTEM_AVAILABLE,
        '0': True
    }
    menu_options = {
        '1': f"Obstacle Detection{'' if system_availability['1'] else ' (Not Available)'}",
        '2': f"Scene Localization{'' if system_availability['2'] else ' (Not Available)'}",
        '3': f"Walking Area Detection{'' if system_availability['3'] else ' (Not Available)'}",
        '4': f"GPS Directions{'' if system_availability['4'] else ' (Not Available)'}",
        '0': "Return to Smart Assistive System Menu"
    }
    
    running = True
    while running:
        choice = get_menu_choice_cv(master_audio_handler, "Navigation Assistant Menu", menu_options)
        
        if choice == 'esc' or choice == '0':
            if confirm_final_action(master_audio_handler, "return to the Main Smart Assistive System Menu"):
                if master_audio_handler: master_audio_handler.speak("Returning to the main menu.")
                running = False
            continue

        if choice in ['1', '2', '3', '4']:
            option_names = {
                '1': "Obstacle Detection", '2': "Scene Localization", 
                '3': "Walking Area Detection", '4': "GPS Directions"
            }
            if not system_availability[choice]:
                main_logger.warning(f"Attempted to select unavailable option: {option_names[choice]}")
                if master_audio_handler: master_audio_handler.speak(f"{option_names[choice]} is not available.")
                continue

            # Handle GPS service selection as a sub-menu
            if choice == '4':
                selected_gps_service = select_gps_service(master_audio_handler)
                if selected_gps_service is None: continue  # User cancelled or service unavailable
                
                main_logger.info("Starting selected GPS Directions service.")
                cv2.destroyWindow(MENU_WINDOW_NAME)
                cv2.waitKey(1)
                try:
                    selected_gps_service(shared_audio_handler=master_audio_handler)
                except Exception as e:
                    main_logger.error(f"Error in GPS service: {e}", exc_info=True)
                    if master_audio_handler: master_audio_handler.speak("An error occurred in the GPS service.")
                
            else: # Handle other systems
                selected_option_name = option_names[choice]
                if confirm_final_action(master_audio_handler, selected_option_name):
                    main_logger.info(f"Starting: {selected_option_name}")
                    cv2.destroyWindow(MENU_WINDOW_NAME)
                    cv2.waitKey(1)
                    try:
                        if choice == '1': run_obstacle_detection_system(shared_audio_handler=master_audio_handler)
                        elif choice == '2': run_scene_localization_system(shared_audio_handler_external=master_audio_handler)
                        elif choice == '3': run_walking_area_system(shared_audio_handler_external=master_audio_handler)
                    except Exception as e:
                        main_logger.error(f"Error in {selected_option_name}: {e}", exc_info=True)
                        if master_audio_handler: master_audio_handler.speak(f"An error occurred in {selected_option_name}.")

            # After a module finishes, restore the menu
            cv2.destroyAllWindows()
            cv2.waitKey(100)
            if running:
                cv2.namedWindow(MENU_WINDOW_NAME)
                cv2.setWindowProperty(MENU_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
                if master_audio_handler: master_audio_handler.speak("Returning to Navigation Assistant menu.")
    
    main_logger.info("Exiting Navigation Assistant.")
    if master_audio_handler:
        master_audio_handler.speak("Goodbye!")
        time.sleep(1) # Give time for goodbye
        if shared_audio_handler is None:
            main_logger.info("Stopping locally created audio handler.")
            master_audio_handler.stop()
        else:
            main_logger.info("Not stopping shared audio handler.")
    
    cv2.destroyAllWindows()
    main_logger.info("Navigation Assistant stopped.")

if __name__ == "__main__":
    try:
        main_menu_loop()
    except KeyboardInterrupt:
        main_logger.info("Program interrupted by user (Ctrl+C).")
    except Exception as e:
        main_logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
    finally:
        main_logger.info("Application shut down.")
        cv2.destroyAllWindows()