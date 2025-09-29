# main.py (On-Screen OpenCV Menu)

import os
import logging
import time
import cv2
import numpy as np # For creating menu image
import sys
import platform
import psutil

# Add current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import core components
from audio_feedback_vision_assitant import AudioFeedbackHandler # Ensure this is the latest stable version
from object_detection_capture import run_object_detection_capture
from object_detection_realtime import run_realtime_spatial_awareness
from object_search import run_object_search
from scene_description_image import run_image_scene_description
from scene_description_video import run_video_scene_description
from image_descrption import run_blip_scene_description # <<< CORRECTED TYPO HERE

# GPU Detection and System Info
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for less console noise, DEBUG can be enabled if needed
    format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmartAssistMain")

logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("pyttsx3.engine").setLevel(logging.INFO)
logging.getLogger("comtypes").setLevel(logging.WARNING)

def detect_optimal_device():
    """Detect and configure the optimal device (GPU/CPU) with detailed logging."""
    device_info = {
        'device': 'cpu',
        'device_name': 'CPU',
        'memory_available': 0,
        'compute_capability': None,
        'cuda_available': False
    }
    
    logger.info("=== SYSTEM DEVICE DETECTION ===")
    
    # System Information
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    # Memory Information
    memory = psutil.virtual_memory()
    logger.info(f"System RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    
    if TORCH_AVAILABLE:
        logger.info("PyTorch is available for device detection.")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA is available! Found {device_count} GPU(s)")
            
            # Get information about the first (primary) GPU
            gpu_id = 0
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            logger.info(f"Primary GPU: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory_gb:.1f} GB")
            
            # Check GPU memory usage
            torch.cuda.empty_cache()  # Clear cache
            gpu_memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            gpu_memory_free = gpu_memory_gb - gpu_memory_allocated
            
            logger.info(f"GPU Memory - Used: {gpu_memory_allocated:.1f} GB, Free: {gpu_memory_free:.1f} GB")
            
            # Additional GPU info if GPUtil is available
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        logger.info(f"GPU Utilization: {gpu.load*100:.1f}%")
                        logger.info(f"GPU Temperature: {gpu.temperature}°C")
                        logger.info(f"GPU Driver: {gpu.driver}")
                except Exception as e:
                    logger.warning(f"Could not get detailed GPU info: {e}")
            
            device_info.update({
                'device': 'cuda',
                'device_name': gpu_name,
                'memory_available': gpu_memory_free,
                'cuda_available': True
            })
            
            logger.info("✅ DEVICE SELECTED: GPU (CUDA)")
            
        else:
            logger.info("CUDA is not available. Checking for other accelerators...")
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("✅ MPS (Apple Silicon) is available!")
                device_info.update({
                    'device': 'mps',
                    'device_name': 'Apple Silicon GPU',
                    'cuda_available': False
                })
                logger.info("✅ DEVICE SELECTED: MPS (Apple Silicon)")
            else:
                logger.info("No GPU acceleration available. Using CPU.")
                device_info['device_name'] = platform.processor()
                logger.info("✅ DEVICE SELECTED: CPU")
    else:
        logger.warning("PyTorch is not available. Defaulting to CPU.")
        logger.info("✅ DEVICE SELECTED: CPU (PyTorch not available)")
    
    # Log final device selection
    logger.info(f"Selected Device: {device_info['device'].upper()}")
    logger.info(f"Device Name: {device_info['device_name']}")
    if device_info['memory_available'] > 0:
        logger.info(f"Available Memory: {device_info['memory_available']:.1f} GB")
    
    logger.info("=== DEVICE DETECTION COMPLETE ===")
    
    return device_info['device'], device_info

def log_performance_stats(operation_name, start_time, end_time, device_info):
    """Log performance statistics for operations."""
    duration = end_time - start_time
    logger.info(f"📊 PERFORMANCE - {operation_name}")
    logger.info(f"⏱️  Duration: {duration:.2f} seconds")
    logger.info(f"🖥️  Device: {device_info['device_name']}")
    
    if device_info['cuda_available'] and TORCH_AVAILABLE:
        try:
            gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            logger.info(f"💾 GPU Memory Used: {gpu_memory_used:.2f} GB")
        except:
            pass
    
    # System resource usage
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    logger.info(f"💻 CPU Usage: {cpu_percent:.1f}%")
    logger.info(f"🧠 RAM Usage: {memory_percent:.1f}%")

# Detect optimal device at startup
DEVICE, DEVICE_INFO = detect_optimal_device()

MODEL_PATH = "yolov8n.pt"  # Default for realtime operations
MODEL_PATH_SNAPSHOT = "yolov8x.pt"  # High accuracy for snapshot analysis
CONFIDENCE_THRESHOLD = 0.5
CAMERA_ID = 1
CAPTURE_DURATION = 5
VIDEO_DESC_DURATION = 7
REALTIME_INTERVAL = 3.0
SEARCH_INTERVAL = 3.0
BLUR_THRESHOLD = 60.0
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "your_api_key") # Replace placeholder

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("The 'requests' library is not installed. Internet check will be skipped.")

def check_internet_connection(timeout=5):
    if not REQUESTS_AVAILABLE: return True
    url = "http://www.google.com"
    try:
        requests.get(url, timeout=timeout, headers={'Cache-Control': 'no-cache'}).raise_for_status()
        logger.info("Internet connection check successful.")
        return True
    except requests.exceptions.RequestException:
        logger.error(f"Internet connection check failed.")
        return False

# --- OpenCV Menu Functions ---
MENU_WINDOW_NAME = "Smart Assistant Menu"
MENU_BG_COLOR = (30, 30, 30) # Dark gray
MENU_TEXT_COLOR = (230, 230, 230) # Light gray
MENU_HIGHLIGHT_COLOR = (100, 180, 255) # Light blue
MENU_FONT = cv2.FONT_HERSHEY_SIMPLEX
MENU_FONT_SCALE = 0.7
MENU_LINE_THICKNESS = 1
MENU_LINE_SPACING = 35
MENU_TOP_MARGIN = 50
MENU_LEFT_MARGIN = 50
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Key codes can vary. These sets cover common values for Windows and Linux/macOS.
# Use cv2.waitKeyEx() to read them.
UP_KEYS = {2490368, 65362, 82}
DOWN_KEYS = {2621440, 65364, 84}
LEFT_KEYS = {2424832, 65361, 81}
RIGHT_KEYS = {2555904, 65363, 83}
ENTER_KEY = 13
ESC_KEY = 27

def create_menu_image(title, options, instructions="", current_selection_index=-1):
    img = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), MENU_BG_COLOR, dtype=np.uint8)
    
    # Title
    cv2.putText(img, title, (MENU_LEFT_MARGIN, MENU_TOP_MARGIN), MENU_FONT, MENU_FONT_SCALE * 1.2, MENU_TEXT_COLOR, MENU_LINE_THICKNESS + 1, cv2.LINE_AA)
    
    # Options
    y_pos = MENU_TOP_MARGIN + MENU_LINE_SPACING * 2
    for i, (key, text) in enumerate(options.items()):
        display_text = f"  {text}" # Indent for selection indicator
        color = MENU_TEXT_COLOR
        if i == current_selection_index:
            color = MENU_HIGHLIGHT_COLOR
            # Add a selection indicator
            cv2.putText(img, ">", (MENU_LEFT_MARGIN - 20, y_pos), MENU_FONT, MENU_FONT_SCALE, color, MENU_LINE_THICKNESS, cv2.LINE_AA)

        cv2.putText(img, display_text, (MENU_LEFT_MARGIN, y_pos), MENU_FONT, MENU_FONT_SCALE, color, MENU_LINE_THICKNESS, cv2.LINE_AA)
        y_pos += MENU_LINE_SPACING
        
    # Instructions
    if instructions:
        y_pos = WINDOW_HEIGHT - 40
        cv2.putText(img, instructions, (MENU_LEFT_MARGIN, y_pos), MENU_FONT, MENU_FONT_SCALE * 0.8, (180,180,180), MENU_LINE_THICKNESS, cv2.LINE_AA)
        
    return img

def get_menu_choice_cv(audio_handler, title, options):
    """Displays an OpenCV menu and gets user choice via arrow key navigation."""
    instructions = "Up/Down: Navigate | Right/Enter: Select | Left: Back | ESC: Quit"
    current_selection_idx = 0
    option_keys = list(options.keys())
    option_values = list(options.values())

    # Brief menu announcement
    if audio_handler:
        audio_handler.speak(f"{title}. {option_values[current_selection_idx]}")

    while True:
        menu_img = create_menu_image(title, options, instructions, current_selection_idx)
        cv2.imshow(MENU_WINDOW_NAME, menu_img)
        
        # Use waitKeyEx to capture arrow keys
        key_pressed = cv2.waitKeyEx(0)
        
        if key_pressed == ESC_KEY:
            return 'esc'

        elif key_pressed in UP_KEYS:
            current_selection_idx = (current_selection_idx - 1 + len(options)) % len(options)
            if audio_handler:
                audio_handler.speak(option_values[current_selection_idx])

        elif key_pressed in DOWN_KEYS:
            current_selection_idx = (current_selection_idx + 1) % len(options)
            if audio_handler:
                audio_handler.speak(option_values[current_selection_idx])
        
        elif key_pressed in RIGHT_KEYS or key_pressed == ENTER_KEY:
            return option_keys[current_selection_idx]

        elif key_pressed in LEFT_KEYS:
            # Find the 'back' or 'return' option, which usually has key '0'
            if '0' in option_keys:
                return '0'

def confirm_final_action(audio_handler, action_name):
    """Confirms final action using Enter (confirm) or Left/ESC (cancel)."""
    if audio_handler:
        audio_handler.speak(f"Start {action_name}? Enter to confirm, Escape to cancel.")

    # Create a simplified menu for confirmation
    confirm_options = {'confirm': "Confirm", 'cancel': "Cancel"}
    instructions = "Enter: Confirm | Left Arrow or ESC: Cancel"
    menu_img = create_menu_image(f"Start {action_name}?", confirm_options, instructions, current_selection_index=0)
    cv2.imshow(MENU_WINDOW_NAME, menu_img)

    while True:
        key_pressed = cv2.waitKeyEx(0)
        
        if key_pressed == ENTER_KEY:
            return True
        elif key_pressed in LEFT_KEYS or key_pressed == ESC_KEY:
            return False

def main(shared_audio_handler=None):
    logger.info("===================================")
    logger.info("Starting Object Detection & Description Module...")
    logger.info(f"Using Device: {DEVICE.upper()} - {DEVICE_INFO['device_name']}")
    logger.info("===================================")
    
    module_audio_handler = shared_audio_handler 

    try:
        if module_audio_handler is None:
            logger.warning("No shared audio handler provided. Attempting to create a local one.")
            try:
                module_audio_handler = AudioFeedbackHandler() 
                if not module_audio_handler.engine:
                    logger.critical("Failed to initialize LOCAL AudioFeedbackHandler. Module audio will be disabled.")
                    module_audio_handler = None
                else:
                    logger.info("LOCAL AudioFeedbackHandler initialized successfully.")
            except Exception as e:
                logger.critical(f"Failed to initialize LOCAL audio handler: {e}")
                module_audio_handler = None
        else:
            logger.info("Using the shared AudioFeedbackHandler from the main application.")

        if module_audio_handler:
            module_audio_handler.speak("Object Detection Module ready.")

        cv2.namedWindow(MENU_WINDOW_NAME)
        cv2.setWindowProperty(MENU_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

        if "YOUR_GOOGLE_API_KEY_HERE" in GOOGLE_API_KEY:
            logger.critical("Warning: Google API Key is a placeholder. Scene description features will be disabled.")
            if module_audio_handler: module_audio_handler.speak("Scene description unavailable.")
        
        gemini_ready = False
        if GOOGLE_API_KEY and "YOUR_GOOGLE_API_KEY_HERE" not in GOOGLE_API_KEY:
            if not check_internet_connection():
                logger.warning("Warning: Internet connection failed. Scene description features might not work.")
                if module_audio_handler: module_audio_handler.speak("No internet. Scene description may not work.")
            else:
                gemini_ready = True
        else: 
            if "YOUR_GOOGLE_API_KEY_HERE" in GOOGLE_API_KEY: 
                if module_audio_handler: module_audio_handler.speak("Scene description disabled.")
        
        running = True
        while running:
            main_menu_options = {
                '1': "Object Detection",
                '2': "Object Search",
                '3': "Scene Description",
                '0': "Return to Main Menu"
            }
            choice = get_menu_choice_cv(module_audio_handler, "Object Detection Menu", main_menu_options)

            if choice == 'esc'or '0' in choice:
                if confirm_final_action(module_audio_handler, "exit"):
                    if module_audio_handler: module_audio_handler.speak("Exiting module.")
                    running = False
                continue
            elif choice == '0':
                # Direct return to main menu without confirmation
                if module_audio_handler: module_audio_handler.speak("Returning to main menu.")
                running = False
                continue

            if choice == '1':  # Object Detection
                obj_det_submenu_options = {
                    '1': "Quick Snapshot",
                    '2': "Continuous Mode",
                    '0': "Back"
                }
                subchoice = get_menu_choice_cv(module_audio_handler, "Object Detection", obj_det_submenu_options)
                
                if subchoice in ['esc', '0']:
                    continue
                    
                if subchoice == "1":
                    if confirm_final_action(module_audio_handler, "snapshot"):
                        cv2.destroyWindow(MENU_WINDOW_NAME)
                        start_time = time.time()
                        run_object_detection_capture(
                            audio_handler=module_audio_handler, duration=CAPTURE_DURATION, camera_id=CAMERA_ID, 
                            model_path=MODEL_PATH_SNAPSHOT, device=DEVICE, conf=CONFIDENCE_THRESHOLD, blur_thresh=BLUR_THRESHOLD
                        )
                        end_time = time.time()
                        log_performance_stats("Quick Snapshot Analysis", start_time, end_time, DEVICE_INFO)
                        cv2.namedWindow(MENU_WINDOW_NAME)
                elif subchoice == "2":
                    if confirm_final_action(module_audio_handler, "continuous mode"):
                        cv2.destroyWindow(MENU_WINDOW_NAME)
                        start_time = time.time()
                        run_realtime_spatial_awareness(
                            audio_handler=module_audio_handler, camera_id=CAMERA_ID, model_path=MODEL_PATH, 
                            device=DEVICE, conf=CONFIDENCE_THRESHOLD, feedback_interval=REALTIME_INTERVAL
                        )
                        end_time = time.time()
                        log_performance_stats("Continuous Item Detection", start_time, end_time, DEVICE_INFO)
                        cv2.namedWindow(MENU_WINDOW_NAME)

            elif choice == '2':  # Object Search
                if confirm_final_action(module_audio_handler, "object search"):
                    cv2.destroyWindow(MENU_WINDOW_NAME)
                    start_time = time.time()
                    run_object_search(
                        audio_handler=module_audio_handler, camera_id=CAMERA_ID, model_path=MODEL_PATH, 
                        device=DEVICE, conf=CONFIDENCE_THRESHOLD, feedback_interval=SEARCH_INTERVAL
                    )
                    end_time = time.time()
                    log_performance_stats("Object Search Mode", start_time, end_time, DEVICE_INFO)
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    time.sleep(1)
                    cv2.namedWindow(MENU_WINDOW_NAME)

            elif choice == '3':  # Scene Description
                if not gemini_ready:
                    if module_audio_handler: 
                        module_audio_handler.speak("Scene description unavailable.")
                    continue
                    
                scene_desc_submenu_options = {
                    '1': "Photo Description",
                    '2': "Video Description",
                    '3': "Quick Caption",
                    '0': "Back"
                }
                subchoice = get_menu_choice_cv(module_audio_handler, "Scene Description", scene_desc_submenu_options)
                
                if subchoice in ['esc', '0']:
                    continue
                    
                if subchoice == "1":
                    if confirm_final_action(module_audio_handler, "photo description"):
                        cv2.destroyWindow(MENU_WINDOW_NAME)
                        start_time = time.time()
                        run_image_scene_description(
                            api_key=GOOGLE_API_KEY, audio_handler=module_audio_handler, duration=CAPTURE_DURATION, 
                            camera_id=CAMERA_ID, blur_thresh=BLUR_THRESHOLD, save_output=True
                        )
                        end_time = time.time()
                        log_performance_stats("Photo Description (Gemini)", start_time, end_time, DEVICE_INFO)
                        cv2.namedWindow(MENU_WINDOW_NAME)
                elif subchoice == "2":
                    if confirm_final_action(module_audio_handler, "video description"):
                        cv2.destroyWindow(MENU_WINDOW_NAME)
                        start_time = time.time()
                        run_video_scene_description(
                            api_key=GOOGLE_API_KEY, audio_handler=module_audio_handler, duration=VIDEO_DESC_DURATION, 
                            camera_id=CAMERA_ID, save_output=True
                        )
                        end_time = time.time()
                        log_performance_stats("Video Description (Gemini)", start_time, end_time, DEVICE_INFO)
                        cv2.namedWindow(MENU_WINDOW_NAME)
                elif subchoice == "3":
                    if confirm_final_action(module_audio_handler, "quick caption"):
                        cv2.destroyWindow(MENU_WINDOW_NAME)
                        start_time = time.time()
                        run_blip_scene_description(
                            audio_handler=module_audio_handler, 
                            duration=CAPTURE_DURATION, 
                            camera_id=CAMERA_ID,
                            model_path="Salesforce/blip-image-captioning-large", 
                            device=DEVICE,
                            blur_thresh=BLUR_THRESHOLD
                        )
                        end_time = time.time()
                        log_performance_stats("Quick Caption Analysis (BLIP)", start_time, end_time, DEVICE_INFO)
                        cv2.namedWindow(MENU_WINDOW_NAME)

    except KeyboardInterrupt:
        logger.info("Ctrl+C detected in main. Exiting Object Detection module.")
        if module_audio_handler: module_audio_handler.speak("Exiting module.")
        running = False
    except Exception as e:
        logger.critical(f"Fatal error in Object Detection module: {e}", exc_info=True)
        if module_audio_handler: module_audio_handler.speak("Critical error occurred.")
        print(f"FATAL ERROR in Object Detection module: {e}")
    finally:
        logger.info("--- Object Detection Module performing final cleanup ---")
        if module_audio_handler:
            logger.info("Stopping AudioFeedbackHandler for modules...")
            if shared_audio_handler is None:
                logger.info("Stopping locally created audio handler.")
                module_audio_handler.stop()
            else:
                logger.info("Not stopping shared audio handler.")
        
        cv2.destroyAllWindows()
        logger.info("All OpenCV windows closed.")
        logger.info("Object Detection & Description Module shutdown complete.")

    print("Object Detection Module shut down.")


if __name__ == "__main__":
    if "YOUR_GOOGLE_API_KEY_HERE" in GOOGLE_API_KEY:
        logger.critical("CRITICAL: GOOGLE_API_KEY is a placeholder in the script. Please replace it.")
    try:
        main()
    except SystemExit as e:
        logger.info(f"Application exited with code {e.code}.")
    except Exception as e:
        print(f"UNHANDLED TOP-LEVEL EXCEPTION: {e}")
        logging.critical(f"UNHANDLED TOP-LEVEL EXCEPTION: {e}", exc_info=True)