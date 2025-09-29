# --- START OF FILE OCR/main_ocr.py ---

import sys
import os
import logging
import time
import cv2
import traceback

# --- This block ensures that the main project directory is on the Python path ---
# This allows us to import 'audio_feedback_vision_assitant'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------------------------------------------------------------

# --- This import should now work correctly ---
from OCR.OCR import OCRProcessor
from audio_feedback_vision_assitant import AudioFeedbackHandler

logger = logging.getLogger(__name__)

def main(shared_audio_handler=None, ocr_processor=None):
    """
    Main function for OCR System. Acts as a bridge to the OCRProcessor class.
    Supports both preloaded and on-demand initialization.
    
    Args:
        shared_audio_handler: Optional shared AudioFeedbackHandler instance
        ocr_processor: Optional preloaded OCRProcessor instance
    
    Returns:
        bool: True if user wants to switch modes, False otherwise
    """
    logger.info("--- Entering OCR Module ---")
    local_audio_handler = None
    switch_mode = False
    
    # Set up audio handler if not provided
    if shared_audio_handler is None:
        logger.warning("No shared audio handler for OCR. Creating a local one for standalone test.")
        try:
            local_audio_handler = AudioFeedbackHandler()
            audio_handler_to_use = local_audio_handler
        except Exception as e:
            logger.error(f"Failed to create local audio handler for OCR: {e}")
            audio_handler_to_use = None
    else:
        logger.info("Using shared audio handler for OCR.")
        audio_handler_to_use = shared_audio_handler

    cap = None
    try:
        if ocr_processor is not None:
            # Use the preloaded OCRProcessor, just set up the camera
            logger.info("Using preloaded OCR processor")
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use DirectShow for faster camera init on Windows
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ocr_processor.cap = cap
            # Skip speaking "OCR system ready" as it's redundant
            switch_mode = ocr_processor.capture_and_process()
        else:
            # Fallback: create a new OCRProcessor
            logger.info("Creating new OCR processor")
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Use DirectShow for faster camera init
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            ocr = OCRProcessor(cap=cap, shared_audio_handler=audio_handler_to_use)
            ocr.speak_text("OCR system ready", "en")
            switch_mode = ocr.capture_and_process()
            
    except Exception as e:
        logger.critical(f"Critical error in OCR system: {e}")
        traceback.print_exc()
        if audio_handler_to_use:
            audio_handler_to_use.speak("A critical error occurred in the OCR system.")
    finally:
        logger.info("--- Exiting OCR Module ---")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("OCR system shutdown")
        
        # If we created a local handler, we are responsible for stopping it
        if local_audio_handler:
            logger.info("Stopping local OCR audio handler.")
            local_audio_handler.stop()
            
        return switch_mode

# This block allows you to test the OCR module by itself
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("Running OCR Module in standalone mode...")
    main()